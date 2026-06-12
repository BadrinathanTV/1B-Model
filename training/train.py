"""
Training Script — Pure BF16 + Stability Suite
===============================================

Native Accelerate training loop with:
  - Liger FusedLinearCrossEntropy (no logit materialization → ~1GB VRAM saved)
  - Z-loss regularization (prevents logit explosion)
  - MTP loss weight annealing (DeepSeek-V3: 0.3→0.1 at 67% training)
  - Per-component gradient norm monitoring (early spike detection)
"""

import argparse
import bisect
import glob
import json
import math
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import time
from dataclasses import asdict
import wandb

# Ensure the script's directory is in the Python path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Local imports are now used because models/ was moved into this directory

import torch
import torch.nn.functional as F

# ── Enable TF32 tensor core acceleration (2-3x faster matmuls on Ampere+) ──
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

from accelerate import Accelerator

from config import SLMConfig
from model import SLMModel
from layers.rope import precompute_freqs_cis, precompute_cos_sin
from optimizers import build_optimizer
from torch.optim.lr_scheduler import LambdaLR

# ─── Fused CE kernel (Liger) ─────────────────────────────────────────────────
try:
    from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction
    LIGER_FUSED_CE = True
except ImportError:
    LIGER_FUSED_CE = False

# Standard CE fallback (uncompiled — torch.compile breaks on dynamic MTP slicing)
_ce_loss_fn = F.cross_entropy


def _fused_ce_loss(hidden_states, lm_head_weight, targets, z_loss_weight=0.0, return_z_loss=False):
    """Compute CE loss via fused kernel — never materializes logits tensor."""
    res = LigerFusedLinearCrossEntropyFunction.apply(
        hidden_states.view(-1, hidden_states.size(-1)),
        lm_head_weight,
        targets.view(-1),
        None,
        None,
        -100,
        z_loss_weight,
        0.0,
        "mean",
        None,
        return_z_loss
    )
    if return_z_loss:
        # Depending on Liger version, res might be a tuple (loss, z_loss)
        return res[0] if isinstance(res, tuple) else res, res[1] if isinstance(res, tuple) else torch.tensor(0.0)
    
    # If not returning z_loss, safely extract the loss whether it's wrapped in a tuple or not
    loss_tensor = res[0] if isinstance(res, tuple) else res
    return loss_tensor


def _standard_ce_loss(hidden_states, lm_head_weight, targets, vocab_size):
    """Fallback: materialize logits then compute CE."""
    logits = F.linear(hidden_states, lm_head_weight)
    return _ce_loss_fn(logits.view(-1, vocab_size), targets.view(-1))


def compute_loss(hidden_states_list, targets, lm_head_weight, config: SLMConfig,
                 step: int = 0) -> tuple[torch.Tensor, dict]:
    """Compute standard next-token prediction CE loss with Z-loss and optional MTP.

    Args:
        hidden_states_list: List of hidden state tensors [main, mtp_1, ...].
            Each is shape (B, S, H). If single tensor, wrap in a list.
        targets: Target token IDs (B, S).
        lm_head_weight: The lm_head weight matrix for fused CE.
        config: Full model config.
        step: Current training step (for MTP loss weight annealing).

    Returns:
        Tuple of (total_loss, metrics_dict).
    """
    use_fused = LIGER_FUSED_CE and hidden_states_list[0].is_cuda
    z_loss = torch.tensor(0.0, device=hidden_states_list[0].device)

    # Standard next-token CE (main head)
    main_hs = hidden_states_list[0]
    if use_fused:
        l_val, z_val = _fused_ce_loss(main_hs, lm_head_weight, targets, config.z_loss_weight, return_z_loss=True)
        z_loss = z_val
        main_loss = l_val - z_val
    else:
        main_loss = _standard_ce_loss(main_hs, lm_head_weight, targets, config.vocab_size)
        if config.z_loss_weight > 0:
            main_logits = F.linear(main_hs.float(), lm_head_weight.float())
            log_z = torch.logsumexp(main_logits, dim=-1)
            z_loss = config.z_loss_weight * (log_z ** 2).mean()

    total_loss = main_loss + z_loss
    mtp_loss = torch.tensor(0.0, device=hidden_states_list[0].device)

    # MTP auxiliary losses (t+2, t+3, ...)
    # Each MTP head i predicts token at position j+i from hidden state at position j.
    # We trim hidden states and targets to align: head i uses hs[:, :-i] vs targets[:, i:]
    # This works whether hidden states are pre-trimmed (from forward_hidden) or raw.
    if len(hidden_states_list) > 1 and config.mtp_depth > 1:
        max_steps = config.training.max_steps
        anneal_frac = 0.67
        if step < max_steps * anneal_frac:
            mtp_weight = 0.3
        else:
            mtp_weight = 0.1

        for i, mtp_hs in enumerate(hidden_states_list[1:], start=1):
            main_len = hidden_states_list[0].shape[1]
            # If mtp_hs is already trimmed (shorter), use it as-is; else trim
            trim = main_len - mtp_hs.shape[1]
            if trim > 0:
                mtp_hs_trimmed = mtp_hs
                mtp_targets = targets[:, i:i + mtp_hs.shape[1]].contiguous()
            else:
                mtp_hs_trimmed = mtp_hs[:, :-i, :].contiguous() if i < main_len else mtp_hs
                mtp_targets = targets[:, i:].contiguous()
            if mtp_hs_trimmed.numel() == 0 or mtp_targets.numel() == 0:
                continue
            if use_fused:
                mtp_ce_res = _fused_ce_loss(mtp_hs_trimmed, lm_head_weight, mtp_targets)
                # Satisfy static type checkers (Pyright/mypy) that might infer a tuple
                mtp_ce = mtp_ce_res[0] if isinstance(mtp_ce_res, tuple) else mtp_ce_res
            else:
                mtp_ce = _standard_ce_loss(mtp_hs_trimmed, lm_head_weight, mtp_targets, config.vocab_size)
            
            # Use type() or isinstance to guarantee it's a tensor for Pyright
            mtp_loss = mtp_loss + mtp_weight * mtp_ce

        total_loss = total_loss + mtp_loss 

    metrics = {
        "main_loss": main_loss.detach(),
        "z_loss": z_loss.detach(),
        "mtp_loss": mtp_loss.detach(),
    }
    return total_loss, metrics


from torch.utils.data import Dataset, DataLoader
import numpy as np

class PretrainingDataset(Dataset):
    """Streams data from memmapped .bin files, falling back to dummy data if empty.

    Standard next-token prediction (NTP).
    """
    def __init__(self, data_dir: str, seq_len: int, vocab_size: int):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.bin")))

        self.chunk_size = seq_len + 1

        self.memmaps = []
        self.cumulative_lengths = [0]

        if self.files:
            for f in self.files:
                m = np.memmap(f, dtype=np.uint16, mode='r')
                self.memmaps.append(m)
                valid_len = len(m) // self.chunk_size
                self.cumulative_lengths.append(self.cumulative_lengths[-1] + valid_len)
            self.total_length = self.cumulative_lengths[-1]
            self.use_dummy = False
        else:
            self.total_length = 1000000 # 1M steps of dummy data
            self.use_dummy = True

    def __len__(self):
        return self.total_length

    def __getitem__(self, index):
        if self.use_dummy:
            inputs = torch.randint(0, self.vocab_size, (self.seq_len,))
            targets = torch.randint(0, self.vocab_size, (self.seq_len,))
            return inputs, targets

        file_idx = bisect.bisect_right(self.cumulative_lengths, index) - 1
        if file_idx >= len(self.files): file_idx = len(self.files) - 1
        local_chunk_idx = index - self.cumulative_lengths[file_idx]
        local_idx = local_chunk_idx * self.chunk_size

        chunk = self.memmaps[file_idx][local_idx : local_idx + self.chunk_size]
        chunk = torch.from_numpy(chunk.astype(np.int64))

        inputs = chunk[:-1]
        targets = chunk[1:]

        return inputs, targets

def evaluate(model, dataloader, accelerator, config, lm_head_weight, freqs_cis, cos_cache, sin_cache, eval_steps=50):
    """Run evaluation and return mean validation loss."""
    model.eval()
    total_loss = 0.0
    steps_run = 0
    data_iter = iter(dataloader)
    with torch.no_grad():
        for _ in range(eval_steps):
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                break
            
            with accelerator.autocast():
                hidden_states_list = model(
                    inputs, freqs_cis=freqs_cis,
                    cos_cache=cos_cache, sin_cache=sin_cache,
                    return_hidden_states=True,
                    use_mtp=True,
                )
                loss, _ = compute_loss(
                    hidden_states_list, targets, lm_head_weight, config,
                )
                # Gather across processes if using multi-GPU
                loss = accelerator.gather(loss).mean()
                total_loss += loss.item()
                steps_run += 1
                
    model.train()
    return total_loss / max(1, steps_run)

def main():
    parser = argparse.ArgumentParser(description="Train the 1B SLM")
    parser.add_argument(
        "--config", type=str, default="training/configs/default.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/mixed_50B_corpus",
        help="Path to pretokenized training dataset",
    )
    parser.add_argument(
        "--val_data_dir", type=str, default=None,
        help="Path to pretokenized validation dataset (optional)",
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints",
        help="Directory to save/load checkpoints",
    )
    parser.add_argument(
        "--checkpoint_interval", type=int, default=500,
        help="Steps between saving checkpoints and running validation",
    )
    parser.add_argument(
        "--compile", action="store_true",
        help="Compile individual TransformerBlocks for higher performance",
    )
    args = parser.parse_args()

    # ── Load config ──
    config = SLMConfig.from_yaml(args.config)
    train_cfg = config.training

    # ── Accelerator ──
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
    )
    accelerator.print(config, flush=True)
    torch.manual_seed(train_cfg.seed)

    # Initialize wandb (only on main process)
    if accelerator.is_main_process:
        wandb.init(
            project="1b-model-pretraining",
            config=asdict(config)
        )

    # ── [1/5] Initialize model in BF16 directly on GPU ──
    accelerator.print("\n[1/5] Initializing model in BF16 on GPU...", flush=True)
    accelerator.print(f"      Init: trunc_normal_(std={config.init_std})", flush=True)
    accelerator.print(f"      Stability: QK-Norm=ON, Z-loss={config.z_loss_weight}, "
                      f"embed_scale={'√d' if config.embed_scale else 'OFF'}", flush=True)
    accelerator.print(f"      Fused CE: {'Liger' if LIGER_FUSED_CE else 'compiled F.cross_entropy'}", flush=True)
    t0 = time.time()
    model = SLMModel(config).to(device=accelerator.device, dtype=torch.bfloat16)
    if args.compile:
        accelerator.print("      Compiling full model (end-to-end)...", flush=True)
        model = torch.compile(model)
    param_count = sum(p.numel() for p in model.parameters())
    vram_mb = torch.cuda.memory_allocated() / 1024**2
    accelerator.print(f"      Done in {time.time()-t0:.1f}s — {param_count/1e9:.3f}B params, {vram_mb:.0f} MB VRAM", flush=True)

    # ── [2/5] Build optimizer ──
    accelerator.print("[2/5] Building optimizer...", flush=True)
    optimizer = build_optimizer(model, config)
    
    def get_custom_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, initial_lr_fraction=0.1):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return initial_lr_fraction + (1.0 - initial_lr_fraction) * float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return LambdaLR(optimizer, lr_lambda)

    scheduler = get_custom_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.optimizer.warmup_steps,
        num_training_steps=config.training.max_steps,
        initial_lr_fraction=0.1  # Starts at 10% of base_lr instead of 0.0
    )
    accelerator.print("      Done.", flush=True)

    # ── [3/5] Prepare via Accelerator ──
    accelerator.print("[3/5] Preparing model and optimizer...", flush=True)
    t0 = time.time()

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    vram_mb = torch.cuda.memory_allocated() / 1024**2
    accelerator.print(f"      Done in {time.time()-t0:.1f}s — VRAM after prepare: {vram_mb:.0f} MB", flush=True)

    # ── [4/5] Precompute RoPE ──
    seq_len = train_cfg.seq_len
    accelerator.print("[4/5] Precomputing RoPE...", flush=True)
    freqs_cis = precompute_freqs_cis(
        dim=config.qk_rope_head_dim,
        end=seq_len,
        theta=config.rope_theta,
        device=accelerator.device,
        yarn_scale=config.yarn_scale_factor if config.use_yarn else 1.0,
        yarn_beta_fast=config.yarn_beta_fast,
        yarn_beta_slow=config.yarn_beta_slow,
        yarn_orig_ctx=config.yarn_original_context,
    )
    # Also precompute cos/sin for Liger Kernel RoPE (4x less memory)
    cos_cache, sin_cache = precompute_cos_sin(
        dim=config.qk_rope_head_dim,
        end=seq_len,
        theta=config.rope_theta,
        device=accelerator.device,
        yarn_scale=config.yarn_scale_factor if config.use_yarn else 1.0,
        yarn_beta_fast=config.yarn_beta_fast,
        yarn_beta_slow=config.yarn_beta_slow,
        yarn_orig_ctx=config.yarn_original_context,
    )
    accelerator.print("      Done.\n", flush=True)

    # ── [4.5] Dataset & DataLoader ──
    accelerator.print("[4.5] Preparing Dataset & DataLoader...", flush=True)

    dataset = PretrainingDataset(
        data_dir=args.data_dir,
        seq_len=seq_len,
        vocab_size=config.vocab_size,
    )
    if dataset.use_dummy:
        accelerator.print("      Warning: No .bin files found in data_dir, falling back to dummy data.", flush=True)

    def make_dataloader(ds, shuffle=True):
        dl = DataLoader(
            ds,
            batch_size=train_cfg.batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )
        return accelerator.prepare(dl)

    dataloader = make_dataloader(dataset, shuffle=True)
    
    # ── Validation Dataloaders ──
    dataloader_val = None
    if args.val_data_dir and os.path.exists(args.val_data_dir):
        accelerator.print("      Setting up validation dataloaders...", flush=True)
        dataset_val = PretrainingDataset(
            data_dir=args.val_data_dir,
            seq_len=seq_len,
            vocab_size=config.vocab_size,
        )
        dataloader_val = make_dataloader(dataset_val, shuffle=False)

    accelerator.print("      Done.\n", flush=True)

    # ── Checkpoint Resume ──
    global_step = 0
    resume_dir = None
    if os.path.exists(args.checkpoint_dir):
        # Find latest checkpoint-X folder
        subdirs = glob.glob(os.path.join(args.checkpoint_dir, "checkpoint-*"))
        if subdirs:
            try:
                subdirs = sorted(subdirs, key=lambda x: int(x.split("-")[-1]))
                resume_dir = subdirs[-1]
                global_step = int(resume_dir.split("-")[-1])
            except ValueError:
                resume_dir = None
        
        if resume_dir:
            try:
                accelerator.load_state(resume_dir)
                accelerator.print(f"Resumed from checkpoint: {resume_dir} at step {global_step}", flush=True)
            except Exception as e:
                accelerator.print(f"Could not load checkpoint from {resume_dir}: {e}", flush=True)
                global_step = 0

    # ── Get lm_head weight for fused CE ──
    unwrapped_model = accelerator.unwrap_model(model)
    lm_head_weight = unwrapped_model.lm_head.weight

    # ── Training loop ──
    grad_accum_steps = getattr(train_cfg, 'gradient_accumulation_steps', 8)
    
    accelerator.print(
        f"Starting training loop ({train_cfg.max_steps} steps, "
        f"grad_accum={grad_accum_steps}, effective_batch={train_cfg.batch_size * grad_accum_steps})...",
        flush=True,
    )

    model.train()
    data_iter = iter(dataloader)

    for step in range(global_step, train_cfg.max_steps):
        # Periodic MTP Curriculum: Interleave MTP training within each checkpoint interval
        interval = args.checkpoint_interval
        mtp_start_step_in_interval = int(interval * 0.8)
        is_mtp_phase = (step % interval) >= mtp_start_step_in_interval
        
        # Log transition within the interval
        if (step % interval) == mtp_start_step_in_interval:
            accelerator.print(f"\n>>> [Step {step}] PERIODIC CURRICULUM: ENTERING MTP PHASE (MTP ON) <<<", flush=True)
        elif (step % interval) == 0 and step > 0:
            accelerator.print(f"\n>>> [Step {step}] PERIODIC CURRICULUM: ENTERING BASE PHASE (MTP OFF) <<<", flush=True)

        t_step = time.time()
        accum_loss = torch.tensor(0.0, device=accelerator.device)
        accum_metrics = {}
        grad_norm = None

        for micro_step in range(grad_accum_steps):
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                inputs, targets = next(data_iter)

            # Mixed-precision context + Gradient accumulation
            with accelerator.autocast(), accelerator.accumulate(model):
                # Return hidden states for fused CE (no logit materialization)
                hidden_states_list = model(
                    inputs, freqs_cis=freqs_cis,
                    cos_cache=cos_cache, sin_cache=sin_cache,
                    return_hidden_states=True,
                    use_mtp=config.use_mtp and is_mtp_phase,
                    target_ids=targets,
                )
                loss, metrics = compute_loss(
                    hidden_states_list, targets, lm_head_weight, config, step=step
                )
                accelerator.backward(loss)
                accum_loss = accum_loss + loss.detach()

                # Accumulate metrics (tensors stay on GPU, no sync)
                for k, v in metrics.items():
                    if isinstance(v, torch.Tensor):
                        accum_metrics[k] = accum_metrics.get(k, torch.tensor(0.0, device=accelerator.device)) + v
                    else:
                        accum_metrics[k] = accum_metrics.get(k, 0.0) + v

                # Gradient clipping + optimizer step
                if accelerator.sync_gradients:
                    # Clip and capture the total grad norm
                    if train_cfg.gradient_clip > 0:
                        grad_norm = accelerator.clip_grad_norm_(
                            model.parameters(), train_cfg.gradient_clip
                        )
                    else:
                        grad_norm = None

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
        
        # Check for catastrophic divergence (Auto-Kill Switch)
        step_loss_val = (accum_loss / grad_accum_steps).item()
        if step_loss_val > 20.0 or math.isnan(step_loss_val):
            raise RuntimeError(
                f"Catastrophic Divergence at step {step + 1}. Loss: {step_loss_val:.4f}. Auto-killing run."
            )

        # Logging — .item() calls only happen here (once per log_interval), not every step
        if (step + 1) % train_cfg.log_interval == 0:
            vram = torch.cuda.memory_allocated() / 1024**2
            elapsed = time.time() - t_step
            tokens = train_cfg.batch_size * grad_accum_steps * seq_len
            tps = tokens / elapsed
            
            avg_loss = (accum_loss / grad_accum_steps).item()
            main_loss_val = (accum_metrics.get('main_loss', torch.tensor(0.0)) / grad_accum_steps).item()
            z_loss_val = (accum_metrics.get('z_loss', torch.tensor(0.0)) / grad_accum_steps).item()
            
            mtp_loss_val = (accum_metrics.get('mtp_loss', torch.tensor(0.0)) / grad_accum_steps).item()
            
            # --- Internal Weight Debugging ---
            with torch.no_grad():
                unwrapped = accelerator.unwrap_model(model)
                embed_norm = unwrapped.embed.word_embeddings.weight.float().norm(2).item() if hasattr(unwrapped, 'embed') else 0.0
                lm_head_norm = unwrapped.lm_head.weight.float().norm(2).item() if hasattr(unwrapped, 'lm_head') else 0.0
                total_w_norm = sum(p.float().norm(2).item()**2 for p in unwrapped.parameters() if p.requires_grad) ** 0.5
            
            accelerator.print(
                f"  Step {step+1}/{train_cfg.max_steps} | "
                f"Loss: {avg_loss:.4f} | "
                f"Main: {main_loss_val:.4f} | "
                f"Z: {z_loss_val:.6f} | "
                f"MTP: {mtp_loss_val:.4f} | "
                f"|W|: {total_w_norm:.2f} | "
                f"|Emb|: {embed_norm:.2f} | "
                f"TPS: {tps:.0f} tok/s | "
                f"Time: {elapsed:.2f}s",
                flush=True,
            )
            if accelerator.is_main_process:
                # Log Python floats, not GPU tensors (avoids CPU-GPU sync)
                log_dict = {
                    "loss": avg_loss,
                    "main_loss": main_loss_val,
                    "z_loss": z_loss_val,
                    "mtp_loss": mtp_loss_val,
                    "vram_mb": vram,
                    "tps": tps,
                    "elapsed_time": elapsed,
                    "debug/weight_norm_total": total_w_norm,
                    "debug/weight_norm_embed": embed_norm,
                    "debug/weight_norm_lm_head": lm_head_norm,
                }
                if grad_norm is not None:
                    if isinstance(grad_norm, torch.Tensor):
                        log_dict["grad_norm"] = grad_norm.item()
                    else:
                        log_dict["grad_norm"] = float(grad_norm)
                
                # Extract dynamic learning rate
                if hasattr(optimizer, "param_groups") and len(optimizer.param_groups) > 0:
                    dyn_lr = scheduler.get_last_lr()[0]
                    log_dict["optim/dynamic_lr"] = float(dyn_lr)
                    
                wandb.log(log_dict, step=step + 1)

        # Checkpointing and Validation
        if (step + 1) % args.checkpoint_interval == 0:
            accelerator.wait_for_everyone()
            # Save Checkpoint
            ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint-{step+1}")
            accelerator.save_state(ckpt_path)
            # Explicitly save as .pt
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), os.path.join(ckpt_path, "model.pt"))
            # Save training state for robust resume
            if accelerator.is_main_process:
                training_state = {
                    "global_step": step + 1,
                }
                with open(os.path.join(ckpt_path, "training_state.json"), "w") as f:
                    json.dump(training_state, f)
            accelerator.print(f"\nSaved checkpoint to {ckpt_path} at step {step+1}", flush=True)
            
            # Run Validation
            if dataloader_val is not None:
                accelerator.print("Running validation...", flush=True)
                val_loss = evaluate(
                    model, dataloader_val, accelerator, config, lm_head_weight,
                    freqs_cis, cos_cache, sin_cache, eval_steps=50
                )
                accelerator.print(f"  Validation Loss: {val_loss:.4f}\n", flush=True)
                if accelerator.is_main_process:
                    wandb.log({"val_loss": val_loss}, step=step + 1)

    # ── Save final checkpoint (Fix #9: don't lose last N steps) ──
    accelerator.wait_for_everyone()

    final_step = train_cfg.max_steps
    ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint-{final_step}")
    accelerator.save_state(ckpt_path)
    
    # Explicitly save final model as .pt
    unwrapped_model = accelerator.unwrap_model(model)
    torch.save(unwrapped_model.state_dict(), os.path.join(ckpt_path, "model.pt"))

    if accelerator.is_main_process:
        training_state = {
            "global_step": final_step,
        }
        with open(os.path.join(ckpt_path, "training_state.json"), "w") as f:
            json.dump(training_state, f)

    accelerator.print(f"\n✓ Training complete! Final checkpoint saved to {ckpt_path}", flush=True)
    if accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()
