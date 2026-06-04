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

# ─── Fused CE kernel (Liger) ─────────────────────────────────────────────────
try:
    from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction
    LIGER_FUSED_CE = True
except ImportError:
    LIGER_FUSED_CE = False

# Compiled CE fallback
_compiled_ce = torch.compile(F.cross_entropy)


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
        return res[0], res[1]
    return res[0]


def _standard_ce_loss(hidden_states, lm_head_weight, targets, vocab_size):
    """Fallback: materialize logits then compute CE."""
    logits = F.linear(hidden_states, lm_head_weight)
    return _compiled_ce(logits.view(-1, vocab_size), targets.view(-1))


def compute_loss(hidden_states_list, targets, lm_head_weight, config: SLMConfig,
                 step: int, is_superposition: bool = False) -> tuple[torch.Tensor, dict]:
    """Compute LM loss with Z-loss, MTP auxiliary losses, MTP annealing, and TST MCE.

    Args:
        hidden_states_list: List of hidden states from MTP module.
        targets: Target token IDs.
            - Superposition phase: (batch, seq_len, tst_group_size) — multi-hot bags
            - Recovery phase: (batch, seq_len) — single next-token
        lm_head_weight: The lm_head weight matrix for fused CE.
        config: Full model config.
        step: Current training step (for MTP annealing).
        is_superposition: Whether we're in TST superposition phase.

    Returns:
        Tuple of (total_loss, metrics_dict).
    """
    main_h = hidden_states_list[0]
    use_fused = LIGER_FUSED_CE and main_h.is_cuda

    z_loss = torch.tensor(0.0, device=main_h.device)

    # ── Main loss ──
    if is_superposition and targets.dim() == 3:
        # TST Multi-Hot Cross-Entropy (MCE)
        # targets shape: (B, S, s)
        s = targets.shape[-1]
        
        if use_fused:
            # Optimize: expand hidden states to match targets and batch the Liger call
            # main_h: (B, S, H) -> (B, S, s, H)
            main_h_expanded = main_h.unsqueeze(2).expand(-1, -1, s, -1).contiguous()
            t_flattened = targets.contiguous()
            
            l_val, z_val = _fused_ce_loss(
                main_h_expanded, lm_head_weight, t_flattened, 
                config.z_loss_weight / s, return_z_loss=True
            )
            # Liger returns the mean over (B*S*s). Since MCE = (1/s) sum CE, 
            # the mean over all items naturally scales by 1/s mathematically.
            z_loss = z_loss + (z_val * s)  # z_val is scaled by config.z_loss_weight/s, so we multiply back
            main_loss = l_val - z_val
        else:
            mce_loss = torch.tensor(0.0, device=main_h.device)
            for j in range(s):
                t_j = targets[:, :, j].contiguous()  # (B, S)
                mce_loss = mce_loss + _standard_ce_loss(main_h, lm_head_weight, t_j, config.vocab_size)
            main_loss = mce_loss / s
    else:
        # Standard next-token CE
        if use_fused:
            l_val, z_val = _fused_ce_loss(main_h, lm_head_weight, targets, config.z_loss_weight, return_z_loss=True)
            z_loss = z_val
            main_loss = l_val - z_val
        else:
            main_loss = _standard_ce_loss(main_h, lm_head_weight, targets, config.vocab_size)

    # ── Z-loss for standard path ──
    if not use_fused and config.z_loss_weight > 0:
        # Materialize logits with full gradient flow for Z-loss regularization
        main_logits = F.linear(main_h.float(), lm_head_weight.float())
        log_z = torch.logsumexp(main_logits, dim=-1)
        z_loss = config.z_loss_weight * (log_z ** 2).mean()

    # ── MTP auxiliary losses with annealing ──
    mtp_loss = torch.tensor(0.0, device=main_h.device)
    mtp_z_loss_total = torch.tensor(0.0, device=main_h.device)
    train_cfg = config.training
    anneal_step = int(train_cfg.max_steps * train_cfg.mtp_anneal_fraction)
    if step < anneal_step:
        mtp_weight = train_cfg.mtp_loss_weight
    else:
        mtp_weight = train_cfg.mtp_loss_weight_final

    if len(hidden_states_list) > 1:
        for i in range(1, len(hidden_states_list)):
            mtp_h = hidden_states_list[i]
            if mtp_h.shape[1] == targets.shape[1]:
                mtp_h = mtp_h[:, :-i, :].contiguous()
            else:
                mtp_h = mtp_h.contiguous()
            if mtp_h.numel() == 0:
                continue
            if is_superposition and targets.dim() == 3:
                # TST Multi-Hot Cross-Entropy (MCE) for MTP
                mtp_targets = targets[:, i:, :].contiguous()
                if mtp_targets.numel() == 0:
                    continue
                s_mtp = mtp_targets.shape[-1]
                
                if use_fused:
                    # Optimize: expand hidden states for MTP
                    mtp_h_expanded = mtp_h.unsqueeze(2).expand(-1, -1, s_mtp, -1).contiguous()
                    t_flattened = mtp_targets.contiguous()
                    
                    l_val, z_val = _fused_ce_loss(
                        mtp_h_expanded, lm_head_weight, t_flattened, 
                        config.z_loss_weight / s_mtp, return_z_loss=True
                    )
                    mtp_z_loss_total = mtp_z_loss_total + (z_val * s_mtp)
                    mtp_loss = mtp_loss + (l_val - z_val)
                else:
                    mtp_mce = torch.tensor(0.0, device=main_h.device)
                    for j in range(s_mtp):
                        t_j = mtp_targets[:, :, j].contiguous()
                        mtp_mce = mtp_mce + _standard_ce_loss(mtp_h, lm_head_weight, t_j, config.vocab_size)
                    
                    if config.z_loss_weight > 0:
                        mtp_logits = F.linear(mtp_h.float(), lm_head_weight.float())
                        log_z_mtp = torch.logsumexp(mtp_logits, dim=-1)
                        mtp_z_loss_total = mtp_z_loss_total + config.z_loss_weight * (log_z_mtp ** 2).mean()
                    mtp_loss = mtp_loss + (mtp_mce / s_mtp)
            else:
                # Standard next-token CE for MTP
                mtp_targets = targets[:, i:].contiguous()
                if mtp_targets.numel() == 0:
                    continue
                if use_fused:
                    l_val, z_val = _fused_ce_loss(mtp_h, lm_head_weight, mtp_targets, config.z_loss_weight, return_z_loss=True)
                    mtp_z_loss_total = mtp_z_loss_total + z_val
                    mtp_loss = mtp_loss + (l_val - z_val)
                else:
                    mtp_loss = mtp_loss + _standard_ce_loss(mtp_h, lm_head_weight, mtp_targets, config.vocab_size)
                    if config.z_loss_weight > 0:
                        mtp_logits = F.linear(mtp_h.float(), lm_head_weight.float())
                        log_z_mtp = torch.logsumexp(mtp_logits, dim=-1)
                        z_val = config.z_loss_weight * (log_z_mtp ** 2).mean()
                        mtp_z_loss_total = mtp_z_loss_total + z_val

    # Scale and add MTP's Z-loss to the total Z-loss metric
    z_loss = z_loss + mtp_weight * mtp_z_loss_total
    total_loss = main_loss + z_loss + mtp_weight * mtp_loss

    # Return raw tensors — .item() causes a CPU-GPU sync that stalls the pipeline.
    # Metrics are extracted at logging time only.
    metrics = {
        "main_loss": main_loss.detach(),
        "z_loss": z_loss.detach(),
        "mtp_loss": mtp_loss.detach(),
        "mtp_weight": mtp_weight,
    }
    return total_loss, metrics


from torch.utils.data import Dataset, DataLoader
import numpy as np

class PretrainingDataset(Dataset):
    """Streams data from memmapped .bin files, falling back to dummy data if empty.

    Supports two TST phases:
      - Superposition phase: inputs are bags of s tokens averaged in embedding layer,
        targets are the FULL next bag of s tokens for MCE loss (shape: seq_len × s).
      - Recovery phase: standard next-token prediction (tst_group_size=1).
    """
    def __init__(self, data_dir: str, input_seq_len: int, seq_len: int,
                 vocab_size: int, tst_group_size: int = 1):
        self.input_seq_len = input_seq_len
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.tst_group_size = tst_group_size
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.bin")))

        # In superposition phase, we need input_seq_len + group_size tokens
        # (the extra group_size tokens form the last target bag)
        self.chunk_size = input_seq_len + tst_group_size

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
        s = self.tst_group_size

        if self.use_dummy:
            if s > 1:
                inputs = torch.randint(0, self.vocab_size, (self.seq_len, s))
                targets = torch.randint(0, self.vocab_size, (self.seq_len, s))
            else:
                inputs = torch.randint(0, self.vocab_size, (self.input_seq_len,))
                targets = torch.randint(0, self.vocab_size, (self.seq_len,))
            return inputs, targets

        file_idx = bisect.bisect_right(self.cumulative_lengths, index) - 1
        if file_idx >= len(self.files): file_idx = len(self.files) - 1
        local_chunk_idx = index - self.cumulative_lengths[file_idx]
        local_idx = local_chunk_idx * self.chunk_size

        chunk = self.memmaps[file_idx][local_idx : local_idx + self.chunk_size]
        chunk = torch.from_numpy(chunk.astype(np.int64))

        # Inputs: first input_seq_len tokens (folded into (seq_len, group_size) if s > 1)
        inputs = chunk[:self.input_seq_len]
        if s > 1:
            inputs = inputs.view(self.seq_len, s)

        if s > 1:
            # TST Superposition Phase: Multi-hot targets
            # For each output position i, the target is the next bag of s tokens
            # Position i corresponds to input bag [i*s .. (i+1)*s-1],
            # so its target bag is [(i+1)*s .. (i+2)*s-1]
            # This gives us seq_len target bags, each of size s
            targets = []
            for i in range(self.seq_len):
                start = (i + 1) * s
                end = start + s
                targets.append(chunk[start:end])
            targets = torch.stack(targets, dim=0)  # (seq_len, s)
        else:
            # Recovery Phase: Standard next-token prediction
            targets = chunk[1 : self.seq_len + 1]

        return inputs, targets

def evaluate(model, dataloader, accelerator, config, lm_head_weight, is_superposition, freqs_cis, cos_cache, sin_cache, eval_steps=50):
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
                    use_mtp=config.use_mtp,
                    return_hidden_states=True,
                    tst_group_size=config.tst_group_size if is_superposition else 1,
                    target_ids=targets,
                )
                loss, _ = compute_loss(
                    hidden_states_list, targets, lm_head_weight, config, 0,
                    is_superposition=is_superposition,
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
        accelerator.print("      Compiling individual TransformerBlocks...", flush=True)
        for i in range(len(model.layers)):
            model.layers[i] = torch.compile(model.layers[i])  # type: ignore
    param_count = sum(p.numel() for p in model.parameters())
    vram_mb = torch.cuda.memory_allocated() / 1024**2
    accelerator.print(f"      Done in {time.time()-t0:.1f}s — {param_count/1e9:.3f}B params, {vram_mb:.0f} MB VRAM", flush=True)

    # ── [2/5] Build optimizer ──
    accelerator.print("[2/5] Building optimizer...", flush=True)
    optimizer = build_optimizer(model, config)
    accelerator.print("      Done.", flush=True)

    # ── [3/5] Prepare via Accelerator ──
    accelerator.print("[3/5] Preparing model and optimizer...", flush=True)
    t0 = time.time()

    model, optimizer = accelerator.prepare(model, optimizer)

    vram_mb = torch.cuda.memory_allocated() / 1024**2
    accelerator.print(f"      Done in {time.time()-t0:.1f}s — VRAM after prepare: {vram_mb:.0f} MB", flush=True)

    # ── [4/5] Precompute RoPE ──
    seq_len = train_cfg.seq_len
    input_seq_len = seq_len * config.tst_group_size
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
    # ── TST Two-Phase Setup ──
    tst_superposition_step = int(train_cfg.max_steps * config.tst_superposition_ratio)
    is_superposition = config.tst_group_size > 1  # Start in superposition if group_size > 1

    accelerator.print("[4.5] Preparing Dataset & DataLoader...", flush=True)
    if is_superposition:
        accelerator.print(f"      TST Two-Phase: Superposition (steps 0-{tst_superposition_step}) → Recovery (steps {tst_superposition_step}-{train_cfg.max_steps})", flush=True)

    # Phase 1 dataset (Superposition): bags of s tokens with MCE targets
    dataset_super = PretrainingDataset(
        data_dir=args.data_dir,
        input_seq_len=input_seq_len,
        seq_len=seq_len,
        vocab_size=config.vocab_size,
        tst_group_size=config.tst_group_size,
    )
    # Phase 2 dataset (Recovery): standard next-token targets
    dataset_recovery = PretrainingDataset(
        data_dir=args.data_dir,
        input_seq_len=seq_len,  # No superposition — 1 token per position
        seq_len=seq_len,
        vocab_size=config.vocab_size,
        tst_group_size=1,
    )
    if dataset_super.use_dummy:
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

    dataloader_super = make_dataloader(dataset_super, shuffle=True)
    dataloader_recovery = make_dataloader(dataset_recovery, shuffle=True)
    
    # ── Validation Dataloaders ──
    dataloader_val_super = None
    dataloader_val_recovery = None
    if args.val_data_dir and os.path.exists(args.val_data_dir):
        accelerator.print("      Setting up validation dataloaders...", flush=True)
        if is_superposition:
            dataset_val_super = PretrainingDataset(
                data_dir=args.val_data_dir,
                input_seq_len=input_seq_len,
                seq_len=seq_len,
                vocab_size=config.vocab_size,
                tst_group_size=config.tst_group_size,
            )
            dataloader_val_super = make_dataloader(dataset_val_super, shuffle=False)

        dataset_val_recovery = PretrainingDataset(
            data_dir=args.val_data_dir,
            input_seq_len=seq_len,
            seq_len=seq_len,
            vocab_size=config.vocab_size,
            tst_group_size=1,
        )
        dataloader_val_recovery = make_dataloader(dataset_val_recovery, shuffle=False)

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

    # Schedule-Free optimizers need .train() mode
    if hasattr(optimizer, 'train'):
        optimizer.train()

    model.train()

    # Determine correct TST phase based on resume step
    if global_step >= tst_superposition_step:
        current_phase = "recovery"
        data_iter = iter(dataloader_recovery)
        phase_switched = True
        if global_step > 0:
            accelerator.print(f"  Resumed in RECOVERY phase (step {global_step} >= transition step {tst_superposition_step})", flush=True)
    else:
        current_phase = "superposition" if is_superposition else "recovery"
        data_iter = iter(dataloader_super if is_superposition else dataloader_recovery)
        phase_switched = False

    for step in range(global_step, train_cfg.max_steps):
        # ── TST Phase Transition Check ──
        if is_superposition and not phase_switched and step >= tst_superposition_step:
            accelerator.print(
                f"\n═══ TST Phase Transition at step {step} ═══\n"
                f"    Switching from SUPERPOSITION (MCE, group_size={config.tst_group_size}) "
                f"→ RECOVERY (standard CE, group_size=1)\n",
                flush=True,
            )
            current_phase = "recovery"
            data_iter = iter(dataloader_recovery)
            phase_switched = True

        in_superposition = (current_phase == "superposition")

        t_step = time.time()
        accum_loss = torch.tensor(0.0, device=accelerator.device)
        accum_metrics = {}
        grad_norm = None

        for micro_step in range(grad_accum_steps):
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                dl = dataloader_super if in_superposition else dataloader_recovery
                data_iter = iter(dl)
                inputs, targets = next(data_iter)

            # Mixed-precision context + Gradient accumulation
            with accelerator.autocast(), accelerator.accumulate(model):
                # Return hidden states for fused CE (no logit materialization)
                hidden_states_list = model(
                    inputs, freqs_cis=freqs_cis,
                    cos_cache=cos_cache, sin_cache=sin_cache,
                    use_mtp=config.use_mtp,
                    return_hidden_states=True,
                    tst_group_size=config.tst_group_size if in_superposition else 1,
                    target_ids=targets,
                )
                loss, metrics = compute_loss(
                    hidden_states_list, targets, lm_head_weight, config, step,
                    is_superposition=in_superposition,
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

                    # Provide closure to fetch loss for Polyak step-size
                    def closure():
                        return accum_loss / grad_accum_steps
                    optimizer.step(closure=closure)
                    optimizer.zero_grad(set_to_none=True)

        # Logging — .item() calls only happen here (once per log_interval), not every step
        if (step + 1) % train_cfg.log_interval == 0:
            vram = torch.cuda.memory_allocated() / 1024**2
            elapsed = time.time() - t_step
            # Fix #2: Use correct group_size based on current phase
            effective_group_size = config.tst_group_size if in_superposition else 1
            tokens = train_cfg.batch_size * grad_accum_steps * seq_len * effective_group_size
            tps = tokens / elapsed
            # Extract metrics to CPU only at log time
            avg_loss = (accum_loss / grad_accum_steps).item()
            main_loss_val = (accum_metrics.get('main_loss', torch.tensor(0.0)) / grad_accum_steps).item()
            z_loss_val = (accum_metrics.get('z_loss', torch.tensor(0.0)) / grad_accum_steps).item()
            mtp_loss_val = (accum_metrics.get('mtp_loss', torch.tensor(0.0)) / grad_accum_steps).item()
            # Calculate current MTP weight based on step
            anneal_step = int(train_cfg.max_steps * train_cfg.mtp_anneal_fraction)
            mtp_weight_val = train_cfg.mtp_loss_weight if step < anneal_step else train_cfg.mtp_loss_weight_final
            accelerator.print(
                f"  Step {step+1}/{train_cfg.max_steps} | "
                f"Loss: {avg_loss:.4f} | "
                f"Main: {main_loss_val:.4f} | "
                f"Z: {z_loss_val:.6f} | "
                f"MTP: {mtp_loss_val:.4f} (w={mtp_weight_val:.2f}) | "
                f"VRAM: {vram:.0f} MB | "
                f"TPS: {tps:.0f} tok/s | "
                f"Time: {elapsed:.2f}s",
                flush=True,
            )
            if accelerator.is_main_process:
                # Fix #15: Log Python floats, not GPU tensors (avoids CPU-GPU sync)
                log_dict = {
                    "loss": avg_loss,
                    "main_loss": main_loss_val,
                    "z_loss": z_loss_val,
                    "mtp_loss": mtp_loss_val,
                    "mtp_weight": mtp_weight_val,
                    "vram_mb": vram,
                    "tps": tps,
                    "elapsed_time": elapsed,
                    "phase": current_phase,
                }
                # Gradient norm monitoring (spike detection)
                if grad_norm is not None:
                    if isinstance(grad_norm, torch.Tensor):
                        log_dict["grad_norm"] = grad_norm.item()
                    else:
                        log_dict["grad_norm"] = float(grad_norm)
                wandb.log(log_dict, step=step + 1)

        # Checkpointing and Validation
        if (step + 1) % args.checkpoint_interval == 0:
            accelerator.wait_for_everyone()
            # Switch to eval mode for Schedule-Free averaged weights
            if hasattr(optimizer, 'eval'):
                optimizer.eval()

            # Save Checkpoint
            ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint-{step+1}")
            accelerator.save_state(ckpt_path)
            # Save training state for robust resume
            if accelerator.is_main_process:
                training_state = {
                    "global_step": step + 1,
                    "phase": current_phase,
                    "phase_switched": phase_switched,
                }
                with open(os.path.join(ckpt_path, "training_state.json"), "w") as f:
                    json.dump(training_state, f)
            accelerator.print(f"\nSaved checkpoint to {ckpt_path} at step {step+1}", flush=True)
            
            # Run Validation
            val_loader = dataloader_val_super if (in_superposition and dataloader_val_super is not None) else dataloader_val_recovery
            if val_loader is not None:
                accelerator.print(f"Running validation ({'superposition' if in_superposition else 'recovery'})...", flush=True)
                val_loss = evaluate(
                    model, val_loader, accelerator, config, lm_head_weight,
                    in_superposition, freqs_cis, cos_cache, sin_cache, eval_steps=50
                )
                accelerator.print(f"  Validation Loss ({'superposition' if in_superposition else 'recovery'}): {val_loss:.4f}\n", flush=True)
                if accelerator.is_main_process:
                    wandb.log({"val_loss": val_loss}, step=step + 1)

            # Switch back to train mode
            if hasattr(optimizer, 'train'):
                optimizer.train()

    # ── Save final checkpoint (Fix #9: don't lose last N steps) ──
    accelerator.wait_for_everyone()
    if hasattr(optimizer, 'eval'):
        optimizer.eval()

    final_step = train_cfg.max_steps
    ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint-{final_step}")
    accelerator.save_state(ckpt_path)
    if accelerator.is_main_process:
        training_state = {
            "global_step": final_step,
            "phase": current_phase,
            "phase_switched": phase_switched,
        }
        with open(os.path.join(ckpt_path, "training_state.json"), "w") as f:
            json.dump(training_state, f)

    accelerator.print(f"\n✓ Training complete! Final checkpoint saved to {ckpt_path}", flush=True)
    if accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()
