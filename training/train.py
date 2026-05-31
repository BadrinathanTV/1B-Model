"""
Training Script — Pure BF16
============================

Native Accelerate training loop with standard BF16 and gradient checkpointing.
"""

import argparse
import os
import sys
import time
from dataclasses import asdict
import wandb

# Ensure the script's directory is in the Python path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Local imports are now used because models/ was moved into this directory

import torch
import torch.nn.functional as F

from accelerate import Accelerator

from config import SLMConfig
from model import SLMModel
from layers.rope import precompute_freqs_cis, precompute_cos_sin
from optimizers import build_optimizer


def compute_loss(logits_list, targets, config: SLMConfig) -> torch.Tensor:
    """Compute LM loss with MTP auxiliary losses."""
    main_logits = logits_list[0]
    loss = _compiled_ce(main_logits.view(-1, config.vocab_size), targets.view(-1))

    if len(logits_list) > 1:
        mtp_weight = config.training.mtp_loss_weight
        for i in range(1, len(logits_list)):
            mtp_logits = logits_list[i][:, :-i, :].contiguous()
            mtp_targets = targets[:, i:].contiguous()
            if mtp_logits.numel() == 0 or mtp_targets.numel() == 0:
                continue
            mtp_loss = _compiled_ce(
                mtp_logits.view(-1, config.vocab_size),
                mtp_targets.view(-1),
            )
            loss += mtp_weight * mtp_loss

    return loss


# Wrap the loss function with torch.compile (2.1x faster CrossEntropy, 0 MB spike)
_compiled_ce = torch.compile(F.cross_entropy)


import glob
from torch.utils.data import Dataset, DataLoader
import numpy as np

class PretrainingDataset(Dataset):
    """Streams data from memmapped .bin files, falling back to dummy data if empty."""
    def __init__(self, data_dir: str, input_seq_len: int, seq_len: int, vocab_size: int):
        self.input_seq_len = input_seq_len
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.bin")))
        
        self.memmaps = []
        self.cumulative_lengths = [0]
        
        # Need enough tokens for input_seq_len + 1 (for targets)
        self.chunk_size = max(input_seq_len, seq_len) + 1 
        
        if self.files:
            for f in self.files:
                m = np.memmap(f, dtype=np.uint16, mode='r')
                self.memmaps.append(m)
                valid_len = max(0, len(m) - self.chunk_size)
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
            inputs = torch.randint(0, self.vocab_size, (self.input_seq_len,))
            targets = torch.randint(0, self.vocab_size, (self.seq_len,))
            return inputs, targets
            
        import bisect
        file_idx = bisect.bisect_right(self.cumulative_lengths, index) - 1
        if file_idx >= len(self.files): file_idx = len(self.files) - 1
        local_idx = index - self.cumulative_lengths[file_idx]
        
        chunk = self.memmaps[file_idx][local_idx : local_idx + self.chunk_size]
        chunk = torch.from_numpy(chunk.astype(np.int64))
        
        # Simple alignment: inputs are the first input_seq_len tokens
        # For TST, target is the token following each group
        inputs = chunk[:self.input_seq_len]
        
        group_size = self.input_seq_len // self.seq_len
        if group_size > 1:
            targets = chunk[group_size : self.input_seq_len + group_size : group_size]
        else:
            targets = chunk[1 : self.seq_len + 1]
        return inputs, targets

def main():
    parser = argparse.ArgumentParser(description="Train the 1B SLM")
    parser.add_argument(
        "--config", type=str, default="training/configs/default.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/mixed_50B_corpus",
        help="Path to pretokenized dataset",
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints",
        help="Directory to save/load checkpoints",
    )
    parser.add_argument(
        "--checkpoint_interval", type=int, default=1000,
        help="Steps between saving checkpoints",
    )
    args = parser.parse_args()

    # ── Load config ──
    config = SLMConfig.from_yaml(args.config)
    train_cfg = config.training

    # ── Accelerator ──
    accelerator = Accelerator(mixed_precision="bf16")
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
    t0 = time.time()
    model = SLMModel(config).to(device=accelerator.device, dtype=torch.bfloat16)
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
    )
    # Also precompute cos/sin for Liger Kernel RoPE (4x less memory)
    cos_cache, sin_cache = precompute_cos_sin(
        dim=config.qk_rope_head_dim,
        end=seq_len,
        theta=config.rope_theta,
        device=accelerator.device,
    )
    accelerator.print("      Done.\n", flush=True)

    # ── [4.5] Dataset & DataLoader ──
    accelerator.print("[4.5] Preparing Dataset & DataLoader...", flush=True)
    dataset = PretrainingDataset(
        data_dir=args.data_dir,
        input_seq_len=input_seq_len,
        seq_len=seq_len,
        vocab_size=config.vocab_size
    )
    if dataset.use_dummy:
        accelerator.print("      Warning: No .bin files found in data_dir, falling back to dummy data.", flush=True)
        
    dataloader = DataLoader(
        dataset, 
        batch_size=train_cfg.batch_size, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True, 
        drop_last=True
    )
    dataloader = accelerator.prepare(dataloader)
    accelerator.print("      Done.\n", flush=True)
    
    # ── Checkpoint Resume ──
    global_step = 0
    if os.path.exists(args.checkpoint_dir):
        # Naive resume: Load the latest checkpoint if possible
        # Accelerator natively handles this well.
        try:
            accelerator.load_state(args.checkpoint_dir)
            accelerator.print(f"Resumed from checkpoint: {args.checkpoint_dir}", flush=True)
            # You would normally parse the step from a file or config here.
        except Exception as e:
            accelerator.print(f"Could not load checkpoint: {e}", flush=True)

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
    
    data_iter = iter(dataloader)

    for step in range(global_step, train_cfg.max_steps):
        t_step = time.time()
        accum_loss = 0.0

        for micro_step in range(grad_accum_steps):
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                inputs, targets = next(data_iter)

            # Mixed-precision context + Gradient accumulation
            with accelerator.autocast(), accelerator.accumulate(model):
                logits_list = model(inputs, freqs_cis=freqs_cis,
                                    cos_cache=cos_cache, sin_cache=sin_cache)
                loss = compute_loss(logits_list, targets, config)
                accelerator.backward(loss)
                accum_loss += loss.item()

                # Gradient clipping
                if accelerator.sync_gradients and train_cfg.gradient_clip > 0:
                    accelerator.clip_grad_norm_(model.parameters(), train_cfg.gradient_clip)

                # Optimizer step (with Polyak closure for SF-NorMuon/AdamC)
                if accelerator.sync_gradients:
                    # Provide closure to fetch loss for Polyak step-size
                    def closure():
                        return loss.item()
                    optimizer.step(closure=closure)
                    optimizer.zero_grad(set_to_none=True)

        # Logging
        if (step + 1) % train_cfg.log_interval == 0:
            vram = torch.cuda.memory_allocated() / 1024**2
            elapsed = time.time() - t_step
            tokens = train_cfg.batch_size * grad_accum_steps * seq_len
            tps = tokens / elapsed
            accelerator.print(
                f"  Step {step+1}/{train_cfg.max_steps} | "
                f"Loss: {accum_loss:.4f} | "
                f"VRAM: {vram:.0f} MB | "
                f"TPS: {tps:.0f} tok/s | "
                f"Time: {elapsed:.2f}s",
                flush=True,
            )
            if accelerator.is_main_process:
                wandb.log({
                    "loss": accum_loss,
                    "vram_mb": vram,
                    "tps": tps,
                    "elapsed_time": elapsed,
                }, step=step + 1)
            
        # Checkpointing
        if (step + 1) % args.checkpoint_interval == 0:
            accelerator.wait_for_everyone()
            # In Schedule-Free, we must save the averaged weights, not the momentum weights.
            # So we briefly switch to eval mode, save, and switch back.
            if hasattr(optimizer, 'eval'):
                optimizer.eval()
                
            accelerator.save_state(args.checkpoint_dir)
            accelerator.print(f"Saved checkpoint to {args.checkpoint_dir} at step {step+1}", flush=True)
            
            if hasattr(optimizer, 'train'):
                optimizer.train()

    # Switch to eval mode for Schedule-Free averaged weights before saving
    if hasattr(optimizer, 'eval'):
        optimizer.eval()

    accelerator.print("\n✓ Training complete! BF16 pipeline is functional.", flush=True)
    if accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()
