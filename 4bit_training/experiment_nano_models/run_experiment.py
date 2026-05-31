"""
Nano GPT Optimizer Experiment Script
====================================

Runs the exact same model architecture at a ~50M parameter scale to 
test different optimizers quickly for a 1B token budget.

Usage:
    python experiment_nano_models/run_experiment.py --optimizer nf_aurora
    python experiment_nano_models/run_experiment.py --optimizer adamw
    python experiment_nano_models/run_experiment.py --optimizer hybrid
"""

import argparse
import sys
from pathlib import Path
import contextlib
import time
import numpy as np
import tiktoken

# Add project root to path so we can import original architecture natively
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F

from config import SLMConfig
from model import SLMModel
from layers.rope import precompute_freqs_cis
from optimizers import build_optimizer

# Reuse NVFP4 utilities from train.py if available
try:
    from train import build_nvfp4_recipe, compute_loss, resolve_device
    import transformer_engine.pytorch as te
    TE_AVAILABLE = True
except Exception as e:
    TE_AVAILABLE = False
    print(f"Warning: NVIDIA Transformer Engine unavailable ({type(e).__name__}). Falling back to standard precision.")
    
    # Fallback definitions if train.py imports fail
    def resolve_device(device_str: str) -> torch.device:
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_str)

    def build_nvfp4_recipe(config: SLMConfig):
        return None
        
    def compute_loss(logits_list, targets, config: SLMConfig) -> torch.Tensor:
        main_logits = logits_list[0].float()
        loss = F.cross_entropy(main_logits.view(-1, config.vocab_size), targets.view(-1))
        if config.mtp_depth > 1:
            mtp_weight = config.training.mtp_loss_weight
            for i in range(1, config.mtp_depth):
                mtp_logits = logits_list[i][:, :-i, :].contiguous().float()
                mtp_targets = targets[:, i:].contiguous()
                if mtp_logits.numel() == 0 or mtp_targets.numel() == 0:
                    continue
                loss += mtp_weight * F.cross_entropy(
                    mtp_logits.view(-1, config.vocab_size),
                    mtp_targets.view(-1),
                )
        return loss


class ShardedDataLoader:
    def __init__(self, data_dir, split, batch_size, seq_len, group_size):
        self.data_dir = Path(data_dir)
        self.split = split
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.group_size = group_size
        self.chunk_size = seq_len * group_size + 1
        
        # Get list of files
        if split == "train":
            self.files = sorted(list(self.data_dir.glob("train_shard_*.bin")))
        else:
            self.files = sorted(list(self.data_dir.glob("val.bin")))
            
        self.use_synthetic = len(self.files) == 0
        if self.use_synthetic:
            print(f"Warning: No {split} binary data found in {data_dir}. Falling back to synthetic generator.")
            return

        self.current_file_idx = 0
        self._load_current_shard()
        
    def _load_current_shard(self):
        filename = self.files[self.current_file_idx]
        self.data = np.memmap(filename, dtype=np.uint16, mode="r")
        self.pointer = 0
        
    def next_batch(self, device):
        if self.use_synthetic:
            x = torch.randint(0, 50257, (self.batch_size, self.seq_len * self.group_size), device=device)
            y = torch.randint(0, 50257, (self.batch_size, self.seq_len), device=device)
            return x, y

        B = self.batch_size
        C = self.chunk_size
        
        # If we reach end of shard, wrap around
        if self.pointer + B * C + 1 > len(self.data):
            if self.split == "train":
                self.current_file_idx = (self.current_file_idx + 1) % len(self.files)
            self._load_current_shard()

        # Extract batch start indices
        ix = np.arange(self.pointer, self.pointer + B * C, C)[:B]
        self.pointer += B * C
        
        if self.pointer + B * C + 1 > len(self.data):
            self.pointer = 0

        # Construct inputs and targets using fast vectorized numpy slicing
        x_list, y_list = [], []
        for i in ix:
            x_list.append(torch.from_numpy(self.data[i : i + C - 1].astype(np.int64)))
            y_list.append(torch.from_numpy(self.data[i + self.group_size : i + self.group_size * self.seq_len + 1 : self.group_size].astype(np.int64)))
            
        x = torch.stack(x_list).to(device, non_blocking=True)
        y = torch.stack(y_list).to(device, non_blocking=True)
        return x, y


def safe_decode(tokens, enc):
    # Clip tokens to valid range for GPT-2 encoder (0 to 50256)
    valid_tokens = [min(max(int(t), 0), 50256) for t in tokens]
    try:
        return enc.decode(valid_tokens)
    except Exception:
        return "[Decoding Error]"


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main():
    parser = argparse.ArgumentParser(description="Nano GPT Optimizer Experiment")
    parser.add_argument(
        "--config", type=str, default="experiment_nano_models/nano_config.yaml",
        help="Path to nano YAML configuration file",
    )
    parser.add_argument(
        "--optimizer", type=str, choices=["nf_aurora", "adamw", "hybrid", "sf_normuon"], default=None,
        help="Override the optimizer defined in the config",
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/mixed_50B_corpus",
        help="Directory where the pre-tokenized binary shards are stored",
    )
    parser.add_argument(
        "--tokens", type=int, default=1_000_000_000,
        help="Target total tokens to train on (default 1 Billion). Automatically calculates max_steps.",
    )
    parser.add_argument(
        "--eval_interval", type=int, default=50,
        help="Evaluate validation loss and print model predictions every N steps",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only run 5 steps to verify setup",
    )
    args = parser.parse_args()

    # ── Load config ──
    config_path = project_root / args.config
    config = SLMConfig.from_yaml(config_path)
    
    if args.optimizer:
        config.optimizer.type = args.optimizer

    train_cfg = config.training
    device = resolve_device(train_cfg.device)
    torch.manual_seed(train_cfg.seed)
    
    # Calculate tokens per step
    tokens_per_step = train_cfg.batch_size * train_cfg.seq_len
    target_steps = args.tokens // tokens_per_step
    
    if args.dry_run:
        target_steps = 50
        print("Dry run enabled. Running for 50 steps only.")
        
    train_cfg.max_steps = target_steps

    print(f"Configuration Loaded. Optimizer: {config.optimizer.type}")
    print(f"Target steps for {args.tokens:,} tokens: {target_steps:,}")

    # Initialize Tokenizer encoder for decoding predictions
    enc = tiktoken.get_encoding("gpt2")

    # ── Initialize dataloaders ──
    print(f"Initializing dataloaders from {args.data_dir}...")
    train_loader = ShardedDataLoader(
        args.data_dir, "train", train_cfg.batch_size, train_cfg.seq_len, config.tst_group_size
    )
    val_loader = ShardedDataLoader(
        args.data_dir, "val", train_cfg.batch_size, train_cfg.seq_len, config.tst_group_size
    )

    # ── Initialize model & optimizer ──
    print(f"\nInitializing Nano SLM on {device}...")
    model = SLMModel(config).to(device)
    
    # Initialize weights to standard LLM defaults (std=0.02) to avoid high initial losses
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.Embedding):
            torch.nn.init.normal_(m.weight, std=0.02)

    model.apply(init_weights)
    
    param_count = count_parameters(model)
    print(f"Total Parameters: {param_count / 1e6:.2f}M")
    
    optimizer = build_optimizer(model, config)
    print(f"Optimizer initialized: {type(optimizer).__name__}")

    # ── NVFP4 recipe ──
    fp8_recipe = build_nvfp4_recipe(config)

    # ── Precompute RoPE ──
    seq_len = train_cfg.seq_len
    input_seq_len = seq_len * config.tst_group_size
    print("Precomputing RoPE frequency coordinates...")
    freqs_cis = precompute_freqs_cis(
        dim=config.qk_rope_head_dim,
        end=seq_len,
        theta=config.rope_theta,
        device=device,
    )

    # ── Training loop ──
    print(f"\nStarting experimental training loop ({train_cfg.max_steps} steps)...")
    
    # Enable train mode for Schedule-Free optimizers
    if hasattr(optimizer, "train"):
        optimizer.train()

    model.train()
    
    t0 = time.time()
    for step in range(train_cfg.max_steps):
        inputs, targets = train_loader.next_batch(device)

        if TE_AVAILABLE and fp8_recipe is not None:
            ctx = te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe)
        else:
            ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()

        with ctx:
            logits_list = model(inputs, freqs_cis=freqs_cis)

        loss = compute_loss(logits_list, targets, config)
        loss.backward()

        if train_cfg.gradient_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.gradient_clip)

        optimizer.step()
        
        # Debug parameter norms
        if step < 5:
            print(f"--- Step {step + 1} parameter norms ---")
            for name, p in model.named_parameters():
                if p.grad is not None:
                    p_norm = p.norm().item()
                    g_norm = p.grad.norm().item()
                    if not torch.isfinite(p).all().item() or p_norm > 100.0 or p_norm < 1e-4:
                        print(f"  WARNING: {name} | W_norm: {p_norm:.4f} | grad_norm: {g_norm:.4f}")
                    elif "embed" in name or "lm_head" in name:
                        print(f"  {name} (Embed) | W_norm: {p_norm:.4f} | grad_norm: {g_norm:.4f}")
                    elif p.ndim < 2:
                        print(f"  {name} (1D) | W_norm: {p_norm:.4f} | grad_norm: {g_norm:.4f}")
                    else:
                        print(f"  {name} (2D) | W_norm: {p_norm:.4f} | grad_norm: {g_norm:.4f}")

        optimizer.zero_grad()

        # Log training loss
        if (step + 1) % train_cfg.log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            tokens_since_last = train_cfg.log_interval * train_cfg.batch_size * train_cfg.seq_len * config.tst_group_size
            tokens_per_sec = tokens_since_last / dt
            print(f"Step {step + 1}/{train_cfg.max_steps} | Train Loss: {loss.item():.4f} | Tok/s: {tokens_per_sec:.0f}")

        # Run Validation & Predictions print
        if (step + 1) % args.eval_interval == 0 or step == 0:
            model.eval()
            if hasattr(optimizer, "eval"):
                optimizer.eval()

            with torch.no_grad():
                val_inputs, val_targets = val_loader.next_batch(device)
                
                if TE_AVAILABLE and fp8_recipe is not None:
                    val_ctx = te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe)
                else:
                    val_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()

                with val_ctx:
                    val_logits_list = model(val_inputs, freqs_cis=freqs_cis)
                
                val_loss = compute_loss(val_logits_list, val_targets, config)
                
                print(f"\n" + "=" * 50)
                print(f"EVALUATION AT STEP {step + 1}")
                print(f"Validation Loss: {val_loss.item():.4f}")
                print("-" * 50)
                
                # Show predictions vs ground truth
                # Print the first sample in the batch
                pred_tokens = torch.argmax(val_logits_list[0][0], dim=-1)
                target_tokens = val_targets[0]
                
                print(f"TARGETS (Ground Truth):")
                print(safe_decode(target_tokens[:50], enc))
                print(f"PREDICTIONS (Model):")
                print(safe_decode(pred_tokens[:50], enc))
                print("=" * 50 + "\n")

            model.train()
            if hasattr(optimizer, "train"):
                optimizer.train()
            # Reset timer to not include evaluation overhead in next step's Tok/s
            t0 = time.time()

    print("\nExperiment loop complete!")


if __name__ == "__main__":
    main()
