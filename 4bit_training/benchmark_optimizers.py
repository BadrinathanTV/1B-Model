#!/usr/bin/env python3
"""
Optimizer Benchmark: NF-Aurora vs SF-NorMuon vs AdamW (Cosine) vs SF-AdamW
==========================================================================

Trains a small GPT-2 style language model (~25M params) on synthetic
language modeling data and compares four optimizer configurations:

  1. AdamW + Cosine Schedule  (horizon-dependent baseline)
  2. SF-AdamW                 (schedule-free, element-wise baseline)
  3. SF-NorMuon               (schedule-free, spectral baseline)
  4. NF-Aurora                (schedule-free, leverage-aware spectral — OURS)

Metrics tracked:
  - Training loss curve
  - Validation loss (evaluated at X_t for SF methods, at W for scheduled)
  - Wall-clock time per step
  - Total training time

Usage:
    python benchmark_optimizers.py [--steps 500] [--device cuda]
"""

import argparse
import math
import time
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Import our optimizers ───────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optimizers import NFAurora, SFNorMuon


# ═══════════════════════════════════════════════════════════════════════════
# MINI-GPT: Self-contained small transformer for benchmarking
# ═══════════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype) * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_heads, max_seq_len=512):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)  # [B, nh, T, hd]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # Use PyTorch SDPA (flash attention when available)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(y)


class MLP(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        hidden = int(dim * mult)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = MLP(dim)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class MiniGPT(nn.Module):
    """Small GPT-2 style model for optimizer benchmarking.

    Config: ~25M parameters
      - vocab_size=32000, dim=512, n_layers=8, n_heads=8
    """

    def __init__(self, vocab_size=32000, dim=512, n_layers=8, n_heads=8, max_seq_len=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([TransformerBlock(dim, n_heads) for _ in range(n_layers)])
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        # Weight tying
        self.head.weight = self.embed.weight
        self.max_seq_len = max_seq_len
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════════════
# DATA: Synthetic language modeling data
# ═══════════════════════════════════════════════════════════════════════════

def make_synthetic_data(vocab_size, seq_len, batch_size, n_batches, device):
    """Generate synthetic LM data with learnable structure (zipf + bigrams).

    Not random noise — uses a synthetic bigram distribution so the model
    can actually learn and reduce loss, making optimizer comparison meaningful.
    """
    # Build a simple bigram transition matrix (sparse, learnable patterns)
    torch.manual_seed(42)
    # Each token has ~5 likely successors
    transition = torch.zeros(vocab_size, vocab_size)
    for i in range(vocab_size):
        # Pick 5 random successors with high probability
        successors = torch.randint(0, vocab_size, (5,))
        transition[i, successors] = 1.0
    transition = transition / transition.sum(dim=1, keepdim=True)

    batches = []
    for _ in range(n_batches):
        # Generate sequences following the bigram model
        seqs = torch.zeros(batch_size, seq_len + 1, dtype=torch.long)
        seqs[:, 0] = torch.randint(0, vocab_size, (batch_size,))
        for t in range(seq_len):
            probs = transition[seqs[:, t]]
            seqs[:, t + 1] = torch.multinomial(probs, 1).squeeze(-1)
        inputs = seqs[:, :-1].to(device)
        targets = seqs[:, 1:].to(device)
        batches.append((inputs, targets))

    return batches


# ═══════════════════════════════════════════════════════════════════════════
# OPTIMIZER FACTORY
# ═══════════════════════════════════════════════════════════════════════════

def create_optimizer(model, opt_name, lr, weight_decay, total_steps, warmup_steps):
    """Create optimizer and optional scheduler."""

    # Separate 2D (matrix) and non-2D (embedding, norm) parameters
    matrix_params = []
    other_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and "embed" not in name and "head" not in name:
            matrix_params.append(p)
        else:
            other_params.append(p)

    scheduler = None

    if opt_name == "adamw_cosine":
        optimizer = torch.optim.AdamW(
            [
                {"params": matrix_params, "lr": lr, "weight_decay": weight_decay},
                {"params": other_params, "lr": lr, "weight_decay": weight_decay},
            ],
            betas=(0.9, 0.95), eps=1e-8,
        )
        # Cosine schedule with warmup
        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif opt_name == "sf_adamw":
        # Schedule-free AdamW (using manual implementation)
        optimizer = _SFAdamW(
            [
                {"params": matrix_params, "lr": lr, "weight_decay": weight_decay},
                {"params": other_params, "lr": lr, "weight_decay": weight_decay},
            ],
            betas=(0.95, 0.99), warmup_steps=warmup_steps,
        )

    elif opt_name == "sf_normuon":
        optimizer = SFNorMuon(
            [
                {"params": matrix_params, "lr": lr, "weight_decay": weight_decay},
                {"params": other_params, "lr": lr, "weight_decay": weight_decay},
            ],
            momentum=0.8, warmup_steps=warmup_steps,
        )

    elif opt_name == "nf_aurora":
        optimizer = NFAurora(
            [
                {"params": matrix_params, "lr": lr, "weight_decay": weight_decay},
                {"params": other_params, "lr": lr, "weight_decay": weight_decay},
            ],
            momentum=0.95, warmup_steps=warmup_steps, nesterov=True,
        )

    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    return optimizer, scheduler


# ─── Simple SF-AdamW implementation ─────────────────────────────────────────
class _SFAdamW(torch.optim.Optimizer):
    """Minimal Schedule-Free AdamW for benchmarking."""

    def __init__(self, params, lr=0.01, betas=(0.95, 0.99),
                 weight_decay=0.05, warmup_steps=2000, eps=1e-8):
        defaults = dict(
            lr=lr, betas=betas, weight_decay=weight_decay,
            warmup_steps=warmup_steps, eps=eps,
            k=0, train_mode=True, weight_sum=0.0,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            if group["train_mode"]:
                beta = group["betas"][0]
                for p in group["params"]:
                    state = self.state[p]
                    if "z" in state:
                        p.lerp_(end=state["z"], weight=1.0 - 1.0 / beta)
                group["train_mode"] = False

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            if not group["train_mode"]:
                beta = group["betas"][0]
                for p in group["params"]:
                    state = self.state[p]
                    if "z" in state:
                        p.lerp_(end=state["z"], weight=1.0 - beta)
                group["train_mode"] = True

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            decay = group["weight_decay"]
            eps = group["eps"]
            k = group["k"]
            warmup = group["warmup_steps"]

            sched = min(1.0, (k + 1) / warmup) if warmup > 0 else 1.0
            lr = group["lr"] * sched
            weight = lr * lr
            weight_sum = group["weight_sum"] = group["weight_sum"] + weight
            ckp1 = weight / weight_sum

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if "z" not in state:
                    state["z"] = p.clone()
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                z = state["z"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                exp_avg.lerp_(grad, 1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                update = exp_avg / (exp_avg_sq.sqrt() + eps)

                x_t = (p.data - (1.0 - beta1) * z) / beta1
                if decay != 0.0:
                    z.sub_(z, alpha=lr * decay)
                z.sub_(update, alpha=lr)
                x_tp1 = (1.0 - ckp1) * x_t + ckp1 * z
                p.data.copy_((1.0 - beta1) * z + beta1 * x_tp1)

            group["k"] = k + 1
        return loss


# ═══════════════════════════════════════════════════════════════════════════
# BENCHMARK RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def train_one_run(model, optimizer, scheduler, train_data, val_data, total_steps, opt_name, device):
    """Train model for total_steps, return metrics."""
    is_sf = opt_name in ("sf_adamw", "sf_normuon", "nf_aurora")
    model.train()
    if is_sf:
        optimizer.train()

    train_losses = []
    val_losses = []
    step_times = []
    n_batches = len(train_data)

    for step in range(total_steps):
        t0 = time.perf_counter()

        inputs, targets = train_data[step % n_batches]
        logits = model(inputs)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

        t1 = time.perf_counter()
        step_times.append(t1 - t0)
        train_losses.append(loss.item())

        # Validation every 50 steps
        if (step + 1) % 50 == 0 or step == 0:
            model.eval()
            if is_sf:
                optimizer.eval()

            with torch.no_grad():
                val_loss_sum = 0.0
                for vi, (vinp, vtgt) in enumerate(val_data):
                    vlogits = model(vinp)
                    vl = F.cross_entropy(vlogits.view(-1, vlogits.size(-1)), vtgt.view(-1))
                    val_loss_sum += vl.item()
                val_loss = val_loss_sum / len(val_data)
            val_losses.append((step + 1, val_loss))

            model.train()
            if is_sf:
                optimizer.train()

            avg_step_time = sum(step_times[-50:]) / len(step_times[-50:])
            print(f"  [{opt_name:>12}] Step {step+1:4d}/{total_steps} | "
                  f"Train: {loss.item():.4f} | Val: {val_loss:.4f} | "
                  f"ms/step: {avg_step_time*1000:.1f}")

    # Final validation
    model.eval()
    if is_sf:
        optimizer.eval()

    with torch.no_grad():
        val_loss_sum = 0.0
        for vinp, vtgt in val_data:
            vlogits = model(vinp)
            vl = F.cross_entropy(vlogits.view(-1, vlogits.size(-1)), vtgt.view(-1))
            val_loss_sum += vl.item()
        final_val = val_loss_sum / len(val_data)

    total_time = sum(step_times)

    return {
        "name": opt_name,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "final_val_loss": final_val,
        "total_time": total_time,
        "avg_step_time": total_time / total_steps,
        "step_times": step_times,
    }


def print_results_table(results):
    """Print a formatted comparison table."""
    print("\n" + "=" * 80)
    print("                    OPTIMIZER BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Optimizer':<16} {'Final Val Loss':>14} {'Avg ms/step':>12} {'Total Time':>12} {'Speedup':>10}")
    print("-" * 80)

    # Sort by final val loss
    results_sorted = sorted(results, key=lambda r: r["final_val_loss"])
    best_time = min(r["total_time"] for r in results)

    for r in results_sorted:
        speedup = best_time / r["total_time"]
        print(f"{r['name']:<16} {r['final_val_loss']:>14.4f} "
              f"{r['avg_step_time']*1000:>12.1f} "
              f"{r['total_time']:>11.1f}s "
              f"{speedup:>9.2f}x")

    print("=" * 80)

    # Print val loss progression
    print("\nValidation Loss Progression:")
    print(f"{'Step':>6}", end="")
    for r in results_sorted:
        print(f"  {r['name']:>14}", end="")
    print()
    print("-" * (6 + 16 * len(results_sorted)))

    # Align by step
    all_steps = sorted(set(s for r in results for s, _ in r["val_losses"]))
    for step in all_steps:
        print(f"{step:>6}", end="")
        for r in results_sorted:
            val = next((v for s, v in r["val_losses"] if s == step), None)
            if val is not None:
                print(f"  {val:>14.4f}", end="")
            else:
                print(f"  {'—':>14}", end="")
        print()

    print()


def main():
    parser = argparse.ArgumentParser(description="Optimizer Benchmark: NF-Aurora vs baselines")
    parser.add_argument("--steps", type=int, default=500, help="Total training steps per optimizer")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length")
    parser.add_argument("--vocab-size", type=int, default=8192, help="Vocabulary size (smaller = faster)")
    parser.add_argument("--dim", type=int, default=384, help="Model dimension")
    parser.add_argument("--n-layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--n-heads", type=int, default=6, help="Number of attention heads")
    parser.add_argument("--lr", type=float, default=0.003, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup steps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--optimizers", type=str, default="all",
                        help="Comma-separated list: adamw_cosine,sf_adamw,sf_normuon,nf_aurora or 'all'")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # ── Generate data ──
    print(f"\nGenerating synthetic bigram data (vocab={args.vocab_size}, seq_len={args.seq_len})...")
    n_train = max(args.steps // 5, 20)
    n_val = 5
    train_data = make_synthetic_data(args.vocab_size, args.seq_len, args.batch_size, n_train, device)
    val_data = make_synthetic_data(args.vocab_size, args.seq_len, args.batch_size, n_val, device)
    print(f"  Train: {n_train} batches, Val: {n_val} batches")

    # ── Determine which optimizers to run ──
    if args.optimizers == "all":
        opt_names = ["adamw_cosine", "sf_adamw", "sf_normuon", "nf_aurora"]
    else:
        opt_names = [x.strip() for x in args.optimizers.split(",")]

    # ── Build reference model and save initial state ──
    print(f"\nBuilding MiniGPT (dim={args.dim}, layers={args.n_layers}, heads={args.n_heads})...")
    torch.manual_seed(args.seed)
    ref_model = MiniGPT(
        vocab_size=args.vocab_size, dim=args.dim,
        n_layers=args.n_layers, n_heads=args.n_heads,
        max_seq_len=args.seq_len,
    ).to(device)
    n_params = ref_model.count_params()
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    init_state = {k: v.clone() for k, v in ref_model.state_dict().items()}

    # ── Run benchmark for each optimizer ──
    results = []
    for opt_name in opt_names:
        print(f"\n{'─' * 60}")
        print(f"  Training with: {opt_name.upper()}")
        print(f"{'─' * 60}")

        # Reset model to same initial weights
        torch.manual_seed(args.seed)
        model = MiniGPT(
            vocab_size=args.vocab_size, dim=args.dim,
            n_layers=args.n_layers, n_heads=args.n_heads,
            max_seq_len=args.seq_len,
        ).to(device)
        model.load_state_dict(init_state)

        # Create optimizer
        optimizer, scheduler = create_optimizer(
            model, opt_name, args.lr, args.weight_decay,
            args.steps, args.warmup,
        )

        # Train
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        result = train_one_run(
            model, optimizer, scheduler, train_data, val_data,
            args.steps, opt_name, device,
        )

        if device.type == "cuda":
            torch.cuda.synchronize()
            result["peak_memory_mb"] = torch.cuda.max_memory_allocated() / 1024**2
            print(f"  Peak GPU Memory: {result['peak_memory_mb']:.0f} MB")

        results.append(result)

    # ── Print comparison ──
    print_results_table(results)

    # ── Memory comparison if CUDA ──
    if device.type == "cuda" and all("peak_memory_mb" in r for r in results):
        print("Memory Usage:")
        for r in sorted(results, key=lambda x: x["peak_memory_mb"]):
            print(f"  {r['name']:<16} {r['peak_memory_mb']:>8.0f} MB")
        print()

    print("Benchmark complete!")


if __name__ == "__main__":
    main()
