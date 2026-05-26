"""
Training Script
================

Config-driven training loop for the 1B SLM.

Usage:
    python train.py                              # uses configs/default.yaml
    python train.py --config configs/1b_nvfp4.yaml  # production preset
"""

import argparse

import torch
import torch.nn.functional as F

from config import SLMConfig
from model import SLMModel
from layers.rope import precompute_freqs_cis
from optimizers import build_optimizer

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import NVFP4BlockScaling
    TE_AVAILABLE = True
except Exception as e:
    TE_AVAILABLE = False
    print(f"Warning: NVIDIA Transformer Engine unavailable ({type(e).__name__}). "
          "Falling back to standard precision.")


def resolve_device(device_str: str) -> torch.device:
    """Resolve device string ('auto', 'cuda', 'cpu') to a torch.device."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def build_nvfp4_recipe(config: SLMConfig):
    """Build the NVFP4 recipe from config, or None if TE is unavailable."""
    if not TE_AVAILABLE:
        return None
    prec = config.precision
    return NVFP4BlockScaling(
        disable_rht=prec.nvfp4_disable_rht,
        disable_stochastic_rounding=prec.nvfp4_disable_stochastic_rounding,
    )


def compute_loss(logits_list, targets, config: SLMConfig) -> torch.Tensor:
    """Compute LM loss with MTP auxiliary losses.

    Args:
        logits_list: List of logit tensors [main, mtp_1, mtp_2, ...].
        targets: Target token IDs (B, seq_len).
        config: Config for vocab_size and mtp settings.

    Returns:
        Combined scalar loss.
    """
    # Standard Language Modeling Loss (t+1)
    main_logits = logits_list[0]
    loss = F.cross_entropy(main_logits.view(-1, config.vocab_size), targets.view(-1))

    # Multi-Token Prediction auxiliary losses (t+2, t+3, ...)
    if config.mtp_depth > 1:
        mtp_weight = config.training.mtp_loss_weight
        for i in range(1, config.mtp_depth):
            mtp_logits = logits_list[i][:, :-i, :].contiguous()
            mtp_targets = targets[:, i:].contiguous()
            # Guard: skip if sequence is too short for this MTP depth
            # (empty tensors produce NaN from cross_entropy)
            if mtp_logits.numel() == 0 or mtp_targets.numel() == 0:
                continue
            mtp_loss = F.cross_entropy(
                mtp_logits.view(-1, config.vocab_size),
                mtp_targets.view(-1),
            )
            loss += mtp_weight * mtp_loss

    return loss


def main():
    parser = argparse.ArgumentParser(description="Train the 1B SLM")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()

    # ── Load config ──
    config = SLMConfig.from_yaml(args.config)
    print(config)

    train_cfg = config.training
    device = resolve_device(train_cfg.device)
    torch.manual_seed(train_cfg.seed)

    # ── Initialize model & optimizer ──
    print(f"\nInitializing 1B SLM on {device}...")
    model = SLMModel(config).to(device)
    optimizer = build_optimizer(model, config)

    # ── NVFP4 recipe ──
    fp8_recipe = build_nvfp4_recipe(config)

    # ── Precompute RoPE ──
    seq_len = train_cfg.seq_len
    input_seq_len = seq_len * config.tst_group_size
    print("Precomputing RoPE frequency coordinates...")
    freqs_cis = precompute_freqs_cis(
        dim=config.qk_rope_head_dim,
        end=seq_len,  # Precompute for seq_len (post-TST compression size)
        theta=config.rope_theta,
        device=device,
    )

    # ── Training loop ──
    print(f"\nStarting training loop ({train_cfg.max_steps} steps)...")
    for step in range(train_cfg.max_steps):
        # Generate dummy data (TST requires inputs group_size times longer)
        inputs = torch.randint(
            0, config.vocab_size,
            (train_cfg.batch_size, input_seq_len),
            device=device,
        )
        targets = torch.randint(
            0, config.vocab_size,
            (train_cfg.batch_size, seq_len),
            device=device,
        )

        # Forward pass (with NVFP4 autocast if available)
        if fp8_recipe is not None:
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                logits_list = model(inputs, freqs_cis=freqs_cis)
        else:
            logits_list = model(inputs, freqs_cis=freqs_cis)

        loss = compute_loss(logits_list, targets, config)
        loss.backward()

        # Apply gradient clipping if configured
        if train_cfg.gradient_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.gradient_clip)

        optimizer.step()
        optimizer.zero_grad()

        if (step + 1) % train_cfg.log_interval == 0:
            print(f"Step {step + 1}/{train_cfg.max_steps}, Loss: {loss.item():.4f}")

    print("\nTraining complete! Architecture routing is functional.")


if __name__ == "__main__":
    main()
