"""
Optimizer Factory
==================

Builds the optimizer from an SLMConfig, automatically routing parameters
to the correct optimizer group.
"""

from __future__ import annotations

import sys
import os

import torch

# Add parent to path for config import when running as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SLMConfig

from .hybrid import HybridSLMOptimizer


def build_optimizer(model: torch.nn.Module, config: SLMConfig) -> torch.optim.Optimizer:
    """Build the optimizer from config, routing params to the correct group.

    Routing logic:
      - Embeddings and output head → AdamW group
      - Biases, routing parameters, norm gains (1D) → AdamW group
      - Hidden matrix layers (2D linear weights) → Aurora group

    Args:
        model: The model whose parameters to optimize.
        config: Full SLMConfig with optimizer settings.

    Returns:
        Configured optimizer instance.
    """
    opt_cfg = config.optimizer
    aurora_cfg = opt_cfg.aurora
    adamw_cfg = opt_cfg.adamw

    aurora_params = []
    adam_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Embeddings and output head → AdamW
        if "embed" in name or "lm_head" in name:
            adam_params.append(param)
        # Biases, routing parameters, and norm gains are 1D → AdamW
        elif param.ndim < 2:
            adam_params.append(param)
        # Hidden matrix layers (2D linear weights) → Aurora
        else:
            aurora_params.append(param)

    param_groups = [
        {
            "params": aurora_params,
            "lr": opt_cfg.base_lr,
            "use_aurora": True, # This flag triggers the SF-NorMuon 2D path
            "weight_decay": aurora_cfg.weight_decay,
        },
        {
            "params": adam_params,
            "lr": opt_cfg.base_lr * adamw_cfg.lr_scale,
            "use_aurora": False, # This flag triggers the AdamC Polyak path
            "weight_decay": adamw_cfg.weight_decay,
        },
    ]

    if opt_cfg.type == "hybrid":
        return HybridSLMOptimizer(param_groups)
    elif opt_cfg.type == "nf_aurora_hybrid":
        from .nf_aurora_hybrid import NFAuroraHybrid
        return NFAuroraHybrid(param_groups)
    elif opt_cfg.type == "nf_normuon_hybrid":
        from .nf_normuon_hybrid import NFNorMuonHybrid
        c_warmup = opt_cfg.warmup_steps * 2
        return NFNorMuonHybrid(
            param_groups,
            sf_beta1=0.9,
            sf_beta1_max=0.965,
            sf_beta1_anneal_steps=opt_cfg.training_steps if hasattr(opt_cfg, 'training_steps') else 40000,
            polyak_beta=0.9,
            c_warmup=c_warmup,
            r=1.0
        )
    elif opt_cfg.type == "nf_aurora":
        from .nf_aurora import NFAurora
        return NFAurora(param_groups)
    elif opt_cfg.type == "sf_normuon":
        from .sf_normuon import SFNorMuon
        # Compute c_warmup based on warmup_steps (2x as recommended by paper)
        c_warmup = opt_cfg.warmup_steps * 2
        # Paper uses same LR for 1D and 2D parameters
        param_groups[1]["lr"] = opt_cfg.base_lr
        return SFNorMuon(
            param_groups,
            sf_beta1=0.9,
            sf_beta1_max=0.965,
            sf_beta1_anneal_steps=opt_cfg.training_steps if hasattr(opt_cfg, 'training_steps') else 40000,
            polyak_beta=0.9, # Standard EMA for Polyak L1 norm
            c_warmup=c_warmup,
            r=1.0 # Long duration weighting
        )
    elif opt_cfg.type == "adamw":
        # standard AdamW (filter out non-AdamW keys)
        adam_groups = []
        for g in param_groups:
            adam_groups.append({
                "params": g["params"],
                "lr": g["lr"],
                "weight_decay": g["weight_decay"],
                "betas": g.get("betas", adamw_cfg.betas),
                "eps": g.get("eps", adamw_cfg.eps),
            })
        return torch.optim.AdamW(adam_groups)
    else:
        raise ValueError(f"Unknown optimizer type: {opt_cfg.type}")
