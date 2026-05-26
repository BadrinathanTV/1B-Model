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
            "momentum": aurora_cfg.momentum,
            "weight_decay": aurora_cfg.weight_decay,
            "use_riemannian": aurora_cfg.use_riemannian,
        },
        {
            "params": adam_params,
            "lr": opt_cfg.base_lr * adamw_cfg.lr_scale,
            "betas": adamw_cfg.betas,
            "eps": adamw_cfg.eps,
            "weight_decay": adamw_cfg.weight_decay,
            "use_riemannian": False,
        },
    ]

    return HybridSLMOptimizer(param_groups)
