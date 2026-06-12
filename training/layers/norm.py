"""
RMSNorm Layer
==============

Provides TritonRMSNorm (fused, fast) with automatic PyTorch fallback.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    
    Optimized to use native PyTorch F.rms_norm. 
    Drops the learned scale (weight) parameter to match modded-nanoGPT, 
    saving memory bandwidth and parameters with identical performance.
    """
    def __init__(self, dim: int, eps: float = 1e-6, use_triton: bool = False):
        super().__init__()
        self.eps = eps
        # No learned weight parameter (reduces memory traffic and parameters)

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), weight=None, eps=self.eps)
