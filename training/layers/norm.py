"""
RMSNorm Layer
==============

Provides TritonRMSNorm (fused, fast) with automatic PyTorch fallback.
"""

try:
    from optimizers.kernels.rms_norm import TritonRMSNorm as RMSNorm
except ImportError:
    # Fallback: pure PyTorch RMSNorm
    import torch
    import torch.nn as nn

    class RMSNorm(nn.Module):
        """Root Mean Square Layer Normalization (PyTorch fallback)."""
        def __init__(self, dim: int, eps: float = 1e-6, use_triton: bool = True):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x):
            variance = x.float().pow(2).mean(-1, keepdim=True)
            x_normed = x * torch.rsqrt(variance + self.eps).to(x.dtype)
            return self.weight * x_normed
