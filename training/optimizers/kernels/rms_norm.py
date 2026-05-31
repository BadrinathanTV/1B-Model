"""
Triton Kernel: Fused RMSNorm
=============================

Fuses the entire RMSNorm computation (variance, rsqrt, scale) into a single
GPU kernel pass, reducing HBM traffic by ~4x compared to the PyTorch version.

Called 48× per forward pass (2× per layer × 24 layers).
"""

import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def _rms_norm_fwd_kernel(
        x_ptr, w_ptr, out_ptr,
        stride_x_row,
        N,  # number of columns (hidden_size)
        eps: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Forward pass: out = (x / rms(x)) * weight."""
        row = tl.program_id(0)
        
        x_row_ptr = x_ptr + row * stride_x_row
        out_row_ptr = out_ptr + row * stride_x_row
        
        # Load the row in blocks
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N
        
        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        
        # Compute RMS: sqrt(mean(x^2) + eps)
        x_sq = x * x
        mean_sq = tl.sum(x_sq, axis=0) / N
        rrms = 1.0 / tl.sqrt(mean_sq + eps)
        
        # Normalize and scale
        out = x * rrms * w
        
        tl.store(out_row_ptr + cols, out, mask=mask)


class TritonRMSNorm(nn.Module):
    """RMSNorm with optional Triton kernel acceleration.
    
    Falls back to pure PyTorch if Triton is not available.
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self._dim = dim
    
    def _triton_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute fused Triton RMSNorm kernel."""
        orig_shape = x.shape
        x_2d = x.contiguous().view(-1, self._dim)
        out = torch.empty_like(x_2d)
        
        n_rows = x_2d.shape[0]
        
        # BLOCK_N must be power of 2 and >= dim
        BLOCK_N = triton.next_power_of_2(self._dim)
        
        _rms_norm_fwd_kernel[(n_rows,)](
            x_2d, self.weight, out,
            stride_x_row=x_2d.stride(0),
            N=self._dim,
            eps=self.eps,
            BLOCK_N=BLOCK_N,
        )
        
        return out.view(orig_shape)
    
    def _pytorch_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fallback pure PyTorch implementation."""
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps).to(x.dtype)
        return self.weight * x_normed
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if TRITON_AVAILABLE and x.is_cuda:
            return self._triton_forward(x)
        return self._pytorch_forward(x)
