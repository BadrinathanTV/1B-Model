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
        x_ptr, w_ptr, out_ptr, rrms_ptr,
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
        tl.store(rrms_ptr + row, rrms)

    @triton.jit
    def _rms_norm_bwd_kernel(
        dx_ptr, dw_ptr, dy_ptr, x_ptr, w_ptr, rrms_ptr,
        stride_x_row,
        N,
        BLOCK_N: tl.constexpr,
    ):
        """Backward pass for RMSNorm."""
        row = tl.program_id(0)
        
        row_stride = row * stride_x_row
        
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N
        
        x = tl.load(x_ptr + row_stride + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + row_stride + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        rrms = tl.load(rrms_ptr + row)
        
        # dx_hat = dy * w
        dx_hat = dy * w
        
        # c2 = -(1/N) * rrms^3 * sum(dx_hat * x)
        dx_hat_dot_x = tl.sum(dx_hat * x, axis=0)
        c2 = -(1.0 / N) * (rrms * rrms * rrms) * dx_hat_dot_x
        
        # dx = rrms * dx_hat + c2 * x
        dx = rrms * dx_hat + c2 * x
        
        tl.store(dx_ptr + row_stride + cols, dx, mask=mask)
        
        # dw = dy * x * rrms (Note: in a real implementation we need to atomically add this across rows)
        # We will compute dw in pure PyTorch for simplicity since atomic adds across many blocks can be slow.

class TritonRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        orig_shape = x.shape
        x_2d = x.contiguous().view(-1, x.shape[-1])
        out = torch.empty_like(x_2d)
        rrms = torch.empty(x_2d.shape[0], device=x.device, dtype=torch.float32)
        
        n_rows = x_2d.shape[0]
        dim = x.shape[-1]
        BLOCK_N = triton.next_power_of_2(dim)
        
        _rms_norm_fwd_kernel[(n_rows,)](
            x_2d, weight, out, rrms,
            stride_x_row=x_2d.stride(0),
            N=dim,
            eps=eps,
            BLOCK_N=BLOCK_N,
        )
        
        ctx.save_for_backward(x_2d, weight, rrms)
        ctx.eps = eps
        ctx.orig_shape = orig_shape
        
        return out.view(orig_shape)

    @staticmethod
    def backward(ctx, dy):
        x_2d, weight, rrms = ctx.saved_tensors
        dy_2d = dy.contiguous().view(-1, x_2d.shape[-1])
        dx_2d = torch.empty_like(x_2d)
        
        n_rows = x_2d.shape[0]
        dim = x_2d.shape[-1]
        BLOCK_N = triton.next_power_of_2(dim)
        
        _rms_norm_bwd_kernel[(n_rows,)](
            dx_2d, None, dy_2d, x_2d, weight, rrms,
            stride_x_row=x_2d.stride(0),
            N=dim,
            BLOCK_N=BLOCK_N,
        )
        
        # dw is computed with PyTorch to avoid atomic operations bottleneck in Triton
        # dw = sum(dy * (x * rrms))
        x_normed = x_2d * rrms.unsqueeze(-1)
        dweight = (dy_2d * x_normed).sum(dim=0)
        
        return dx_2d.view(ctx.orig_shape), dweight, None

class TritonRMSNorm(nn.Module):
    """RMSNorm with optional Triton kernel acceleration.
    
    Falls back to pure PyTorch if Triton is not available.
    """
    
    def __init__(self, dim: int, eps: float = 1e-6, use_triton: bool = True):
        super().__init__()
        self.eps = eps
        self.use_triton = use_triton
        self.weight = nn.Parameter(torch.ones(dim))
        self._dim = dim
    
    def _triton_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute fused Triton RMSNorm kernel via autograd Function."""
        return TritonRMSNormFunction.apply(x, self.weight, self.eps)
    
    def _pytorch_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fallback pure PyTorch implementation."""
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps).to(x.dtype)
        return self.weight * x_normed
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if TRITON_AVAILABLE and x.is_cuda and self.use_triton:
            return self._triton_forward(x)
        return self._pytorch_forward(x)
