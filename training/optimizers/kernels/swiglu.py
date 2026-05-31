"""
Triton Kernel: Fused SwiGLU Activation
=======================================

Fuses silu(gate) * up into a single kernel, avoiding materialization of
the intermediate silu tensor. Called 24× per forward pass.
"""

import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def _swiglu_fwd_kernel(
        gate_ptr, up_ptr, out_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Compute silu(gate) * up in a single fused pass."""
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        gate = tl.load(gate_ptr + offsets, mask=mask).to(tl.float32)
        up = tl.load(up_ptr + offsets, mask=mask).to(tl.float32)
        
        # SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
        silu_gate = gate / (1.0 + tl.exp(-gate))
        
        out = silu_gate * up
        
        tl.store(out_ptr + offsets, out.to(tl.bfloat16), mask=mask)


def swiglu_forward(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused SwiGLU: silu(gate) * up.
    
    Args:
        gate: Output of gate_proj linear layer.
        up: Output of up_proj linear layer.
        
    Returns:
        silu(gate) * up, same shape as inputs.
    """
    if not TRITON_AVAILABLE or not gate.is_cuda:
        # Fallback to PyTorch
        import torch.nn.functional as F
        return F.silu(gate) * up
    
    assert gate.shape == up.shape
    out = torch.empty_like(gate)
    n_elements = gate.numel()
    
    BLOCK_SIZE = 4096
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    _swiglu_fwd_kernel[grid](
        gate.view(-1), up.view(-1), out.view(-1),
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out
