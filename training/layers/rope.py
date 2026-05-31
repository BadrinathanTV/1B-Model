"""
RoPE — Rotary Position Embeddings (Hybrid Kernel)
===================================================

Uses Liger Kernel RoPE when available (4x less memory spike),
falls back to PyTorch complex arithmetic otherwise.
"""

import torch

try:
    from liger_kernel.ops.rope import LigerRopeFunction
    LIGER_ROPE = True
except ImportError:
    LIGER_ROPE = False


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device: torch.device = None) -> torch.Tensor:
    """Precompute the complex exponential freqs_cis for rotary position embeddings.
    
    Args:
        dim: Dimension of the rotary embedding (must be even).
        end: Maximum sequence length.
        theta: Frequency scale factor.
        device: Target device for computation and storage.
        
    Returns:
        freqs_cis: Complex tensor of shape [end, dim // 2].
    """
    assert dim % 2 == 0, "RoPE dimension must be even"
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    t = torch.arange(end, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs) # [end, dim // 2]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs) # Complex tensor
    return freqs_cis


def precompute_cos_sin(dim: int, end: int, theta: float = 10000.0, device: torch.device = None, dtype=torch.bfloat16):
    """Precompute cos/sin caches for Liger RoPE kernel.
    
    Returns:
        cos_cache, sin_cache: Tensors of shape [end, dim // 2].
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    t = torch.arange(end, device=device, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape freqs_cis to be broadcastable with x."""
    ndim = x.ndim
    assert ndim >= 2, "Tensor must have at least 2 dimensions"
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor,
                     cos_cache: torch.Tensor | None = None, sin_cache: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors.
    
    Uses Liger Kernel when available (4x less memory), falls back to
    PyTorch complex arithmetic otherwise.
    """
    seq_len = xq.shape[1]
    
    # Liger RoPE kernel
    if LIGER_ROPE and cos_cache is not None and sin_cache is not None and xq.is_cuda:
        # Liger expects [batch, num_heads, seq_len, head_dim]
        xq_liger = xq.transpose(1, 2)
        xk_liger = xk.transpose(1, 2)
        
        # Liger expects cos/sin to be [1, seq_len, head_dim] or [1, seq_len, head_dim//2]
        cos_s = cos_cache[:seq_len].unsqueeze(0)
        sin_s = sin_cache[:seq_len].unsqueeze(0)
        
        xq_out, xk_out = LigerRopeFunction.apply(xq_liger, xk_liger, cos_s, sin_s, None, 0)
        
        # Transpose back to [batch, seq_len, num_heads, head_dim]
        return xq_out.transpose(1, 2), xk_out.transpose(1, 2)
    
    # PyTorch fallback: LLaMA-style (half-half) rotation to match Liger
    if seq_len > freqs_cis.shape[0]:
        raise ValueError(
            f"Input sequence length ({seq_len}) exceeds precomputed RoPE cache "
            f"({freqs_cis.shape[0]}). Increase 'end' in precompute_freqs_cis()."
        )
        
    if cos_cache is None or sin_cache is None:
        # If caches not provided, extract from freqs_cis
        cos_s = freqs_cis[:seq_len].real
        sin_s = freqs_cis[:seq_len].imag
    else:
        cos_s = cos_cache[:seq_len]
        sin_s = sin_cache[:seq_len]
        
    # Broadcast cos_s and sin_s to [1, seq_len, 1, head_dim/2]
    cos_s = cos_s.view(1, seq_len, 1, -1)
    sin_s = sin_s.view(1, seq_len, 1, -1)
    
    # LLaMA half-half rotation function
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
        
    # Duplicate cos/sin for both halves, or multiply halves directly
    # Better: duplicate cos and sin so they match the full head_dim
    cos_full = torch.cat((cos_s, cos_s), dim=-1)
    sin_full = torch.cat((sin_s, sin_s), dim=-1)
    
    xq_out = (xq.float() * cos_full) + (rotate_half(xq.float()) * sin_full)
    xk_out = (xk.float() * cos_full) + (rotate_half(xk.float()) * sin_full)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)
