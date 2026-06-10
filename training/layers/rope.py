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


import math

def _compute_yarn_freqs(dim: int, theta: float, scale: float, beta_fast: float, beta_slow: float, original_context: int, device: torch.device):
    """Compute YaRN (Yet another RoPE extensioN) inverse frequencies and attention scale."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    
    if scale <= 1.0:
        return freqs, 1.0

    # Find the interpolation ranges
    def find_correction_dim(num_rotations):
        return (dim * math.log(original_context / (num_rotations * 2 * math.pi))) / (2 * math.log(theta))
    
    low = max(math.floor(find_correction_dim(beta_fast)), 0)
    high = min(math.ceil(find_correction_dim(beta_slow)), dim - 1)
    
    # Calculate linear ramp mask
    inv_freq_mask = torch.arange(0, dim, 2, dtype=torch.float32, device=device)
    inv_freq_mask = 1.0 - (inv_freq_mask - low) / max(high - low, 1e-4)
    inv_freq_mask = torch.clamp(inv_freq_mask, 0.0, 1.0)
    
    # Apply piece-wise interpolation
    freqs_interp = freqs / scale
    freqs_yarn = freqs * inv_freq_mask + freqs_interp * (1.0 - inv_freq_mask)
    
    # Calculate temperature scale for attention logits (we scale cos/sin by sqrt(t) so Q*K gets scaled by t)
    # The paper uses t = 0.1 * ln(s) + 1.0
    mscale = float(0.1 * math.log(scale) + 1.0)
    
    return freqs_yarn, mscale

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device: torch.device = None,
                         yarn_scale: float = 1.0, yarn_beta_fast: float = 32.0, yarn_beta_slow: float = 1.0, yarn_orig_ctx: int = 2048) -> torch.Tensor:
    """Precompute the complex exponential freqs_cis for rotary position embeddings."""
    assert dim % 2 == 0, "RoPE dimension must be even"
    freqs, mscale = _compute_yarn_freqs(dim, theta, yarn_scale, yarn_beta_fast, yarn_beta_slow, yarn_orig_ctx, device)
    
    t = torch.arange(end, device=device, dtype=torch.float32)
    angles = torch.outer(t, freqs) # [end, dim // 2]
    
    # Scale amplitude by mscale so dot products are scaled by mscale^2
    freqs_cis = torch.polar(torch.ones_like(angles) * mscale, angles) # Complex tensor
    return freqs_cis


def precompute_cos_sin(dim: int, end: int, theta: float = 10000.0, device: torch.device = None, dtype=torch.bfloat16,
                       yarn_scale: float = 1.0, yarn_beta_fast: float = 32.0, yarn_beta_slow: float = 1.0, yarn_orig_ctx: int = 2048):
    """Precompute cos/sin caches for Liger RoPE kernel."""
    freqs, mscale = _compute_yarn_freqs(dim, theta, yarn_scale, yarn_beta_fast, yarn_beta_slow, yarn_orig_ctx, device)
    
    t = torch.arange(end, device=device, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    
    return (angles.cos() * mscale).to(dtype), (angles.sin() * mscale).to(dtype)


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
        
        # FIX 1: Duplicate cos/sin to match the full head_dim for the Triton kernel
        cos_full_liger = torch.cat([cos_cache[:seq_len], cos_cache[:seq_len]], dim=-1)
        sin_full_liger = torch.cat([sin_cache[:seq_len], sin_cache[:seq_len]], dim=-1)
        
        # FIX 2: Cast to xq's dtype to prevent silent Triton C++ compilation crashes
        cos_s = cos_full_liger.unsqueeze(0).to(xq.dtype)
        sin_s = sin_full_liger.unsqueeze(0).to(xq.dtype)
        
        # FIX 3: Use unsqueeze_dim=1 (Liger standard) to properly broadcast over num_heads
        xq_out, xk_out = LigerRopeFunction.apply(xq_liger, xk_liger, cos_s, sin_s, None, 1)
        
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
