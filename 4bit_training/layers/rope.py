import torch

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

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape freqs_cis to be broadcastable with x."""
    ndim = x.ndim
    assert ndim >= 2, "Tensor must have at least 2 dimensions"
    # x has shape [batch, seq_len, heads, dim // 2] (complex)
    # freqs_cis has shape [seq_len, dim // 2]
    # We need to reshape freqs_cis to [1, seq_len, 1, dim // 2]
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors.
    
    Args:
        xq: Query tensor of shape [batch, seq_len, heads, qk_rope_head_dim].
        xk: Key tensor of shape [batch, seq_len, heads, qk_rope_head_dim].
        freqs_cis: Complex tensor of shape [seq_len, qk_rope_head_dim // 2].
        
    Returns:
        xq_out, xk_out: Rotated query and key tensors of the same shape and dtype.
    """
    # Cast to float32 for precision during complex multiplication
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # Broadcast freqs_cis to the correct rank
    # xq_ has shape [batch, seq_len, heads, dim // 2]
    # Slice freqs_cis to match sequence length of input in case of cached decoding or shorter seqs
    seq_len = xq.shape[1]
    if seq_len > freqs_cis.shape[0]:
        raise ValueError(
            f"Input sequence length ({seq_len}) exceeds precomputed RoPE cache "
            f"({freqs_cis.shape[0]}). Increase 'end' in precompute_freqs_cis()."
        )
    freqs_cis_sliced = freqs_cis[:seq_len]
    freqs_cis_broadcast = reshape_for_broadcast(freqs_cis_sliced, xq_)
    
    # Rotate
    xq_out = torch.view_as_real(xq_ * freqs_cis_broadcast).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis_broadcast).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)
