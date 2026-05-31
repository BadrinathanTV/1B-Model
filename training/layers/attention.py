import torch
import torch.nn as nn
import torch.nn.functional as F

from config import SLMConfig
from .norm import RMSNorm
from .rope import apply_rotary_emb

class MultiHeadLatentAttention(nn.Module):
    """Multi-Head Latent Attention (MLA) with KV and Query compression."""
    def __init__(self, config: SLMConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        
        # Query compression
        self.q_down_proj = nn.Linear(config.hidden_size, self.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(self.q_lora_rank, config.rms_norm_eps)
        self.q_up_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.v_head_dim, bias=False)
        
        # KV compression
        self.kv_down_proj = nn.Linear(config.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_norm = RMSNorm(self.kv_lora_rank, config.rms_norm_eps)
        
        # Up projections for K and V
        self.k_up_proj = nn.Linear(self.kv_lora_rank, self.num_heads * self.v_head_dim, bias=False)
        self.v_up_proj = nn.Linear(self.kv_lora_rank, self.num_heads * self.v_head_dim, bias=False)
        
        # Query RoPE
        self.q_rope_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_rope_head_dim, bias=False)
        
        # Output projection
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, config.hidden_size, bias=False)
        
    def forward(self, x, freqs_cis=None, cos_cache=None, sin_cache=None):
        batch_size, seq_len, _ = x.shape
        
        # --- Query Path ---
        q_c = self.q_norm(self.q_down_proj(x))
        q = self.q_up_proj(q_c).view(batch_size, seq_len, self.num_heads, self.v_head_dim)
        q_rope = self.q_rope_proj(q_c).view(batch_size, seq_len, self.num_heads, self.qk_rope_head_dim)
        
        # --- KV Path ---
        kv_c_rope = self.kv_down_proj(x)
        kv_c, k_rope = torch.split(kv_c_rope, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_c = self.kv_norm(kv_c)
        
        k = self.k_up_proj(kv_c).view(batch_size, seq_len, self.num_heads, self.v_head_dim)
        v = self.v_up_proj(kv_c).view(batch_size, seq_len, self.num_heads, self.v_head_dim)
        
        # Expand k_rope to all heads
        k_rope = k_rope.view(batch_size, seq_len, 1, self.qk_rope_head_dim).expand(-1, -1, self.num_heads, -1)
        
        # Apply RoPE rotary embeddings (required for positional awareness)
        if freqs_cis is None:
            raise ValueError(
                "freqs_cis must be provided for RoPE. Precompute with "
                "precompute_freqs_cis() and pass to model.forward()."
            )
        q_rope, k_rope = apply_rotary_emb(q_rope, k_rope, freqs_cis,
                                           cos_cache=cos_cache, sin_cache=sin_cache)
            
        # Concatenate RoPE dimensions and regular latent dimensions
        q_full = torch.cat([q, q_rope], dim=-1)
        k_full = torch.cat([k, k_rope], dim=-1)
        
        # Transpose for SDPA: (batch, heads, seq, dim)
        q_full = q_full.transpose(1, 2)
        k_full = k_full.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Use cuDNN/Triton optimized SDPA 
        attn_output = F.scaled_dot_product_attention(q_full, k_full, v, is_causal=True)
        
        # Reshape and project out
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn_output)
