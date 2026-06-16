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

        # QK-Norm: per-head normalization to prevent attention entropy collapse
        qk_head_dim = self.v_head_dim + self.qk_rope_head_dim
        # Disable Triton for this specific norm: Triton forces a 2D reshape which launches
        # 122,880 tiny thread blocks per layer. PyTorch handles this 4D reduction much faster.
        self.q_head_norm = RMSNorm(qk_head_dim, config.rms_norm_eps, use_triton=False)
        self.k_head_norm = RMSNorm(qk_head_dim, config.rms_norm_eps, use_triton=False)
        
    def forward(self, x, freqs_cis=None, cos_cache=None, sin_cache=None, past_key_value=None, use_cache=False):
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
        
        # FIX 1: Add .contiguous() after expand to prevent Liger kernel stride-0 crash
        k_rope = k_rope.view(batch_size, seq_len, 1, self.qk_rope_head_dim).expand(-1, -1, self.num_heads, -1).contiguous()
        
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

        # QK-Norm: normalize per-head before SDPA (after RoPE concat)
        q_full = self.q_head_norm(q_full)
        k_full = self.k_head_norm(k_full)
        
        # Transpose for SDPA and ensure memory is contiguous for FlashAttention speed
        q_full = q_full.transpose(1, 2).contiguous()
        k_full = k_full.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        
        # FIX 2: DeepSeek Padding Trick for FlashAttention
        # Pad V's head dimension with zeros to match Q/K, otherwise SDPA silently 
        # disables FlashAttention and falls back to Math (instant OOM).
        v_padded = F.pad(v, (0, self.qk_rope_head_dim))
        
        # --- KV Cache Logic ---
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k_full = torch.cat([past_k, k_full], dim=2)
            v_padded = torch.cat([past_v, v_padded], dim=2)
        
        # We only return the updated cache during inference (when past_key_value is passed or use_cache is True)
        present_key_value = (k_full, v_padded) if (use_cache or past_key_value is not None) else None
        
        # is_causal must be False if we are in decode phase (query length < key length)
        is_causal = (seq_len > 1 and q_full.size(2) == k_full.size(2))
        
        # Use cuDNN/Triton optimized SDPA 
        attn_output = F.scaled_dot_product_attention(q_full, k_full, v_padded, is_causal=is_causal)
        
        # Slice off the padding we just added
        attn_output = attn_output[..., :self.v_head_dim]
        
        # Reshape and project out
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # If cache was provided or requested, return output + new cache. Otherwise just return output.
        if use_cache or past_key_value is not None:
            return self.o_proj(attn_output), present_key_value
        return self.o_proj(attn_output)
