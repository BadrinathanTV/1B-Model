"""
Transformer Components
=======================

TransformerBlock and SLMModel with:
  - Pre-Norm + Post-Norm (attention & FFN sublayers)
  - QK-Norm (in attention module)
  - Truncated normal weight initialization (DeepSeek-V3 style)
  - Gradient checkpointing
"""

import math
from contextlib import nullcontext

import torch
import torch.nn as nn

from config import SLMConfig
from layers.norm import RMSNorm
from layers.attention import MultiHeadLatentAttention
from layers.ffn import DenseFFN
from layers.residual import DeltaAttentionResidual
from models.embedding import TokenSuperpositionEmbedding
from models.mtp import MTPModule


class TransformerBlock(nn.Module):
    """Transformer block with Delta Attention Residuals and standard BF16 precision.

    Uses both Pre-Norm (before sublayer) and Post-Norm (after sublayer) for
    tighter activation magnitude control ("Spike No More", extended norm studies).
    """

    def __init__(self, config: SLMConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.max_delta_history = config.max_delta_history

        # Pre-norms (standard)
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        # Post-norms (controls output magnitude of sublayers)
        self.post_attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        self.attn = MultiHeadLatentAttention(config)
        self.ffn = DenseFFN(config)

        # Delta Attention Residual routing block
        self.delta_residual = DeltaAttentionResidual(config)

        # Learned query parameters (zero-initialized as per paper)
        self.routing_q_attn = nn.Parameter(torch.zeros(config.hidden_size))
        self.routing_q_ffn = nn.Parameter(torch.zeros(config.hidden_size))

    def _deltas_list(self, deltas_buf, num_deltas):
        """Convert fixed-size buffer back to list for delta_residual."""
        if num_deltas == 0:
            return []
        return [deltas_buf[i] for i in range(num_deltas)]

    def _push_delta(self, deltas_buf, num_deltas, new_delta):
        """Push a new delta into the fixed-size ring buffer."""
        if num_deltas < self.max_delta_history:
            deltas_buf[num_deltas] = new_delta
            return deltas_buf, num_deltas + 1
        else:
            # Shift left and insert at end (FIFO)
            deltas_buf = torch.cat([deltas_buf[1:], new_delta.unsqueeze(0)], dim=0)
            return deltas_buf, num_deltas

    def forward(self, x, deltas_buf, num_deltas, freqs_cis=None, cos_cache=None, sin_cache=None):
        # 1. Delta routing enriches x with past deltas for attention input
        deltas_list = self._deltas_list(deltas_buf, num_deltas)
        h_attn = self.delta_residual(x, deltas_list, self.routing_q_attn)
        # 2. Attention computation with post-norm
        v_attn = self.post_attn_norm(
            self.attn(self.attn_norm(h_attn), freqs_cis=freqs_cis,
                      cos_cache=cos_cache, sin_cache=sin_cache)
        )
        # 3. Standard residual: add sublayer output to main stream (NOT h_attn)
        x = x + v_attn
        deltas_buf, num_deltas = self._push_delta(deltas_buf, num_deltas, v_attn)

        # 4. Delta routing enriches x with past deltas for FFN input
        deltas_list = self._deltas_list(deltas_buf, num_deltas)
        h_ffn = self.delta_residual(x, deltas_list, self.routing_q_ffn)
        # 5. FFN computation with post-norm
        v_ffn = self.post_ffn_norm(self.ffn(self.ffn_norm(h_ffn)))
        # 6. Standard residual: add sublayer output to main stream (NOT h_ffn)
        x = x + v_ffn
        deltas_buf, num_deltas = self._push_delta(deltas_buf, num_deltas, v_ffn)

        return x, deltas_buf, num_deltas


class SLMModel(nn.Module):
    """The full 1B SLM integrating Delta Residuals, MLA, Dense FFN, TST, and MTP.

    Stability features:
      - Truncated normal initialization (sigma=config.init_std)
      - Residual sublayer output scaling by 1/sqrt(2*num_layers)
      - Post-attention/FFN normalization
      - QK-Norm (inside MLA)
    """

    def __init__(self, config: SLMConfig):
        super().__init__()
        self.config = config
        self.embed = TokenSuperpositionEmbedding(config)
        self.layers = nn.ModuleList([
            TransformerBlock(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie weights if configured
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed.word_embeddings.weight

        # Output logit scaling / temperature
        base_scale = config.output_logit_scale
        if config.embed_scale:
            base_scale = base_scale / math.sqrt(config.hidden_size)

        if config.output_logit_scale_trainable:
            self.logit_scale = nn.Parameter(torch.tensor(base_scale))
        else:
            self.register_buffer("logit_scale", torch.tensor(base_scale))

        self.mtp = MTPModule(config, self.lm_head)

        # Apply weight initialization AFTER all modules are created
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """DeepSeek-V3 style truncated normal initialization.

        Also scales residual sublayer outputs (o_proj, down_proj) by
        1/sqrt(2*num_layers) to keep residual stream magnitude stable
        across depth (GPT-2 / "Spike No More" recommendation).
        """
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-2*std, b=2*std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-2*std, b=2*std)

        # Scale residual output projections for depth stability
        residual_scale = 1.0 / math.sqrt(2.0 * self.config.num_hidden_layers)
        if isinstance(module, MultiHeadLatentAttention):
            with torch.no_grad():
                module.o_proj.weight.mul_(residual_scale)
        elif isinstance(module, DenseFFN):
            with torch.no_grad():
                module.down_proj.weight.mul_(residual_scale)

    def forward(self, input_ids, freqs_cis=None, cos_cache=None, sin_cache=None,
                use_mtp: bool = True, return_hidden_states: bool = False,
                tst_group_size: int = None):
        x = self.embed(input_ids, group_size_override=tst_group_size)

        # Pre-allocate fixed-size delta buffer to avoid torch.compile recompilation
        # Shape: (max_delta_history, batch_size, seq_len, hidden_size)
        batch_size, seq_len, hidden_size = x.shape
        max_hist = self.config.max_delta_history
        deltas_buf = torch.zeros(max_hist, batch_size, seq_len, hidden_size,
                                 device=x.device, dtype=x.dtype)
        num_deltas = 0

        from torch.utils.checkpoint import checkpoint
        for i, layer in enumerate(self.layers):
            interval = getattr(self.config, "gradient_checkpointing_interval", 1)
            if self.training and self.config.gradient_checkpointing and (i % interval == 0):
                def custom_forward(x_in, db_in, nd_in):
                    return layer(x_in, db_in.clone(), int(nd_in.item()),
                                 freqs_cis=freqs_cis,
                                 cos_cache=cos_cache, sin_cache=sin_cache)
                nd_tensor = torch.tensor(num_deltas, device=x.device)
                from torch.utils.checkpoint import checkpoint
                x, deltas_buf, num_deltas = checkpoint(
                    custom_forward, x, deltas_buf, nd_tensor,
                    use_reentrant=True
                )  # type: ignore
                num_deltas = int(num_deltas) if isinstance(num_deltas, torch.Tensor) else num_deltas
            else:
                x, deltas_buf, num_deltas = layer(
                    x, deltas_buf, num_deltas,
                    freqs_cis=freqs_cis,
                    cos_cache=cos_cache, sin_cache=sin_cache
                )

        x = self.norm(x)

        # Apply config-level MTP toggle if either the flag or the config is False
        use_mtp = use_mtp and self.config.use_mtp

        # Training: return hidden states for fused CE (avoids materializing logits)
        # Inference: return logits directly
        if return_hidden_states:
            return self.mtp.forward_hidden(x, use_mtp=use_mtp, logit_scale=self.logit_scale)
        return self.mtp(x, use_mtp=use_mtp, logit_scale=self.logit_scale)
