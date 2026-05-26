"""
Transformer Components
=======================

TransformerBlock and SLMModel with configurable precision routing
and deduplicated forward logic.
"""

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

try:
    import transformer_engine.pytorch as te
    TE_AVAILABLE = True
except Exception:
    TE_AVAILABLE = False


class TransformerBlock(nn.Module):
    """Transformer block with Delta Attention Residuals and configurable precision.

    The precision boundary (which layers run in full BF16 vs NVFP4) is
    controlled by config.precision.high_precision_start_layers and
    config.precision.high_precision_end_layers — no more hardcoded magic numbers.
    """

    def __init__(self, config: SLMConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = MultiHeadLatentAttention(config)

        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn = DenseFFN(config)

        # Delta Attention Residual routing block
        self.delta_residual = DeltaAttentionResidual(config)

        # Learned query parameters (zero-initialized as per paper)
        self.routing_q_attn = nn.Parameter(torch.zeros(config.hidden_size))
        self.routing_q_ffn = nn.Parameter(torch.zeros(config.hidden_size))

    def _is_high_precision(self) -> bool:
        """Check if this layer should run in high precision (BF16)."""
        prec = self.config.precision
        n_layers = self.config.num_hidden_layers
        in_start = self.layer_idx < prec.high_precision_start_layers
        in_end = self.layer_idx >= (n_layers - prec.high_precision_end_layers)
        return in_start or in_end

    def _precision_context(self):
        """Return the appropriate precision context manager.

        High-precision layers disable fp8_autocast; normal layers use
        whatever the caller set up (passthrough via nullcontext).
        """
        if self._is_high_precision() and TE_AVAILABLE:
            return te.fp8_autocast(enabled=False)
        return nullcontext()

    def forward(self, x, deltas, freqs_cis=None):
        with self._precision_context():
            # 1. Delta routing enriches x with past deltas for attention input
            h_attn = self.delta_residual(x, deltas, self.routing_q_attn)
            # 2. Attention computation
            v_attn = self.attn(self.attn_norm(h_attn), freqs_cis=freqs_cis)
            # 3. Standard residual: add sublayer output to main stream (NOT h_attn)
            x = x + v_attn
            deltas.append(v_attn)
            if self.config.max_delta_history > 0:
                while len(deltas) > self.config.max_delta_history:
                    deltas.pop(0)

            # 4. Delta routing enriches x with past deltas for FFN input
            h_ffn = self.delta_residual(x, deltas, self.routing_q_ffn)
            # 5. FFN computation
            v_ffn = self.ffn(self.ffn_norm(h_ffn))
            # 6. Standard residual: add sublayer output to main stream (NOT h_ffn)
            x = x + v_ffn
            deltas.append(v_ffn)
            if self.config.max_delta_history > 0:
                while len(deltas) > self.config.max_delta_history:
                    deltas.pop(0)

        return x, deltas


class SLMModel(nn.Module):
    """The full 1B SLM integrating Delta Residuals, MLA, Dense FFN, TST, and MTP."""

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
            
        self.mtp = MTPModule(config, self.lm_head)

    def forward(self, input_ids, freqs_cis=None):
        x = self.embed(input_ids)

        # Maintain a list of deltas across all layers for Delta Attention Residuals routing
        deltas = []

        for layer in self.layers:
            x, deltas = layer(x, deltas, freqs_cis=freqs_cis)

        x = self.norm(x)
        logits_list = self.mtp(x)
        return logits_list
