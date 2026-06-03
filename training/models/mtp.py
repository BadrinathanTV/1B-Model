import torch
import torch.nn as nn
import torch.nn.functional as F

from config import SLMConfig
from layers.norm import RMSNorm


class MTPProjection(nn.Module):
    """Single MTP projection head using a SwiGLU gating mechanism.

    Each head: RMSNorm -> SwiGLU (gate_proj * up_proj) -> down_proj (with residual).
    This prevents the linear collapse that occurs when stacking projections.
    """
    def __init__(self, config: SLMConfig):
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.gate_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.down_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, x):
        h = self.norm(x)
        h = self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
        return x + h  # Residual connection


class MTPModule(nn.Module):
    """Multi-Token Prediction (MTP) projection heads.

    Supports two modes:
      - forward():        returns logits_list    (for inference)
      - forward_hidden(): returns hidden states  (for fused CE during training)
    """
    def __init__(self, config: SLMConfig, lm_head: nn.Linear):
        super().__init__()
        self.mtp_depth = config.mtp_depth
        self.lm_head = lm_head
        if self.mtp_depth > 1:
            self.projs = nn.ModuleList([
                MTPProjection(config) for _ in range(self.mtp_depth - 1)
            ])

    def forward(self, hidden_states, use_mtp: bool = True, logit_scale = 1.0):
        """Standard forward — returns logits list (for inference)."""
        logits_list = [self.lm_head(hidden_states) * logit_scale]
        curr_states = hidden_states

        # Predict t+2, t+3, ...
        if use_mtp and self.mtp_depth > 1:
            for proj in self.projs:
                curr_states = proj(curr_states)
                logits_list.append(self.lm_head(curr_states) * logit_scale)

        return logits_list

    def forward_hidden(self, hidden_states, use_mtp: bool = True, logit_scale = 1.0):
        """Training forward — returns hidden states for fused cross-entropy.

        Avoids materializing the full [B*S, vocab_size] logits tensor,
        saving ~1GB VRAM per micro-step with vocab_size=64000.
        """
        states_list = [hidden_states * logit_scale]
        curr_states = hidden_states

        if use_mtp and self.mtp_depth > 1:
            for proj in self.projs:
                curr_states = proj(curr_states)
                states_list.append(curr_states * logit_scale)

        return states_list
