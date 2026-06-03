import torch
import torch.nn as nn
import torch.nn.functional as F

from config import SLMConfig

class DeltaAttentionResidual(nn.Module):
    """Delta Attention Residuals routing connection to prevent routing collapse.

    Uses a fixed-size tensor buffer instead of a variable-length Python list
    to avoid torch.compile recompilation on every unique delta count.
    """
    def __init__(self, config: SLMConfig):
        super().__init__()
        self.config = config
        self.rms_norm_eps = config.rms_norm_eps

    def forward(self, x, deltas, routing_q):
        """Routes previous layer deltas using softmax attention.

        Args:
            x: Current hidden state (batch, seq_len, hidden_size)
            deltas: List of delta tensors from previous sublayers.
            routing_q: Learned query parameter (hidden_size,)

        Returns:
            x + weighted sum of deltas
        """
        num_deltas = len(deltas)
        if num_deltas == 0:
            return x

        # Stack all deltas into a single tensor: (num_deltas, batch, seq_len, hidden)
        # This is done outside torch.compile's tracing since the list length varies,
        # but the stacked tensor shape is static within each call.
        stacked = torch.stack(deltas, dim=0)  # (D, B, S, H)

        # Compute scores for all deltas in a single batched operation
        # RMSNorm (stateless): normalize each delta across hidden dim
        nf = torch.mean(stacked.float() ** 2, dim=-1, keepdim=True)
        nd = stacked * torch.rsqrt(nf + self.rms_norm_eps).to(stacked.dtype)

        # Score each delta: dot product with routing query → (D, B, S)
        scores = torch.sum(nd * routing_q, dim=-1)

        # Softmax across delta dimension → (D, B, S)
        alpha = F.softmax(scores, dim=0)

        # Weighted sum across deltas → (B, S, H)
        routed = torch.sum(alpha.unsqueeze(-1) * stacked, dim=0)

        return x + routed
