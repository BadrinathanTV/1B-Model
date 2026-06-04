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

    def forward_static(self, x, deltas_buf, num_deltas, routing_q):
        """Compile-friendly branchless routing over a fixed-size delta buffer.

        Mathematically identical to forward(), but operates on a pre-allocated
        tensor buffer with a mask instead of a variable-length Python list.
        This avoids torch.compile recompilations caused by changing list lengths.

        Args:
            x: Current hidden state (batch, seq_len, hidden_size)
            deltas_buf: Fixed-size buffer (max_hist, batch, seq_len, hidden_size)
            num_deltas: Number of active slots (scalar tensor on device)
            routing_q: Learned query parameter (hidden_size,)

        Returns:
            x + weighted sum of active deltas
        """
        max_hist = deltas_buf.shape[0]

        # RMSNorm (stateless): normalize ALL slots (inactive slots are zeros,
        # their scores will be masked to -inf anyway)
        nf = torch.mean(deltas_buf.float() ** 2, dim=-1, keepdim=True)
        nd = deltas_buf * torch.rsqrt(nf + self.rms_norm_eps).to(deltas_buf.dtype)

        # Score each slot: dot product with routing query → (max_hist, B, S)
        scores = torch.sum(nd * routing_q, dim=-1)

        # Boolean mask for active slots: shape (max_hist, 1, 1)
        indices = torch.arange(max_hist, device=deltas_buf.device)
        mask = (indices < num_deltas).view(-1, 1, 1)

        # FIX 1 & 3: Bypass the -inf mask ONLY if the buffer is completely empty.
        # This prevents F.softmax from receiving an all '-inf' tensor and outputting NaN.
        # The output will safely be zeroed out in the next step anyway.
        safe_mask = mask | (num_deltas == 0)
        
        # Use exact float('-inf') to ensure active slots always sum strictly to 1.0
        masked_scores = torch.where(
            safe_mask, 
            scores, 
            torch.tensor(float('-inf'), device=scores.device, dtype=scores.dtype)
        )

        # Softmax across delta dimension (dim=0) → (max_hist, B, S)
        alpha = F.softmax(masked_scores, dim=0)

        # FIX 2: Cast mask to alpha's dtype to prevent silent upcasting to float32
        alpha = alpha * mask.to(alpha.dtype)

        # Weighted sum across all slots → (B, S, H)
        routed = torch.sum(alpha.unsqueeze(-1) * deltas_buf, dim=0)

        return x + routed
