import torch
import torch.nn as nn
import torch.nn.functional as F

from config import SLMConfig

class DeltaAttentionResidual(nn.Module):
    """Delta Attention Residuals routing connection to prevent routing collapse."""
    def __init__(self, config: SLMConfig):
        super().__init__()
        self.config = config
        
    def forward(self, x, deltas, routing_q):
        """Routes previous layer deltas using softmax attention."""
        if not deltas:
            return x
            
        # Stack deltas: (batch_size, seq_len, num_deltas, hidden_size)
        stacked_deltas = torch.stack(deltas, dim=2)
        
        # RMSNorm on the stacked deltas (stateless norm across hidden dim)
        # Upcast to float32 to prevent overflow during squared mean
        norm_factor = torch.mean(stacked_deltas.float() ** 2, dim=-1, keepdim=True)
        normed_deltas = stacked_deltas * torch.rsqrt(norm_factor + self.config.rms_norm_eps).to(stacked_deltas.dtype)
        
        # Compute softmax scores using the learned query parameter
        # Shape: (batch_size, seq_len, num_deltas)
        scores = torch.einsum('bsdh,h->bsd', normed_deltas, routing_q)
        alpha = F.softmax(scores, dim=-1)
        
        # Compute the weighted sum of deltas and add to the current residual stream
        routed = torch.einsum('bsd,bsdh->bsh', alpha, stacked_deltas)
        return x + routed
