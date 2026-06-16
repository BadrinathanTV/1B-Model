import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeltaAttentionResidual(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps

    def init_state(self, batch_size: int, seq_len: int, max_deltas: int, dtype: torch.dtype, device: torch.device):
        """Allocates static-sized buffers for the generation phase."""
        # [max_deltas, B, S, H]
        delta_buffer = torch.zeros((max_deltas, batch_size, seq_len, self.hidden_size), 
                                   dtype=dtype, device=device)
        # [max_deltas, B, S]
        active_mask = torch.zeros((max_deltas, batch_size, seq_len), 
                                  dtype=torch.bool, device=device)
        # 0D tensor to track insertions on-device without graph breaks
        current_idx = torch.tensor(0, dtype=torch.long, device=device)
        
        return delta_buffer, active_mask, current_idx

    def update_state(self, new_delta: torch.Tensor, delta_buffer: torch.Tensor, active_mask: torch.Tensor, current_idx: torch.Tensor):
        """Inserts a new delta using a ring buffer to cap memory."""
        max_deltas = delta_buffer.size(0)
        idx = current_idx % max_deltas  # Overwrite oldest when full
        
        # In-place updates maintain static shapes for the compiler
        delta_buffer[idx] = new_delta
        active_mask[idx] = True
        
        return delta_buffer, active_mask, current_idx + 1

    def forward(self, x: torch.Tensor, routing_q: torch.Tensor, past_deltas=None, delta_buffer=None, active_mask=None):
        """Dual-phase forward pass: lists for training, static buffers for compiled generation."""
        # ---------------------------------------------------------
        # PHASE 1: Training (List-based, dynamic)
        # ---------------------------------------------------------
        if past_deltas is not None:
            if not past_deltas:
                return x
            
            deltas_stack = torch.stack(past_deltas, dim=0)
            num_deltas, B, S, H = deltas_stack.shape
            
            # RMSNorm (stateless)
            nd = F.rms_norm(deltas_stack, (H,), weight=None, eps=self.rms_norm_eps)

            # Score each slot: use flattened F.linear to avoid einsum broadcast OOM
            nd_flat = nd.view(-1, H)
            scores_flat = F.linear(nd_flat, routing_q) / math.sqrt(H)
            scores = scores_flat.view(num_deltas, B, S)

            alpha = F.softmax(scores, dim=0)

            # Route via batched matrix multiply (bmm) to avoid einsum OOM
            alpha_flat = alpha.permute(1, 2, 0).reshape(-1, 1, num_deltas)
            deltas_flat = deltas_stack.permute(1, 2, 0, 3).reshape(-1, num_deltas, H)
            routed = torch.bmm(alpha_flat, deltas_flat).view(B, S, H)

            return x + routed

        # ---------------------------------------------------------
        # PHASE 2: Generation (Static buffers, compiled-ready)
        # ---------------------------------------------------------
        assert delta_buffer is not None and active_mask is not None, "Must provide delta_buffer and active_mask if use_cache=True"
        
        # 1. Normalize the entire static buffer
        nd = F.rms_norm(delta_buffer, (self.hidden_size,), weight=None, eps=self.rms_norm_eps)

        # 2. Compute scores directly via einsum (safe during generation as B=1 usually)
        scores = torch.einsum('nbsh, h -> nbs', nd, routing_q) / math.sqrt(self.hidden_size)

        # 3. Mask out uninitialized slots with -inf
        scores = scores.masked_fill(~active_mask, float('-inf'))

        # 4. Softmax
        alpha = F.softmax(scores, dim=0)

        # 5. Route via weighted sum
        routed = torch.einsum('nbs, nbsh -> bsh', alpha, delta_buffer)

        # 6. The NaN Safeguard
        has_active = active_mask.any(dim=0).unsqueeze(-1)  # (B, S, 1)
        routed = torch.nan_to_num(routed, nan=0.0)

        return torch.where(has_active, x + routed, x)
