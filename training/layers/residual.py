import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeltaAttentionResidual(nn.Module):
    """Delta Attention Residual with Dynamic Bottleneck Routing.

    Instead of a single static learned vector per sublayer, each token
    independently generates its own routing query via a bottleneck projection:
        x → down_proj(H, rank) → SiLU → up_proj(rank, H) → routing_q [B, S, H]

    This allows every token to choose which past layer deltas are most relevant
    to its current context, making the residual stream routing highly expressive.

    The up_proj is zero-initialized so at init time all scores are 0,
    softmax gives uniform weights, and the module averages all deltas equally —
    matching the original static zero-vector behavior.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps
        self.routing_rank = getattr(config, 'delta_routing_rank', 256)

        # Dynamic routing bottleneck: x → down → SiLU → up → routing_q
        self.routing_down = nn.Linear(self.hidden_size, self.routing_rank, bias=False)
        self.routing_up = nn.Linear(self.routing_rank, self.hidden_size, bias=False)

        # Zero-init the up projection so the module starts as uniform routing
        # (equivalent to the old zero-initialized nn.Parameter behavior)
        nn.init.zeros_(self.routing_up.weight)

    def compute_routing_q(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-token dynamic routing query from current hidden state.
        
        Args:
            x: [B, S, H] current hidden state
            
        Returns:
            routing_q: [B, S, H] dynamic routing query
        """
        return self.routing_up(F.silu(self.routing_down(x)))

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
        """Dual-phase forward pass: lists for training, static buffers for compiled generation.
        
        Args:
            x: [B, S, H] current hidden state
            routing_q: [B, S, H] dynamic per-token routing query (from compute_routing_q)
            past_deltas: list of [B, S, H] tensors (training phase) or None
            delta_buffer: [max_deltas, B, S, H] static buffer (generation phase) or None
            active_mask: [max_deltas, B, S] bool mask (generation phase) or None
        """
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

            # Support both 1D (legacy/static) and 3D (dynamic per-token) routing queries
            if routing_q.dim() == 1:
                nd_flat = nd.view(-1, H)
                scores_flat = F.linear(nd_flat, routing_q) / math.sqrt(H)
                scores = scores_flat.view(num_deltas, B, S)
            else:
                # Dynamic per-token scoring via bmm (memory-safe, no broadcast OOM)
                # routing_q: [B, S, H] → [B*S, 1, H]
                rq = routing_q.reshape(-1, 1, H)
                # nd: [N, B, S, H] → [B*S, N, H]
                nd_t = nd.permute(1, 2, 0, 3).reshape(-1, num_deltas, H)
                # bmm: [B*S, 1, H] @ [B*S, H, N] → [B*S, 1, N] → [B*S, N]
                scores_flat = torch.bmm(rq, nd_t.transpose(-1, -2)).squeeze(1) / math.sqrt(H)
                # [B*S, N] → [N, B, S]
                scores = scores_flat.view(B, S, num_deltas).permute(2, 0, 1)

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

        # 2. Dynamic per-token scoring (handles 1D and 3D routing queries)
        if routing_q.dim() == 1:
            scores = torch.einsum('nbsh, h -> nbs', nd, routing_q) / math.sqrt(self.hidden_size)
        else:
            scores = torch.einsum('nbsh, bsh -> nbs', nd, routing_q) / math.sqrt(self.hidden_size)

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
