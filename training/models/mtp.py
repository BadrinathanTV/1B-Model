import torch
import torch.nn as nn
import torch.nn.functional as F

from config import SLMConfig
from layers.norm import RMSNorm
from layers.ffn import DenseFFN
from torch.utils.checkpoint import checkpoint


class DeepSeekMTPBlock(nn.Module):
    """
    DeepSeek-V3 Sequential MTP Chain Block.
    Unlike Medusa, this fuses the current latent state with the ACTUAL 
    next token embedding to maintain the causal Markov chain.
    """
    def __init__(self, config: SLMConfig):
        super().__init__()
        # DeepSeek concatenates the hidden state (H) and token embedding (H) -> 2H
        self.concat_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        
        # DeepSeek uses a block (often a simplified Transformer or MLP)
        # We use your highly optimized FFN to maintain throughput.
        self.ffn = DenseFFN(config)
        
        # Stability Fix: Zero-initialize the output of the FFN's down_proj 
        # so this block acts as an identity function at step 0.
        nn.init.zeros_(self.ffn.down_proj.weight)

    def forward(self, h, token_emb):
        # 1. Concatenate latent state and actual token embedding
        z = torch.cat([h, token_emb], dim=-1)
        
        # 2. Project back to hidden size and normalize
        h_next = self.norm(self.concat_proj(z))
        
        # 3. Apply residual FFN
        return h_next + self.ffn(h_next)


class MTPModule(nn.Module):
    """
    DeepSeek-V3 Multi-Token Prediction Architecture.
    STRICTLY requires a shared embedding layer and LM Head.
    """
    def __init__(self, config: SLMConfig, lm_head: nn.Linear, shared_emb: nn.Module | None = None):
        super().__init__()
        self.mtp_depth = config.mtp_depth
        self.lm_head = lm_head
        
        self.shared_emb: nn.Module
        if shared_emb is None:
            # Fallback for compatibility/testing if shared_emb is not provided
            self.shared_emb = nn.Embedding(config.vocab_size, config.hidden_size)
            if lm_head is not None and lm_head.weight.shape == (config.vocab_size, config.hidden_size):
                self.shared_emb.weight = lm_head.weight
        else:
            self.shared_emb = shared_emb
        
        if self.mtp_depth > 1:
            self.blocks = nn.ModuleList([
                DeepSeekMTPBlock(config) for _ in range(self.mtp_depth - 1)
            ])
            self.out_norms = nn.ModuleList([
                RMSNorm(config.hidden_size, config.rms_norm_eps) for _ in range(self.mtp_depth - 1)
            ])

    def _get_embeddings(self, ids: torch.Tensor) -> torch.Tensor:
        if hasattr(self.shared_emb, "word_embeddings"):
            word_emb_layer = getattr(self.shared_emb, "word_embeddings")
            embs = word_emb_layer(ids)
            if hasattr(self.shared_emb, "scale"):
                scale_val = getattr(self.shared_emb, "scale")
                embs = embs * scale_val
        else:
            embs = self.shared_emb(ids)
        return embs

    def forward(self, hidden_states, use_mtp: bool = True, logit_scale = 1.0):
        """Standard forward — returns logits list (for inference)."""
        logits_list = [self.lm_head(hidden_states) * logit_scale]
        curr_states = hidden_states

        if use_mtp and self.mtp_depth > 1:
            for block, norm in zip(self.blocks, self.out_norms):
                # For inference, get the predicted tokens from the last logits
                # shape: (B, S, V) -> (B, S)
                pred_ids = logits_list[-1].argmax(dim=-1)
                pred_embs = self._get_embeddings(pred_ids)
                
                # Combine current states with the predicted token embedding
                curr_states = block(curr_states, pred_embs)
                logits_list.append(self.lm_head(norm(curr_states)) * logit_scale)

        return logits_list

    def forward_hidden(self, hidden_states, use_mtp: bool = True, logit_scale = 1.0, target_ids: torch.Tensor | None = None):
        """Training forward — returns hidden states list for fused cross-entropy.

        We use teacher forcing if target_ids is provided.
        """
        states_list = [hidden_states * logit_scale]
        curr_states = hidden_states

        if use_mtp and self.mtp_depth > 1:
            if target_ids is None:
                # If target_ids is not provided, we fallback to predicting with argmax
                # to prevent crashes (e.g. in some unit tests).
                for block, norm in zip(self.blocks, self.out_norms):
                    # We compute logits to find the argmax
                    pred_ids = (self.lm_head(curr_states) * logit_scale).argmax(dim=-1)
                    pred_embs = self._get_embeddings(pred_ids)
                    
                    # We shrink the sequence by 1 to maintain causal alignment even in fallback mode
                    curr_states = curr_states[:, :-1, :]
                    pred_embs = pred_embs[:, :-1, :]
                    
                    if curr_states.requires_grad:
                        curr_states = checkpoint(block, curr_states, pred_embs, use_reentrant=False)
                    else:
                        curr_states = block(curr_states, pred_embs)
                    states_list.append(norm(curr_states) * logit_scale)
            else:
                # Teacher forcing: target_ids is the sequence of targets.
                # Since target_ids matches the next token labels (t+1), we embed them directly.
                target_embs = self._get_embeddings(target_ids)
                for block, norm in zip(self.blocks, self.out_norms):
                    # Shrink the sequence by 1 to maintain causal alignment
                    curr_states = curr_states[:, :-1, :]
                    curr_targets = target_embs[:, :-1, :]
                    
                    # Combine latent state (t) with target embedding (t+1) to predict (t+2)
                    if curr_states.requires_grad:
                        curr_states = checkpoint(block, curr_states, curr_targets, use_reentrant=False)
                    else:
                        curr_states = block(curr_states, curr_targets)
                    states_list.append(norm(curr_states) * logit_scale)
                    
                    # Since we shrunk the sequence, target_embs must also shrink for the next MTP layer
                    target_embs = target_embs[:, 1:, :]

        return states_list
