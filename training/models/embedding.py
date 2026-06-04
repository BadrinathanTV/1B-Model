import math
import torch
import torch.nn as nn
from config import SLMConfig


class TokenSuperpositionEmbedding(nn.Module):
    """Embeddings supporting Token-Superposition Training (TST)."""
    
    def __init__(self, config: SLMConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.group_size = config.tst_group_size
        self.scale = math.sqrt(config.hidden_size) if config.embed_scale else 1.0

    def compress_embeddings(self, embeds, group_size_override=None):
        """Helper method to compress raw embeddings into TST superposition space.
        
        This allows external modules (like MTP) to align their target streams
        with the compressed hidden states of the transformer trunk.
        """
        group_size = group_size_override if group_size_override is not None else self.group_size
        
        if group_size > 1:
            batch_size, seq_len, hidden_size = embeds.shape
            
            # FIX 1: Compile-safe symbolic guard (No graph breaks)
            assert seq_len % group_size == 0, f"Seq len {seq_len} must be divisible by {group_size}"
            
            new_seq_len = seq_len // group_size
            embeds = embeds.view(batch_size, new_seq_len, group_size, hidden_size)
            
            # FIX 2: Mixed-precision safety for accumulation
            orig_dtype = embeds.dtype
            embeds = embeds.float().mean(dim=2).to(orig_dtype)
            
        return embeds

    def forward(self, input_ids, group_size_override=None):
        embeds = self.word_embeddings(input_ids) * self.scale
        
        if input_ids.dim() == 3:
            # 3D input: (batch_size, seq_len_folded, group_size)
            orig_dtype = embeds.dtype
            embeds = embeds.float().mean(dim=2).to(orig_dtype)
        else:
            # 2D input: route through the helper for unified logic
            embeds = self.compress_embeddings(embeds, group_size_override=group_size_override)
            
        return embeds
