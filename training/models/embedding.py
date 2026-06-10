import math
import torch
import torch.nn as nn
from config import SLMConfig


# Backward-compatible alias for imports that still reference the old name
TokenSuperpositionEmbedding = None  # set below after class definition


class Embedding(nn.Module):
    """Token embedding layer with optional √hidden_size scaling (Gemma/DeepSeek style)."""
    
    def __init__(self, config: SLMConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.scale = math.sqrt(config.hidden_size) if config.embed_scale else 1.0

    def forward(self, input_ids):
        embeds = self.word_embeddings(input_ids)
        if self.scale != 1.0:
            embeds = embeds * self.scale
        return embeds


# Backward-compatible alias
TokenSuperpositionEmbedding = Embedding
