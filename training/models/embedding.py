import math

import torch.nn as nn

from config import SLMConfig


class TokenSuperpositionEmbedding(nn.Module):
    """Embeddings supporting Token-Superposition Training (TST).

    Stability: optionally scales embeddings by √hidden_size to balance signal
    magnitude vs. positional encoding (standard practice in Transformer literature).
    """
    def __init__(self, config: SLMConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.group_size = config.tst_group_size
        self.scale = math.sqrt(config.hidden_size) if config.embed_scale else 1.0

    def forward(self, input_ids, group_size_override=None):
        embeds = self.word_embeddings(input_ids) * self.scale
        group_size = group_size_override if group_size_override is not None else self.group_size
        if group_size > 1:
            batch_size, seq_len, hidden_size = embeds.shape
            if seq_len % group_size != 0:
                raise ValueError(
                    f"Input sequence length ({seq_len}) must be divisible by "
                    f"tst_group_size ({group_size}). Got remainder "
                    f"{seq_len % group_size}."
                )
            new_seq_len = seq_len // group_size
            embeds = embeds.view(batch_size, new_seq_len, group_size, hidden_size)
            # Average consecutive tokens as per TST paper
            embeds = embeds.mean(dim=2)
        return embeds
