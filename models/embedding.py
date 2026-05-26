import torch.nn as nn

from config import SLMConfig

class TokenSuperpositionEmbedding(nn.Module):
    """Embeddings supporting Token-Superposition Training (TST)."""
    def __init__(self, config: SLMConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.group_size = config.tst_group_size
        
    def forward(self, input_ids):
        embeds = self.word_embeddings(input_ids)
        if self.group_size > 1:
            batch_size, seq_len, hidden_size = embeds.shape
            if seq_len % self.group_size != 0:
                raise ValueError(
                    f"Input sequence length ({seq_len}) must be divisible by "
                    f"tst_group_size ({self.group_size}). Got remainder "
                    f"{seq_len % self.group_size}."
                )
            new_seq_len = seq_len // self.group_size
            embeds = embeds.view(batch_size, new_seq_len, self.group_size, hidden_size)
            # Average consecutive tokens as per TST paper
            embeds = embeds.mean(dim=2)
        return embeds
