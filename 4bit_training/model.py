# Core exports to maintain backwards compatibility while remaining modular
from config import SLMConfig
from layers.norm import RMSNorm
from layers.attention import MultiHeadLatentAttention
from layers.ffn import DenseFFN
from layers.residual import DeltaAttentionResidual
from layers.rope import precompute_freqs_cis, apply_rotary_emb
from models.embedding import TokenSuperpositionEmbedding
from models.mtp import MTPModule
from models.transformer import TransformerBlock, SLMModel
