"""
SLM Configuration
=================

Centralized, validated configuration for the 1B SLM.

Supports:
  - YAML file loading:   SLMConfig.from_yaml("configs/default.yaml")
  - YAML serialization:  config.save_yaml("run_config.yaml")
  - Programmatic construction with validation
"""

from __future__ import annotations


from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


# ─── Sub-configs ──────────────────────────────────────────────────────────────

@dataclass
class AuroraConfig:
    """Riemannian Aurora optimizer hyperparameters (for 2D matrix weights)."""
    momentum: float = 0.95
    weight_decay: float = 0.1
    nesterov: bool = True
    use_riemannian: bool = True
    update_clip: float = 0.1
    pp_iterations: int = 2
    pp_beta: float = 0.5


@dataclass
class AdamWConfig:
    """4-bit quantized AdamW hyperparameters (for 1D params, embeddings, norms)."""
    lr_scale: float = 1.0
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-10
    weight_decay: float = 0.1


@dataclass
class OptimizerConfig:
    """Top-level optimizer configuration with param-group routing."""
    type: str = "hybrid"            # "hybrid" | "adamw"
    base_lr: float = 1e-3
    warmup_steps: int = 2000
    aurora: AuroraConfig = field(default_factory=AuroraConfig)
    adamw: AdamWConfig = field(default_factory=AdamWConfig)

    def __post_init__(self):
        if isinstance(self.aurora, dict):
            self.aurora = AuroraConfig(**self.aurora)
        if isinstance(self.adamw, dict):
            # Handle list -> tuple for betas
            if "betas" in self.adamw and isinstance(self.adamw["betas"], list):
                self.adamw["betas"] = tuple(self.adamw["betas"])
            self.adamw = AdamWConfig(**self.adamw)


@dataclass
class TrainingConfig:
    """Training loop settings."""
    batch_size: int = 2
    seq_len: int = 512
    max_steps: int = 5
    gradient_accumulation_steps: int = 8
    gradient_clip: float = 1.0
    log_interval: int = 1
    seed: int = 42
    device: str = "auto"            # "auto" | "cuda" | "cpu"
    learning_rate: float = 1e-3     # Alias for optimizer.base_lr (used by some tests)


# ─── Main Config ─────────────────────────────────────────────────────────────

@dataclass
class SLMConfig:
    """Complete configuration for the 1B SLM.

    Contains model architecture, optimizer, and training settings in a single
    validated object.

    Example:
        config = SLMConfig.from_yaml("configs/default.yaml")
        config = SLMConfig(vocab_size=32000, hidden_size=1024)
    """

    # Model architecture
    vocab_size: int = 64000
    hidden_size: int = 1536
    num_hidden_layers: int = 24
    num_attention_heads: int = 12
    intermediate_size: int = 4096
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-6

    # MLA (Multi-Head Latent Attention)
    kv_lora_rank: int = 256
    q_lora_rank: int = 1024
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    rope_theta: float = 10000.0

    # YaRN Context Extension
    use_yarn: bool = False             # Enable YaRN RoPE scaling
    yarn_scale_factor: float = 1.0     # Context scale (e.g. 16.0 for 32k if trained on 2k)
    yarn_original_context: int = 2048  # Original pre-training max context
    yarn_beta_fast: float = 32.0       # YaRN beta fast (high frequency cutoff)
    yarn_beta_slow: float = 1.0        # YaRN beta slow (low frequency cutoff)

    # MTP (Multi-Token Prediction)
    mtp_depth: int = 1                 # Number of future tokens to predict (1 = standard NTP)
    use_mtp: bool = True               # Whether to use MTP heads during inference

    # Stability: Weight Initialization
    init_std: float = 0.02             # Truncated normal sigma (DeepSeek-V3 uses 0.006)

    # Stability: Z-Loss (logit regularization)
    z_loss_weight: float = 1e-4        # Penalty on logit magnitude (PaLM/Gemma style)

    # Stability: Embedding scaling
    embed_scale: bool = True           # Multiply embeddings by sqrt(hidden_size)

    # Stability: Output head logit scaling / temperature
    output_logit_scale: float = 1.0    # Additional scale factor for output logits
    output_logit_scale_trainable: bool = False # Make the scale factor a trainable parameter

    # Weight Tying & Memory Optimizations
    tie_word_embeddings: bool = True
    max_delta_history: int = 0  # 0 means keep full history
    gradient_checkpointing: bool = True # Set to False for massive speedup if VRAM allows
    gradient_checkpointing_interval: int = 1 # Checkpoint every N layers (higher = faster but more memory)

    # Sub-configs
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        """Convert nested dicts to dataclasses and run validation."""
        if isinstance(self.optimizer, dict):
            self.optimizer = OptimizerConfig(**self.optimizer)
        if isinstance(self.training, dict):
            self.training = TrainingConfig(**self.training)
        self._validate()

    def _validate(self):
        """Validate configuration consistency."""
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        if self.qk_rope_head_dim % 2 != 0:
            raise ValueError(
                f"qk_rope_head_dim ({self.qk_rope_head_dim}) must be even for complex RoPE"
            )
        if self.output_logit_scale <= 0.0:
            raise ValueError(f"output_logit_scale must be positive, got {self.output_logit_scale}")
        if self.max_delta_history < 0:
            raise ValueError(f"max_delta_history must be >= 0, got {self.max_delta_history}")
        if self.mtp_depth < 1:
            raise ValueError(f"mtp_depth must be >= 1, got {self.mtp_depth}")

        if self.optimizer.type not in ("hybrid", "adamw"):
            raise ValueError(
                f"optimizer.type must be 'hybrid' or 'adamw', "
                f"got '{self.optimizer.type}'"
            )

    # ─── Serialization ────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SLMConfig":
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Fully validated SLMConfig instance.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        if raw is None:
            raw = {}

        # Flatten the 'model' section into top-level fields
        model_cfg = raw.pop("model", {})
        flat = {**model_cfg}

        # Keep sub-configs as nested dicts
        if "optimizer" in raw:
            flat["optimizer"] = raw["optimizer"]
        if "training" in raw:
            flat["training"] = raw["training"]

        return cls(**flat)

    def save_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file for reproducibility.

        Args:
            path: Destination path for the YAML file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self._to_serializable_dict()

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def _to_serializable_dict(self) -> dict[str, Any]:
        """Convert to nested dict matching YAML structure."""
        # Model fields (everything except sub-configs)
        model_fields = {}
        sub_config_names = {"optimizer", "training"}
        for k, v in asdict(self).items():
            if k not in sub_config_names:
                model_fields[k] = v

        return {
            "model": model_fields,
            "optimizer": _optimizer_to_dict(self.optimizer),
            "training": asdict(self.training),
        }

    def __repr__(self) -> str:
        return (
            f"SLMConfig(\n"
            f"  arch: {self.num_hidden_layers}L / {self.hidden_size}d / "
            f"{self.num_attention_heads}H / {self.vocab_size}V\n"
            f"  MLA: q_lora={self.q_lora_rank}, kv_lora={self.kv_lora_rank}, "
            f"rope_dim={self.qk_rope_head_dim}\n"
            f"  MTP: depth={self.mtp_depth}\n"
            f"  optimizer: {self.optimizer.type} @ lr={self.optimizer.base_lr}\n"
            f"  training: bs={self.training.batch_size}, seq={self.training.seq_len}, "
            f"steps={self.training.max_steps}\n"
            f")"
        )


def _optimizer_to_dict(opt: OptimizerConfig) -> dict:
    """Convert OptimizerConfig to dict, ensuring betas is a list for YAML."""
    d = asdict(opt)
    if "adamw" in d and "betas" in d["adamw"]:
        d["adamw"]["betas"] = list(d["adamw"]["betas"])
    return d
