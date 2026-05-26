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

import copy
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


# ─── Sub-configs ──────────────────────────────────────────────────────────────

@dataclass
class PrecisionConfig:
    """NVFP4 and mixed-precision routing settings."""
    high_precision_start_layers: int = 2
    high_precision_end_layers: int = 4
    nvfp4_disable_rht: bool = True
    nvfp4_disable_stochastic_rounding: bool = True


@dataclass
class AuroraConfig:
    """Riemannian Aurora optimizer hyperparameters (for 2D matrix weights)."""
    momentum: float = 0.95
    weight_decay: float = 0.1
    nesterov: bool = True
    use_riemannian: bool = True


@dataclass
class AdamWConfig:
    """4-bit quantized AdamW hyperparameters (for 1D params, embeddings, norms)."""
    lr_scale: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-10
    weight_decay: float = 0.1


@dataclass
class OptimizerConfig:
    """Top-level optimizer configuration with param-group routing."""
    type: str = "hybrid"            # "hybrid" | "nf_aurora" | "adamw"
    base_lr: float = 1e-3
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
    mtp_loss_weight: float = 0.3
    gradient_clip: float = 1.0
    log_interval: int = 1
    seed: int = 42
    device: str = "auto"            # "auto" | "cuda" | "cpu"


# ─── Main Config ─────────────────────────────────────────────────────────────

@dataclass
class SLMConfig:
    """Complete configuration for the 1B SLM.

    Contains model architecture, precision routing, optimizer, and training
    settings in a single validated object.

    Example:
        config = SLMConfig.from_yaml("configs/default.yaml")
        config = SLMConfig(vocab_size=32000, hidden_size=1024)
    """

    # Model architecture
    vocab_size: int = 64000
    hidden_size: int = 2048
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 8192
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-6

    # MLA (Multi-Head Latent Attention)
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    rope_theta: float = 10000.0

    # TST (Token Superposition Training)
    tst_group_size: int = 4

    # MTP (Multi-Token Prediction)
    mtp_depth: int = 3

    # Weight Tying & Memory Optimizations
    tie_word_embeddings: bool = True
    max_delta_history: int = 0  # 0 means keep full history

    # Sub-configs
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        """Convert nested dicts to dataclasses and run validation."""
        if isinstance(self.precision, dict):
            self.precision = PrecisionConfig(**self.precision)
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
        if self.tst_group_size < 1:
            raise ValueError(f"tst_group_size must be >= 1, got {self.tst_group_size}")
        if self.mtp_depth < 1:
            raise ValueError(f"mtp_depth must be >= 1, got {self.mtp_depth}")
        if self.max_delta_history < 0:
            raise ValueError(f"max_delta_history must be >= 0, got {self.max_delta_history}")

        total_hp = self.precision.high_precision_start_layers + self.precision.high_precision_end_layers
        if total_hp > self.num_hidden_layers:
            raise ValueError(
                f"high_precision layers ({total_hp}) exceed total layers ({self.num_hidden_layers})"
            )

        if self.optimizer.type not in ("hybrid", "nf_aurora", "adamw"):
            raise ValueError(
                f"optimizer.type must be 'hybrid', 'nf_aurora', or 'adamw', "
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
        if "precision" in raw:
            flat["precision"] = raw["precision"]
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
        sub_config_names = {"precision", "optimizer", "training"}
        for k, v in asdict(self).items():
            if k not in sub_config_names:
                model_fields[k] = v

        return {
            "model": model_fields,
            "precision": asdict(self.precision),
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
            f"  TST: group_size={self.tst_group_size}\n"
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
