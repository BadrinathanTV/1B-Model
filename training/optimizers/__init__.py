"""
Optimizers Package
==================

Production optimizers for the 1B SLM:
  - HybridSLMOptimizer: Routes 2D→Aurora, 1D→AdamW
  - build_optimizer: Config-driven factory
"""

from .hybrid import HybridSLMOptimizer
from .factory import build_optimizer

__all__ = [
    "HybridSLMOptimizer",
    "build_optimizer",
]
