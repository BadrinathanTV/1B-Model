"""
Optimizers Package
===================

Production optimizers for the 1B SLM:
  - HybridSLMOptimizer: Routes 2D→Aurora, 1D→AdamW4bit
  - NFAurora: Schedule-Free + Aurora (experimental)
  - SFNorMuon: Schedule-Free NorMuon (benchmark baseline)
  - build_optimizer: Config-driven factory
"""

from .hybrid import HybridSLMOptimizer
from .nf_aurora import NFAurora
from .sf_normuon import SFNorMuon
from .factory import build_optimizer

__all__ = [
    "HybridSLMOptimizer",
    "NFAurora",
    "SFNorMuon",
    "build_optimizer",
]
