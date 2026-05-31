"""
Optimizers Package
===================

Production optimizers for the 1B SLM:
  - HybridSLMOptimizer: Routes 2D→Aurora, 1D→AdamW4bit
  - SFAurora: Schedule-Free Aurora (leverage-uniform spectral optimizer)
  - SFNorMuon: Schedule-Free NorMuon (per-neuron normalized spectral optimizer)
  - build_optimizer: Config-driven factory
"""

from .hybrid import HybridSLMOptimizer
from .nf_aurora import NFAurora
from .nf_aurora_hybrid import NFAuroraHybrid
from .nf_normuon_hybrid import NFNorMuonHybrid
from .sf_normuon import SFNorMuon
from .factory import build_optimizer

# Backward compatibility alias
SFAurora = NFAurora

__all__ = [
    "HybridSLMOptimizer",
    "NFAurora",
    "NFAuroraHybrid",
    "NFNorMuonHybrid",
    "SFAurora",
    "SFNorMuon",
    "build_optimizer",
]
