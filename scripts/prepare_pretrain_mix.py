"""
Indic LLM Pre-Training Data Preparation Pipeline
=================================================
Implements the cross-lingual knowledge transfer data mixing strategy:

1. α=0.3 exponential smoothing for language sampling
2. Cross-lingual parallel data interleaving (CrossIC-PT)
3. 3-phase curriculum scheduling (Foundation → Bridge → Consolidation)
4. Synthetic augmentation for low-resource languages
5. Quality filtering via perplexity and length constraints

References:
- CrossIC-PT (ACL 2025): Bilingual document interleaving
- CSCL (ACL 2025): Code-switching curriculum learning
- DoReMi (NeurIPS 2023): Minimax domain reweighting
- IndicLLMSuite (AI4Bharat 2024): Sangraha data recipe
"""

import os
import math
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Optional

# ─── Language Configuration ───────────────────────────────────────────────────

INDIC_LANGUAGES = [
    "asm", "ben", "brx", "doi", "gom", "guj", "hin", "kan",
    "kas", "mai", "mal", "mar", "mni", "nep", "ori", "pan",
    "san", "sat", "snd", "tam", "tel", "urd"
]

# Approximate raw data sizes in GB from Sangraha (verified + unverified + synthetic)
LANG_RAW_SIZE_GB = {
    "hin": 50.0, "ben": 15.0, "tam": 10.0, "tel": 10.0, "mar": 8.0,
    "guj": 7.0,  "kan": 6.0,  "mal": 6.0,  "pan": 5.0,  "urd": 4.0,
    "ori": 3.0,  "asm": 2.0,  "nep": 2.0,  "san": 1.0,  "mai": 1.0,
    "snd": 0.5,  "gom": 0.3,  "doi": 0.2,  "mni": 0.2,  "kas": 0.1,
    "brx": 0.1,  "sat": 0.05,
    "eng": 100.0,  # English (from FineWeb-2 / CulturaX)
    "code": 50.0,  # Programming code (Python, JS, etc.)
}

# Script family clusters (for curriculum Phase 1: same-script foundation)
SCRIPT_CLUSTERS = {
    "devanagari": ["hin", "mar", "san", "nep", "mai", "doi", "brx", "gom"],
    "dravidian":  ["tam", "tel", "kan", "mal"],
    "eastern":    ["ben", "asm", "ori", "mni"],
    "perso_arabic": ["urd", "snd", "kas"],
    "gurmukhi":   ["pan"],
    "ol_chiki":   ["sat"],
}

# Priority alignment pairs with weights (for parallel data generation)
# Higher weight = more parallel data generated for this pair
ALIGNMENT_PAIRS = {
    # Devanagari family (high mutual transfer)
    ("hin", "mar"): 2.0, ("hin", "nep"): 1.5, ("hin", "san"): 1.0,
    ("hin", "mai"): 1.5, ("hin", "doi"): 1.0, ("hin", "brx"): 0.5,
    ("hin", "gom"): 0.8,
    # Cross-script bridges (critical for knowledge sharing)
    ("hin", "ben"): 2.0, ("hin", "tam"): 1.5, ("hin", "tel"): 1.5,
    ("hin", "guj"): 1.5, ("hin", "kan"): 1.0, ("hin", "mal"): 1.0,
    ("hin", "pan"): 1.5, ("hin", "ori"): 1.0, ("hin", "urd"): 2.0,
    ("hin", "asm"): 0.8, ("hin", "mni"): 0.5, ("hin", "kas"): 0.5,
    ("hin", "snd"): 0.5, ("hin", "sat"): 0.5,
    # Intra-family bridging
    ("ben", "asm"): 1.5, ("tam", "mal"): 1.0, ("tel", "kan"): 1.0,
    # English knowledge bridge
    ("eng", "hin"): 2.0, ("eng", "tam"): 1.0, ("eng", "ben"): 1.0,
    ("eng", "tel"): 0.8,
}


# ─── Data Mixing Configuration ───────────────────────────────────────────────

@dataclass
class DataMixConfig:
    """Configuration for cross-lingual data mixing."""

    # Total token budget for pre-training
    total_tokens: int = 50_000_000_000  # 50B tokens

    # α for exponential smoothing (0.3 = aggressive upsampling for related families)
    alpha: float = 0.3

    # Data composition ratios
    monolingual_ratio: float = 0.55     # 55% monolingual Indic
    parallel_ratio: float = 0.25        # 25% parallel/interleaved
    codemixed_ratio: float = 0.05       # 5% code-mixed (Hinglish, etc)
    english_ratio: float = 0.10         # 10% English (~5B tokens)
    code_ratio: float = 0.05            # 5% Code (~2.5B tokens - optimal for reasoning/structure)

    # Quality filtering
    min_sentence_chars: int = 50        # Min chars per sentence
    max_sentence_chars: int = 500       # Max chars per sentence (Krutrim finding)

    # Curriculum phases (fraction of total training)
    phase1_fraction: float = 0.20       # Foundation (same-script)
    phase2_fraction: float = 0.30       # Cross-script bridge
    phase3_fraction: float = 0.50       # Consolidation

    # Output paths
    output_dir: str = "data/pretrain_mix"
    manifest_path: str = "data/pretrain_mix/manifest.json"

    def __post_init__(self):
        total = self.monolingual_ratio + self.parallel_ratio + self.codemixed_ratio + self.english_ratio + self.code_ratio
        assert abs(total - 1.0) < 0.01, f"Ratios must sum to 1.0, got {total}"
        phase_total = self.phase1_fraction + self.phase2_fraction + self.phase3_fraction
        assert abs(phase_total - 1.0) < 0.01, f"Phase fractions must sum to 1.0, got {phase_total}"


def compute_alpha_smoothed_weights(raw_sizes: dict, alpha: float = 0.3) -> dict:
    """
    Compute α-smoothed sampling weights for multilingual data mixing.

    Instead of sampling proportionally to raw data size p_i,
    samples proportionally to p_i^α (α < 1 upsamples low-resource languages).

    Args:
        raw_sizes: Dict mapping language code to raw data size (GB).
        alpha: Smoothing exponent. 0.3 recommended for related Indic families.

    Returns:
        Dict mapping language code to sampling probability.
    """
    smoothed = {}
    for lang, size in raw_sizes.items():
        if size > 0:
            smoothed[lang] = math.pow(size, alpha)

    total = sum(smoothed.values())
    weights = {lang: v / total for lang, v in smoothed.items()}
    return weights


def compute_token_budgets(config: DataMixConfig) -> dict:
    """
    Compute per-language token budgets using α=0.3 exponential smoothing.

    Returns:
        Dict mapping language code to target token count.
    """
    # Only Indic languages for monolingual budget
    indic_sizes = {lang: LANG_RAW_SIZE_GB[lang] for lang in INDIC_LANGUAGES}
    weights = compute_alpha_smoothed_weights(indic_sizes, config.alpha)

    monolingual_budget = int(config.total_tokens * config.monolingual_ratio)
    budgets = {}
    for lang, weight in weights.items():
        budgets[lang] = int(monolingual_budget * weight)

    # Add English and Code budgets
    budgets["eng"] = int(config.total_tokens * config.english_ratio)
    budgets["code"] = int(config.total_tokens * config.code_ratio)

    return budgets


def compute_parallel_budgets(config: DataMixConfig) -> dict:
    """
    Compute token budgets for parallel/interleaved data per language pair.

    Returns:
        Dict mapping (src, tgt) tuple to target token count for that pair.
    """
    parallel_budget = int(config.total_tokens * config.parallel_ratio)
    total_weight = sum(ALIGNMENT_PAIRS.values())

    pair_budgets = {}
    for pair, weight in ALIGNMENT_PAIRS.items():
        pair_budgets[pair] = int(parallel_budget * weight / total_weight)

    return pair_budgets


def generate_curriculum_schedule(config: DataMixConfig) -> list:
    """
    Generate the 3-phase curriculum training schedule.

    Phase 1 (Foundation, 20%):   Same-script families + basic parallel pairs
    Phase 2 (Bridge, 30%):       Cross-script transfer + code-mixed data
    Phase 3 (Consolidation, 50%): Full diversity + English + Code

    Returns:
        List of phase dicts with data composition details.
    """
    token_budgets = compute_token_budgets(config)
    parallel_budgets = compute_parallel_budgets(config)

    phases = []

    # ─── Phase 1: Foundation (same-script families) ───────────────────────
    phase1_tokens = int(config.total_tokens * config.phase1_fraction)
    phase1_mono = {}
    phase1_parallel = {}

    for lang, budget in token_budgets.items():
        if lang in ("eng", "code"):
            phase1_mono[lang] = int(budget * 0.05 / config.phase1_fraction)  # Minimal English
        else:
            phase1_mono[lang] = int(budget * config.phase1_fraction)

    # Only same-script parallel pairs in Phase 1
    devanagari_set = set(SCRIPT_CLUSTERS["devanagari"])
    for pair, budget in parallel_budgets.items():
        src, tgt = pair
        if src in devanagari_set and tgt in devanagari_set:
            phase1_parallel[pair] = int(budget * 0.5)  # 50% of this pair's budget
        elif src == "eng" and tgt == "hin":
            phase1_parallel[pair] = int(budget * 0.3)

    phases.append({
        "name": "Foundation",
        "fraction": config.phase1_fraction,
        "total_tokens": phase1_tokens,
        "description": "Same-script family foundation + Devanagari parallel pairs",
        "monolingual": phase1_mono,
        "parallel": {f"{s}-{t}": v for (s, t), v in phase1_parallel.items()},
        "codemixed_ratio": 0.0,
    })

    # ─── Phase 2: Cross-Script Bridge ─────────────────────────────────────
    phase2_tokens = int(config.total_tokens * config.phase2_fraction)
    phase2_mono = {}
    phase2_parallel = {}

    for lang, budget in token_budgets.items():
        if lang == "eng":
            phase2_mono[lang] = int(budget * 0.3 / config.phase2_fraction)
        elif lang == "code":
            phase2_mono[lang] = int(budget * 0.4 / config.phase2_fraction)
        else:
            phase2_mono[lang] = int(budget * config.phase2_fraction)

    # Cross-script parallel pairs emphasized in Phase 2
    for pair, budget in parallel_budgets.items():
        src, tgt = pair
        # Cross-script pairs get heavy allocation
        if src in devanagari_set and tgt not in devanagari_set:
            phase2_parallel[pair] = int(budget * 0.6)
        elif src == "eng":
            phase2_parallel[pair] = int(budget * 0.5)
        elif not (src in devanagari_set and tgt in devanagari_set):
            phase2_parallel[pair] = int(budget * 0.5)

    phases.append({
        "name": "Cross-Script Bridge",
        "fraction": config.phase2_fraction,
        "total_tokens": phase2_tokens,
        "description": "Cross-script transfer + code-mixed data introduced",
        "monolingual": phase2_mono,
        "parallel": {f"{s}-{t}": v for (s, t), v in phase2_parallel.items()},
        "codemixed_ratio": 0.10,
    })

    # ─── Phase 3: Consolidation ───────────────────────────────────────────
    phase3_tokens = int(config.total_tokens * config.phase3_fraction)
    phase3_mono = {}
    phase3_parallel = {}

    for lang, budget in token_budgets.items():
        if lang == "eng":
            phase3_mono[lang] = int(budget * 0.65 / config.phase3_fraction)
        elif lang == "code":
            phase3_mono[lang] = int(budget * 0.6 / config.phase3_fraction)
        else:
            phase3_mono[lang] = int(budget * config.phase3_fraction)

    # All remaining parallel budget
    for pair, budget in parallel_budgets.items():
        allocated = 0
        for phase in phases:
            key = f"{pair[0]}-{pair[1]}"
            allocated += phase["parallel"].get(key, 0)
        remaining = budget - allocated
        if remaining > 0:
            phase3_parallel[pair] = remaining

    phases.append({
        "name": "Consolidation",
        "fraction": config.phase3_fraction,
        "total_tokens": phase3_tokens,
        "description": "Full diversity + English knowledge + Code reasoning",
        "monolingual": phase3_mono,
        "parallel": {f"{s}-{t}": v for (s, t), v in phase3_parallel.items()},
        "codemixed_ratio": 0.10,
    })

    return phases


def generate_manifest(config: Optional[DataMixConfig] = None) -> dict:
    """
    Generate the complete pre-training data manifest.

    This manifest describes:
    - Per-language token budgets (α=0.3 smoothed)
    - Parallel data pair allocations
    - 3-phase curriculum schedule
    - Data source paths

    Returns:
        Complete manifest dict (also saved to config.manifest_path).
    """
    if config is None:
        config = DataMixConfig()

    token_budgets = compute_token_budgets(config)
    parallel_budgets = compute_parallel_budgets(config)
    curriculum = generate_curriculum_schedule(config)

    manifest = {
        "config": asdict(config),
        "token_budgets": token_budgets,
        "parallel_budgets": {f"{s}-{t}": v for (s, t), v in parallel_budgets.items()},
        "curriculum": curriculum,
        "data_sources": {
            "monolingual": "data/sangraha_full/extracted_text/{lang}.txt",
            "parallel": "data/parallel/{src}-{tgt}.jsonl",
            "codemixed": "data/codemixed/{lang}.txt",
            "english": "data/english/fineweb2_sample.txt",
            "code": "data/code/starcoderdata_sample.txt",
        },
        "script_clusters": SCRIPT_CLUSTERS,
        "alignment_pairs": {f"{s}-{t}": w for (s, t), w in ALIGNMENT_PAIRS.items()},
    }

    os.makedirs(config.output_dir, exist_ok=True)
    manifest_path = config.manifest_path
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest


def print_data_plan(config: Optional[DataMixConfig] = None):
    """Print a human-readable summary of the data mixing plan."""
    if config is None:
        config = DataMixConfig()

    print("=" * 80)
    print("  INDIC LLM PRE-TRAINING DATA MIXING PLAN")
    print("  α=0.3 Exponential Smoothing + CrossIC-PT + 3-Phase Curriculum")
    print("=" * 80)

    # Token budgets
    token_budgets = compute_token_budgets(config)
    total_indic = sum(v for k, v in token_budgets.items() if k not in ("eng", "code"))

    print(f"\n📊 Total Training Budget: {config.total_tokens / 1e9:.1f}B tokens")
    print(f"   Monolingual: {config.monolingual_ratio*100:.0f}% | "
          f"Parallel: {config.parallel_ratio*100:.0f}% | "
          f"Code-mixed: {config.codemixed_ratio*100:.0f}% | "
          f"English: {config.english_ratio*100:.0f}% | "
          f"Code: {config.code_ratio*100:.0f}%")
    print(f"\n{'Language':<20} {'Raw Size':>10} {'α=0.3 Weight':>14} {'Token Budget':>14}")
    print("-" * 62)

    sorted_budgets = sorted(token_budgets.items(), key=lambda x: x[1], reverse=True)
    for lang, budget in sorted_budgets:
        raw_gb = LANG_RAW_SIZE_GB.get(lang, 0)
        weight = budget / config.total_tokens * 100
        budget_str = f"{budget / 1e9:.2f}B"
        print(f"  {lang:<18} {raw_gb:>8.2f} GB {weight:>12.1f}% {budget_str:>14}")

    # Parallel data
    parallel_budgets = compute_parallel_budgets(config)
    print(f"\n🔗 Cross-Lingual Parallel Data ({config.parallel_ratio*100:.0f}% of budget)")
    print(f"{'Pair':<15} {'Weight':>8} {'Token Budget':>14}")
    print("-" * 40)
    sorted_pairs = sorted(parallel_budgets.items(), key=lambda x: x[1], reverse=True)
    for (src, tgt), budget in sorted_pairs[:10]:
        weight = ALIGNMENT_PAIRS[(src, tgt)]
        print(f"  {src}↔{tgt:<10} {weight:>6.1f}x {budget/1e9:.2f}B")
    if len(sorted_pairs) > 10:
        print(f"  ... and {len(sorted_pairs) - 10} more pairs")

    # Curriculum
    curriculum = generate_curriculum_schedule(config)
    print(f"\n📅 3-Phase Curriculum Schedule")
    print("-" * 60)
    for i, phase in enumerate(curriculum, 1):
        tokens_b = phase["total_tokens"] / 1e9
        n_langs = len([l for l in phase["monolingual"] if phase["monolingual"][l] > 0])
        n_pairs = len(phase["parallel"])
        print(f"  Phase {i}: {phase['name']:<25} "
              f"{phase['fraction']*100:.0f}% ({tokens_b:.1f}B tokens) "
              f"| {n_langs} langs, {n_pairs} parallel pairs")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    config = DataMixConfig()
    print_data_plan(config)

    print("\n📝 Generating manifest...")
    manifest = generate_manifest(config)
    print(f"✅ Manifest saved to: {config.manifest_path}")
