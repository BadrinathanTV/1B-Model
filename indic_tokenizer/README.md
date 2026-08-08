# Indic SentencePiece Tokenizer Suite (64k Vocab)

This directory contains the pipeline for downloading the AI4Bharat Sangraha dataset, training a 64,000 vocabulary Indic-only SentencePiece tokenizer, and evaluating fertility rates.

## Directory Structure
- `download_sangraha_dataset.py`: Downloads all 2,492 parquet files (~705 GB) from Sangraha (`ai4bharat/sangraha`) covering 22 Indian scheduled languages across all splits (`verified`, `unverified`, `synthetic`), featuring automatic retry loops, progress tracking, disk space checking, and text extraction.
- `train_indic_sentencepiece.py`: Streams/loads Sangraha datasets, applies Temperature Scaling ($T=3.0$), and trains SentencePiece (64k vocab, script isolation, byte fallback) exported to HuggingFace format.
- `evaluate_indic_fertility.py`: Benchmarks subword fertility rate (tokens per word) across Indic script families.

## Commands

### 1. Download Dataset

```bash
# Download COMPLETE 705 GB Sangraha Indic dataset (2,492 parquet files across 22 Indic languages)
uv run python indic_tokenizer/download_sangraha_dataset.py --full --target_dir data/sangraha_full

# Download dataset AND extract plain text for training
uv run python indic_tokenizer/download_sangraha_dataset.py --full --target_dir data/sangraha_full --extract_text

# Subset download (e.g. 100 MB per language for fast tokenizer training)
uv run python indic_tokenizer/download_sangraha_dataset.py --target_mb_per_lang 100 --extract_text
```

### 2. Train Tokenizer
```bash
uv run python indic_tokenizer/train_indic_sentencepiece.py
```

### 3. Evaluate Fertility Rate
```bash
uv run python indic_tokenizer/evaluate_indic_fertility.py
```
