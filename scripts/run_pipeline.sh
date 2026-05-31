#!/bin/bash
set -e

# echo "========================================="
# echo "Step 1: Training Custom Tokenizer"
# echo "========================================="
# uv run python scripts/train_custom_tokenizer.py

echo "========================================="
echo "Step 2: Building AST-FIM Pretraining Corpus"
echo "========================================="
uv run python scripts/build_pretraining_corpus.py

echo "========================================="
echo "Step 3: Training the 1B Model"
echo "========================================="
uv run python training/train.py --config training/configs/default.yaml
