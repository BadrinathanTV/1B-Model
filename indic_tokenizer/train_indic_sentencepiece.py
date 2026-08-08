"""
Indic-Only SentencePiece 64k Tokenizer Training Pipeline for Sangraha Dataset
=============================================================================
This script handles:
1. Downloading & streaming 22 Indic language subsets from Sangraha dataset splits.
2. Fallback logic: 'verified' -> 'unverified' -> 'synthetic'.
3. Temperature Scaling (T = 3.0) up-sampling for dataset balancing.
4. SentencePiece training with script isolation, byte fallback, and 64k vocab size.
5. HuggingFace PreTrainedTokenizerFast conversion.
"""

import os
import math
import random
import glob
from collections import defaultdict

# 22 Indic Languages covered in AI4Bharat Sangraha
INDIC_LANGUAGES = [
    "asm", "ben", "brx", "doi", "gom", "guj", "hin", "kan", 
    "kas", "mai", "mal", "mar", "mni", "nep", "ori", "pan", 
    "san", "sat", "snd", "tam", "tel", "urd"
]

SPLIT_FALLBACK_ORDER = ["verified", "unverified", "synthetic"]
TARGET_BYTES_PER_LANG = 100 * 1024 * 1024  # 100 MB per language pool target
TEMPERATURE = 3.0  # Up-sampling temperature factor (T = 3.0)
VOCAB_SIZE = 64000  # 64k base vocab size (leaves room for uint16 special tokens)
OUTPUT_DIR = "models/indic_sentencepiece_64k"
CACHE_DIR = "corpus_cache"

def build_temperature_scaled_corpus():
    """Streams datasets from Sangraha, applies fallback hierarchy, and performs Temperature Up-Scaling."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    sampled_corpus_path = os.path.join(CACHE_DIR, "sampled_indic_corpus.txt")
    
    if os.path.exists(sampled_corpus_path) and os.path.getsize(sampled_corpus_path) > 10 * 1024 * 1024:
        print(f"✅ Reusing existing sampled Indic corpus at '{sampled_corpus_path}'")
        return sampled_corpus_path

    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        raise ImportError("Please install datasets and huggingface_hub: pip install datasets huggingface_hub")

    print("\n--- 1. Gathering Sangraha Data with Fallback Hierarchy ---")
    lang_files = defaultdict(list)
    lang_bytes = defaultdict(int)

    # 1. Check local extracted text first
    extracted_text_dirs = [
        "data/sangraha_full/extracted_text",
        "data/sangraha/extracted_text"
    ]

    # 2. Pre-fetch remote dataset file index from HF
    api = HfApi()
    try:
        print("🔍 Fetching Sangraha file index from Hugging Face...")
        all_repo_files = api.list_repo_files("ai4bharat/sangraha", repo_type="dataset")
        repo_parquet_files = [f for f in all_repo_files if f.endswith(".parquet")]
    except Exception as e:
        print(f"⚠️ Could not fetch HF repo index: {e}")
        repo_parquet_files = []

    LANG_ALIASES = {"nep": ["nep", "npi"], "ori": ["ori", "ory"]}

    for lang in INDIC_LANGUAGES:
        collected = 0
        print(f"\nProcessing language: {lang.upper()}")

        # Check local extracted text file first
        local_txt = None
        for edir in extracted_text_dirs:
            candidate = os.path.join(edir, f"{lang}.txt")
            if os.path.exists(candidate) and os.path.getsize(candidate) > 0:
                local_txt = candidate
                break

        if local_txt:
            print(f"  📁 Found local extracted text file: '{local_txt}'")
            split_file = os.path.join(CACHE_DIR, f"{lang}_local.txt")
            with open(local_txt, "r", encoding="utf-8") as in_f, open(split_file, "w", encoding="utf-8") as out_f:
                for line in in_f:
                    text = line.strip()
                    if text:
                        out_f.write(text + "\n")
                        collected += len(text.encode("utf-8"))
                        if collected >= TARGET_BYTES_PER_LANG:
                            break
            if os.path.exists(split_file) and os.path.getsize(split_file) > 0:
                lang_files[lang].append(split_file)
                print(f"    Added {os.path.getsize(split_file)/(1024*1024):.2f} MB from local text")
                lang_bytes[lang] = collected
                continue

        # Downloading remote parquet files via hf_hub_download + pyarrow
        possible_codes = LANG_ALIASES.get(lang, [lang])
        for split in SPLIT_FALLBACK_ORDER:
            if collected >= TARGET_BYTES_PER_LANG:
                break
            
            matched_files = []
            for fpath in repo_parquet_files:
                parts = fpath.split('/')
                if len(parts) >= 2 and parts[0] == split:
                    folder_base = parts[1].split('_')[0]
                    if folder_base in possible_codes:
                        matched_files.append(fpath)

            if not matched_files:
                continue

            print(f"  Fetching split '{split}' for {lang} ({len(matched_files)} files available)...")
            split_file = os.path.join(CACHE_DIR, f"{lang}_{split}.txt")
            
            with open(split_file, "w", encoding="utf-8") as out_f:
                for fpath in matched_files:
                    if collected >= TARGET_BYTES_PER_LANG:
                        break
                    try:
                        local_pfile = hf_hub_download(repo_id="ai4bharat/sangraha", filename=fpath, repo_type="dataset")
                        import pyarrow.parquet as pq
                        table = pq.read_table(local_pfile, columns=["text"])
                        for text_val in table["text"].to_pylist():
                            text = text_val.strip() if text_val else ""
                            if text:
                                out_f.write(text + "\n")
                                collected += len(text.encode("utf-8"))
                                if collected >= TARGET_BYTES_PER_LANG:
                                    break
                    except Exception as e:
                        print(f"    Warning: Failed to process {fpath}: {e}")

            if os.path.exists(split_file):
                if os.path.getsize(split_file) > 0:
                    lang_files[lang].append(split_file)
                    print(f"    Added {os.path.getsize(split_file)/(1024*1024):.2f} MB from {split}")
                else:
                    os.remove(split_file)

        lang_bytes[lang] = collected

    # Calculate Temperature Scaling probabilities
    print("\n--- 2. Computing Temperature-Scaled Probabilities (T = 3.0) ---")
    total_scaled = sum(math.pow(size, 1.0 / TEMPERATURE) for size in lang_bytes.values() if size > 0)
    if total_scaled == 0:
        raise RuntimeError("No data was collected from Sangraha dataset.")

    probs = {lang: (math.pow(size, 1.0 / TEMPERATURE) / total_scaled) for lang, size in lang_bytes.items() if size > 0}

    for lang in sorted(probs.keys()):
        raw_mb = lang_bytes[lang] / (1024 * 1024)
        print(f"  {lang:<5}: Raw = {raw_mb:>6.2f} MB | Sampling Probability = {probs[lang]*100:>5.2f}%")

    print("\n--- 3. Sampling Lines into Consolidated Corpus ---")
    TOTAL_TARGET_LINES = 3_000_000
    
    with open(sampled_corpus_path, "w", encoding="utf-8") as out_f:
        for lang, prob in probs.items():
            target_lines = int(TOTAL_TARGET_LINES * prob)
            lines_written = 0
            for file_path in lang_files[lang]:
                if lines_written >= target_lines:
                    break
                with open(file_path, "r", encoding="utf-8") as in_f:
                    for line in in_f:
                        out_f.write(line)
                        lines_written += 1
                        if lines_written >= target_lines:
                            break

    print(f"✅ Multi-script corpus generated: {sampled_corpus_path} ({os.path.getsize(sampled_corpus_path)/(1024*1024):.2f} MB)")
    return sampled_corpus_path

def train_sentencepiece(corpus_path):
    """Trains a 64k SentencePiece BPE tokenizer with Script isolation and Byte Fallback."""
    try:
        import sentencepiece as spm
    except ImportError:
        raise ImportError("Please install sentencepiece: pip install sentencepiece")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_prefix = os.path.join(OUTPUT_DIR, "spm_indic_64k")

    print("\n--- 4. Training SentencePiece Model ---")
    print(f"  Vocab Size: {VOCAB_SIZE}")
    print("  Byte Fallback: True")
    print("  Split by Unicode Script: True")
    print("  Split Digits: True")

    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=model_prefix,
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        character_coverage=0.9995,
        byte_fallback=True,
        split_digits=True,
        split_by_unicode_script=True,
        split_by_whitespace=True,
        max_sentence_length=16384,
        input_sentence_size=5000000,
        shuffle_input_sentence=True,
        num_threads=16,
        user_defined_symbols=[
            "<|im_start|>", "<|im_end|>", 
            "<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>",
            "<|thought|>", "<|end_thought|>"
        ]
    )

    print("✅ SentencePiece binary model generated successfully!")
    return f"{model_prefix}.model"

def convert_to_huggingface(spm_model_path):
    """Converts trained SentencePiece model into HuggingFace PreTrainedTokenizerFast format."""
    print("\n--- 5. Exporting to HuggingFace PreTrainedTokenizerFast ---")
    try:
        from transformers import LlamaTokenizerFast
        tokenizer = LlamaTokenizerFast(
            vocab_file=spm_model_path,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>"
        )
    except Exception as e:
        print(f"⚠️ Falling back to PreTrainedTokenizerFast: {e}")
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=spm_model_path,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>"
        )

    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"🎉 Final Indic Tokenizer ready at: {OUTPUT_DIR}")
    print(f"   Total Vocab Size: {len(tokenizer)}")

if __name__ == "__main__":
    corpus = build_temperature_scaled_corpus()
    spm_model = train_sentencepiece(corpus)
    convert_to_huggingface(spm_model)
