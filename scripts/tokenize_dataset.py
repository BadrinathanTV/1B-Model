"""
Tokenize and Pack Dataset for Indic LLM Pre-training
====================================================
Reads the data mixing manifest and creates pre-tokenized .bin memmap files
organized by curriculum phase.

Optimized for speed:
- Uses HuggingFace fast tokenizers (Rust-backed)
- Writes directly to numpy memmaps
- Chunks files to 2GB max for easy handling

Usage:
    uv run python scripts/tokenize_dataset.py
"""

import os
import json
import glob
import numpy as np
import time
from transformers import AutoTokenizer
from tqdm import tqdm

# ─── Configuration ─────────────────────────────────────────────────────────────

MANIFEST_PATH = "data/pretrain_mix/manifest.json"
TOKENIZER_PATH = "models/indic_sentencepiece_64k"
OUTPUT_DIR = "data/tokenized_curriculum"
MAX_BIN_TOKENS = 100_000_000  # ~200MB per .bin file (uint16)

# ─── Data Generators ─────────────────────────────────────────────────────────
# In a real environment, these would stream from HuggingFace or local parquet files.
# We use generator functions to simulate infinite streaming.

def get_monolingual_stream(lang: str):
    """
    Yields raw text strings for a specific language from local files.
    """
    if lang == "eng":
        candidate_paths = ["data/english/fineweb_edu.txt"]
    elif lang == "code":
        candidate_paths = ["data/code/starcoder_python.txt", "data/code/codeparrot_clean.txt"]
    else:
        # Check all possible locations where Sangraha text might be stored
        candidate_paths = [
            f"data/sangraha_full/extracted_text/{lang}.txt",
            f"data/sangraha/extracted_text/{lang}.txt",
            f"corpus_cache/{lang}_local.txt",
        ] + glob.glob(f"corpus_cache/{lang}_*.txt")

    file_path = None
    for cp in candidate_paths:
        if os.path.exists(cp) and os.path.getsize(cp) > 0:
            file_path = cp
            break
        
    if not file_path:
        # Fallback if file doesn't exist to prevent crashing the whole pipeline
        print(f"⚠️ Warning: Data for {lang} not found in expected paths. Skipping.")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if lang == "code":
                # Restore newlines for code that were escaped during download
                text = text.replace("\\n", "\n")
            if text:
                yield text

def get_parallel_stream(src: str, tgt: str):
    """
    Yields interleaved bilingual text from IndicTrans2 parallel corpus.
    Format: <|lang:src|> Text <|lang:tgt|> Translation
    """
    file_path = f"data/parallel/{src}-{tgt}.jsonl"
    if not os.path.exists(file_path):
        # We also check the reverse direction just in case
        rev_path = f"data/parallel/{tgt}-{src}.jsonl"
        if os.path.exists(rev_path):
            file_path = rev_path
        else:
            print(f"⚠️ Warning: Parallel data {src}-{tgt} not found. Skipping.")
            return

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                # Assuming JSONL has 'src_text' and 'tgt_text' keys
                # Or 'en' and 'hi' keys. We use generic fetching if possible
                src_text = data.get(src, data.get('src_text', ''))
                tgt_text = data.get(tgt, data.get('tgt_text', ''))
                
                if src_text and tgt_text:
                    yield f"<|lang:{src}|> {src_text} <|lang:{tgt}|> {tgt_text}"
            except Exception:
                continue

# ─── Tokenization Pipeline ───────────────────────────────────────────────────

def pack_tokens(token_stream, output_prefix: str, target_tokens: int, tokenizer):
    """
    Consumes a token stream and packs it into .bin memmap files.
    """
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    
    tokens_written = 0
    file_idx = 0
    buffer = []
    buffer_size = 1_000_000  # Write in 1M token chunks

    pbar = tqdm(total=target_tokens, desc=os.path.basename(output_prefix), unit="tok", leave=False)

    while tokens_written < target_tokens:
        try:
            text = next(token_stream)
            # Tokenize without adding special tokens (handled by document boundaries if needed)
            tokens = tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(tokens)
            buffer.append(tokenizer.eos_token_id)  # Separate documents with EOS
        except StopIteration:
            break

        if len(buffer) >= buffer_size or (tokens_written + len(buffer)) >= target_tokens:
            # How many tokens can we actually write before hitting the target?
            to_write = min(len(buffer), target_tokens - tokens_written)
            chunk = buffer[:to_write]
            buffer = buffer[to_write:]

            # Which bin file are we writing to?
            bin_path = f"{output_prefix}_{file_idx:04d}.bin"
            
            # If file doesn't exist, create it. If it does, append.
            # Using append mode with numpy memmap requires careful sizing. 
            # A simpler approach for sequential writing is appending to a standard file:
            arr = np.array(chunk, dtype=np.uint16)
            with open(bin_path, "ab") as f:
                f.write(arr.tobytes())

            tokens_written += len(chunk)
            pbar.update(len(chunk))

            # Rotate file if it exceeds MAX_BIN_TOKENS
            if os.path.getsize(bin_path) >= MAX_BIN_TOKENS * 2:  # 2 bytes per uint16 token
                file_idx += 1

    pbar.close()
    return tokens_written

# ─── Main Execution ──────────────────────────────────────────────────────────

def main():
    if not os.path.exists(MANIFEST_PATH):
        print(f"❌ Manifest not found at {MANIFEST_PATH}. Run prepare_pretrain_mix.py first.")
        return

    print("📖 Loading Manifest and Tokenizer...")
    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    except Exception as e:
        print(f"❌ Failed to load tokenizer from {TOKENIZER_PATH}: {e}")
        return

    print(f"✅ Tokenizer loaded. Vocab size: {len(tokenizer)}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    curriculum = manifest.get("curriculum", [])
    
    total_phases = len(curriculum)
    print(f"\n🚀 Starting Tokenization across {total_phases} Phases...\n")

    for phase_idx, phase in enumerate(curriculum, 1):
        phase_name = phase["name"].replace(" ", "_").lower()
        print(f"{"="*50}")
        print(f" Phase {phase_idx}: {phase['name']}")
        print(f" Target Tokens: {phase['total_tokens'] / 1e9:.2f}B")
        print(f"{"="*50}")

        phase_dir = os.path.join(OUTPUT_DIR, f"phase_{phase_idx}_{phase_name}")
        
        # 1. Process Monolingual Data
        mono_budgets = phase.get("monolingual", {})
        for lang, target_tokens in mono_budgets.items():
            if target_tokens <= 0:
                continue
            
            # For demonstration, we'll process a small fraction so the script finishes quickly
            # In production, remove this cap!
            target_tokens = min(target_tokens, 50_000) # Cap at 50k tokens per lang for testing
            
            out_prefix = os.path.join(phase_dir, "monolingual", lang)
            stream = get_monolingual_stream(lang)
            pack_tokens(stream, out_prefix, target_tokens, tokenizer)

        # 2. Process Parallel Interleaved Data
        parallel_budgets = phase.get("parallel", {})
        for pair, target_tokens in parallel_budgets.items():
            if target_tokens <= 0:
                continue
            
            target_tokens = min(target_tokens, 20_000) # Cap at 20k tokens per pair for testing
            
            out_prefix = os.path.join(phase_dir, "parallel", pair)
            src, tgt = pair.split("-")
            stream = get_parallel_stream(src, tgt)
            pack_tokens(stream, out_prefix, target_tokens, tokenizer)
            
    print(f"\n🎉 Dataset tokenization complete! Data saved to {OUTPUT_DIR}")
    print("Next step: Update training/train.py to load phases dynamically.")

if __name__ == "__main__":
    main()
