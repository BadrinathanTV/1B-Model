import os
import time
import random
import argparse
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

BASE_DIR = "data/raw_corpus"

TARGET_TOKENS = {
    "starcoder/python": 14_000_000_000,
    "starcoder/sql": 3_500_000_000,
    "finemath": 17_500_000_000,
    "fineweb_edu": 10_000_000_000,
    "tamil": 5_000_000_000
}

DATASET_CONFIGS = {
    "starcoder/python": {"path": "bigcode/starcoderdata", "name": None, "data_dir": "python", "col": "content"},
    "starcoder/sql": {"path": "bigcode/starcoderdata", "name": None, "data_dir": "sql", "col": "content"},
    "finemath": {"path": "HuggingFaceFW/finemath", "name": "finemath-3B", "col": "text"},
    "fineweb_edu": {"path": "HuggingFaceFW/fineweb-edu", "name": "sample-10BT", "col": "text"},
    "tamil": {"path": "wikimedia/wikipedia", "name": "20231101.ta", "col": "text"}
}

def download_domain(name, target_tokens, tokenizer):
    config = DATASET_CONFIGS[name]
    output_dir = os.path.join(BASE_DIR, name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n🌐 Streaming and saving {name} until reaching {target_tokens/1e9:.2f}B tokens...")
    
    kwargs = {}
    if config["name"]:
        kwargs["name"] = config["name"]
    if "data_dir" in config:
        kwargs["data_dir"] = config["data_dir"]
        
    try:
        ds = load_dataset(config["path"], split="train", streaming=True, **kwargs)
    except Exception as e:
        print(f"❌ Failed to load stream for {name}: {e}")
        return
        
    chunk_texts = []
    chunk_tokens = 0
    total_tokens_written = 0
    file_idx = 0
    
    # Batch size for writing to disk to avoid excessive memory usage
    BATCH_TOKEN_TARGET = 50_000_000  # Write every 50M tokens
    
    for item in ds:
        col = config["col"]
        if col not in item:
            for alt in ["content", "text", "code"]:
                if alt in item:
                    col = alt
                    break
        text = item.get(col, "")
        if not text or not text.strip():
            continue
            
        # Tokenize to count exactly
        token_count = len(tokenizer.encode(text, add_special_tokens=True))
        
        chunk_texts.append(text)
        chunk_tokens += token_count
        
        if chunk_tokens >= BATCH_TOKEN_TARGET:
            # Save batch to parquet file
            table = pa.Table.from_arrays([pa.array(chunk_texts)], names=["text"])
            out_file = os.path.join(output_dir, f"part_{file_idx:04d}.parquet")
            pq.write_table(table, out_file)
            
            total_tokens_written += chunk_tokens
            print(f"✅ Wrote {chunk_tokens/1e6:.1f}M tokens to {out_file}. Progress: {total_tokens_written/1e9:.3f}B / {target_tokens/1e9:.2f}B")
            
            # Reset chunk buffers
            chunk_texts = []
            chunk_tokens = 0
            file_idx += 1
            
            if total_tokens_written >= target_tokens:
                break
                
    # Write remaining buffer
    if chunk_texts and total_tokens_written < target_tokens:
        table = pa.Table.from_arrays([pa.array(chunk_texts)], names=["text"])
        out_file = os.path.join(output_dir, f"part_{file_idx:04d}.parquet")
        pq.write_table(table, out_file)
        total_tokens_written += chunk_tokens
        print(f"✅ Wrote remaining {chunk_tokens/1e6:.1f}M tokens to {out_file}. Total: {total_tokens_written/1e9:.3f}B")

    print(f"🎉 Completed {name}! Total written: {total_tokens_written/1e9:.3f}B tokens.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default=None, help="Specific domain to download: starcoder/python, starcoder/sql, finemath, fineweb_edu, tamil")
    args = parser.parse_args()
    
    print("Loading Tokenizer...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained("models/tokenizer_bpe_65528_agentic_reasoning")
    
    if args.domain:
        if args.domain in TARGET_TOKENS:
            download_domain(args.domain, TARGET_TOKENS[args.domain], tokenizer)
        else:
            print(f"Invalid domain: {args.domain}. Choices: {list(TARGET_TOKENS.keys())}")
    else:
        for name, target in TARGET_TOKENS.items():
            download_domain(name, target, tokenizer)
            
if __name__ == "__main__":
    main()
