"""
Download Auxiliary Pre-Training Data (English & Code)
======================================================
Downloads high-quality English (FineWeb-Edu) and Code (StarCoderData) 
datasets from HuggingFace to augment the Indic LLM pre-training curriculum.

Outputs are saved as plain text files line-by-line for fast tokenization.
"""

import os
import argparse
from datasets import load_dataset
from tqdm import tqdm

def download_english_data(output_dir: str, target_tokens_billions: float = 5.0):
    """
    Downloads highly curated educational English data from FineWeb-Edu.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "fineweb_edu.txt")
    
    if os.path.exists(out_file) and os.path.getsize(out_file) > 1024 * 1024:
        print(f"✅ English data already exists at {out_file}")
        return

    print(f"\n📚 Downloading {target_tokens_billions}B tokens of English (FineWeb-Edu)...")
    
    # 1 Billion tokens is roughly ~3.5GB to 4GB of raw text.
    target_bytes = int(target_tokens_billions * 3.5 * 1024 * 1024 * 1024)
    bytes_written = 0

    # Use streaming to avoid downloading massive caches
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    
    with open(out_file, "w", encoding="utf-8") as f:
        pbar = tqdm(total=target_bytes, desc="English Data", unit="B", unit_scale=True)
        for row in ds:
            text = row["text"].strip().replace("\n", " ")
            if len(text) > 50:
                f.write(text + "\n")
                bytes_added = len(text.encode("utf-8"))
                bytes_written += bytes_added
                pbar.update(bytes_added)
                
            if bytes_written >= target_bytes:
                break
        pbar.close()

def download_code_data(output_dir: str, target_tokens_billions: float = 2.5):
    """
    Downloads high-quality programming code (Python) from StarCoder.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "starcoder_python.txt")
    
    if os.path.exists(out_file) and os.path.getsize(out_file) > 1024 * 1024:
        print(f"✅ Code data already exists at {out_file}")
        return

    print(f"\n💻 Downloading {target_tokens_billions}B tokens of Code (StarCoder - Python)...")
    
    target_bytes = int(target_tokens_billions * 3.5 * 1024 * 1024 * 1024)
    bytes_written = 0

    # Stream Python code from 100% open, non-gated codeparrot/codeparrot-clean dataset
    ds = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
    
    with open(out_file, "w", encoding="utf-8") as f:
        pbar = tqdm(total=target_bytes, desc="Code Data", unit="B", unit_scale=True)
        for row in ds:
            # We don't strip newlines for code, as whitespace is semantic in Python
            text = row["content"].strip()
            if len(text) > 100:
                # Replace internal newlines with a literal representation if we want strict line-by-line,
                # or just write as is and separate files. To keep streaming robust, we can wrap code in tags
                # and keep newlines, OR just write one Python file content per line by escaping newlines.
                # Let's escape newlines to keep the "one document per line" format for our tokenizer stream.
                text_escaped = text.replace("\n", "\\n")
                f.write(text_escaped + "\n")
                
                bytes_added = len(text_escaped.encode("utf-8"))
                bytes_written += bytes_added
                pbar.update(bytes_added)
                
            if bytes_written >= target_bytes:
                break
        pbar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eng_tokens", type=float, default=5.0, help="Target English tokens in Billions")
    parser.add_argument("--code_tokens", type=float, default=2.5, help="Target Code tokens in Billions")
    args = parser.parse_args()

    download_english_data("data/english", target_tokens_billions=args.eng_tokens)
    download_code_data("data/code", target_tokens_billions=args.code_tokens)
    print("\n🎉 Auxiliary data download complete!")
