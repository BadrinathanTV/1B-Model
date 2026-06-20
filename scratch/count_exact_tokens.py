import os
import sys
import pyarrow.parquet as pq
from transformers import PreTrainedTokenizerFast

def count_tokens_in_file(file_path, tokenizer):
    try:
        table = pq.read_table(file_path, columns=["text"])
        total_tokens = 0
        texts = table["text"].to_pylist()
        for i, text in enumerate(texts):
            if text:
                tokens = tokenizer.encode(text, add_special_tokens=True)
                total_tokens += len(tokens)
        return total_tokens, len(texts)
    except Exception as e:
        return str(e), 0

def main():
    print("Loading tokenizer...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained("models/tokenizer_bpe_65528_agentic_reasoning")
    
    sample_files = [
        "data/raw_corpus/fineweb_edu/part_0000.parquet",
        "data/raw_corpus/finemath/part_0000.parquet",
        "data/raw_corpus/starcoder/python/part_0000.parquet",
        "data/raw_corpus/starcoder/sql/part_0000.parquet"
    ]
    
    print("\nCounting exact tokens in one sample parquet file from each domain:")
    print("-" * 85)
    print(f"{'Sample File':<50} | {'Docs':<8} | {'Exact Token Count':<18}")
    print("-" * 85)
    
    for f_path in sample_files:
        if os.path.exists(f_path):
            tokens, docs = count_tokens_in_file(f_path, tokenizer)
            print(f"{f_path:<50} | {docs:<8} | {tokens:,}")
        else:
            print(f"{f_path:<50} | File not found!")
    print("-" * 85)

if __name__ == "__main__":
    main()
