"""
Dataset Downloader using Token/GB estimation.

For each dataset:
1. Identify available parquet files from the Hugging Face repo.
2. If a specific subset is available, use it directly.
3. Download a single sample file to calculate the tokens per GB ratio.
4. Using the ratio, calculate the total disk size (GB) required to reach the target tokens.
5. Select and download the exact number of files needed to hit the target GB.
"""

import os
import argparse
from huggingface_hub import hf_hub_download, list_repo_tree, snapshot_download
import pyarrow.parquet as pq
from transformers import PreTrainedTokenizerFast

BASE_DIR = "data/raw_corpus"

# Includes a 10% surplus
TARGET_TOKENS = {
    "starcoder/python": int(15_000_000_000 * 1.1),
    "starcoder/sql": int(4_000_000_000 * 1.1),
    "finemath": int(18_500_000_000 * 1.1),
    "fineweb_edu": int(16_500_000_000 * 1.1)
}

# Define how to access each dataset
DATASET_CONFIGS = {
    "starcoder/python": [{"repo_id": "bigcode/starcoderdata", "repo_type": "dataset", "data_dir": "python", "col": "content"}],
    "starcoder/sql": [{"repo_id": "bigcode/starcoderdata", "repo_type": "dataset", "data_dir": "sql", "col": "content"}],
    "finemath": [
        {"repo_id": "HuggingFaceTB/finemath", "repo_type": "dataset", "data_dir": "finemath-4plus", "col": "text"},
        {"repo_id": "HuggingFaceTB/finemath", "repo_type": "dataset", "data_dir": "finemath-3plus", "col": "text"}
    ],
    "fineweb_edu": [{"repo_id": "HuggingFaceFW/fineweb-edu", "repo_type": "dataset", "data_dir": "sample/100BT", "col": "text"}]
}

def get_parquet_files(repo_id, data_dir):
    """List all parquet files and their sizes in a given repo directory."""
    try:
        tree = list_repo_tree(repo_id, repo_type="dataset", path_in_repo=data_dir)
        files = [f for f in tree if f.path.endswith(".parquet") and f.size is not None]
        return sorted(files, key=lambda x: x.path)
    except Exception as e:
        print(f"Error listing files for {repo_id}/{data_dir}: {e}")
        return []

def calculate_tokens_per_gb(file_path, col_name, tokenizer):
    """Calculate the number of tokens per GB for a given parquet file."""
    file_size_bytes = os.path.getsize(file_path)
    file_size_gb = file_size_bytes / (1024**3)
    
    print("Reading sample file in batches to count tokens...")
    parquet_file = pq.ParquetFile(file_path)
    
    total_tokens = 0
    total_docs = 0
    
    # Process row groups one by one to keep memory usage low
    for i in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(i, columns=[col_name])
        texts = [t for t in table[col_name].to_pylist() if t]
        
        # Tokenize this batch
        batch_size = 10000
        for j in range(0, len(texts), batch_size):
            batch = texts[j:j+batch_size]
            encoded = tokenizer(batch, add_special_tokens=True)
            total_tokens += sum(len(ids) for ids in encoded["input_ids"])
            
        total_docs += len(texts)
        print(f"  Processed row group {i+1}/{parquet_file.num_row_groups} ({total_docs} docs)...")
        
    print(f"Total documents tokenized: {total_docs}")
    tokens_per_gb = total_tokens / file_size_gb
    return tokens_per_gb, total_tokens, file_size_gb

def download_domain(domain_name, target_tokens, tokenizer):
    print(f"\n{'='*60}")
    print(f"Processing Domain: {domain_name}")
    print(f"Target Tokens: {target_tokens:,}")
    print(f"{'='*60}")
    
    configs = DATASET_CONFIGS[domain_name]
    output_dir = os.path.join(BASE_DIR, domain_name.replace("/", "_"))
    os.makedirs(output_dir, exist_ok=True)
    
    tokens_remaining = target_tokens
    
    for config in configs:
        if tokens_remaining <= 0:
            break
            
        repo_id = config["repo_id"]
        data_dir = config["data_dir"]
        col = config["col"]
        
        print(f"\n🔍 Searching for parquet files in {repo_id}/{data_dir}...")
        hf_files = get_parquet_files(repo_id, data_dir)
        
        if not hf_files:
            print(f"No parquet files found in {repo_id}/{data_dir}")
            continue
            
        print(f"Found {len(hf_files)} parquet files.")
        
        # 1. Download a single big sample file to calculate tokens/GB
        # Let's take the first file
        sample_file_info = hf_files[0]
        print(f"Downloading sample file: {sample_file_info.path} ({sample_file_info.size / 1024**2:.2f} MB)")
        sample_local_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=sample_file_info.path,
            local_dir=output_dir
        )
        
        # 2. Calculate tokens/GB
        tokens_per_gb, sample_tokens, sample_gb = calculate_tokens_per_gb(sample_local_path, col, tokenizer)
        print("Sample File Stats:")
        print(f"  - Size: {sample_gb:.4f} GB")
        print(f"  - Tokens: {sample_tokens:,}")
        print(f"  - Tokens per GB: {tokens_per_gb:,.0f}")
        
        tokens_remaining -= sample_tokens
        
        if tokens_remaining <= 0:
            print("✅ Sample file alone satisfied target tokens for this source.")
            continue
            
        # Determine exact subset logic using accurate tokens_per_gb
        total_subset_gb = sum(f.size for f in hf_files) / 1024**3
        estimated_subset_tokens = total_subset_gb * tokens_per_gb
        
        if estimated_subset_tokens < tokens_remaining * 1.2: 
            # If the entire subset is roughly what we need or less, just download it all
            print("Subset size is roughly equal to or less than our remaining target. Using entire exact subset directly.")
            # We already downloaded the first file, so download the rest
            selected_files = [f.path for f in hf_files[1:]]
            
            if selected_files:
                print(f"Starting download of {len(selected_files)} remaining files in subset...")
                snapshot_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    allow_patterns=selected_files,
                    local_dir=output_dir,
                    max_workers=16
                )
            
            # Deduct the remaining files' estimated tokens
            remaining_gb = sum(f.size for f in hf_files[1:]) / 1024**3
            tokens_remaining -= remaining_gb * tokens_per_gb
            print(f"✅ Finished exact subset. Estimated tokens remaining: {max(0, tokens_remaining):,.0f}")
            continue

        # 3. Calculate how many GBs we need to download to hit tokens_remaining
        gb_needed = tokens_remaining / tokens_per_gb
        print(f"Remaining Tokens: {tokens_remaining:,}")
        print(f"Estimated GB needed: {gb_needed:.2f} GB")
        
        # 4. Select files to meet the GB requirement
        selected_files = []
        accumulated_bytes = 0
        target_bytes = gb_needed * (1024**3)
        
        for f_info in hf_files[1:]:
            selected_files.append(f_info.path)
            accumulated_bytes += f_info.size
            if accumulated_bytes >= target_bytes:
                break
                
        if accumulated_bytes < target_bytes:
            print("⚠️ Warning: Dataset might not be large enough. Selected all available files.")
        
        print(f"Selected {len(selected_files)} additional files to download (~{accumulated_bytes / 1024**3:.2f} GB)")
        
        # 5. Download the selected files
        if selected_files:
            print("Starting download of selected files...")
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                allow_patterns=selected_files,
                local_dir=output_dir,
                max_workers=16
            )
            
        # Estimate new remaining tokens based on exact size downloaded
        estimated_tokens_downloaded = (accumulated_bytes / (1024**3)) * tokens_per_gb
        tokens_remaining -= estimated_tokens_downloaded
        print(f"✅ Finished source {data_dir}. Estimated tokens remaining: {max(0, tokens_remaining):,.0f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default=None, help="Specific domain to download: starcoder/python, starcoder/sql, finemath, fineweb_edu")
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
            
    print("\n🎉 All requested datasets have been dynamically sized and downloaded!")

if __name__ == "__main__":
    main()
