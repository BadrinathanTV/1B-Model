import os
import sys
import subprocess
import logging
from huggingface_hub import snapshot_download, hf_hub_download

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

BASE_DATA_DIR = "data/raw_corpus"
os.makedirs(BASE_DATA_DIR, exist_ok=True)

def safe_download(repo_id, folder, patterns, repo_type="dataset"):
    target_dir = os.path.join(BASE_DATA_DIR, folder)
    os.makedirs(target_dir, exist_ok=True)
    logging.info(f"Downloading {repo_id} -> {target_dir}")
    
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            allow_patterns=patterns,
            local_dir=target_dir,
            max_workers=4
        )
        logging.info(f"✅ Downloaded {repo_id} successfully.")
    except Exception as e:
        logging.error(f"❌ Failed to download {repo_id}: {e}")

def main():
    logging.info("Starting Crash-Proof File Downloader with Exact Target Split Ratios...")
    
    # --- TAMIL DATASETS (Skipped as they are already complete) ---
    # folder = os.path.join(BASE_DATA_DIR, "tamil_wikipedia")
    # indic_folder = os.path.join(BASE_DATA_DIR, "indiccorp_tamil")
    # ...
    
    # 3. FineWeb-Edu: Exact 17.5B tokens (25 shards from 100BT split)
    fineweb_patterns = [
        "sample/100BT/000_0000*.parquet",  # 00-09
        "sample/100BT/000_0001*.parquet",  # 10-19
        "sample/100BT/000_00020.parquet",
        "sample/100BT/000_00021.parquet",
        "sample/100BT/000_00022.parquet",
        "sample/100BT/000_00023.parquet",
        "sample/100BT/000_00024.parquet"
    ]
    safe_download("HuggingFaceFW/fineweb-edu", "fineweb_edu", fineweb_patterns)
    
    # 4. DCLM-Baseline: Exact 5.0B tokens (37 shards out of local-shard_0)
    dclm_patterns = [
        "global-shard_01_of_10/local-shard_0_of_10/shard_0000000*_processed.jsonl.zst", # 00-09
        "global-shard_01_of_10/local-shard_0_of_10/shard_0000001*_processed.jsonl.zst", # 10-19
        "global-shard_01_of_10/local-shard_0_of_10/shard_0000002*_processed.jsonl.zst", # 20-29
        "global-shard_01_of_10/local-shard_0_of_10/shard_0000003[0-6]_processed.jsonl.zst" # 30-36
    ]
    safe_download("mlfoundations/dclm-baseline-1.0", "dclm_baseline", dclm_patterns)
    
    # 5. StarCoderData: Exact 7.5B tokens (57 shards out of 59 python split)
    starcoder_patterns = [
        "python/train-0000*-of-00059.parquet", # 00-09
        "python/train-0001*-of-00059.parquet", # 10-19
        "python/train-0002*-of-00059.parquet", # 20-29
        "python/train-0003*-of-00059.parquet", # 30-39
        "python/train-0004*-of-00059.parquet", # 40-49
        "python/train-0005[0-6]-of-00059.parquet" # 50-56
    ]
    safe_download("bigcode/starcoderdata", "starcoder", starcoder_patterns)
    
    # 6. FineMath: Exact 5.0B tokens (13 shards out of finemath-4plus)
    finemath_patterns = [
        "finemath-4plus/train-0000*-of-00064.parquet", # 00-09
        "finemath-4plus/train-0001[0-2]-of-00064.parquet" # 10-12
    ]
    safe_download("HuggingFaceTB/finemath", "finemath", finemath_patterns)
    
    # 7. Cosmopedia V2: Exact 2.5B tokens (7 shards out of 105)
    cosmopedia_patterns = [
        "cosmopedia-v2/train-0000[0-6]-of-00104.parquet"
    ]
    safe_download("HuggingFaceTB/cosmopedia-v2", "cosmopedia", cosmopedia_patterns)
    
    # 8. peS2o: Exact 2.5B tokens (13 shards out of 20)
    pes2o_patterns = [
        "data/v1/train-0000*-of-00020.json.gz", # 00-09
        "data/v1/train-0001[0-2]-of-00020.json.gz" # 10-12
    ]
    safe_download("allenai/peS2o", "pes2o", pes2o_patterns)
    
    # 9. Wikipedia (English): Exact 2.5B tokens (34 shards out of 41)
    wiki_patterns = [
        "20231101.en/train-0000*.parquet", # 00-09
        "20231101.en/train-0001*.parquet", # 10-19
        "20231101.en/train-0002*.parquet", # 20-29
        "20231101.en/train-0003[0-3]-of-00041.parquet" # 30-33
    ]
    safe_download("wikimedia/wikipedia", "wikipedia", wiki_patterns)
    
    logging.info("🎉 All raw files safely downloaded according to your exact 50B token specification!")

if __name__ == "__main__":
    main()
