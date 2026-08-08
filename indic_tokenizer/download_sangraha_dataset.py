"""
Sangraha Full Indic Dataset Downloader (705 GB)
==============================================
Downloads the complete ~705 GB AI4Bharat Sangraha Indic dataset (2,492 parquet files across 22 Indian scheduled languages).
Excludes English, code, and math datasets.

Features:
- Full coverage of 22 Indic languages across 'verified', 'unverified', and 'synthetic' splits.
- Handles language code variations (e.g. npi/nep for Nepali, ory/ori for Odia).
- Includes native scripts (_Deva, _Beng, _Taml, etc.) and transliterated scripts (_Latn).
- Automatic retry loop (up to 5 retries with backoff) for resilient 705GB downloads.
- Disk space check & overall progress tracker.
- Resumable: Skips already completed files cleanly.

Usage:
  # Download complete 705GB dataset
  uv run python indic_tokenizer/download_sangraha_dataset.py --full --target_dir data/sangraha_full

  # Download subset (e.g., 500MB per language for fast testing)
  uv run python indic_tokenizer/download_sangraha_dataset.py --target_mb_per_lang 500
"""

import os
import sys
import time
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "ai4bharat/sangraha"
SPLIT_FALLBACK_ORDER = ["verified", "unverified", "synthetic"]

# Language alias mapping to standard codes
LANG_ALIASES = {
    "npi": "nep",  # Nepali
    "ory": "ori",  # Odia
}

# 22 Indian Scheduled Languages
INDIC_LANGUAGES = {
    "asm", "ben", "brx", "doi", "gom", "guj", "hin", "kan", 
    "kas", "mai", "mal", "mar", "mni", "nep", "ori", "pan", 
    "san", "sat", "snd", "tam", "tel", "urd"
}

def map_folder_to_indic_lang(folder_name):
    """Maps a repo subfolder name (e.g., 'hin_Deva', 'npi_Latn', 'tam') to an Indic language code."""
    if folder_name.startswith("eng"):
        return None  # Exclude English

    base_code = folder_name.split("_")[0]
    normalized_code = LANG_ALIASES.get(base_code, base_code)

    if normalized_code in INDIC_LANGUAGES:
        return normalized_code
    return None

def fetch_all_indic_parquet_files():
    """Queries Hugging Face API and discovers all 2,492 parquet files belonging to Indic languages."""
    print("🔍 Fetching repository file index from Hugging Face ('ai4bharat/sangraha')...")
    api = HfApi()
    all_files = api.list_repo_files(REPO_ID, repo_type="dataset")

    indic_files = []
    skipped_eng = 0

    for file_path in all_files:
        if not file_path.endswith(".parquet"):
            continue

        parts = file_path.split("/")
        if len(parts) < 2:
            continue

        folder = parts[1]
        lang_code = map_folder_to_indic_lang(folder)

        if lang_code:
            indic_files.append((file_path, parts[0], lang_code))
        elif folder.startswith("eng"):
            skipped_eng += 1

    print(f"✅ Discovered {len(indic_files)} Indic parquet files (~705 GB total dataset).")
    print(f"ℹ️  Filtered out {skipped_eng} English files.")
    return indic_files

def check_disk_space(target_dir, min_gb_required=700):
    """Checks if free disk space is sufficient for downloading."""
    os.makedirs(target_dir, exist_ok=True)
    total, used, free = shutil.disk_usage(target_dir)
    free_gb = free / (1024 ** 3)
    print(f"💾 Storage Status: {free_gb:.2f} GB free on partition containing '{target_dir}'.")
    if free_gb < min_gb_required:
        print(f"⚠️ WARNING: Free disk space ({free_gb:.1f} GB) is less than recommended {min_gb_required} GB for full Sangraha dataset!")
    return free_gb

def download_file_with_retry(file_path, target_dir, max_retries=5):
    """Downloads a single parquet file with exponential backoff retries."""
    local_path = os.path.join(target_dir, file_path)

    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return file_path, os.path.getsize(local_path), False  # Already cached

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    for attempt in range(1, max_retries + 1):
        try:
            downloaded_file = hf_hub_download(
                repo_id=REPO_ID,
                filename=file_path,
                repo_type="dataset",
                local_dir=target_dir,
                local_dir_use_symlinks=False
            )
            size = os.path.getsize(downloaded_file)
            return file_path, size, True
        except Exception as e:
            if attempt == max_retries:
                print(f"\n❌ Failed to download {file_path} after {max_retries} attempts: {e}")
                return file_path, 0, False
            time.sleep(2 ** attempt)

def extract_parquet_to_text(parquet_path, text_out_path):
    """Extracts 'text' column from a Parquet file and appends to plain text file."""
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(parquet_path, columns=["text"])
        texts = table["text"].to_pylist()
        
        with open(text_out_path, "a", encoding="utf-8") as f:
            for line in texts:
                if line and line.strip():
                    f.write(line.strip() + "\n")
        return len(texts)
    except Exception as e:
        print(f"⚠️ Error extracting text from {parquet_path}: {e}")
        return 0

def run_download_705gb(target_dir, target_mb_per_lang=None, max_workers=8, extract_text=False):
    check_disk_space(target_dir, min_gb_required=700 if not target_mb_per_lang else 5)
    indic_files = fetch_all_indic_parquet_files()

    if target_mb_per_lang:
        print(f"🎯 Target cap per language: {target_mb_per_lang} MB")
        # Filter files per language up to target_mb_per_lang
        lang_sizes = {lang: 0 for lang in INDIC_LANGUAGES}
        target_bytes = target_mb_per_lang * 1024 * 1024
        filtered_files = []
        for file_path, split, lang in indic_files:
            if lang_sizes[lang] < target_bytes:
                filtered_files.append((file_path, split, lang))
                lang_sizes[lang] += 15 * 1024 * 1024  # Est size per file
        indic_files = filtered_files

    total_files = len(indic_files)
    print(f"\n========================================================")
    print(f"🚀 Starting Download: {total_files} files across 22 Indic languages")
    print(f"  Destination: {os.path.abspath(target_dir)}")
    print(f"  Parallel Threads: {max_workers}")
    print(f"========================================================\n")

    completed_count = 0
    total_bytes_downloaded = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(download_file_with_retry, fitem[0], target_dir): fitem 
            for fitem in indic_files
        }

        for future in as_completed(future_to_file):
            fpath, split, lang = future_to_file[future]
            fpath_res, size, is_new = future.result()
            
            completed_count += 1
            total_bytes_downloaded += size
            
            elapsed = time.time() - start_time
            gb_so_far = total_bytes_downloaded / (1024 ** 3)
            mb_s = (total_bytes_downloaded / (1024 ** 2)) / elapsed if elapsed > 0 else 0

            status_str = "NEW" if is_new else "CACHED"
            print(f"[{completed_count}/{total_files}] ({gb_so_far:.2f} GB | {mb_s:.1f} MB/s) [{status_str}] {fpath}")

            if extract_text and size > 0:
                local_pfile = os.path.join(target_dir, fpath)
                txt_out = os.path.join(target_dir, "extracted_text", f"{lang}.txt")
                os.makedirs(os.path.dirname(txt_out), exist_ok=True)
                extract_parquet_to_text(local_pfile, txt_out)

    print("\n========================================================")
    print(f"🎉 DOWNLOAD COMPLETE: {completed_count}/{total_files} files processed.")
    print(f"📊 Total Dataset Volume: {total_bytes_downloaded / (1024**3):.2f} GB")
    print("========================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sangraha Full 705GB Indic Dataset Downloader")
    parser.add_argument("--target_dir", type=str, default="data/sangraha_full", help="Target output directory")
    parser.add_argument("--full", action="store_true", help="Download complete 705GB dataset without size limit")
    parser.add_argument("--target_mb_per_lang", type=int, default=None, help="Optional size cap per language in MB")
    parser.add_argument("--extract_text", action="store_true", help="Extract parquet documents to plain text files")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel download threads")

    args = parser.parse_args()

    limit = None if args.full or args.target_mb_per_lang is None else args.target_mb_per_lang
    run_download_705gb(
        target_dir=args.target_dir,
        target_mb_per_lang=limit,
        max_workers=args.workers,
        extract_text=args.extract_text
    )
