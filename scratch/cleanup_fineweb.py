import os
import shutil
import glob

# 1. Delete CC-MAIN-2024-18
p18 = "data/raw_corpus/fineweb_edu/data/CC-MAIN-2024-18"
if os.path.exists(p18):
    shutil.rmtree(p18)
    print("Deleted CC-MAIN-2024-18 directory.")

# 2. Delete Cache
cache = "data/raw_corpus/fineweb_edu/.cache"
if os.path.exists(cache):
    shutil.rmtree(cache)
    print("Deleted cache directory.")

# 3. Keep only 000_00000.parquet to 000_00009.parquet
files = glob.glob("data/raw_corpus/fineweb_edu/data/CC-MAIN-2024-10/*.parquet")
kept = 0
deleted = 0
for f in files:
    basename = os.path.basename(f)
    # Check if name is 000_00000.parquet to 000_00009.parquet
    if len(basename) >= 9 and basename.startswith("000_0000") and basename[8].isdigit() and int(basename[8]) <= 9:
        kept += 1
    else:
        os.remove(f)
        deleted += 1

print(f"Cleanup done! Kept: {kept} files, Deleted: {deleted} files.")
