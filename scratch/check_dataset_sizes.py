import os
import pyarrow.parquet as pq

BASE_DIR = "data/raw_corpus"

def count_corpus_files():
    if not os.path.exists(BASE_DIR):
        print(f"Directory {BASE_DIR} does not exist!")
        return
        
    print(f"{'Domain Directory':<30} | {'Parquet Files':<15} | {'Total Size (GB)':<15}")
    print("-" * 66)
    
    total_files_all = 0
    total_size_all = 0
    
    for domain in sorted(os.listdir(BASE_DIR)):
        domain_path = os.path.join(BASE_DIR, domain)
        if not os.path.isdir(domain_path):
            continue
            
        parquet_files = []
        for root, dirs, files in os.walk(domain_path):
            for file in files:
                if file.endswith(".parquet"):
                    parquet_files.append(os.path.join(root, file))
                    
        total_size_bytes = sum(os.path.getsize(f) for f in parquet_files)
        total_size_gb = total_size_bytes / (1024 ** 3)
        
        print(f"{domain:<30} | {len(parquet_files):<15} | {total_size_gb:<15.4f}")
        
        total_files_all += len(parquet_files)
        total_size_all += total_size_gb
        
    print("-" * 66)
    print(f"{'Total':<30} | {total_files_all:<15} | {total_size_all:<15.4f}")

if __name__ == "__main__":
    count_corpus_files()
