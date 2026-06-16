import os
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

TARGETS = {
    "starcoder_python": {"target_tokens": 14.0, "path": "data/raw_corpus/starcoder/python"},
    "starcoder_sql": {"target_tokens": 3.5, "path": "data/raw_corpus/starcoder/sql"},
    "finemath": {"target_tokens": 10.5, "path": "data/raw_corpus/finemath"},
    "pes2o": {"target_tokens": 7.0, "path": "data/raw_corpus/pes2o"},
    "fineweb_edu": {"target_tokens": 5.0, "path": "data/raw_corpus/fineweb_edu"},
    "cosmopedia": {"target_tokens": 3.0, "path": "data/raw_corpus/cosmopedia"},
    "wikipedia": {"target_tokens": 2.0, "path": "data/raw_corpus/wikipedia"},
    "indiccorp_tamil": {"target_tokens": 4.0, "path": "data/raw_corpus/indiccorp_tamil"},
    "tamil_wikipedia": {"target_tokens": 1.0, "path": "data/raw_corpus/tamil_wikipedia"},
}

def get_directory_size(path):
    total_size = 0
    if not os.path.exists(path):
        return 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size

def estimate_tokens(size_bytes):
    # Standard heuristic: 1 byte of compressed/raw text data corresponds to approx 0.25 to 0.3 tokens
    return (size_bytes * 0.28) / 1e9  # in Billions of tokens

def download_sql_if_missing():
    sql_path = TARGETS["starcoder_sql"]["path"]
    os.makedirs(sql_path, exist_ok=True)
    files = [f for f in os.listdir(sql_path) if f.endswith(".parquet")]
    
    if len(files) == 0:
        print("\n📥 Downloading Gretel AI Synthetic SQL dataset (high quality & non-gated)...")
        try:
            dataset = load_dataset("gretelai/synthetic_text_to_sql", split="train")
            formatted_texts = []
            for item in dataset:
                context = item.get("sql_context", "")
                query = item.get("sql_query", "")
                explanation = item.get("sql_explanation", "")
                doc = f"-- Database Schema:\n{context}\n\n-- Query Goal:\n-- {explanation}\n\n-- SQL Query:\n{query}\n"
                formatted_texts.append(doc)
                
            table = pa.Table.from_arrays([pa.array(formatted_texts)], names=["text"])
            out_file = os.path.join(sql_path, "synthetic_sql.parquet")
            pq.write_table(table, out_file)
            print(f"✅ Downloaded and saved synthetic SQL dataset: {out_file}")
        except Exception as e:
            print(f"❌ Error downloading SQL: {e}")
    else:
        print("\n✓ SQL dataset already exists. Skipping download.")

def main():
    print("=" * 80)
    print("                    DATASET PROPORTIONS & READINESS CHECK")
    print("=" * 80)
    
    # Check/Download SQL first
    download_sql_if_missing()
    
    print("\n" + "-" * 80)
    print(f"{'Dataset':<20} | {'Size (GB)':<10} | {'Est. Tokens (B)':<15} | {'Target (B)':<12} | {'Required Epochs':<15}")
    print("-" * 80)
    
    total_est_tokens = 0
    total_target_tokens = 0
    
    for name, info in TARGETS.items():
        path = info["path"]
        target = info["target_tokens"]
        size_bytes = get_directory_size(path)
        size_gb = size_bytes / (1024**3)
        est_tokens = estimate_tokens(size_bytes)
        
        epochs = target / est_tokens if est_tokens > 0 else float('inf')
        epoch_str = f"{epochs:.2f}x" if epochs != float('inf') else "Missing"
        
        total_est_tokens += est_tokens
        total_target_tokens += target
        
        print(f"{name:<20} | {size_gb:<10.2f} | {est_tokens:<15.2f} | {target:<12.2f} | {epoch_str:<15}")
        
    print("-" * 80)
    print(f"{'TOTAL':<20} | {'-':<10} | {total_est_tokens:<15.2f} | {total_target_tokens:<12.2f} | -")
    print("=" * 80)
    
    print("\n💡 Note on Epochs:")
    print("If Required Epochs > 1.0x, the WeightedDomainGenerator will automatically stream the dataset")
    print("multiple times (epochs) to meet the target tokens during pretraining. This is standard practice")
    print("for high-quality data (math and code) to reinforce logical reasoning.")

if __name__ == "__main__":
    main()
