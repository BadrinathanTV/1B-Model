import os
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

OUTPUT_DIR = "data/raw_corpus/starcoder/sql"

def download_sql_dataset():
    print("Downloading synthetic SQL dataset from Gretel AI...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load public non-gated dataset
    dataset = load_dataset("gretelai/synthetic_text_to_sql", split="train")
    
    formatted_texts = []
    print("Formatting SQL samples...")
    for item in dataset:
        context = item.get("sql_context", "")
        query = item.get("sql_query", "")
        explanation = item.get("sql_explanation", "")
        
        # Format as a clean SQL document
        doc = f"-- Database Schema:\n{context}\n\n-- Query Goal:\n-- {explanation}\n\n-- SQL Query:\n{query}\n"
        formatted_texts.append(doc)
        
    print(f"Constructed {len(formatted_texts)} SQL documents.")
    
    # Save as a single parquet file
    table = pa.Table.from_arrays([pa.array(formatted_texts)], names=["text"])
    out_file = os.path.join(OUTPUT_DIR, "synthetic_sql.parquet")
    pq.write_table(table, out_file)
    print(f"✅ SQL dataset saved to {out_file}")

if __name__ == "__main__":
    download_sql_dataset()
