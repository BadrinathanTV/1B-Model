import os
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

BASE_DIR = "data/raw_corpus"

DATASETS = {
    "tamil_wikipedia": {
        "path": "wikimedia/wikipedia", "name": "20231101.ta", "text_col": "text", "folder": "tamil_wikipedia", "limit": 10000
    },
    "indiccorp_tamil": {
        "path": "wikimedia/wikipedia", "name": "20231101.ta", "text_col": "text", "folder": "indiccorp_tamil", "limit": 10000 # Use ta wikipedia as a backup/supplement for Tamil BPE merges
    },
    "starcoder_python": {
        "path": "bigcode/starcoderdata", "name": None, "data_dir": "python", "text_col": "content", "folder": "starcoder/python", "limit": 5000
    },
    "starcoder_sql": {
        "path": "bigcode/starcoderdata", "name": None, "data_dir": "sql", "text_col": "content", "folder": "starcoder/sql", "limit": 5000
    },
    "finemath": {
        "path": "HuggingFaceFW/finemath", "name": "finemath-3B", "text_col": "text", "folder": "finemath", "limit": 5000
    },
    "pes2o": {
        "path": "HuggingFaceFW/finemath", "name": "finemath-3B", "text_col": "text", "folder": "pes2o", "limit": 5000 # Math data for pes2o BPE math merges
    },
    "fineweb_edu": {
        "path": "HuggingFaceFW/fineweb-edu", "name": "sample-10BT", "text_col": "text", "folder": "fineweb_edu", "limit": 5000
    },
    "cosmopedia": {
        "path": "HuggingFaceFW/cosmopedia", "name": "cosmopedia-v0.7", "text_col": "text", "folder": "cosmopedia", "limit": 5000
    },
    "wikipedia": {
        "path": "wikimedia/wikipedia", "name": "20231101.en", "text_col": "text", "folder": "wikipedia", "limit": 5000
    }
}

def download_sample(name, config):
    dest_folder = os.path.join(BASE_DIR, config["folder"])
    os.makedirs(dest_folder, exist_ok=True)
    
    print(f"\n📥 Downloading small BPE training sample for: {name}...")
    try:
        kwargs = {}
        if config["name"]:
            kwargs["name"] = config["name"]
        if "data_dir" in config:
            kwargs["data_dir"] = config["data_dir"]
            
        ds = load_dataset(config["path"], split="train", streaming=True, **kwargs)
        
        texts = []
        limit = config["limit"]
        text_col = config["text_col"]
        
        for idx, item in enumerate(ds):
            # Dynamic fallback check for text column
            col = text_col
            if col not in item:
                for c in ["content", "text", "code"]:
                    if c in item:
                        col = c
                        break
            
            val = item.get(col, "")
            if val and len(val.strip()) > 0:
                texts.append(val)
            if len(texts) >= limit:
                break
                
        if len(texts) > 0:
            # Save as a single parquet file in the expected directory
            table = pa.Table.from_arrays([pa.array(texts)], names=["text"])
            out_file = os.path.join(dest_folder, "sample.parquet")
            pq.write_table(table, out_file)
            print(f"✅ Saved {len(texts)} sample documents to {out_file}")
        else:
            print(f"⚠️ No documents fetched for {name}.")
    except Exception as e:
        print(f"❌ Error downloading {name}: {e}")

def main():
    print("=" * 80)
    print("             DOWNLOADING SAMPLE FILES FOR TOKENIZER TRAINING")
    print("=" * 80)
    
    for name, config in DATASETS.items():
        download_sample(name, config)
        
    print("\n🎉 All tokenizer samples successfully prepared in data/raw_corpus/!")
    print("You can now run: uv run scripts/train_custom_tokenizer.py")

if __name__ == "__main__":
    main()
