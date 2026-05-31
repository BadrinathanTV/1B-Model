import os
import argparse
import numpy as np
import tiktoken
import time
from datasets import load_dataset
from tqdm import tqdm

# Ratio distribution for the 8 datasets
DATASET_CONFIGS = {
    "FineWeb-Edu": {
        "path": "HuggingFaceFW/fineweb-edu",
        "name": "sample-10BT",
        "split": "train",
        "ratio": 0.40,
        "text_key": "text",
    },
    "DCLM-Baseline": {
        "path": "mlfoundations/dclm-baseline-1.0-parquet",
        "split": "train",
        "ratio": 0.15,
        "text_key": "text",
    },
    "StarCoderData": {
        "path": "bigcode/starcoderdata",
        "split": "train",
        "data_dir": "python",
        "ratio": 0.15,
        "text_key": "content",
        "gated": True,
    },
    "FineMath": {
        "path": "HuggingFaceTB/finemath",
        "name": "finemath-3plus",
        "split": "train",
        "ratio": 0.10,
        "text_key": "text",
    },
    "Cosmopedia-v2": {
        "path": "HuggingFaceTB/smollm-corpus",
        "name": "cosmopedia-v2",
        "split": "train",
        "ratio": 0.07,
        "text_key": "text",
    },
    "peS2o": {
        "path": "allenai/peS2o",
        "split": "train",
        "revision": "refs/convert/parquet",
        "ratio": 0.05,
        "text_key": "text",
    },
    "PG-19": {
        "path": "emozilla/pg19",
        "split": "train",
        "ratio": 0.05,
        "text_key": "text",
    },
    "Wikipedia": {
        "path": "wikipedia",
        "name": "20220301.en",
        "split": "train",
        "ratio": 0.03,
        "text_key": "text",
    },
}

class DatasetCycleStream:
    """Wrapper to safely loop and stream Hugging Face datasets."""
    def __init__(self, load_fn, name):
        self.load_fn = load_fn
        self.name = name
        self.iterator = None
        self._reset()

    def _reset(self):
        print(f"[{self.name}] Initializing/re-starting streaming iterator...")
        self.iterator = iter(self.load_fn())

    def __iter__(self):
        return self

    def __next__(self):
        for attempt in range(3):
            try:
                if self.iterator is None:
                    self._reset()
                return next(self.iterator)
            except StopIteration:
                self._reset()
            except Exception as e:
                print(f"Warning: error streaming {self.name} (attempt {attempt + 1}/3): {e}")
                time.sleep(2)
                self.iterator = None
        raise RuntimeError(f"Failed to stream from dataset {self.name} after 3 attempts.")


def get_load_fn(cfg, hf_token=None):
    """Returns a parameterless function that loads the streaming dataset."""
    path = cfg["path"]
    kwargs = {
        "split": cfg.get("split", "train"),
        "streaming": True,
    }
    if "name" in cfg:
        kwargs["name"] = cfg["name"]
    if "revision" in cfg:
        kwargs["revision"] = cfg["revision"]
    if "data_dir" in cfg:
        kwargs["data_dir"] = cfg["data_dir"]
    if hf_token:
        kwargs["token"] = hf_token

    return lambda: load_dataset(path, **kwargs)


def fallback_starcoder_load_fn():
    """Fallback function for StarCoderData using python instructions."""
    return lambda: load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train", streaming=True)


def tokenize_doc(doc, text_key, enc, eot, is_starcoder_fallback=False):
    """Tokenize a document and prepend EOT token."""
    if is_starcoder_fallback:
        inst = doc.get("instruction", "")
        inp = doc.get("input", "")
        out = doc.get("output", "")
        text = f"Instruction: {inst}\n"
        if inp:
            text += f"Input: {inp}\n"
        text += f"Response: {out}"
    else:
        text = doc.get(text_key, "")

    tokens = [eot]
    tokens.extend(enc.encode_ordinary(text))
    return tokens


def main():
    parser = argparse.ArgumentParser(description="Download, tokenize, and interleave pretraining datasets")
    parser.add_argument(
        "--output_dir", type=str, default="data/mixed_50B_corpus",
        help="Directory to save the tokenized binary shards"
    )
    parser.add_argument(
        "--shard_size", type=int, default=100_000_000,
        help="Number of tokens per training shard (default 100M)"
    )
    parser.add_argument(
        "--target_tokens", type=int, default=1_000_000_000,
        help="Target total training tokens to collect (default 1B)"
    )
    parser.add_argument(
        "--val_tokens", type=int, default=10_000_000,
        help="Number of tokens for validation set (default 10M)"
    )
    parser.add_argument(
        "--hf_token", type=str, default=os.environ.get("HF_TOKEN", None),
        help="Optional Hugging Face access token for gated datasets"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Use GPT-2 tokenizer via tiktoken
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens['<|endoftext|>']

    # Initialize streams
    streams = {}
    is_fallback_starcoder = False

    for name, cfg in DATASET_CONFIGS.items():
        if name == "StarCoderData":
            # Attempt to initialize gated StarCoderData
            try:
                load_fn = get_load_fn(cfg, hf_token=args.hf_token)
                # Try to pull one element to verify access
                test_iter = iter(load_fn())
                next(test_iter)
                streams[name] = DatasetCycleStream(load_fn, name)
                print(f"Successfully authenticated and initialized gated dataset: {name}")
            except Exception as e:
                print(f"\n[WARNING] Could not access gated dataset {name}: {e}")
                print("Falling back to public Python instructions dataset: iamtarun/python_code_instructions_18k_alpaca")
                load_fn = fallback_starcoder_load_fn()
                streams[name] = DatasetCycleStream(load_fn, name + "_fallback")
                is_fallback_starcoder = True
        else:
            load_fn = get_load_fn(cfg, hf_token=args.hf_token)
            streams[name] = DatasetCycleStream(load_fn, name)

    # Initialize collection statistics
    ratios = {name: cfg["ratio"] for name, cfg in DATASET_CONFIGS.items()}
    collected_counts = {name: 0 for name in DATASET_CONFIGS.keys()}

    total_target = args.target_tokens + args.val_tokens
    print(f"\nTargeting total tokens: {total_target:,} (Train: {args.target_tokens:,}, Val: {args.val_tokens:,})")
    print("Dataset Target Ratios:")
    for name, ratio in ratios.items():
        print(f"  - {name}: {ratio * 100:.1f}% ({int(total_target * ratio):,} tokens)")

    pbar = tqdm(total=total_target, desc="Collected Tokens")

    shard_index = 0
    token_buffer = []
    total_tokens_collected = 0

    def write_shard(filename, tokens):
        arr = np.array(tokens, dtype=np.uint16)
        filepath = os.path.join(args.output_dir, filename)
        print(f"\nWriting {len(arr):,} tokens to {filepath}...")
        with open(filepath, "wb") as f:
            f.write(arr.tobytes())

    # Document-level weighted interleaving loop
    while total_tokens_collected < total_target:
        # Find the dataset that is furthest behind its target ratio
        # Avoid division by zero by handling initialized state
        selected_name = min(
            DATASET_CONFIGS.keys(),
            key=lambda name: (collected_counts[name] / ratios[name]) if ratios[name] > 0 else float('inf')
        )

        # Pull a document from the selected dataset stream
        try:
            stream = streams[selected_name]
            doc = next(stream)
        except Exception as e:
            print(f"Error fetching from stream {selected_name}: {e}. Skipping document.")
            continue

        # Tokenize the document
        text_key = DATASET_CONFIGS[selected_name].get("text_key", "text")
        is_fallback = (selected_name == "StarCoderData" and is_fallback_starcoder)
        tokens = tokenize_doc(doc, text_key, enc, eot, is_fallback)

        if not tokens or len(tokens) <= 1:
            continue

        token_buffer.extend(tokens)
        collected_counts[selected_name] += len(tokens)
        total_tokens_collected += len(tokens)
        pbar.update(len(tokens))

        # Write validation shard first if we have enough tokens
        if shard_index == 0 and len(token_buffer) >= args.val_tokens:
            val_tokens_list = token_buffer[:args.val_tokens]
            write_shard("val.bin", val_tokens_list)
            token_buffer = token_buffer[args.val_tokens:]
            shard_index += 1

        # Write regular training shards
        while len(token_buffer) >= args.shard_size:
            shard_tokens = token_buffer[:args.shard_size]
            write_shard(f"train_shard_{shard_index:03d}.bin", shard_tokens)
            token_buffer = token_buffer[args.shard_size:]
            shard_index += 1

    # Write any remaining tokens to the final shard if significant
    if len(token_buffer) > 0 and total_tokens_collected >= total_target:
        write_shard(f"train_shard_{shard_index:03d}.bin", token_buffer)

    pbar.close()
    print("\nDownload and tokenization complete!")
    print(f"Shards saved to {args.output_dir}/")
    print("Final collected token counts per dataset:")
    for name, count in collected_counts.items():
        actual_ratio = count / total_tokens_collected if total_tokens_collected > 0 else 0
        print(f"  - {name}: {count:,} tokens ({actual_ratio * 100:.2f}%)")

if __name__ == "__main__":
    main()
