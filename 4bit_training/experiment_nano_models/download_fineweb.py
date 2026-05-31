import os
import argparse
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Download and tokenize FineWeb-Edu 1B tokens")
    parser.add_argument(
        "--output_dir", type=str, default="data/fineweb1B",
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
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Use GPT-2 tokenizer via tiktoken
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens['<|endoftext|>'] # end of text token ID (50256)

    def tokenize(doc):
        # Tokenizes a single document and appends the EOT token
        tokens = [eot] # start with EOT
        tokens.extend(enc.encode_ordinary(doc['text']))
        return tokens

    # Stream the FineWeb-Edu 10BT sample (since it has plenty of data to get 1B tokens)
    print("Initializing FineWeb-Edu streaming...")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True
    )

    # Shard processing parameters
    total_tokens_collected = 0
    shard_index = 0
    token_buffer = []

    def write_shard(filename, tokens):
        arr = np.array(tokens, dtype=np.uint16)
        filepath = os.path.join(args.output_dir, filename)
        print(f"Writing {len(arr):,} tokens to {filepath}...")
        with open(filepath, "wb") as f:
            f.write(arr.tobytes())

    print(f"Downloading and tokenizing to target {args.target_tokens + args.val_tokens:,} tokens...")
    pbar = tqdm(total=args.target_tokens + args.val_tokens, desc="Tokens")

    for doc in dataset:
        tokens = tokenize(doc)
        token_buffer.extend(tokens)
        pbar.update(len(tokens))
        total_tokens_collected += len(tokens)

        # Write validation shard first if we have enough tokens
        if shard_index == 0 and len(token_buffer) >= args.val_tokens:
            val_tokens = token_buffer[:args.val_tokens]
            write_shard("val.bin", val_tokens)
            token_buffer = token_buffer[args.val_tokens:]
            shard_index += 1

        # Write regular training shards
        while len(token_buffer) >= args.shard_size:
            shard_tokens = token_buffer[:args.shard_size]
            write_shard(f"train_shard_{shard_index:03d}.bin", shard_tokens)
            token_buffer = token_buffer[args.shard_size:]
            shard_index += 1

        if total_tokens_collected >= args.target_tokens + args.val_tokens:
            break

    # Write any remaining tokens to a final shard if significant
    if len(token_buffer) > 0 and total_tokens_collected < args.target_tokens + args.val_tokens:
        write_shard(f"train_shard_{shard_index:03d}.bin", token_buffer)

    pbar.close()
    print("Download and tokenization complete!")
    print(f"Shards saved to {args.output_dir}/")

if __name__ == "__main__":
    main()
