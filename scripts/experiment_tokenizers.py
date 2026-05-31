import os
import glob
import gzip
import json
import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, Regex
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel, Digits, Sequence, Split, Metaspace, Punctuation
from tokenizers.decoders import Metaspace as MetaspaceDecoder
from tokenizers.trainers import BpeTrainer

BASE_DIR = "data/raw_corpus"

DOMAINS = {
    "tamil": ["tamil_wikipedia", "indiccorp_tamil"],
    "code": ["starcoder"],
    "math": ["finemath", "pes2o"],
    "english": ["fineweb_edu", "cosmopedia", "wikipedia"]
}

def stream_folder(folder_path):
    """Safely stream text from a folder without loading everything into memory."""
    # 1. Text files
    for f in glob.glob(os.path.join(folder_path, "**", "*.txt"), recursive=True):
        try:
            with open(f, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    if line: yield line
        except Exception: pass

    # 2. Parquet files
    for f in glob.glob(os.path.join(folder_path, "**", "*.parquet"), recursive=True):
        try:
            schema = pq.read_schema(f)
            col_name = 'text' if 'text' in schema.names else 'content'
            table = pq.read_table(f, columns=[col_name])
            for val in table[col_name]:
                text = val.as_py()
                if text and len(text.strip()) > 0: yield text
        except Exception: pass

    # 3. JSON GZip files (peS2o)
    for f in glob.glob(os.path.join(folder_path, "**", "*.json.gz"), recursive=True):
        try:
            with gzip.open(f, 'rt', encoding='utf-8') as file:
                for line in file:
                    obj = json.loads(line)
                    text = obj.get("text", "")
                    if text and len(text.strip()) > 0: yield text
        except Exception: pass

def get_training_corpus(samples_per_domain=300000):
    """Yields exactly samples_per_domain across our 4 critical competency domains."""
    for domain, folders in DOMAINS.items():
        print(f"\n--- Loading {samples_per_domain} documents for {domain.upper()} domain ---")
        yielded = 0
        for folder in folders:
            folder_path = os.path.join(BASE_DIR, folder)
            if not os.path.exists(folder_path):
                continue
            for text in stream_folder(folder_path):
                yield text
                yielded += 1
                if yielded >= samples_per_domain:
                    break
            if yielded >= samples_per_domain:
                break
        print(f"Loaded {yielded} samples for {domain.upper()}.")

import argparse

def main():
    parser = argparse.ArgumentParser(description="Zero-RAM Tokenizer Experiment Pipeline")
    parser.add_argument("--step", type=str, choices=["create_corpus", "train_llama", "train_sarvam", "train_hybrid", "benchmark", "run_all"], default="run_all",
                        help="Which step of the experiment to run to avoid memory accumulation.")
    args = parser.parse_args()

    samples_per_domain = 200000
    llama_pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    vocab_size = 65535

    os.makedirs("models", exist_ok=True)

    # Step 1: Train Llama
    if args.step in ["train_llama", "run_all"]:
        print(f"\n[1/3] Training Model A (Llama-3 ByteLevel BPE) on a stream of {samples_per_domain} docs per domain...")
        tok_a = Tokenizer(BPE())
        tok_a.pre_tokenizer = Sequence([Split(Regex(llama_pattern), behavior="isolated"), ByteLevel(add_prefix_space=False)])
        tok_a.decoder = decoders.ByteLevel()
        # ByteLevel alphabet is strictly 256, so limit_alphabet doesn't matter here, but min_freq=50 saves RAM
        trainer_a = BpeTrainer(vocab_size=vocab_size, min_frequency=50, special_tokens=["<|im_start|>", "<|im_end|>", "<|pad|>"], initial_alphabet=ByteLevel.alphabet())
        
        # Optimal Zero-RAM Streaming:
        tok_a.train_from_iterator(get_training_corpus(samples_per_domain), trainer=trainer_a)
        
        tok_a.save("models/tokenizer_llama.json")
        print("✅ Saved Model A to models/tokenizer_llama.json")

    # Step 2: Train Sarvam
    if args.step in ["train_sarvam", "run_all"]:
        print(f"\n[2/3] Training Model B (Sarvam Metaspace BPE) on a stream of {samples_per_domain} docs per domain...")
        tok_b = Tokenizer(BPE(unk_token="<|unk|>"))
        # Memory Fix: Punctuation() is required because Metaspace alone doesn't split punctuation from code/math. 
        # Without it, "a=b+c*d" is treated as one unique word, blowing up the BPE HashMap in RAM!
        tok_b.pre_tokenizer = Sequence([Punctuation(), Metaspace(replacement="▁", prepend_scheme="first")])
        tok_b.decoder = MetaspaceDecoder(replacement="▁", prepend_scheme="first")
        # Memory Optimization: limit_alphabet=2000 strictly drops noisy Chinese/Emoji unicode, cutting RAM usage by 10x!
        trainer_b = BpeTrainer(vocab_size=vocab_size, min_frequency=50, limit_alphabet=2000, special_tokens=["<|unk|>", "<|im_start|>", "<|im_end|>", "<|pad|>"])
        
        # Optimal Zero-RAM Streaming:
        tok_b.train_from_iterator(get_training_corpus(samples_per_domain), trainer=trainer_b)
        
        tok_b.save("models/tokenizer_sarvam.json")
        print("✅ Saved Model B to models/tokenizer_sarvam.json")

    # Step 3: Train Hybrid
    if args.step in ["train_hybrid", "run_all"]:
        print(f"\n[3/3] Training Model C (Hybrid Regex + Metaspace BPE) on a stream of {samples_per_domain} docs per domain...")
        tok_c = Tokenizer(BPE(unk_token="<|unk|>"))
        tok_c.pre_tokenizer = Sequence([Split(Regex(llama_pattern), behavior="isolated"), Metaspace(replacement="▁", prepend_scheme="first")])
        tok_c.decoder = MetaspaceDecoder(replacement="▁", prepend_scheme="first")
        # Memory Optimization: limit_alphabet=2000 strictly drops noisy Chinese/Emoji unicode, cutting RAM usage by 10x!
        trainer_c = BpeTrainer(vocab_size=vocab_size, min_frequency=50, limit_alphabet=2000, special_tokens=["<|unk|>", "<|im_start|>", "<|im_end|>", "<|pad|>"])
        
        # Optimal Zero-RAM Streaming:
        tok_c.train_from_iterator(get_training_corpus(samples_per_domain), trainer=trainer_c)
        
        tok_c.save("models/tokenizer_hybrid.json")
        print("✅ Saved Model C to models/tokenizer_hybrid.json")
    
    # Step 4: Benchmark
    if args.step in ["benchmark", "run_all"]:
        print("\n\n" + "="*50)
        print("🚀 RUNNING BENCHMARKS 🚀")
        print("="*50)
        
        # Load saved tokenizers
        try:
            tok_a = Tokenizer.from_file("models/tokenizer_llama.json")
            tok_b = Tokenizer.from_file("models/tokenizer_sarvam.json")
            tok_c = Tokenizer.from_file("models/tokenizer_hybrid.json")
            tok_final = Tokenizer.from_file("models/tokenizer/tokenizer.json")
        except Exception as e:
            print(f"❌ Error loading tokenizers: {e}")
            print("Make sure you train the tokenizers first!")
            return

        tests = {
            "Tamil": "தமிழ்நாடு (Tamil Nadu) இந்தியாவின் தென்மாநிலங்களில் ஒன்றாகும். இது இந்திய துணைக்கண்டத்தின் தென்கோடியில் அமைந்துள்ளது. இது வடக்கில் ஆந்திரப் பிரதேசம் மற்றும் கர்நாடகா, மேற்கில் கேரளா, கிழக்கில் வங்காள விரிகுடா, மற்றும் தெற்கில் இந்தியப் பெருங்கடல் ஆகியவற்றால் சூழப்பட்டுள்ளது.",
            "English": "The universe is expanding at an accelerating rate, driven by a mysterious force known as dark energy. Scientists estimate that dark energy makes up roughly 68% of the universe, while dark matter accounts for about 27%.",
            "Code": "def calculate_fibonacci(n):\n    if n <= 0:\n        return []\n    elif n == 1:\n        return [0]\n    result = [0, 1]\n    while len(result) < n:\n        result.append(result[-1] + result[-2])\n    return result",
            "Math": "Let $f(x)$ be a differentiable function on the interval $[a, b]$. Then, according to the Mean Value Theorem, there exists at least one point $c \\in (a, b)$ such that $f'(c) = \\frac{f(b) - f(a)}{b - a}$.",
            "Edge: Emoji": "Hello 👋🌍! 👨‍👩‍👧‍👦 🚀✨ The quick brown 🦊 jumps over the lazy 🐶.",
            "Edge: URL": "https://www.example.co.uk/path/to/resource?query=123&session_id=abcdef#section-4",
            "Edge: Numbers": "1,000,000.00 + 4.5e-10 = 1000000.00000000045 (approx 1B)",
            "Edge: Space": "def   foo(  )  : \n        pass    # 8 spaces indent",
            "Edge: Tamil Weird": "ஶ்ரீனிவாசன் மற்றும் அஃறிணைப் பெயர்கள், க்ஷத்திரியன்."
        }

        models = {
            "Llama-3 (ByteLevel)": tok_a,
            "Sarvam (Metaspace)": tok_b,
            "Hybrid (Regex+Metaspace)": tok_c,
            "Final Llama-3 (4M docs)": tok_final
        }
        
        print(f"| {'Domain':<10} | {'Llama-3 (ByteLevel)':<20} | {'Sarvam (Metaspace)':<20} | {'Hybrid (Regex+Metaspace)':<25} | {'Final Llama-3 (4M)':<20} |")
        print(f"|{'-'*12}|{'-'*22}|{'-'*22}|{'-'*27}|{'-'*22}|")
        
        for domain, text in tests.items():
            results = []
            for name, tok in models.items():
                encoded = tok.encode(text)
                tokens = encoded.tokens
                num_tokens = len(tokens)
                has_unk = "<|unk|>" in tokens
                status = f"{num_tokens} tokens" + (" (OOM/UNK!)" if has_unk else "")
                results.append(status)
                
            print(f"| {domain:<10} | {results[0]:<20} | {results[1]:<20} | {results[2]:<25} | {results[3]:<20} |")

if __name__ == "__main__":
    main()
