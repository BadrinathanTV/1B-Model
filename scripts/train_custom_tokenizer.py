import os
import glob
import gzip
import json
import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, Regex
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel, Digits, Sequence, Split
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

def get_training_corpus(chars_per_domain=500_000_000):
    """Yields text balanced by character count across domains."""
    for domain, folders in DOMAINS.items():
        print(f"\n--- Sampling ~{chars_per_domain/1e6:.0f}M chars for {domain.upper()} domain ---")
        char_count = 0
        for folder in folders:
            folder_path = os.path.join(BASE_DIR, folder)
            if not os.path.exists(folder_path):
                print(f"  Skipping {folder} (not found)")
                continue
                
            print(f"  Reading from {folder}...")
            for text in stream_folder(folder_path):
                yield text
                char_count += len(text)
                if char_count >= chars_per_domain:
                    break
            if char_count >= chars_per_domain:
                break
        print(f"Finished {domain.upper()} domain with ~{char_count/1e6:.1f}M chars.")

def batch_iterator(generator, batch_size=10000):
    """Yields lists of strings instead of single strings to drastically reduce Python-to-Rust GIL overhead."""
    batch = []
    for item in generator:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def train_tokenizer():
    print("\nInitializing Final Llama-3 ByteLevel BPE Tokenizer...")
    
    # 1. Base BPE Model
    # Set to 65528. Since all special/reserved tokens are passed to BpeTrainer,
    # the trainer includes them in this total. The final vocab size will be exactly 65528.
    vocab_size = 65528
    tokenizer = Tokenizer(BPE())
    
    # 2. Pre-Tokenizer & Decoder (Llama-3 Architecture)
    llama_pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    tokenizer.pre_tokenizer = Sequence([Split(Regex(llama_pattern), behavior="isolated"), ByteLevel(add_prefix_space=False)])
    tokenizer.decoder = decoders.ByteLevel()
    
    # Comprehensive Future-Proof Special Tokens
    base_specials = ["<|im_start|>", "<|im_end|>", "<|pad|>", "<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"]
    reasoning_specials = ["<|thought|>", "<|end_thought|>", "<|step|>", "<|end_step|>"]
    agentic_specials = ["<|tool_call|>", "<|tool_response|>", "<|python_run|>", "<|python_output|>"]
    rag_specials = ["<|context_start|>", "<|context_end|>", "<|cite|>"]
    multimodal_specials = ["<|image|>", "<|audio|>"]
    reserved_specials = [f"<|reserved_{i}|>" for i in range(50)]
    
    all_special_tokens = base_specials + reasoning_specials + agentic_specials + rag_specials + multimodal_specials + reserved_specials

    # 3. Trainer with highly optimized BPE parameters
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=50,
        max_token_length=32, # Crucial: Prevents wasting vocab slots on massive 100+ byte sequences
        special_tokens=all_special_tokens,
        initial_alphabet=ByteLevel.alphabet()
    )
    
    # Force 100% CPU Parallelism in Rust
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    
    print("\nTraining tokenizer on 4,000,000 balanced documents (~15 GB text) using SSD Virtual RAM...")
    print("This will take 10-30 minutes. Please wait while the Rust backend computes merges...")
    
    # Execute high-speed batched training
    corpus_gen = get_training_corpus(chars_per_domain=500_000_000)
    batched_gen = batch_iterator(corpus_gen, batch_size=10000)
    
    tokenizer.train_from_iterator(batched_gen, trainer=trainer)
    
    # Save results
    output_dir = "models/tokenizer_bpe_65528_agentic_reasoning"
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save(os.path.join(output_dir, "tokenizer.json"))
    
    raw_vocab_size = tokenizer.get_vocab_size()
    print(f"\n✅ Raw BPE tokenizer trained. Vocab size: {raw_vocab_size} (max ID: {raw_vocab_size - 1})")
    
    # Export to Transformers format for seamless local/HuggingFace inference
    try:
        from transformers import PreTrainedTokenizerFast
        fast_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token="<|im_start|>",
            eos_token="<|im_end|>",
            pad_token="<|pad|>",
            additional_special_tokens=all_special_tokens[3:] # Exclude bos, eos, pad
        )
        
        # Critical: Verify total vocab fits within uint16 (0-65535)
        wrapped_vocab_size = len(fast_tokenizer)
        print(f"   Wrapped vocab size (after PreTrainedTokenizerFast): {wrapped_vocab_size}")
        
        if wrapped_vocab_size > 65536:
            raise ValueError(
                f"FATAL: Wrapped tokenizer vocab size is {wrapped_vocab_size}, "
                f"which exceeds uint16 limit of 65536! "
                f"Reduce BpeTrainer vocab_size (currently {vocab_size}) by at least "
                f"{wrapped_vocab_size - 65536} to fix this."
            )
        
        # Verify special token IDs (test a subset)
        for tok_str in ["<|fim_prefix|>", "<|thought|>", "<|tool_call|>", "<|reserved_0|>"]:
            tid = fast_tokenizer.encode(tok_str, add_special_tokens=False)
            if tid:
                print(f"   {tok_str} -> ID {tid[0]}")
        
        fast_tokenizer.save_pretrained(output_dir)
        print(f"✅ Tokenizer saved to {output_dir}/ (total vocab: {wrapped_vocab_size}, max ID: {wrapped_vocab_size - 1})")
    except ImportError:
        print("Note: 'transformers' library not installed, skipping HuggingFace format export.")

if __name__ == "__main__":
    train_tokenizer()
