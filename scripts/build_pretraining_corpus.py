import os
import random
import time
import multiprocessing as mp
import numpy as np
from transformers import PreTrainedTokenizerFast
import tree_sitter_python as tspython
from tree_sitter import Language, Parser
from datasets import load_dataset
import pyarrow.parquet as pq

# FIM Rates
FIM_RATE_CODE = 0.50
FIM_RATE_TEXT = 0.10 # Lower rate for natural language to protect L2R coherence
AST_FIM_RATIO = 0.70 # 70% AST-FIM (structural), 30% random (mid-token)

def format_fim(prefix, middle, suffix):
    if random.random() < 0.5:
        # PSM format (Prefix-Suffix-Middle)
        return f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{middle}<|im_end|>"
    else:
        # SPM format (Suffix-Prefix-Middle)
        return f"<|fim_suffix|>{suffix}<|fim_prefix|>{prefix}<|fim_middle|>{middle}<|im_end|>"

def stream_folder(folder_path, column_name="text"):
    import glob
    import pyarrow.parquet as pq
    for f in glob.glob(os.path.join(folder_path, "**", "*.parquet"), recursive=True):
        try:
            parquet_file = pq.ParquetFile(f)
            for batch in parquet_file.iter_batches(batch_size=1000, columns=[column_name]):
                for val in batch[column_name]:
                    text = val.as_py()
                    if text and len(text.strip()) > 0:
                        yield text
        except Exception:
            pass

def get_ast_nodes(root_node, valid_nodes):
    stack = [root_node]
    while stack:
        node = stack.pop()
        if node.is_named and (node.end_byte - node.start_byte) > 10:
            valid_nodes.append(node)
        stack.extend(node.children)

def apply_ast_fim(text, parser):
    try:
        tree = parser.parse(bytes(text, "utf8"))
        valid_nodes = []
        get_ast_nodes(tree.root_node, valid_nodes)
        
        if not valid_nodes:
            return apply_random_fim(text)
            
        weights = [n.end_byte - n.start_byte for n in valid_nodes]
        total_weight = sum(weights)
        if total_weight == 0:
            return apply_random_fim(text)
            
        probs = [w / total_weight for w in weights]
        selected_node = random.choices(valid_nodes, weights=probs, k=1)[0]
        
        start_byte = selected_node.start_byte
        end_byte = selected_node.end_byte
        
        text_bytes = bytes(text, "utf8")
        prefix = text_bytes[:start_byte].decode("utf8", errors="ignore")
        middle = text_bytes[start_byte:end_byte].decode("utf8", errors="ignore")
        suffix = text_bytes[end_byte:].decode("utf8", errors="ignore")
        
        return format_fim(prefix, middle, suffix)
    except Exception:
        return apply_random_fim(text)

def apply_random_fim(text):
    if len(text) < 10:
        return text + "<|im_end|>"
    
    chars = list(text)
    split1 = random.randint(0, len(chars) - 2)
    split2 = random.randint(split1 + 1, len(chars) - 1)
    
    prefix = "".join(chars[:split1])
    middle = "".join(chars[split1:split2])
    suffix = "".join(chars[split2:])
    
    return format_fim(prefix, middle, suffix)

def process_document(args):
    text, domain = args
    
    global THREAD_PARSER, THREAD_TOKENIZER
    if 'THREAD_PARSER' not in globals() and domain == "code_python":
        try:
            PY_LANGUAGE = Language(tspython.language())
            THREAD_PARSER = Parser(PY_LANGUAGE)
        except Exception:
            THREAD_PARSER = None
            
    if 'THREAD_TOKENIZER' not in globals():
        THREAD_TOKENIZER = PreTrainedTokenizerFast.from_pretrained("models/tokenizer_bpe_65528_agentic_reasoning")
        
    if domain == "code_python":
        if random.random() < FIM_RATE_CODE:
            if random.random() < AST_FIM_RATIO and THREAD_PARSER is not None:
                text = apply_ast_fim(text, THREAD_PARSER)
            else:
                text = apply_random_fim(text)
    elif domain == "code_sql":
        if random.random() < FIM_RATE_CODE:
            text = apply_random_fim(text)
    else:
        # Math, English
        if random.random() < FIM_RATE_TEXT:
            text = apply_random_fim(text)
                
    processed_text = text + "<|im_end|>"
    return THREAD_TOKENIZER.encode(processed_text, add_special_tokens=True)

class WeightedDomainGenerator:
    def __init__(self, target_weights, base_dir):
        self.weights = dict(target_weights)
        self.generators = {}
        self.base_dir = base_dir
        
        # Local directories mapped to domains and their corresponding column name
        folders = {
            "math": (["finemath"], "text"),
            "code_python": (["starcoder_python"], "content"),
            "code_sql": (["starcoder_sql"], "content"),
            "english": (["fineweb_edu"], "text")
        }
        
        for domain, (f_list, col_name) in folders.items():
            self.generators[domain] = self._create_domain_generator(f_list, col_name)
            
    def _create_domain_generator(self, folders, col_name):
        # 1 Epoch / 1 Pass: Iterate over the folders exactly once, no outer while True loop.
        for folder in folders:
            folder_path = os.path.join(self.base_dir, folder)
            if not os.path.exists(folder_path):
                print(f"⚠️ Warning: Domain path {folder_path} does not exist.")
                continue
            for text in stream_folder(folder_path, col_name):
                yield text
        
    def __iter__(self):
        return self
        
    def __next__(self):
        while self.weights:
            # Re-normalize probabilities for remaining active domains
            domains = list(self.weights.keys())
            probs = list(self.weights.values())
            total_prob = sum(probs)
            if total_prob == 0:
                break
            normalized_probs = [p / total_prob for p in probs]
            
            sampled_domain = random.choices(domains, weights=normalized_probs, k=1)[0]
            gen = self.generators[sampled_domain]
            try:
                text = next(gen)
                return text, sampled_domain
            except StopIteration:
                # Domain is fully exhausted after 1 epoch
                print(f"\n📢 Domain '{sampled_domain}' fully exhausted (1 pass complete). Removing from active stream.")
                del self.weights[sampled_domain]
                
        raise StopIteration("All domains are fully exhausted. 1-pass pretraining corpus generation complete.")

def chunk_iterator(iterator, size):
    chunk = []
    for item in iterator:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk



def build_corpus(samples_per_domain=float('inf'), chunk_size=1000, resume=True, split_phases=False):
    BASE_DIR = "data/raw_corpus"
    OUTPUT_DIR_PHASE1 = "data/phase1_corpus"
    OUTPUT_DIR_PHASE2 = "data/phase2_corpus"
    OUTPUT_DIR_PHASE3 = "data/phase3_corpus"
    
    if split_phases:
        os.makedirs(OUTPUT_DIR_PHASE1, exist_ok=True)
        os.makedirs(OUTPUT_DIR_PHASE2, exist_ok=True)
        os.makedirs(OUTPUT_DIR_PHASE3, exist_ok=True)
        current_output_dir = OUTPUT_DIR_PHASE1
        current_phase = 1
    else:
        os.makedirs("data/mixed_50B_corpus", exist_ok=True)
        current_output_dir = "data/mixed_50B_corpus"
    
    print("Loading Tokenizer...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained("models/tokenizer_bpe_65528_agentic_reasoning")
    
    # Target weights: Math 35%, Code Python 28%, Code SQL 7%, English 30%
    target_weights = {
        "math": 0.35,
        "code_python": 0.28,
        "code_sql": 0.07,
        "english": 0.30
    }
    
    generator = WeightedDomainGenerator(target_weights, BASE_DIR)
    
    def task_generator():
        yielded = 0
        for text, domain in generator:
            yield (text, domain)
            yielded += 1
            if yielded >= samples_per_domain:
                break

    print("Starting Multiprocessing Pipeline...")
    pool = mp.Pool(processes=os.cpu_count())
    
    file_idx = 0
    MAX_TOKENS_PER_FILE = 250_000_000 # ~500MB per bin file
    
    total_tokens_written = 0
    PHASE1_LIMIT = 35_000_000_000  # Phase 1: 0 - 35B
    PHASE2_LIMIT = 46_000_000_000  # Phase 2: 35B - 46B
    
    buffer = np.zeros(MAX_TOKENS_PER_FILE, dtype=np.uint16)
    buffer_idx = 0
    
    global_stop = False
    try:
        for raw_chunk in chunk_iterator(task_generator(), 10000):
            if global_stop:
                break
            
            encoded_batch = pool.map(process_document, raw_chunk)
            
            for tokens in encoded_batch:
                # Safety validation to prevent silent uint16 wrap-around/overflow
                if len(tokens) > 0 and max(tokens) > 65535:
                    raise ValueError(
                        f"Token ID {max(tokens)} exceeds uint16 limit (65535)! "
                        f"Vocabulary exceeds safety boundary."
                    )

                if total_tokens_written >= 50_000_000_000:
                    global_stop = True
                    break
                    
                if buffer_idx + len(tokens) >= MAX_TOKENS_PER_FILE:
                    space_left = MAX_TOKENS_PER_FILE - buffer_idx
                    buffer[buffer_idx:] = tokens[:space_left]
                    
                    out_path = os.path.join(current_output_dir, f"corpus_{file_idx:04d}.bin")
                    mmap = np.memmap(out_path, dtype=np.uint16, mode='w+', shape=(MAX_TOKENS_PER_FILE,))
                    mmap[:] = buffer[:]
                    mmap.flush()
                    print(f"✅ Wrote {MAX_TOKENS_PER_FILE} tokens to {out_path}")
                    
                    total_tokens_written += MAX_TOKENS_PER_FILE
                    file_idx += 1
                    buffer_idx = 0
                    
                    if split_phases:
                        if current_phase == 1 and total_tokens_written >= PHASE1_LIMIT:
                            current_phase = 2
                            current_output_dir = OUTPUT_DIR_PHASE2
                            file_idx = 0 # reset file idx for new folder
                            print("\n>>> TRANSITIONING TO PHASE 2 DATASET <<<\n")
                        elif current_phase == 2 and total_tokens_written >= PHASE2_LIMIT:
                            current_phase = 3
                            current_output_dir = OUTPUT_DIR_PHASE3
                            file_idx = 0 # reset file idx for new folder
                            print("\n>>> TRANSITIONING TO PHASE 3 DATASET <<<\n")
                    
                    remaining_tokens = tokens[space_left:]
                    if len(remaining_tokens) > 0:
                        buffer[:len(remaining_tokens)] = remaining_tokens
                        buffer_idx = len(remaining_tokens)
                else:
                    buffer[buffer_idx:buffer_idx+len(tokens)] = tokens
                    buffer_idx += len(tokens)
    except KeyboardInterrupt:
        print("\nInterrupt received. Closing pool...")
    finally:
        pool.close()
        pool.join()
        
    # Write remaining
    if buffer_idx > 0:
        out_path = os.path.join(current_output_dir, f"corpus_{file_idx:04d}.bin")
        mmap = np.memmap(out_path, dtype=np.uint16, mode='w+', shape=(buffer_idx,))
        mmap[:] = buffer[:buffer_idx]
        mmap.flush()
        print(f"✅ Wrote {buffer_idx} tokens to {out_path}")
        
    print("🎉 Corpus generation complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_phases", action="store_true", help="Split output into phase1, phase2, phase3 for curriculum learning")
    args = parser.parse_args()
    build_corpus(samples_per_domain=float('inf'), resume=False, split_phases=args.split_phases)
