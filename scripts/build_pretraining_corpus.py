import os
import glob
import gzip
import json
import random
import multiprocessing as mp
from functools import partial
import numpy as np
import pyarrow.parquet as pq
from transformers import PreTrainedTokenizerFast
import tree_sitter_python as tspython
from tree_sitter import Language, Parser

BASE_DIR = "data/raw_corpus"
OUTPUT_DIR = "data/mixed_50B_corpus"

DOMAINS = {
    "tamil": ["tamil_wikipedia", "indiccorp_tamil"],
    "code": ["starcoder"],
    "math": ["finemath", "pes2o"],
    "english": ["fineweb_edu", "cosmopedia", "wikipedia"]
}

# FIM Rates
FIM_RATE_CODE = 0.70
AST_FIM_RATIO = 0.90 # 90% of FIM is AST-FIM, 10% is random

def stream_folder(folder_path, skip_patterns=None):
    """Safely stream text from a folder without loading everything into memory."""
    for f in glob.glob(os.path.join(folder_path, "**", "*.txt"), recursive=True):
        if skip_patterns and any(p in f for p in skip_patterns): continue
        try:
            with open(f, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    if line: yield line
        except Exception: pass

    for f in glob.glob(os.path.join(folder_path, "**", "*.parquet"), recursive=True):
        if skip_patterns and any(p in f for p in skip_patterns): continue
        try:
            schema = pq.read_schema(f)
            col_name = 'text' if 'text' in schema.names else 'content'
            table = pq.read_table(f, columns=[col_name])
            for val in table[col_name]:
                text = val.as_py()
                if text and len(text.strip()) > 0: yield text
        except Exception: pass

    for f in glob.glob(os.path.join(folder_path, "**", "*.json.gz"), recursive=True):
        if skip_patterns and any(p in f for p in skip_patterns): continue
        try:
            with gzip.open(f, 'rt', encoding='utf-8') as file:
                for line in file:
                    obj = json.loads(line)
                    text = obj.get("text", "")
                    if text and len(text.strip()) > 0: yield text
        except Exception: pass

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
            
        # Select node with probability proportional to size
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
        
        return f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{middle}<|im_end|>"
    except Exception:
        # Fallback to random FIM on parse failure
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
    
    return f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{middle}<|im_end|>"

def process_document(args):
    text, domain = args
    
    if domain == "code":
        if random.random() < FIM_RATE_CODE:
            if random.random() < AST_FIM_RATIO:
                # Tree-sitter objects are not picklable, must initialize per-process
                global THREAD_PARSER
                if 'THREAD_PARSER' not in globals():
                    PY_LANGUAGE = Language(tspython.language())
                    THREAD_PARSER = Parser(PY_LANGUAGE)
                return apply_ast_fim(text, THREAD_PARSER)
            else:
                return apply_random_fim(text)
                
    return text + "<|im_end|>"

def chunk_iterator(iterator, size):
    chunk = []
    for item in iterator:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

def build_corpus(samples_per_domain=float('inf'), chunk_size=3000, resume=True):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading Tokenizer...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained("models/tokenizer")
    
    # Generate tasks
    def task_generator():
        for domain, folders in DOMAINS.items():
            if resume and domain in ["tamil", "code"]:
                continue
            yielded = 0
            for folder in folders:
                if resume and folder == "finemath":
                    continue
                folder_path = os.path.join(BASE_DIR, folder)
                if not os.path.exists(folder_path): continue
                
                skip_patterns = None
                if resume and folder == "pes2o":
                    # skip train-00000 to train-00013
                    skip_patterns = [f"train-{i:05d}" for i in range(14)]
                    
                for text in stream_folder(folder_path, skip_patterns):
                    yield (text, domain)
                    yielded += 1
                    if yielded >= samples_per_domain:
                        break
                if yielded >= samples_per_domain:
                    break

    print("Starting Multiprocessing Pipeline...")
    # Set processes=8 and chunk_size=3000 to guarantee stability and prevent OOM killer
    pool = mp.Pool(processes=8)
    
    file_idx = 149 if resume else 0
    MAX_TOKENS_PER_FILE = 250_000_000 # ~500MB per bin file
    
    buffer = np.zeros(MAX_TOKENS_PER_FILE, dtype=np.uint16)
    buffer_idx = 0
    
    for raw_chunk in chunk_iterator(task_generator(), chunk_size):
        batch_results = pool.map(process_document, raw_chunk)
        encoded_batch = tokenizer(batch_results, add_special_tokens=True)["input_ids"]
        
        # Safety validation to prevent silent uint16 wrap-around/overflow
        for tokens in encoded_batch:
            if len(tokens) > 0 and max(tokens) > 65535:
                raise ValueError(
                    f"Token ID {max(tokens)} exceeds uint16 limit (65535)! "
                    f"This will cause severe dataset corruption via silent wrap-around. "
                    f"Please retrain your tokenizer with a smaller vocabulary size."
                )
                
        for tokens in encoded_batch:
            
            if buffer_idx + len(tokens) >= MAX_TOKENS_PER_FILE:
                space_left = MAX_TOKENS_PER_FILE - buffer_idx
                buffer[buffer_idx:] = tokens[:space_left]
                
                out_path = os.path.join(OUTPUT_DIR, f"corpus_{file_idx:04d}.bin")
                mmap = np.memmap(out_path, dtype=np.uint16, mode='w+', shape=(MAX_TOKENS_PER_FILE,))
                mmap[:] = buffer[:]
                mmap.flush()
                print(f"✅ Wrote {MAX_TOKENS_PER_FILE} tokens to {out_path}")
                
                file_idx += 1
                buffer_idx = 0
                
                remaining_tokens = tokens[space_left:]
                if len(remaining_tokens) > 0:
                    buffer[:len(remaining_tokens)] = remaining_tokens
                    buffer_idx = len(remaining_tokens)
            else:
                buffer[buffer_idx:buffer_idx+len(tokens)] = tokens
                buffer_idx += len(tokens)

    # Write remaining
    if buffer_idx > 0:
        out_path = os.path.join(OUTPUT_DIR, f"corpus_{file_idx:04d}.bin")
        mmap = np.memmap(out_path, dtype=np.uint16, mode='w+', shape=(buffer_idx,))
        mmap[:] = buffer[:buffer_idx]
        mmap.flush()
        print(f"✅ Wrote {buffer_idx} tokens to {out_path}")

    pool.close()
    pool.join()
    print("🎉 Corpus generation complete!")

if __name__ == "__main__":
    build_corpus(samples_per_domain=float('inf')) # Process the entire 50B dataset
