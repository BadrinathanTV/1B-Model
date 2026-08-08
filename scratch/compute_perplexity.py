import os
import sys
import torch
import torch.nn.functional as F
import math
import yaml
from transformers import PreTrainedTokenizerFast

# Add training/ directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training")))

from config import SLMConfig
from model import SLMModel
from layers.rope import precompute_freqs_cis, precompute_cos_sin

def load_config(config_path):
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    model_cfg = cfg_dict.get("model", {})
    if "training" not in cfg_dict:
        cfg_dict["training"] = {}
    
    return SLMConfig(
        vocab_size=model_cfg.get("vocab_size", 65536),
        hidden_size=model_cfg.get("hidden_size", 1280),
        num_hidden_layers=model_cfg.get("num_hidden_layers", 28),
        num_attention_heads=model_cfg.get("num_attention_heads", 10),
        intermediate_size=model_cfg.get("intermediate_size", 4096),
        max_position_embeddings=model_cfg.get("max_position_embeddings", 8192),
        rms_norm_eps=float(model_cfg.get("rms_norm_eps", 1e-6)),
        kv_lora_rank=model_cfg.get("kv_lora_rank", 320),
        q_lora_rank=model_cfg.get("q_lora_rank", 960),
        qk_rope_head_dim=model_cfg.get("qk_rope_head_dim", 64),
        v_head_dim=model_cfg.get("v_head_dim", 128),
        rope_theta=model_cfg.get("rope_theta", 10000.0),
        mtp_depth=model_cfg.get("mtp_depth", 2),
        use_mtp=model_cfg.get("use_mtp", False),
        init_std=model_cfg.get("init_std", 0.02),
        z_loss_weight=float(model_cfg.get("z_loss_weight", 1e-4)),
        embed_scale=model_cfg.get("embed_scale", True),
        output_logit_scale=model_cfg.get("output_logit_scale", 1.0),
        output_logit_scale_trainable=model_cfg.get("output_logit_scale_trainable", False),
        tie_word_embeddings=model_cfg.get("tie_word_embeddings", True),
        max_delta_history=model_cfg.get("max_delta_history", 2),
        training=cfg_dict.get("training", {})
    )

def evaluate_perplexity(checkpoint_path, config, tokenizer, tokens, device):
    print(f"\n--- Evaluating {checkpoint_path} ---")
    
    # Initialize model
    model = SLMModel(config).to(device)
    
    # Load weights
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint {checkpoint_path} not found.")
        return None, None
        
    state_dict = torch.load(checkpoint_path, map_location=device)
    fixed_state = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            fixed_state[k[10:]] = v
        else:
            fixed_state[k] = v
            
    model.load_state_dict(fixed_state, strict=True)
    model.eval()
    
    seq_len = 512
    chunk_size = seq_len + 1
    
    # Chunk tokens
    num_chunks = len(tokens) // chunk_size
    print(f"Total tokens: {len(tokens)}, evaluating on {num_chunks} chunks of size {seq_len}...")
    
    if num_chunks == 0:
        print("Error: Text is too short.")
        return None, None
        
    # Precompute RoPE caches
    freqs_cis = precompute_freqs_cis(
        dim=config.qk_rope_head_dim,
        end=seq_len,
        theta=config.rope_theta,
        device=device,
        yarn_scale=config.yarn_scale_factor if getattr(config, "use_yarn", False) else 1.0,
        yarn_beta_fast=getattr(config, "yarn_beta_fast", 32.0),
        yarn_beta_slow=getattr(config, "yarn_beta_slow", 1.0),
        yarn_orig_ctx=getattr(config, "yarn_original_context", 512),
    )
    cos_cache, sin_cache = precompute_cos_sin(
        dim=config.qk_rope_head_dim,
        end=seq_len,
        theta=config.rope_theta,
        device=device,
        yarn_scale=config.yarn_scale_factor if getattr(config, "use_yarn", False) else 1.0,
        yarn_beta_fast=getattr(config, "yarn_beta_fast", 32.0),
        yarn_beta_slow=getattr(config, "yarn_beta_slow", 1.0),
        yarn_orig_ctx=getattr(config, "yarn_original_context", 512),
    )
    
    total_nll = 0.0
    total_tokens_evaluated = 0
    
    with torch.no_grad():
        for i in range(num_chunks):
            start_idx = i * chunk_size
            chunk = tokens[start_idx : start_idx + chunk_size]
            
            # Prepare inputs & targets
            inputs = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
            targets = torch.tensor([chunk[1:]], dtype=torch.long, device=device)
            
            # Forward pass
            # Returns [logits] because use_mtp=False
            logits_list = model(
                inputs,
                freqs_cis=freqs_cis,
                cos_cache=cos_cache,
                sin_cache=sin_cache,
                use_mtp=False
            )
            logits = logits_list[0]
            
            # Compute loss
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), targets.view(-1), reduction='sum')
            
            total_nll += loss.item()
            total_tokens_evaluated += targets.numel()
            
    avg_loss = total_nll / total_tokens_evaluated
    perplexity = math.exp(avg_loss)
    
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"Perplexity:      {perplexity:.4f}")
    
    return avg_loss, perplexity

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running evaluation on {device}...")
    
    config_path = "training/configs/default.yaml"
    config = load_config(config_path)
    
    tokenizer_path = "models/tokenizer_bpe_65528_agentic_reasoning"
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    
    # Read reference paper text
    paper_path = "nvfp4_paper.txt"
    if not os.path.exists(paper_path):
        print(f"Error: {paper_path} not found.")
        sys.exit(1)
        
    with open(paper_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    tokens = tokenizer.encode(text)
    
    results = {}
    checkpoints = [
        "model_500_re-run.pt",
        "model_1000_re_run.pt",
        "model_1500_re_run.pt",
        "model_2500_re_run.pt",
        "model_9500_re_run.pt"
    ]
    for checkpoint in checkpoints:
        if os.path.exists(checkpoint):
            loss, ppl = evaluate_perplexity(checkpoint, config, tokenizer, tokens, device)
            results[checkpoint] = (loss, ppl)
        else:
            print(f"Warning: {checkpoint} not found in workspace.")
            
    print("\n================ SUMMARY ================ ")
    for k, v in results.items():
        if v[0] is not None:
            print(f"{k:25s} | Validation Loss: {v[0]:.4f} | Perplexity: {v[1]:.4f}")
    print("========================================= ")

if __name__ == "__main__":
    main()
