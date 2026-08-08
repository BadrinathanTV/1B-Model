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
        use_mtp=model_cfg.get("use_mtp", True),
        init_std=model_cfg.get("init_std", 0.02),
        z_loss_weight=float(model_cfg.get("z_loss_weight", 1e-4)),
        embed_scale=model_cfg.get("embed_scale", True),
        output_logit_scale=model_cfg.get("output_logit_scale", 1.0),
        output_logit_scale_trainable=model_cfg.get("output_logit_scale_trainable", False),
        tie_word_embeddings=model_cfg.get("tie_word_embeddings", True),
        max_delta_history=model_cfg.get("max_delta_history", 2),
        training=cfg_dict.get("training", {})
    )

# Override local config function for robustness
def load_config_direct(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]
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
        use_mtp=model_cfg.get("use_mtp", True),
        init_std=model_cfg.get("init_std", 0.02),
        z_loss_weight=float(model_cfg.get("z_loss_weight", 1e-4)),
        embed_scale=model_cfg.get("embed_scale", True),
        output_logit_scale=model_cfg.get("output_logit_scale", 1.0),
        output_logit_scale_trainable=model_cfg.get("output_logit_scale_trainable", False),
        tie_word_embeddings=model_cfg.get("tie_word_embeddings", True),
        max_delta_history=model_cfg.get("max_delta_history", 2),
        training=cfg.get("training", {})
    )

def evaluate_mtp_perplexity(checkpoint_path, config, tokenizer, tokens, device):
    print(f"\nEvaluating: {checkpoint_path}")
    
    # Initialize model
    model = SLMModel(config).to(device)
    
    # Load weights
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
    num_chunks = len(tokens) // chunk_size
    
    # Precompute RoPE caches
    freqs_cis = precompute_freqs_cis(
        dim=config.qk_rope_head_dim,
        end=seq_len,
        theta=config.rope_theta,
        device=device,
    )
    cos_cache, sin_cache = precompute_cos_sin(
        dim=config.qk_rope_head_dim,
        end=seq_len,
        theta=config.rope_theta,
        device=device,
    )
    
    # We will accumulate:
    # 1. Main loss (next-token prediction) when use_mtp=False
    # 2. Main loss when use_mtp=True
    # 3. MTP loss (t+2 prediction) when use_mtp=True
    
    nll_no_mtp = 0.0
    nll_with_mtp_main = 0.0
    nll_with_mtp_aux = 0.0
    
    tokens_main = 0
    tokens_aux = 0
    
    lm_head_weight = model.lm_head.weight
    
    with torch.no_grad():
        for i in range(num_chunks):
            start_idx = i * chunk_size
            chunk = tokens[start_idx : start_idx + chunk_size]
            
            inputs = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
            targets = torch.tensor([chunk[1:]], dtype=torch.long, device=device)
            
            # --- Case 1: Without MTP ---
            hidden_no_mtp = model(
                inputs, freqs_cis=freqs_cis, cos_cache=cos_cache, sin_cache=sin_cache,
                return_hidden_states=True, use_mtp=False
            )
            logits_no_mtp = F.linear(hidden_no_mtp[0], lm_head_weight)
            loss_no_mtp = F.cross_entropy(logits_no_mtp.view(-1, config.vocab_size), targets.view(-1), reduction='sum')
            nll_no_mtp += loss_no_mtp.item()
            tokens_main += targets.numel()
            
            # --- Case 2: With MTP ---
            hidden_with_mtp = model(
                inputs, freqs_cis=freqs_cis, cos_cache=cos_cache, sin_cache=sin_cache,
                return_hidden_states=True, use_mtp=True, target_ids=targets
            )
            
            # Main head loss (predicting t+1)
            logits_main = F.linear(hidden_with_mtp[0], lm_head_weight)
            loss_main = F.cross_entropy(logits_main.view(-1, config.vocab_size), targets.view(-1), reduction='sum')
            nll_with_mtp_main += loss_main.item()
            
            # MTP head loss (predicting t+2)
            # Alignment: hidden_with_mtp[1] is shape (1, seq_len-1, H)
            # targets for t+2 prediction are targets[:, 1:] which is shape (1, seq_len-1)
            logits_aux = F.linear(hidden_with_mtp[1], lm_head_weight)
            targets_aux = targets[:, 1:].contiguous()
            loss_aux = F.cross_entropy(logits_aux.view(-1, config.vocab_size), targets_aux.view(-1), reduction='sum')
            nll_with_mtp_aux += loss_aux.item()
            tokens_aux += targets_aux.numel()
            
    loss_no_mtp_avg = nll_no_mtp / tokens_main
    ppl_no_mtp = math.exp(loss_no_mtp_avg)
    
    loss_with_mtp_main_avg = nll_with_mtp_main / tokens_main
    ppl_with_mtp_main = math.exp(loss_with_mtp_main_avg)
    
    loss_with_mtp_aux_avg = nll_with_mtp_aux / tokens_aux
    ppl_with_mtp_aux = math.exp(loss_with_mtp_aux_avg)
    
    # Joint MTP loss as defined in training (main_loss + 0.3 * mtp_loss)
    loss_joint = loss_with_mtp_main_avg + 0.3 * loss_with_mtp_aux_avg
    ppl_joint = math.exp(loss_joint)
    
    print(f"  [Without MTP]")
    print(f"    Base Head (t+1) Loss: {loss_no_mtp_avg:.4f} | Perplexity: {ppl_no_mtp:.4f}")
    print(f"  [With MTP]")
    print(f"    Base Head (t+1) Loss: {loss_with_mtp_main_avg:.4f} | Perplexity: {ppl_with_mtp_main:.4f}")
    print(f"    MTP Head  (t+2) Loss: {loss_with_mtp_aux_avg:.4f} | Perplexity: {ppl_with_mtp_aux:.4f}")
    print(f"    Joint Loss (Base+0.3MTP): {loss_joint:.4f} | Perplexity: {ppl_joint:.4f}")
    
    return {
        "no_mtp_base": (loss_no_mtp_avg, ppl_no_mtp),
        "with_mtp_base": (loss_with_mtp_main_avg, ppl_with_mtp_main),
        "with_mtp_aux": (loss_with_mtp_aux_avg, ppl_with_mtp_aux),
        "joint": (loss_joint, ppl_joint)
    }

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_config_direct("training/configs/default.yaml")
    
    tokenizer = PreTrainedTokenizerFast.from_pretrained("models/tokenizer_bpe_65528_agentic_reasoning")
    
    with open("nvfp4_paper.txt", "r", encoding="utf-8") as f:
        text = f.read()
    tokens = tokenizer.encode(text)
    
    checkpoints = [
        "model_13500_re-run.pt"
    ]
    
    all_results = {}
    for cp in checkpoints:
        if os.path.exists(cp):
            all_results[cp] = evaluate_mtp_perplexity(cp, config, tokenizer, tokens, device)
            
    print("\n========================= DETAILED COMPARISON ========================= ")
    for cp, res in all_results.items():
        print(f"\nCheckpoint: {cp}")
        print(f"  * Base Head Perplexity (No MTP):         {res['no_mtp_base'][1]:.4f} (Loss: {res['no_mtp_base'][0]:.4f})")
        print(f"  * Base Head Perplexity (With MTP):       {res['with_mtp_base'][1]:.4f} (Loss: {res['with_mtp_base'][0]:.4f})")
        print(f"  * MTP (t+2) Head Perplexity:             {res['with_mtp_aux'][1]:.4f} (Loss: {res['with_mtp_aux'][0]:.4f})")
        print(f"  * Joint Perplexity (Base + 0.3 * MTP):   {res['joint'][1]:.4f} (Loss: {res['joint'][0]:.4f})")
    print("======================================================================== ")

if __name__ == "__main__":
    main()
