import os
import sys
import torch
import yaml
from transformers import PreTrainedTokenizerFast

# Add training/ directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training")))

from config import SLMConfig
from model import SLMModel
from layers.rope import precompute_freqs_cis, precompute_cos_sin

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

def generate(model, tokenizer, prompt, max_new_tokens=1024, temperature=1.0, device="cuda"):
    model.eval()
    tokens = tokenizer.encode(prompt)
    max_seq_len = len(tokens) + max_new_tokens + 5
    
    freqs_cis = precompute_freqs_cis(model.config.qk_rope_head_dim, max_seq_len, model.config.rope_theta, device)
    cos_cache, sin_cache = precompute_cos_sin(model.config.qk_rope_head_dim, max_seq_len, model.config.rope_theta, device)
    
    past_key_values = [None] * model.config.num_hidden_layers
    generated = list(tokens)
    next_token = None
    
    # Stream the prompt itself
    print(prompt, end="", flush=True)
    
    for step in range(max_new_tokens):
        is_prefill = (step == 0)
        if is_prefill:
            current_input_ids = torch.tensor([generated], dtype=torch.long, device=device)
            step_freqs = freqs_cis[:len(tokens)]
            step_cos = cos_cache[:len(tokens)]
            step_sin = sin_cache[:len(tokens)]
        else:
            current_input_ids = torch.tensor([[next_token]], dtype=torch.long, device=device)
            step_freqs = freqs_cis[len(tokens) + step - 1 : len(tokens) + step]
            step_cos = cos_cache[len(tokens) + step - 1 : len(tokens) + step]
            step_sin = sin_cache[len(tokens) + step - 1 : len(tokens) + step]
            
        with torch.no_grad():
            out = model(
                current_input_ids,
                freqs_cis=step_freqs,
                cos_cache=step_cos,
                sin_cache=step_sin,
                use_cache=True,
                past_key_values=past_key_values,
                use_mtp=False
            )
            
        logits = out[0] if isinstance(out, tuple) else out
        if isinstance(out, tuple):
            past_key_values = out[1]
            
        next_token_logits = logits[0][0, -1, :]
        if temperature == 0.0:
            next_token = torch.argmax(next_token_logits).item()
        else:
            next_token_logits = next_token_logits / temperature
            top_k = 50
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            next_token_logits[indices_to_remove] = -float("Inf")
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
        generated.append(next_token)
        
        # Decode and print token immediately
        token_str = tokenizer.decode([next_token])
        print(token_str, end="", flush=True)
        
        if next_token == tokenizer.eos_token_id:
            break
            
    print()  # Newline after finished generation
    return tokenizer.decode(generated)

def main():
    # Force line buffering for standard output to ensure seamless streaming
    sys.stdout.reconfigure(line_buffering=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running streaming generation benchmarks on {device}...\n", flush=True)
    
    config = load_config_direct("training/configs/default.yaml")
    tokenizer = PreTrainedTokenizerFast.from_pretrained("models/tokenizer_bpe_65528_agentic_reasoning")
    
    checkpoints = [
        "model_13500_re-run.pt"
    ]
    
    prompts = {
        "English": [
            "Tamil Nadu is",
            "Artificial Intelligence"
        ],
        "Math": [
            "Sum of first n numbers",
            "If x + 5 = 12,"
        ],
        "SQL": [
            "SELECT name, age FROM",
            "SELECT department, COUNT(*)"
        ],
        "Code": [
            "def fibonacci(n):",
            "def bubble_sort(arr):"
        ]
    }
    
    for checkpoint_path in checkpoints:
        if not os.path.exists(checkpoint_path):
            print(f"Skipping {checkpoint_path} (not found)\n", flush=True)
            continue
            
        print(f"=======================================================", flush=True)
        print(f"EVALUATING CHECKPOINT: {checkpoint_path} (temp=1.0)", flush=True)
        print(f"=======================================================\n", flush=True)
        
        # Load model weights
        model = SLMModel(config).to(device=device, dtype=torch.bfloat16)
        state_dict = torch.load(checkpoint_path, map_location=device)
        fixed_state = {}
        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                fixed_state[k[10:]] = v
            else:
                fixed_state[k] = v
        model.load_state_dict(fixed_state, strict=True)
        
        for category, category_prompts in prompts.items():
            print(f"--- Category: {category} ---", flush=True)
            for prompt in category_prompts:
                print("Output: ", end="", flush=True)
                generate(model, tokenizer, prompt, max_new_tokens=1024, temperature=1.0, device=device)
                print(flush=True)
            print("-" * 50, flush=True)
            
        # Free memory immediately to prevent CUDA OOM
        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
