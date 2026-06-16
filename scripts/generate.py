#!/usr/bin/env python3
"""
Text Generation Script
======================

Loads a trained model checkpoint and performs autoregressive text generation.
Includes support for temperature, top-k, and top-p (nucleus) sampling.
"""

import os
import sys
import time
import argparse
import random
import yaml
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast

# Add training/ directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training")))

from config import SLMConfig
from model import SLMModel
from layers.rope import precompute_freqs_cis, precompute_cos_sin


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text from SLM checkpoint.")
    parser.add_argument(
        "--config",
        type=str,
        default="training/configs/dry_run.yaml",
        help="Path to model config YAML file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint folder or file (e.g. checkpoints/pytorch_model.bin).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The beauty of Tamil Nadu is",
        help="Input prompt for generation.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0.0 for greedy decoding).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling threshold (0 to disable).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling threshold (1.0 to disable).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path):
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    model_cfg = cfg_dict.get("model", {})
    if "training" not in cfg_dict:
        cfg_dict["training"] = {}
    
    return SLMConfig(
        vocab_size=model_cfg.get("vocab_size", 128256),
        hidden_size=model_cfg.get("hidden_size", 1280),
        num_hidden_layers=model_cfg.get("num_hidden_layers", 24),
        num_attention_heads=model_cfg.get("num_attention_heads", 10),
        intermediate_size=model_cfg.get("intermediate_size", 5120),
        max_position_embeddings=model_cfg.get("max_position_embeddings", 8192),
        rms_norm_eps=float(model_cfg.get("rms_norm_eps", 1e-6)),
        kv_lora_rank=model_cfg.get("kv_lora_rank", 320),
        q_lora_rank=model_cfg.get("q_lora_rank", 960),
        qk_rope_head_dim=model_cfg.get("qk_rope_head_dim", 64),
        v_head_dim=model_cfg.get("v_head_dim", 128),
        rope_theta=model_cfg.get("rope_theta", 10000.0),
        mtp_depth=model_cfg.get("mtp_depth", 1),
        use_mtp=model_cfg.get("use_mtp", True),
        init_std=model_cfg.get("init_std", 0.02),
        z_loss_weight=float(model_cfg.get("z_loss_weight", 1e-4)),
        embed_scale=model_cfg.get("embed_scale", True),
        output_logit_scale=model_cfg.get("output_logit_scale", 1.0),
        output_logit_scale_trainable=model_cfg.get("output_logit_scale_trainable", False),
        tie_word_embeddings=model_cfg.get("tie_word_embeddings", True),
        max_delta_history=model_cfg.get("max_delta_history", 4),
        training=cfg_dict.get("training", {})
    )


def sample(logits, temperature=0.7, top_k=50, top_p=0.9):
    """Filter and sample from a distribution of logits."""
    if temperature == 0.0:
        return torch.argmax(logits).item()

    logits = logits / temperature

    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float("Inf")

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float("Inf")

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


def generate_autoregressive(model, prompt_tokens, max_new_tokens, config, tokenizer, temperature, top_k, top_p):
    """Standard autoregressive generation."""
    device = next(model.parameters()).device
    model.eval()

    # Precompute caches up to the maximum potential length
    max_seq_len = len(prompt_tokens) + max_new_tokens + 5
    freqs_cis = precompute_freqs_cis(config.qk_rope_head_dim, max_seq_len, config.rope_theta, device)
    cos_cache, sin_cache = precompute_cos_sin(config.qk_rope_head_dim, max_seq_len, config.rope_theta, device)

    generated = list(prompt_tokens)
    print(tokenizer.decode(generated), end="", flush=True)

    start_time = time.time()
    for _ in range(max_new_tokens):
        curr_len = len(generated)
        input_ids = torch.tensor([generated], dtype=torch.long, device=device)

        with torch.no_grad():
            f_cis = freqs_cis[:curr_len]
            c_cache = cos_cache[:curr_len]
            s_cache = sin_cache[:curr_len]
            
            logits_list = model(
                input_ids,
                freqs_cis=f_cis,
                cos_cache=c_cache,
                sin_cache=s_cache,
                use_mtp=False,
            )

        # Autoregressive next token is predicted by the main head (index 0)
        next_token_logits = logits_list[0][0, -1, :]
        next_token = sample(next_token_logits, temperature, top_k, top_p)
        generated.append(next_token)
        print(tokenizer.decode([next_token]), end="", flush=True)

    elapsed = time.time() - start_time
    print("\n" + "=" * 50)
    print(f"Standard autoregressive decoding: {max_new_tokens} tokens in {elapsed:.2f}s ({max_new_tokens / elapsed:.2f} tokens/sec)")
    print("=" * 50)
    return generated


def main():
    args = parse_args()
    set_seed(args.seed)

    print(f"Loading config from {args.config}...")
    config = load_config(args.config)
    print(f"Model dimensions: Hidden: {config.hidden_size}, Layers: {config.num_hidden_layers}, Attention Heads: {config.num_attention_heads}")
    print(f"MTP: Depth = {config.mtp_depth}")
    print(f"Output Logit Scaling / Temp Scale: {config.output_logit_scale} (Trainable: {config.output_logit_scale_trainable})")

    # Load tokenizer
    tokenizer_path = "models/tokenizer"
    if not os.path.exists(tokenizer_path):
        print(f"Error: Tokenizer directory '{tokenizer_path}' does not exist.")
        sys.exit(1)
    
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

    # Initialize model
    print(f"Initializing SLMModel on {args.device}...")
    model = SLMModel(config).to(args.device)

    # Checkpoint loading
    if args.checkpoint:
        if os.path.isdir(args.checkpoint):
            pt_path = os.path.join(args.checkpoint, "model.pt")
            if os.path.exists(pt_path):
                bin_path = pt_path
            else:
                bin_path = os.path.join(args.checkpoint, "pytorch_model.bin")
        else:
            bin_path = args.checkpoint

        if os.path.exists(bin_path):
            print(f"Loading checkpoint weights from {bin_path}...")
            state_dict = torch.load(bin_path, map_location=args.device)
            # Remove keys prefix if saved via Accelerator
            fixed_state = {}
            for k, v in state_dict.items():
                if k.startswith("_orig_mod."):
                    fixed_state[k[10:]] = v
                else:
                    fixed_state[k] = v
            model.load_state_dict(fixed_state, strict=True)
            print("✓ Checkpoint loaded successfully.")
        else:
            print(f"Warning: Checkpoint '{bin_path}' not found. Generating with random initializations.")
    else:
        print("Note: No checkpoint provided. Generating with randomly initialized weights.")

    prompt = args.prompt
    prompt_tokens = tokenizer.encode(prompt)
    print(f"Encoded prompt: {prompt_tokens} (Length: {len(prompt_tokens)})")

    print("Starting standard autoregressive decoding...")
    generate_autoregressive(
        model=model,
        prompt_tokens=prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        config=config,
        tokenizer=tokenizer,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )


if __name__ == "__main__":
    main()
