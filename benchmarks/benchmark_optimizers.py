import argparse
import time
import os
import sys
import torch
import torch.nn.functional as F

# Resolve imports for training
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../training")))

from config import SLMConfig
from model import SLMModel
from optimizers import build_optimizer
from layers.rope import precompute_freqs_cis, precompute_cos_sin

# Wrap loss in compile just like train.py
_compiled_ce = torch.compile(F.cross_entropy)

def compute_loss(logits_list, targets, config: SLMConfig) -> torch.Tensor:
    main_logits = logits_list[0]
    loss = _compiled_ce(main_logits.view(-1, config.vocab_size), targets.view(-1))
    if config.mtp_depth > 1:
        mtp_weight = config.training.mtp_loss_weight
        for i in range(1, config.mtp_depth):
            mtp_logits = logits_list[i][:, :-i, :].contiguous()
            mtp_targets = targets[:, i:].contiguous()
            if mtp_logits.numel() == 0 or mtp_targets.numel() == 0: continue
            loss += mtp_weight * _compiled_ce(mtp_logits.view(-1, config.vocab_size), mtp_targets.view(-1))
    return loss

def run_benchmark(opt_type, config, device, steps=50, warmup=10):
    print(f"\n{'='*50}\nBenchmarking: {opt_type.upper()}\n{'='*50}")
    
    # 1. Build Model
    torch.manual_seed(42)
    model = SLMModel(config).to(device=device, dtype=torch.bfloat16)
    
    # 2. Build Optimizer
    config.optimizer.type = opt_type
    optimizer = build_optimizer(model, config)
    if hasattr(optimizer, 'train'):
        optimizer.train()
    model.train()
    
    # 3. Dummy Data
    bs = config.training.batch_size
    seq = config.training.seq_len
    # Need inputs and targets
    inputs = torch.randint(0, config.vocab_size, (bs, seq * config.tst_group_size), device=device)
    targets = torch.randint(0, config.vocab_size, (bs, seq), device=device)
    
    # Precompute RoPE
    freqs_cis = precompute_freqs_cis(config.qk_rope_head_dim, seq, config.rope_theta, device)
    cos_cache, sin_cache = precompute_cos_sin(config.qk_rope_head_dim, seq, config.rope_theta, device)
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    # Warmup
    print(f"Running {warmup} warmup steps...")
    for _ in range(warmup):
        logits_list = model(inputs, freqs_cis=freqs_cis, cos_cache=cos_cache, sin_cache=sin_cache)
        loss = compute_loss(logits_list, targets, config)
        loss.backward()
        def closure(): return loss.item()
        optimizer.step(closure=closure) if hasattr(optimizer, 'step') else optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    
    torch.cuda.synchronize()
    start_vram = torch.cuda.memory_allocated() / 1024**2
    
    # Benchmark
    print(f"Running {steps} benchmark steps...")
    t0 = time.time()
    for _ in range(steps):
        logits_list = model(inputs, freqs_cis=freqs_cis, cos_cache=cos_cache, sin_cache=sin_cache)
        loss = compute_loss(logits_list, targets, config)
        loss.backward()
        def closure(): return loss.item()
        optimizer.step(closure=closure) if hasattr(optimizer, 'step') else optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
    torch.cuda.synchronize()
    t1 = time.time()
    
    peak_vram = torch.cuda.max_memory_allocated() / 1024**2
    time_per_step = (t1 - t0) / steps * 1000
    
    print(f"Time per step : {time_per_step:.1f} ms")
    print(f"Peak VRAM     : {peak_vram:.0f} MB")
    
    # Free memory
    del model, optimizer, inputs, targets, loss, logits_list
    torch.cuda.empty_cache()
    
    return time_per_step, peak_vram

def main():
    device = "cuda"
    
    # Scale down slightly to fit in 16GB VRAM without DeepSpeed/Gradient Checkpointing
    config = SLMConfig()
    config.num_hidden_layers = 12  # ~500M params for testing
    config.training.batch_size = 2
    config.training.seq_len = 512
    
    print(f"Benchmarking on {device}")
    
    optimizers_to_test = ["nf_aurora_hybrid", "nf_normuon_hybrid"]
    results = {}
    
    for opt in optimizers_to_test:
        try:
            ms, vram = run_benchmark(opt, config, device)
            results[opt] = {"ms": ms, "vram": vram}
        except Exception as e:
            print(f"Failed to benchmark {opt}: {e}")
            results[opt] = {"ms": float('inf'), "vram": float('inf')}
            
    print("\n\n" + "="*50)
    print("FINAL BENCHMARK RESULTS (Custom Triton/Liger Stack)")
    print("="*50)
    print(f"{'Optimizer':<15} | {'ms / step':<10} | {'Peak VRAM':<10} | {'Speedup':<10}")
    print("-"*50)
    
    best_ms = min(r["ms"] for r in results.values())
    for opt, res in results.items():
        if res["ms"] == float('inf'):
            print(f"{opt:<15} | {'FAILED':<10} | {'FAILED':<10} | {'N/A':<10}")
        else:
            speedup = best_ms / res["ms"]
            print(f"{opt:<15} | {res['ms']:<10.1f} | {res['vram']:<10.0f} | {speedup:<9.2f}x")

if __name__ == "__main__":
    main()
