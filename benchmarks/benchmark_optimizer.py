import os
import sys
import time
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../training")))
from optimizers.hybrid import HybridSLMOptimizer

def benchmark_optimizer(n_params=200_000_000, steps=20):
    print(f"\n--- Benchmarking HybridSLMOptimizer ---")
    
    # 200M params to simulate a large 1D tensor
    p = torch.randn(n_params, device="cuda", dtype=torch.bfloat16)
    p.requires_grad_(True)
    p.grad = torch.randn_like(p) * 0.01
    
    param_groups = [{"params": [p], "lr": 1e-3, "use_aurora": False, "weight_decay": 0.0, "eps": 1e-8, "betas": (0.9, 0.95)}]
    opt = HybridSLMOptimizer(param_groups, warmup_steps=0)
    
    # Warmup
    for _ in range(3):
        opt.step()
        
    torch.cuda.synchronize()
    
    start_vram = torch.cuda.memory_allocated() / 1024**2
    torch.cuda.reset_peak_memory_stats()
    
    t0 = time.time()
    for _ in range(steps):
        opt.step()
    torch.cuda.synchronize()
    t1 = time.time()
    
    peak_vram = torch.cuda.max_memory_allocated() / 1024**2
    
    time_per_step = (t1 - t0) / steps * 1000
    vram_spike = peak_vram - start_vram
    
    print(f"Time per step: {time_per_step:.2f} ms")
    print(f"VRAM Spike:    {vram_spike:.2f} MB")
    
    return time_per_step

if __name__ == "__main__":
    print("Initializing benchmark (200 Million params)...")
    try:
        py_time = benchmark_optimizer()
        print(f"\nResult: {py_time:.2f} ms/step")
    except Exception as e:
        print(f"Error during benchmark: {e}")
