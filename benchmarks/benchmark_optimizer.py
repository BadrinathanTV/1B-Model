import os
import sys
import time
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../training")))
from optimizers.nf_aurora import NFAurora

# Force Triton off for Python evaluation
import optimizers.kernels.sf_adamw_4bit
original_triton = optimizers.kernels.sf_adamw_4bit.TRITON_AVAILABLE

def benchmark_optimizer(use_triton, n_params=200_000_000, steps=20):
    optimizers.kernels.sf_adamw_4bit.TRITON_AVAILABLE = use_triton
    
    print(f"\\n--- Benchmarking {'Triton Kernel' if use_triton else 'Python Chunked'} ---")
    
    # 200M params to simulate a massive DeepSpeed 1D tensor
    p = torch.randn(n_params, device="cuda", dtype=torch.bfloat16)
    p.requires_grad_(True)
    p.grad = torch.randn_like(p) * 0.01
    
    # Only 1D params to isolate the 4-bit AdamW performance
    opt = NFAurora([p], lr=1e-3)
    
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
        py_time = benchmark_optimizer(use_triton=False)
        triton_time = benchmark_optimizer(use_triton=True)
        
        print("\\n=== Results ===")
        print(f"Python: {py_time:.2f} ms/step")
        print(f"Triton: {triton_time:.2f} ms/step")
        print(f"Speedup: {py_time / triton_time:.2f}x faster")
    except Exception as e:
        print(f"Error during benchmark: {e}")
