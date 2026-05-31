import time
import torch
import torch.nn.functional as F

try:
    from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
    LIGER_AVAILABLE = True
except ImportError:
    LIGER_AVAILABLE = False
    print("Liger Kernel not found. Please install liger-kernel.")

def run_benchmark(name, loss_fn, logits, targets, steps=50, warmup=10):
    print(f"\n{'='*50}\nBenchmarking: {name}\n{'='*50}")
    
    # Optimizer for dummy parameter to force backward pass
    dummy_param = torch.nn.Parameter(logits.clone())
    optimizer = torch.optim.AdamW([dummy_param], lr=0.01)
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    print(f"Running {warmup} warmup steps...")
    for _ in range(warmup):
        loss = loss_fn(dummy_param.view(-1, dummy_param.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    
    torch.cuda.synchronize()
    
    print(f"Running {steps} benchmark steps...")
    t0 = time.time()
    for _ in range(steps):
        loss = loss_fn(dummy_param.view(-1, dummy_param.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
    torch.cuda.synchronize()
    t1 = time.time()
    
    peak_vram = torch.cuda.max_memory_allocated() / 1024**2
    time_per_step = (t1 - t0) / steps * 1000
    
    print(f"Time per step : {time_per_step:.2f} ms")
    print(f"Peak VRAM     : {peak_vram:.0f} MB")
    
    del dummy_param, optimizer, loss
    torch.cuda.empty_cache()
    
    return time_per_step, peak_vram

def main():
    device = "cuda"
    
    # Massive scale test (simulating a large batch or long context)
    batch_size = 4
    seq_len = 2048
    vocab_size = 32000
    
    print(f"Benchmarking CrossEntropy on {device}")
    print(f"Batch: {batch_size}, Seq: {seq_len}, Vocab: {vocab_size}")
    
    # Allocate dummy logits and targets
    torch.manual_seed(42)
    logits = torch.randn((batch_size, seq_len, vocab_size), device=device, dtype=torch.bfloat16)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    results = {}
    
    # 1. Eager PyTorch
    def eager_loss(pred, targ):
        return F.cross_entropy(pred, targ)
    
    ms, vram = run_benchmark("Eager PyTorch CE", eager_loss, logits, targets)
    results["Eager PyTorch"] = {"ms": ms, "vram": vram}
    
    # 2. torch.compile PyTorch
    compiled_loss = torch.compile(F.cross_entropy)
    ms, vram = run_benchmark("torch.compile CE", compiled_loss, logits, targets)
    results["torch.compile"] = {"ms": ms, "vram": vram}
    
    # 3. Liger Kernel
    if LIGER_AVAILABLE:
        liger_loss = LigerCrossEntropyLoss()
        ms, vram = run_benchmark("Liger Kernel CE", liger_loss, logits, targets)
        results["Liger Kernel"] = {"ms": ms, "vram": vram}
        
    print("\n\n" + "="*50)
    print("FINAL KERNEL BENCHMARK RESULTS")
    print("="*50)
    print(f"{'Kernel':<20} | {'ms / step':<10} | {'Peak VRAM':<10}")
    print("-" * 50)
    
    for name, res in results.items():
        print(f"{name:<20} | {res['ms']:<10.2f} | {res['vram']:<10.0f} MB")

if __name__ == "__main__":
    main()
