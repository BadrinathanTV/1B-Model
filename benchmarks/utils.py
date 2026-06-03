import torch
import time

def benchmark(fn, *args, num_warmup=10, num_iters=100, **kwargs):
    # Ensure gradients are cleared
    for a in args:
        if isinstance(a, torch.Tensor) and a.grad is not None:
            a.grad = None
            
    # Warmup
    for _ in range(num_warmup):
        fn(*args, **kwargs)
        for a in args:
            if isinstance(a, torch.Tensor) and a.grad is not None:
                a.grad = None
    
    torch.cuda.synchronize()
    
    # Timing
    torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()
    
    for _ in range(num_iters):
        fn(*args, **kwargs)
        for a in args:
            if isinstance(a, torch.Tensor) and a.grad is not None:
                a.grad = None
        
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    max_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
    avg_time = (end_time - start_time) / num_iters * 1000 # ms
    
    return avg_time, max_memory
