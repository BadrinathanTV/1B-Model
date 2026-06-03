import sys
import os
import torch
import torch.nn as nn
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from benchmarks.utils import benchmark

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'training')))
from optimizers import build_optimizer
from config import SLMConfig
from models.transformer import SLMModel

def create_model_and_optimizer(opt_type, device, dtype):
    config = SLMConfig()
    # scale down just a little bit for faster setup if needed, but keeping it 1B is best for realistic timing
    config.hidden_size = 1280
    config.num_hidden_layers = 12 # half depth for benchmarking just the step time
    
    config.optimizer.type = opt_type
    
    model = SLMModel(config).to(device=device, dtype=dtype)
    optimizer = build_optimizer(model, config)
    
    return model, optimizer

def main():
    print("=== Optimizer Step Benchmark ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    
    opts_to_test = ["sf_normuon", "adamw"]
    
    results = {}
    
    for opt_type in opts_to_test:
        print(f"\nSetting up {opt_type}...")
        model, optimizer = create_model_and_optimizer(opt_type, device, dtype)
        
        # Populate dummy gradients
        for p in model.parameters():
            if p.requires_grad:
                p.grad = torch.randn_like(p)
                
        def step_fn():
            if opt_type == "sf_normuon":
                def closure(): return 1.0 # dummy loss
                optimizer.step(closure=closure)
            else:
                optimizer.step()
                
        print(f"Benchmarking {opt_type}...")
        
        t, m = benchmark(step_fn, num_warmup=2, num_iters=20)
        results[opt_type] = (t, m)
        
        # Clean up to free VRAM
        del model
        del optimizer
        torch.cuda.empty_cache()
        
    print("\nResults:")
    print(f"{'Optimizer':<20} | {'Time (ms/step)':<15} | {'Peak VRAM (MB)':<15}")
    print("-" * 55)
    for opt_type, (t, m) in results.items():
        print(f"{opt_type:<20} | {t:<15.2f} | {m:<15.2f}")

if __name__ == '__main__':
    main()
