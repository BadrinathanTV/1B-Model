import sys
import os
import torch
import torch.nn as nn
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from benchmarks.utils import benchmark

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'training')))
from layers.norm import RMSNorm as TritonRMSNorm
from benchmarks.bench_ce import BENCH_CONFIG

class PyTorchRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps).to(x.dtype)
        return self.weight * x_normed

def main():
    print("=== RMSNorm Benchmark ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    B = BENCH_CONFIG["batch_size"]
    S = BENCH_CONFIG["seq_len"]
    H = BENCH_CONFIG["hidden_size"]
    dtype = BENCH_CONFIG["dtype"]
    
    x = torch.randn(B, S, H, device=device, dtype=dtype, requires_grad=True)
    grad_output = torch.randn_like(x)
    
    def run_norm(norm_layer):
        y = norm_layer(x)
        y.backward(grad_output)
        return y
        
    print("Benchmarking TritonRMSNorm...")
    triton_norm = TritonRMSNorm(H).to(device=device, dtype=dtype)
    t_triton, m_triton = benchmark(run_norm, triton_norm)
    
    print("Benchmarking PyTorchRMSNorm...")
    pt_norm = PyTorchRMSNorm(H).to(device=device, dtype=dtype)
    t_pt, m_pt = benchmark(run_norm, pt_norm)
    
    print("Benchmarking Compiled PyTorchRMSNorm...")
    compiled_pt_norm = torch.compile(pt_norm)
    t_comp, m_comp = benchmark(run_norm, compiled_pt_norm)
    
    print("\nResults:")
    print(f"{'Kernel':<30} | {'Time (ms/step)':<15} | {'Peak VRAM (MB)':<15}")
    print("-" * 65)
    print(f"{'TritonRMSNorm':<30} | {t_triton:<15.2f} | {m_triton:<15.2f}")
    print(f"{'PyTorchRMSNorm':<30} | {t_pt:<15.2f} | {m_pt:<15.2f}")
    print(f"{'Compiled PyTorchRMSNorm':<30} | {t_comp:<15.2f} | {m_comp:<15.2f}")

if __name__ == '__main__':
    main()
