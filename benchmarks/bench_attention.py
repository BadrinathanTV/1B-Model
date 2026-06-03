import sys
import os
import torch
import torch.nn.functional as F
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from benchmarks.utils import benchmark

try:
    from torch.nn.attention.flex_attention import flex_attention
    FLEX_AVAILABLE = True
except ImportError:
    FLEX_AVAILABLE = False

from benchmarks.bench_ce import BENCH_CONFIG

def main():
    print("=== Attention (SDPA vs FlexAttention) Benchmark ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    B = BENCH_CONFIG["batch_size"]
    S = BENCH_CONFIG["seq_len"]
    H_H = 10 # num_heads
    H_D = 192 # v_head_dim + qk_rope_head_dim
    dtype = BENCH_CONFIG["dtype"]
    
    # [batch, heads, seq, head_dim]
    q = torch.randn(B, H_H, S, H_D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H_H, S, H_D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H_H, S, H_D, device=device, dtype=dtype, requires_grad=True)
    
    grad_output = torch.randn_like(q)
    
    def run_sdpa():
        # use is_causal=True
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out.backward(grad_output)
        return out
        
    def run_sdpa_compiled():
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out.backward(grad_output)
        return out
        
    compiled_sdpa = torch.compile(run_sdpa_compiled)
    
    def causal_mod(score, b, h, q_idx, kv_idx):
        return torch.where(q_idx >= kv_idx, score, float("-inf"))
        
    def run_flex():
        out = flex_attention(q, k, v, score_mod=causal_mod)
        out.backward(grad_output)
        return out
        
    print("Benchmarking SDPA (Native)...")
    t_sdpa, m_sdpa = benchmark(run_sdpa)
    
    print("Benchmarking SDPA (Compiled)...")
    t_comp, m_comp = benchmark(compiled_sdpa)
    
    if FLEX_AVAILABLE:
        print("Benchmarking FlexAttention...")
        t_flex, m_flex = benchmark(run_flex)
    else:
        t_flex, m_flex = 0.0, 0.0
        
    print("\nResults:")
    print(f"{'Kernel':<30} | {'Time (ms/step)':<15} | {'Peak VRAM (MB)':<15}")
    print("-" * 65)
    print(f"{'Native SDPA':<30} | {t_sdpa:<15.2f} | {m_sdpa:<15.2f}")
    print(f"{'Compiled SDPA':<30} | {t_comp:<15.2f} | {m_comp:<15.2f}")
    if FLEX_AVAILABLE:
        print(f"{'FlexAttention':<30} | {t_flex:<15.2f} | {m_flex:<15.2f}")

if __name__ == '__main__':
    main()
