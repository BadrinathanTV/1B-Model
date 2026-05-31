import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Resolve imports for training
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../training")))

from layers.rope import apply_rotary_emb, precompute_freqs_cis, precompute_cos_sin
import layers.rope as rope_module
from config import SLMConfig

try:
    from optimizers.kernels.rms_norm import TritonRMSNorm
    TRITON_RMSNORM_AVAILABLE = True
except ImportError:
    TRITON_RMSNORM_AVAILABLE = False

try:
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction
    LIGER_SWIGLU_AVAILABLE = True
except ImportError:
    LIGER_SWIGLU_AVAILABLE = False

try:
    from liger_kernel.ops.rope import LigerRopeFunction
    LIGER_ROPE_AVAILABLE = True
except ImportError:
    LIGER_ROPE_AVAILABLE = False

try:
    from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
    LIGER_CE_AVAILABLE = True
except ImportError:
    LIGER_CE_AVAILABLE = False

def time_op(op, name, warmup=20, steps=100):
    torch.cuda.synchronize()
    # Warmup
    for _ in range(warmup):
        out = op()
        if isinstance(out, tuple):
            loss = sum(t.float().sum() for t in out if t is not None and t.requires_grad)
        else:
            loss = out.float().sum()
        if loss.requires_grad:
            loss.backward()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    t0 = time.time()
    for _ in range(steps):
        out = op()
        if isinstance(out, tuple):
            loss = sum(t.float().sum() for t in out if t is not None and t.requires_grad)
        else:
            loss = out.float().sum()
        if loss.requires_grad:
            loss.backward()
    torch.cuda.synchronize()
    t1 = time.time()
    
    ms = (t1 - t0) / steps * 1000
    vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return ms, vram

def benchmark_rmsnorm():
    print("\n--- Benchmarking RMSNorm ---")
    dim = 2048
    batch_size = 4
    seq_len = 2048
    
    x = torch.randn(batch_size, seq_len, dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    
    # Fallback
    class EagerRMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim, device='cuda', dtype=torch.bfloat16))
        def forward(self, x):
            variance = x.float().pow(2).mean(-1, keepdim=True)
            return self.weight * (x * torch.rsqrt(variance + self.eps)).to(x.dtype)
            
    eager_norm = EagerRMSNorm(dim).cuda().to(dtype=torch.bfloat16)
    
    def run_eager():
        if x.grad is not None: x.grad.zero_()
        eager_norm.weight.grad = None
        return eager_norm(x)
        
    ms_eager, vram_eager = time_op(run_eager, "RMSNorm Eager")
    print(f"Eager PyTorch RMSNorm: {ms_eager:.3f} ms | Peak VRAM: {vram_eager:.2f} MB")
    
    if TRITON_RMSNORM_AVAILABLE:
        triton_norm = TritonRMSNorm(dim).cuda().to(dtype=torch.bfloat16)
        def run_triton():
            if x.grad is not None: x.grad.zero_()
            triton_norm.weight.grad = None
            return triton_norm(x)
        ms_triton, vram_triton = time_op(run_triton, "RMSNorm Triton")
        speedup = ms_eager / ms_triton
        print(f"Triton RMSNorm      : {ms_triton:.3f} ms | Peak VRAM: {vram_triton:.2f} MB ({speedup:.2f}x speedup)")
    else:
        print("Triton RMSNorm not available.")

def benchmark_rope():
    print("\n--- Benchmarking RoPE ---")
    batch_size = 4
    seq_len = 2048
    num_heads = 32
    head_dim = 128
    
    xq = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    xk = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    
    freqs_cis = precompute_freqs_cis(head_dim, seq_len, 10000.0, device='cuda')
    cos_cache, sin_cache = precompute_cos_sin(head_dim, seq_len, 10000.0, device='cuda', dtype=torch.bfloat16)
    
    def run_eager():
        if xq.grad is not None: xq.grad.zero_()
        if xk.grad is not None: xk.grad.zero_()
        rope_module.LIGER_ROPE = False
        return apply_rotary_emb(xq, xk, freqs_cis, cos_cache, sin_cache)
        
    ms_eager, vram_eager = time_op(run_eager, "RoPE Fallback")
    print(f"Fallback RoPE  : {ms_eager:.3f} ms | Peak VRAM: {vram_eager:.2f} MB")
    
    if LIGER_ROPE_AVAILABLE:
        def run_liger():
            if xq.grad is not None: xq.grad.zero_()
            if xk.grad is not None: xk.grad.zero_()
            rope_module.LIGER_ROPE = True
            return apply_rotary_emb(xq, xk, freqs_cis, cos_cache, sin_cache)
        ms_liger, vram_liger = time_op(run_liger, "RoPE Liger")
        speedup = ms_eager / ms_liger
        print(f"Liger RoPE     : {ms_liger:.3f} ms | Peak VRAM: {vram_liger:.2f} MB ({speedup:.2f}x speedup)")
    else:
        print("Liger RoPE not available.")

def benchmark_swiglu():
    print("\n--- Benchmarking SwiGLU (FFN) ---")
    batch_size = 4
    seq_len = 2048
    intermediate_size = 5632
    
    gate = torch.randn(batch_size, seq_len, intermediate_size, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    up = torch.randn(batch_size, seq_len, intermediate_size, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    
    def run_eager():
        if gate.grad is not None: gate.grad.zero_()
        if up.grad is not None: up.grad.zero_()
        return F.silu(gate) * up
        
    ms_eager, vram_eager = time_op(run_eager, "SwiGLU Eager")
    print(f"Eager SwiGLU  : {ms_eager:.3f} ms | Peak VRAM: {vram_eager:.2f} MB")
    
    if LIGER_SWIGLU_AVAILABLE:
        def run_liger():
            if gate.grad is not None: gate.grad.zero_()
            if up.grad is not None: up.grad.zero_()
            return LigerSiLUMulFunction.apply(gate, up)
        ms_liger, vram_liger = time_op(run_liger, "SwiGLU Liger")
        speedup = ms_eager / ms_liger
        print(f"Liger SwiGLU  : {ms_liger:.3f} ms | Peak VRAM: {vram_liger:.2f} MB ({speedup:.2f}x speedup)")
    else:
        print("Liger SwiGLU not available.")

def benchmark_cross_entropy():
    print("\n--- Benchmarking CrossEntropy Loss ---")
    batch_size = 4
    seq_len = 2048
    vocab_size = 32000
    
    logits = torch.randn(batch_size * seq_len, vocab_size, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    targets = torch.randint(0, vocab_size, (batch_size * seq_len,), device='cuda')
    
    def run_eager():
        if logits.grad is not None: logits.grad.zero_()
        return F.cross_entropy(logits, targets)
        
    ms_eager, vram_eager = time_op(run_eager, "CrossEntropy Eager")
    print(f"Eager CE   : {ms_eager:.3f} ms | Peak VRAM: {vram_eager:.2f} MB")
    
    if LIGER_CE_AVAILABLE:
        liger_loss = LigerCrossEntropyLoss()
        def run_liger():
            if logits.grad is not None: logits.grad.zero_()
            return liger_loss(logits, targets)
        ms_liger, vram_liger = time_op(run_liger, "CrossEntropy Liger")
        speedup = ms_eager / ms_liger
        print(f"Liger CE   : {ms_liger:.3f} ms | Peak VRAM: {vram_liger:.2f} MB ({speedup:.2f}x speedup)")
    else:
        print("Liger CE not available.")

if __name__ == "__main__":
    benchmark_rmsnorm()
    benchmark_rope()
    benchmark_swiglu()
    benchmark_cross_entropy()
