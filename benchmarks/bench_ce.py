import sys
import os
import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from benchmarks.utils import benchmark

try:
    from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction
    LIGER_AVAILABLE = True
except ImportError:
    LIGER_AVAILABLE = False

BENCH_CONFIG = {
    "batch_size": 4,
    "seq_len": 2048,
    "hidden_size": 1280,
    "vocab_size": 64000,
    "dtype": torch.bfloat16,
}

def check_accuracy(h, w, t):
    # Reference in fp32
    h_f32 = h.float().detach().clone().requires_grad_(True)
    w_f32 = w.float().detach().clone().requires_grad_(True)
    
    logits_f32 = F.linear(h_f32, w_f32)
    loss_f32 = F.cross_entropy(logits_f32, t)
    loss_f32.backward()
    
    # Standard CE bf16
    h_bf16 = h.detach().clone().requires_grad_(True)
    w_bf16 = w.detach().clone().requires_grad_(True)
    logits_bf16 = F.linear(h_bf16, w_bf16)
    loss_bf16 = F.cross_entropy(logits_bf16, t)
    loss_bf16.backward()
    
    std_loss_err = (loss_f32.item() - loss_bf16.item())
    
    liger_loss_err = 0.0
    if LIGER_AVAILABLE:
        h_lig = h.detach().clone().requires_grad_(True)
        w_lig = w.detach().clone().requires_grad_(True)
        loss_lig = LigerFusedLinearCrossEntropyFunction.apply(h_lig, w_lig, t)[0]
        loss_lig.backward()
        liger_loss_err = (loss_f32.item() - loss_lig.item())
        
    print(f"Accuracy vs FP32 Reference:")
    print(f"  Standard BF16 loss error: {abs(std_loss_err):.6f}")
    if LIGER_AVAILABLE:
        print(f"  Liger BF16 loss error:    {abs(liger_loss_err):.6f}")
    print()

def main():
    print("=== Cross Entropy Benchmark ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    B = BENCH_CONFIG["batch_size"]
    S = BENCH_CONFIG["seq_len"]
    H = BENCH_CONFIG["hidden_size"]
    V = BENCH_CONFIG["vocab_size"]
    dtype = BENCH_CONFIG["dtype"]
    
    print(f"Config: B={B}, S={S}, H={H}, V={V}")
    
    hidden_states = torch.randn(B * S, H, device=device, dtype=dtype, requires_grad=True)
    targets = torch.randint(0, V, (B * S,), device=device)
    lm_head_weight = torch.randn(V, H, device=device, dtype=dtype, requires_grad=True)
    
    check_accuracy(hidden_states, lm_head_weight, targets)
    
    def standard_ce(h, w, t):
        logits = F.linear(h, w)
        loss = F.cross_entropy(logits, t)
        loss.backward()
        return loss

    compiled_ce_forward = torch.compile(F.cross_entropy)
    def compiled_ce(h, w, t):
        logits = F.linear(h, w)
        loss = compiled_ce_forward(logits, t)
        loss.backward()
        return loss

    def liger_ce(h, w, t):
        loss = LigerFusedLinearCrossEntropyFunction.apply(h, w, t)[0]
        loss.backward()
        return loss
        
    # Baseline
    print("Benchmarking Standard CE...")
    t_std, m_std = benchmark(standard_ce, hidden_states, lm_head_weight, targets)
    
    # Compiled
    print("Benchmarking Compiled CE...")
    t_comp, m_comp = benchmark(compiled_ce, hidden_states, lm_head_weight, targets)
    
    # Liger
    if LIGER_AVAILABLE:
        print("Benchmarking Liger CE...")
        t_lig, m_lig = benchmark(liger_ce, hidden_states, lm_head_weight, targets)
    else:
        t_lig, m_lig = 0.0, 0.0
        
    print("\nResults:")
    print(f"{'Kernel':<25} | {'Time (ms/step)':<15} | {'Peak VRAM (MB)':<15}")
    print("-" * 60)
    print(f"{'Standard CE (BF16)':<25} | {t_std:<15.2f} | {m_std:<15.2f}")
    print(f"{'Compiled CE (BF16)':<25} | {t_comp:<15.2f} | {m_comp:<15.2f}")
    if LIGER_AVAILABLE:
        print(f"{'Liger Fused CE (BF16)':<25} | {t_lig:<15.2f} | {m_lig:<15.2f}")
        
if __name__ == '__main__':
    main()
