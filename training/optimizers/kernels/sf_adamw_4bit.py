"""
Triton Kernel for 4-bit Schedule-Free AdamW
===========================================

Fuses dequantization, EMA updates, block requantization, and SF-AdamW math
into a single GPU kernel for maximum throughput.
"""

import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


@triton.jit
def _sf_adamw_4bit_kernel(
    p_ptr, grad_ptr, z_ptr, q_v_ptr, scale_v_ptr,
    beta, beta2, decay, eta_t, ckp1,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused Triton kernel for the 4-bit Schedule-Free AdamW fallback.
    Each block processes exactly BLOCK_SIZE elements and computes a single scale.
    """
    # Block index
    pid = tl.program_id(axis=0)
    
    # Offsets for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for bounds checking
    mask = offsets < n_elements
    
    # Load pointers
    # Note: scale_v_ptr has 1 element per BLOCK_SIZE
    scale_ptr = scale_v_ptr + pid
    
    p = tl.load(p_ptr + offsets, mask=mask)
    grad = tl.load(grad_ptr + offsets, mask=mask)
    z = tl.load(z_ptr + offsets, mask=mask)
    
    # Load quantized states
    q_v = tl.load(q_v_ptr + offsets, mask=mask)
    scale_v = tl.load(scale_ptr)  # scalar load
    
    # 1. Dequantize second moment
    v = q_v * scale_v
    
    # 2. Update EMA (SF-AdamW uses raw gradient)
    g2 = grad * grad
    v_new = beta2 * v + (1.0 - beta2) * g2
    
    # 3. Block Quantization
    # Compute max over the block for the new scale
    max_v = tl.max(v_new, axis=0)
    scale_new = tl.maximum(max_v / 15.0, 1e-12)
    
    # Quantize and clamp to [0, 15]
    # Manual round to nearest integer: int(x + 0.5)
    q_new_f = v_new / scale_new
    q_new_i = (q_new_f + 0.5).to(tl.int32)
    q_new_i = tl.minimum(tl.maximum(q_new_i, 0), 15)
    q_new_u8 = q_new_i.to(tl.uint8)
    
    # 4. Dequantize again for the step update
    v_step = q_new_i.to(tl.float32) * scale_new
    
    # Safety bound against 4-bit division-by-zero
    min_v = (1.0 - beta2) * g2
    v_step = tl.maximum(v_step, min_v)
    
    # 5. Compute the 1D update
    denom = tl.math.sqrt(v_step) + 1e-8
    update_1d = grad / denom
    
    # 6. Schedule-Free Dynamics
    x_t = (p - (1.0 - beta) * z) / beta
    z_new = z - (eta_t * decay) * z - (eta_t * update_1d)
    x_tp1 = (1.0 - ckp1) * x_t + ckp1 * z_new
    p_new = (1.0 - beta) * z_new + beta * x_tp1
    
    # 7. Write back
    tl.store(q_v_ptr + offsets, q_new_u8, mask=mask)
    tl.store(scale_ptr, scale_new)  # scalar write
    tl.store(z_ptr + offsets, z_new, mask=mask)
    tl.store(p_ptr + offsets, p_new, mask=mask)


def step_sf_adamw_4bit(p, grad, z, state, lr, beta, decay, ckp1):
    """
    Launch the Triton kernel on a 1D parameter tensor.
    Handles dynamic block sizes matching the chunk size (default: 10M).
    But for optimal Triton performance, we use a BLOCK_SIZE of 4096 or 8192
    to maximize SM occupancy and achieve fine-grained 4-bit scaling.
    """
    if not TRITON_AVAILABLE:
        raise ImportError("Triton is not available.")
        
    n_elements = p.numel()
    
    # Triton prefers power-of-2 block sizes. 
    # Smaller block sizes mean more fine-grained 4-bit scales (higher precision).
    # 4096 is a good balance between precision and kernel launch overhead.
    BLOCK_SIZE = 4096
    
    # Flatten everything
    p_flat = p.view(-1)
    g_flat = grad.view(-1)
    z_flat = z.view(-1)
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    if "q_exp_avg_sq" not in state:
        state["q_exp_avg_sq"] = torch.zeros_like(p_flat, dtype=torch.uint8)
        state["scale_exp_avg_sq"] = torch.zeros((num_blocks,), device=p.device, dtype=torch.bfloat16)
        
    q_flat = state["q_exp_avg_sq"].view(-1)
    scale_flat = state["scale_exp_avg_sq"]
    
    # Ensure scales match block count (e.g. if loaded from checkpoint with different BLOCK_SIZE)
    if scale_flat.numel() != num_blocks:
        # Resize scales by replicating or slicing if block size changed
        # This is unlikely during uninterrupted pretraining, but handled gracefully.
        new_scale = torch.zeros((num_blocks,), device=p.device, dtype=scale_flat.dtype)
        min_len = min(scale_flat.numel(), num_blocks)
        new_scale[:min_len] = scale_flat[:min_len]
        state["scale_exp_avg_sq"] = new_scale
        scale_flat = new_scale

    beta2 = 0.99
    eta_t = lr * math.sqrt(1.0 - beta2)
    
    _sf_adamw_4bit_kernel[grid](
        p_flat, g_flat, z_flat, q_flat, scale_flat,
        float(beta), float(beta2), float(decay), float(eta_t), float(ckp1),
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
