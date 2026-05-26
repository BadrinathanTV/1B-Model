# NVFP4 Pretraining & Inference Guide — 1B SLM on RTX 5080 (Blackwell)

This guide documents the finalized technical recipe for pretraining and serving a **1B parameter Small Language Model** using NVIDIA's native 4-bit floating point format (**NVFP4**) on the **RTX 5080 (SM120)**.

> [!IMPORTANT]
> **SM120 vs. SM100 Architectural Divergence**
> The RTX 5080 uses compute capability **SM120** (consumer Blackwell / GB203). It **lacks** Tensor Memory (TMEM) and the `tcgen05` instruction set present in datacenter Blackwell **SM100** (B100/B200/B300).
> * **SM100** utilizes autonomous, per-thread, TMEM-based Tensor Core instructions (`tcgen05.mma`).
> * **SM120** relies exclusively on warp-synchronous, register-to-register **`mma.sync`** instructions (`mma.sync.aligned.m16n8k64` or `m16n8k32` with `.e2m1` formats).
> Kernels compiled for SM100 **will trigger illegal instruction crashes** on SM120. All software layers must be configured for register-based execution.

---

## 1. Hardware: SM120 vs SM100 — Deep Spec Comparison

| Feature | RTX 5080 (SM120 / sm_120f) | B200 (SM100 / sm_100a) |
| :--- | :--- | :--- |
| **Tensor Memory (TMEM)** | ❌ Physically Absent | ✅ 256 KB per SM |
| **Tensor Core Pipeline** | `mma.sync` (register-to-register) | `tcgen05` (TMEM-to-TMEM) |
| **FP4 Tensor Cores** | ✅ Native (5th Gen, 2x dense math) | ✅ Native (5th Gen, 2x dense math) |
| **Memory Architecture** | 16 GB GDDR7 (960 GB/s) | 192 GB HBM3e (8.0 TB/s) |
| **TMA Support** | ✅ `cp.async.bulk` (Global-to-Smem) | ✅ Full TMA (TMEM/Smem routing) |
| **Key Implication** | No FA3/FA4/FlashMLA; use Triton/cuDNN | Native datacenter FlashAttention suite |

---

## 2. NVIDIA Transformer Engine (TE) SM120 Integration & Critical Workarounds

As of **v2.15** (May 2026), Transformer Engine officially supports Blackwell consumer architectures (`sm_120`), resolving previous compiler crashes. However, out-of-the-box training recipes designed for datacenter Blackwell will still crash on the RTX 5080 due to architectural differences.

### ⚠️ The RHT & Stochastic Rounding Crash
The default `NVFP4BlockScaling` recipe utilizes **Random Hadamard Transforms (RHT)** to smooth outliers and **Stochastic Rounding (SR)** to avoid gradient underflow. On SM120, these operations invoke specific TMEM/UMMA layouts that are physically unsupported, resulting in `CUDA error: invalid argument` or shared memory allocation failures at runtime.

### The Code-Level Fix
To execute NVFP4 pretraining on the RTX 5080, you must explicitly disable RHT and Stochastic Rounding when creating the scaling recipe:

```python
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import NVFP4BlockScaling

# Define the recipe with SM120 compatible workarounds
recipe = NVFP4BlockScaling(
    disable_rht=True,                  # Bypasses unsupported Hadamard kernels
    disable_stochastic_rounding=True   # Reverts to standard deterministic rounding
)

# Wrap your training step in the autocast context
with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
    outputs = model(inputs)
```

> [!WARNING]
> **Convergence Trade-off:** Disabling RHT and Stochastic Rounding exposes the model to raw 4-bit quantization noise and potential gradient bias. You must monitor loss curves closely and consider slightly reducing your learning rate or employing tighter scaling block bounds to prevent divergence.

### Source Compilation for SM120
To build TE v2.15+ from source for the RTX 5080, target the `120` or `120a` designation explicitly:

```bash
export NVTE_FRAMEWORK=pytorch
export NVTE_CUDA_ARCHS="120"  # Or "120a" if needed by specific toolkit variants
export MAX_JOBS=4             # Restrict parallel compilation jobs to avoid OOM
pip install --no-build-isolation .
```

---

## 3. Triton FP4 Custom Kernels (`tl.dot_scaled`)

OpenAI Triton (v3.3+) supports 4-bit block-scaled GEMMs natively on Blackwell SM120 through the `tl.dot_scaled` API.

### Triton Implementation Rules:
1. **Data Packing:** FP4 elements must be packed into standard `uint8` tensors (two `e2m1` 4-bit elements per byte) along the reduction dimension ($K$).
2. **Scale Factor Layouts:** Scaled factors are loaded in a contiguous, packed 5D layout `(M//128, K//VEC_SIZE//4, 32, 16)` to optimize memory coalescing and prevent bank conflicts, and then reshaped into the 2D layout required by `tl.dot_scaled`.
3. **`rhs_k_pack` Parameter:** When feeding FP4 data into `tl.dot_scaled`, you must configure the right-hand-side to pack along the K-dimension:

```python
# Triton kernel snippet
# a: FP4 input packed in uint8, scale_a: FP8 E4M3 scale
# b: FP4 weight packed in uint8, scale_b: FP8 E4M3 scale
accumulator = tl.dot_scaled(
    a, scale_a, "e2m1", 
    b.T, scale_b, "e2m1", 
    accumulator,
    rhs_k_pack=True  # Strictly required for FP4 inputs
)
```

Triton's compiler automatically targets the RTX 5080's registers, compiling `tl.dot_scaled` down to `mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X` instructions.

---

## 4. CUDA C++ Custom Kernels & CUTLASS 3.x

If writing custom CUDA C++ kernels to extend the `gau-nernst` codebase for your attention blocks:

* **Header Definitions:** The core definitions for register-based Blackwell Tensor Core execution are located in CUTLASS at `include/cute/arch/mma_sm120.hpp`.
* **Scale Formatting:** The custom CUDA mainloop expects scale factors in an interleaved layout (`SfKMajorAtom`).
* **Avoid Datacenter Examples:** Do not copy examples from `72b_blackwell_nvfp4_nvfp4_gemm.cu` directly, as they invoke TMEM-based `tcgen05.mma` instructions. Instead, adapt register-based `mma.sync` flows from CUTLASS 3.x that target `sm_120`.

---

## 5. Finalized Attention Stack for RTX 5080 (SM120)

| Kernel | Training (Fwd+Bwd) | Inference (Fwd) | SM120 Status & Implementation |
| :--- | :--- | :--- | :--- |
| **PyTorch SDPA (cuDNN)** | ✅ | ✅ | **Highly Recommended for Training.** Automatically selects fused, register-based attention paths tuned for Blackwell. |
| **torch.compile + Triton** | ✅ | ✅ | **Highly Recommended for Custom Blocks.** Captured via AOTAutograd to compile fused attention blocks with zero launch overhead. |
| **SageAttention 3** | ⚠️ (8-bit bwd only) | ✅ (4-bit fwd) | **Highly Recommended for Inference.** Achieves ~1038 TOPS on consumer Blackwell via register-level FP4 quantization. Training is limited to 8-bit fine-tuning. |
| **FlashInfer** | ❌ | ✅ | **For Serving.** Ensure CUDA 13.0+ and `compute_120f` are set to prevent TMA grouped GEMM initialization failures. |
| **FlashAttention-4 / 3** | ❌ | ❌ | **Incompatible.** Hardware requires TMEM/tcgen05 instructions which are physically absent on SM120. |

---

## 6. Software Environment Requirements (RTX 5080/SM120)

To avoid silent performance degradations or falling back to Ada Lovelace (`sm_89`) compilation profiles, align your environment to these minimum specs:

```bash
CUDA Toolkit:        >= 12.8 (13.0+ recommended for native SageAttention 3 and FlashInfer)
PyTorch:             >= 2.7.0 (2.11+ nightly recommended to avoid early Blackwell compiler JIT bugs)
cuDNN:               >= 9.13.1 (Contains optimized fused Blackwell SDPA paths)
Triton:              >= 3.3.0
NVIDIA Drivers:      >= R525+

# Verify your environment explicitly exposes native sm_120 support
python -c "import torch; print(torch.cuda.get_arch_list())"
# Ensure 'sm_120' or 'sm_120a' is explicitly outputted.
```
