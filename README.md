# 1B SLM: Production-Grade Blackwell & NVFP4 Pretraining Suite

A high-performance, production-grade 1.1 Billion parameter Small Language Model (SLM) architecture and pretraining pipeline optimized for NVIDIA Blackwell (SM120) architectures and consumer GPUs. It incorporates state-of-the-art architectures (Multi-Head Latent Attention, Delta Attention Residuals), training efficiency mechanisms (Token-Superposition Training, Multi-Token Prediction), and a customized high-performance hybrid optimizer suite.

---

## Key Architectural Features

### 1. Multi-Head Latent Attention (MLA)
To resolve the key-value (KV) cache memory bottleneck during long-context inference, the model implements Decoupled MLA:
* **KV Compression:** Compresses Keys and Values into a low-rank latent space dimension $d_c$ ($KV$ rank = 512).
* **Decoupled RoPE:** Keeps positional embeddings separated from the compressed semantic latent space by projecting a dedicated query/key part for RoPE, which is concatenated prior to computing Scaled Dot-Product Attention (SDPA). 

### 2. Delta Attention Residuals
Replaces uniform additive residual paths with learned softmax-routed residual connections over sub-layer changes (deltas):
* **Selective Routing:** Uses zero-initialized query parameters (`routing_q_attn`, `routing_q_ffn`) to dynamically route input tokens through past layers' deltas.
* **Pruned History:** Supports capped delta history tracking via `max_delta_history` to prevent $O(L)$ unbounded memory accumulation in deep architectures.

### 3. Token-Superposition Training (TST)
Allows training-time sequence length compression by averaging consecutive tokens over a group size ($G$):
* Compresses the sequence length in the embedding layer (reducing memory/flops by $G\times$).
* Enforces strict sequence length divisibility checks to ensure exact, error-free token grouping.

### 4. Multi-Token Prediction (MTP)
An auxiliary prediction framework that predicts multiple future tokens ($t+1, t+2, \dots$) in parallel:
* Implements stacked projection modules using a **SwiGLU** gating mechanism (`RMSNorm -> SwiGLU(gate * up) -> down` with residual skip) to prevent representation collapse.
* Includes NaN-guards during training to gracefully bypass auxiliary loss calculation if sequences are shorter than prediction depth.

---

## Precision & Quantization System

Designed to run efficiently on SM120 (Blackwell) GPUs using NVIDIA's 4-bit block-scaled quantization (NVFP4):
* **Precision Routing:** Automatically routes critical entry and exit blocks (configured via `high_precision_start_layers` and `high_precision_end_layers`) to run in high precision `bfloat16`/`float32` for convergence stability.
* **Intermediate Layers:** Autocasts intermediate Transformer blocks to `NVFP4` block-scaled format.
* **FP32 Casting:** Normalization layers (`RMSNorm`) and softmax calculations are kept in full `float32` precision to avoid underflow/overflow.

---

## Custom Optimizer Suite

The model uses a dual-optimization paradigm built from the ground up:

1. **Riemannian & Vanilla Aurora:** A leverage-aware spectral optimizer for 2D rectangular matrix weights, utilizing high-quality Newton-Schulz iteration (quintic convergence) to compute balanced polar decompositions.
2. **4-bit Quantized AdamW:** Quantizes the `exp_avg` (symmetric 4-bit, $[-7, 7]$) and `exp_avg_sq` (unsigned 4-bit, $[0, 15]$) states to save ~60% optimizer state memory for 1D biases, gains, and embeddings.
3. **NF-Aurora:** A Schedule-Free leverage-aware spectral optimizer that eliminates learning rate decay schedule tuning while providing anytime-training capabilities.
4. **SF-NorMuon:** Baseline schedule-free spectral optimizer used for benchmarking performance.

---

## Repository Structure

```
.
├── configs/
│   ├── default.yaml            # Default development config
│   └── 1b_nvfp4.yaml          # Production-grade 1.1B preset
├── layers/
│   ├── __init__.py
│   ├── attention.py            # Multi-Head Latent Attention
│   ├── ffn.py                  # Feed-Forward Dense Layer
│   ├── norm.py                 # Stable RMSNorm
│   ├── residual.py             # Delta Attention Residual Routing
│   └── rope.py                 # Complex precomputed RoPE
├── models/
│   ├── __init__.py
│   ├── embedding.py            # TST Embedding Layer
│   ├── mtp.py                  # MTP Head Projections
│   └── transformer.py          # TransformerBlock and SLMModel
├── optimizers/
│   ├── __init__.py
│   ├── aurora.py               # Vanilla and Riemannian Aurora
│   ├── hybrid.py               # HybridSLMOptimizer & 4-bit AdamW
│   ├── polar.py                # Newton-Schulz quintic polar
│   ├── nf_aurora.py            # Schedule-free NF-Aurora
│   ├── sf_normuon.py           # Baseline SF-NorMuon
│   └── factory.py              # Parameter routing builder
├── config.py                   # YAML Configuration parser
├── train.py                    # Main pretraining script
├── benchmark_optimizers.py     # Optimizer benchmarking suite
└── pyproject.toml              # Dependencies and metadata
```

---

## Getting Started

### 1. Prerequisites & Installation
This project requires Python 3.10+ and a CUDA-enabled GPU (Blackwell RTX 5080/5090 or Hopper/Ada Lovelace architectures for NVFP4 acceleration).

Install package dependencies using uv:
```bash
# Setup virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e .
```

To run with NVFP4 hardware acceleration, ensure NVIDIA Transformer Engine (`transformer_engine`) is installed. The codebase falls back to standard precision when `transformer_engine` is unavailable.

### 2. Configuration Options
Configurations are managed via YAML files:
* **Model Configuration:** Setup vocab size, hidden dimensions, layer count, MLA rank, TST group sizes, and MTP prediction depth.
* **Optimizations:** Enable/disable weight tying (`tie_word_embeddings`) and limit delta residual context length (`max_delta_history`).
* **Optimizer Configuration:** Route between `hybrid`, `nf_aurora`, or standard `adamw`.

### 3. Launching Pretraining
Run the training script with your target configuration:
```bash
# Launch with default configuration (24 layers, hybrid optimizer)
python train.py --config configs/default.yaml

# Launch production-grade 1B NVFP4 training run
python train.py --config configs/1b_nvfp4.yaml
```

### 4. Running Benchmarks
Benchmark optimizer convergence on Mini-GPT:
```bash
python benchmark_optimizers.py --steps 1000 --device cuda
```
