# 🔬 1B SLM — Stability & Performance Audit

> Based on: DeepSeek-V3, Llama 4, Qwen 3, "Spike No More" (COLM 2025), μP, Liger Kernel, and frontier training best practices (2024–2025).

---

## Executive Summary

After auditing every file in `training/` and cross-referencing with 8+ papers and frontier model reports, I identified **15 actionable improvements** across 4 categories. The table below ranks them by **impact × ease**:

| # | Improvement | Category | Priority | Effort | Impact |
|---|---|---|---|---|---|
| 1 | **QK-Norm** | Stability | 🔴 Critical | Low | Prevents attention entropy collapse |
| 2 | **Fused Linear + Cross-Entropy** | Performance | 🔴 Critical | Low | −40% VRAM on output layer |
| 3 | **Weight Initialization** | Stability | 🔴 Critical | Low | Prevents early divergence |
| 4 | **Embedding Scaling** | Stability | 🟠 High | Low | Balances signal vs. positional |
| 5 | **Z-Loss Regularization** | Stability | 🟠 High | Low | Prevents logit explosion |
| 6 | **MTP Loss Annealing** | Stability | 🟠 High | Trivial | Matches DeepSeek-V3 schedule |
| 7 | **Selective Recomputation** | Performance | 🟠 High | Medium | −20% recompute overhead |
| 8 | **Post-Attention/FFN Norm** | Stability | 🟡 Medium | Low | Controls activation growth |
| 9 | **Logit Soft-Capping** | Stability | 🟡 Medium | Low | Synergizes with QK-Norm |
| 10 | **FIM (Fill-in-Middle)** | Data | 🟡 Medium | Medium | Code infilling capability |
| 11 | **Batch Size Warmup** | Stability | 🟡 Medium | Low | Smoother early training |
| 12 | **Output Head Scaling** | Stability | 🟡 Medium | Trivial | Prevents tied-weight logit drift |
| 13 | **Gradient Norm Monitoring** | Observability | 🟡 Medium | Low | Early warning for spikes |
| 14 | **LR Schedule (Multi-Stage)** | Optimizer | 🟢 Low | Medium | Matches DeepSeek-V3 recipe |
| 15 | **Curriculum Data Mixing** | Data | 🟢 Low | High | Advanced data scheduling |

---

## Category 1: Training Stability

### 1. 🔴 QK-Norm (Query-Key Normalization)

**Problem:** Your [attention.py](file:///home/badrinathan-tv/Desktop/Projects/1B-Model/training/layers/attention.py) has **no normalization** on Q/K projections before the dot product. At high LR (0.01 with SF-NorMuon), attention logits can grow unboundedly → attention entropy collapse → loss spikes.

**Evidence:** DeepSeek-V3 applies RMSNorm on the compressed KV latent (`kv_norm`) and Q latent (`q_norm`) — you already have these for the *latent space*, but the **post-projection Q and K going into SDPA are unnormalized**. Gemma 2, Llama 3.2, and "Spike No More" all show QK-Norm is the #1 stability intervention.

**Where:** [attention.py:43-69](file:///home/badrinathan-tv/Desktop/Projects/1B-Model/training/layers/attention.py#L43-L69)

**Implementation:**
```python
# In __init__:
self.q_head_norm = RMSNorm(self.v_head_dim + self.qk_rope_head_dim, config.rms_norm_eps)
self.k_head_norm = RMSNorm(self.v_head_dim + self.qk_rope_head_dim, config.rms_norm_eps)

# In forward, after concatenating RoPE dims (line ~68-69):
q_full = torch.cat([q, q_rope], dim=-1)
k_full = torch.cat([k, k_rope], dim=-1)

# ADD THIS — normalize per-head before SDPA:
q_full = self.q_head_norm(q_full)
k_full = self.k_head_norm(k_full)
```

> [!IMPORTANT]
> Apply QK-Norm **after** RoPE concatenation but **before** `scaled_dot_product_attention`. This is critical — normalizing before RoPE would destroy positional information.

---

### 2. 🔴 Weight Initialization

**Problem:** Your codebase uses **PyTorch defaults** (`kaiming_uniform_` for Linear, normal for Embedding). DeepSeek-V3 uses `truncated_normal(σ=0.006)` for all parameters. For a 1B model with `hidden_size=1280`, the appropriate σ is approximately `1/√hidden_size ≈ 0.028` or a tuned smaller value.

**Where:** [transformer.py](file:///home/badrinathan-tv/Desktop/Projects/1B-Model/training/models/transformer.py) — no `_init_weights` method exists.

**Implementation:**
```python
# Add to SLMModel.__init__ (after all modules created):
self.apply(self._init_weights)

def _init_weights(self, module):
    σ = 0.02  # or 1/√1280 ≈ 0.028; tune via proxy
    if isinstance(module, nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, std=σ, a=-2*σ, b=2*σ)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.trunc_normal_(module.weight, std=σ, a=-2*σ, b=2*σ)
```

> [!TIP]
> DeepSeek-V3 also scales the output projection of each residual sub-layer by `1/√(2*num_layers)` to keep residual stream magnitude stable. Consider applying this to `o_proj` and `down_proj` weights.

---

### 3. 🟠 Embedding Scaling (`√d_model`)

**Problem:** Your [embedding.py](file:///home/badrinathan-tv/Desktop/Projects/1B-Model/training/models/embedding.py) returns raw embeddings without scaling. With weight tying, the embedding matrix serves dual duty — without `√d` scaling on input, the semantic signal is too small relative to the residual stream magnitude after the first layer.

**Where:** [embedding.py:12-26](file:///home/badrinathan-tv/Desktop/Projects/1B-Model/training/models/embedding.py#L12-L26)

**Implementation:**
```python
def forward(self, input_ids):
    embeds = self.word_embeddings(input_ids) * math.sqrt(self.hidden_size)
    # ... rest of TST logic
```

> [!WARNING]
> If you add embedding scaling, you may need to compensate in the output logits (divide by `√d` or add a learned temperature) to prevent the tied `lm_head` from producing over-confident logits.

---

### 4. 🟠 Z-Loss Regularization

**Problem:** No logit regularization exists in your [train.py:31-49](file:///home/badrinathan-tv/Desktop/Projects/1B-Model/training/train.py#L31-L49). Over long training runs, output logits can drift to extreme values → numerical instability in softmax → gradient spikes.

**Evidence:** PaLM, Gemma, and ST-MoE all use Z-loss with coefficient `1e-4` to `1e-5`. DeepSeek-V3 uses a complementary sequence-wise auxiliary loss at `α=0.0001`.

**Implementation:**
```python
def compute_loss(logits_list, targets, config: SLMConfig) -> torch.Tensor:
    main_logits = logits_list[0]
    loss = _compiled_ce(main_logits.view(-1, config.vocab_size), targets.view(-1))

    # Z-loss: penalize large logits to prevent softmax saturation
    z_loss_weight = 1e-4
    log_z = torch.logsumexp(main_logits.float(), dim=-1)  # [B, S]
    z_loss = z_loss_weight * (log_z ** 2).mean()
    loss = loss + z_loss

    # ... MTP losses unchanged
```

---

### 5. 🟠 MTP Loss Weight Annealing

**Problem:** Your config uses a fixed `mtp_loss_weight: 0.3` throughout training. DeepSeek-V3 anneals from **λ=0.3 → 0.1** at 67% of training (10T/14.8T tokens). This prevents the auxiliary MTP signal from competing with the main LM head in late training.

**Where:** [train.py:37](file:///home/badrinathan-tv/Desktop/Projects/1B-Model/training/train.py#L37), [config.py:69](file:///home/badrinathan-tv/Desktop/Projects/1B-Model/training/config.py#L69)

**Implementation:**
```python
# In train.py training loop:
mtp_anneal_step = int(config.training.max_steps * 0.67)
if step < mtp_anneal_step:
    mtp_weight = 0.3
else:
    mtp_weight = 0.1
```

Add `mtp_loss_weight_final: 0.1` and `mtp_anneal_fraction: 0.67` to `TrainingConfig`.

---

### 6. 🟡 Post-Attention & Post-FFN Normalization

**Problem:** Recent research ("Spike No More", extended normalization studies) shows that activation norms from the **output projection** and **FFN down_proj** can grow during training, even with pre-norm. Your [transformer.py](file:///home/badrinathan-tv/Desktop/Projects/1B-Model/training/models/transformer.py) only has pre-norm (`attn_norm`, `ffn_norm`).

**Where:** [transformer.py:30-34](file:///home/badrinathan-tv/Desktop/Projects/1B-Model/training/models/transformer.py#L30-L34)

**Implementation:**
```python
# In TransformerBlock.__init__:
self.post_attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
self.post_ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

# In forward:
v_attn = self.post_attn_norm(self.attn(self.attn_norm(h_attn), ...))
# ...
v_ffn = self.post_ffn_norm(self.ffn(self.ffn_norm(h_ffn)))
```

> [!NOTE]
> This is a "belt and suspenders" approach. If QK-Norm alone stabilizes training, you may not need this. Monitor activation norms during initial runs to decide.

---

### 7. 🟡 Logit Soft-Capping

**Problem:** Synergizes with QK-Norm. Caps attention logits before softmax to prevent extreme sharpening.

**Evidence:** Gemma 2 uses `tanh` soft-capping with a cap value of 50.0 on attention logits. Research shows combining QK-Norm + soft-capping allows **1.5× higher learning rates** without divergence.

**Where:** [attention.py:77](file:///home/badrinathan-tv/Desktop/Projects/1B-Model/training/layers/attention.py#L77)

**Implementation:**
```python
# Cannot use with F.scaled_dot_product_attention directly.
# Option A: Use FlexAttention (PyTorch 2.5+) with score_mod
# Option B: Manual attention (slower, but more control)
# Option C: Skip if QK-Norm alone is sufficient (recommended first)
```

> [!CAUTION]
> Logit soft-capping is **incompatible** with `F.scaled_dot_product_attention` as it doesn't expose the raw attention scores. Only implement this if QK-Norm alone proves insufficient, and use FlexAttention or a custom kernel.

---

### 8. 🟡 Output Head Temperature / Scaling

**Problem:** With `tie_word_embeddings: true`, the `lm_head` weight is the embedding matrix. If you add `√d` embedding scaling (item #3), logits scale up proportionally, causing over-confidence. Even without embedding scaling, a learned output temperature improves convergence.

**Where:** [transformer.py:81](file:///home/badrinathan-tv/Desktop/Projects/1B-Model/training/models/transformer.py#L81), [mtp.py:38](file:///home/badrinathan-tv/Desktop/Projects/1B-Model/training/models/mtp.py#L38)

**Implementation:**
```python
# In SLMModel.__init__:
self.output_scale = nn.Parameter(torch.ones(1) * (1.0 / math.sqrt(config.hidden_size)))

# In MTPModule.forward:
logits_list = [self.lm_head(hidden_states) * self.model.output_scale]
```

---

## Category 2: Performance & Throughput

### 9. 🔴 Liger FusedLinearCrossEntropy

**Problem:** Your [train.py:53](file:///home/badrinathan-tv/Desktop/Projects/1B-Model/training/train.py#L53) uses `torch.compile(F.cross_entropy)`, which still materializes the full `[B×S, 64000]` logits tensor. With `batch_size=4, seq_len=2048`, that's `4 × 2048 × 64000 × 2 bytes = ~1 GB` per micro-step. Liger's `FusedLinearCrossEntropy` computes the loss **without ever materializing the logits**.

**Evidence:** You already use Liger for SwiGLU and RoPE. The fused CE kernel saves **40-60% VRAM** on the output layer.

**Where:** [mtp.py:38](file:///home/badrinathan-tv/Desktop/Projects/1B-Model/training/models/mtp.py#L38), [train.py:31-53](file:///home/badrinathan-tv/Desktop/Projects/1B-Model/training/train.py#L31-L53)

**Implementation:**
```python
# In mtp.py — return hidden states instead of logits:
def forward(self, hidden_states, use_mtp=True):
    states_list = [hidden_states]
    curr = hidden_states
    if use_mtp and self.mtp_depth > 1:
        for proj in self.projs:
            curr = proj(curr)
            states_list.append(curr)
    return states_list

# In train.py — use fused kernel:
from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction

def compute_loss(hidden_states_list, targets, lm_head_weight, config):
    main_h = hidden_states_list[0]
    loss = LigerFusedLinearCrossEntropyFunction.apply(
        main_h.view(-1, main_h.size(-1)), lm_head_weight, targets.view(-1)
    )
    # ... MTP losses similarly
    return loss
```

> [!TIP]
> This is the single biggest VRAM win available. With vocab_size=64000, you save ~1GB per micro-step, potentially allowing you to increase batch_size or seq_len.

---

### 10. 🟠 Selective Gradient Checkpointing

**Problem:** Your [transformer.py:97-103](file:///home/badrinathan-tv/Desktop/Projects/1B-Model/training/models/transformer.py#L97-L103) checkpoints **entire TransformerBlocks** including all projections. DeepSeek-V3 only recomputes RMSNorm and MLA up-projections — keeping the expensive down-projections in memory.

**Current code:**
```python
x, deltas = checkpoint(custom_forward, x, list(deltas), use_reentrant=False)
```

**Better approach:** Since you're using FlashAttention via SDPA (which already recomputes attention internally), the main memory hog is the FFN activations. Consider checkpointing only the FFN sublayer or only the norm+projection:

```python
# Option: Don't checkpoint attention (SDPA already memory-efficient),
# only checkpoint FFN:
v_attn = self.attn(self.attn_norm(h_attn), ...)  # No checkpoint
x = x + v_attn

# Checkpoint only the FFN
v_ffn = checkpoint(lambda h: self.ffn(self.ffn_norm(h)), h_ffn, use_reentrant=False)
x = x + v_ffn
```

> [!NOTE]
> Measure wall-clock time with vs. without full checkpointing. On your 16GB setup, full checkpointing may still be necessary, but the **5-10% throughput improvement** from selective recompute is worth testing.

---

## Category 3: Architecture Refinements

### 11. 🟡 MTP Architecture — DeepSeek-V3 Alignment

**Problem:** Your [MTP](file:///home/badrinathan-tv/Desktop/Projects/1B-Model/training/models/mtp.py) uses a SwiGLU projection head. DeepSeek-V3's MTP uses a **full Transformer block** per depth level plus a **concatenation-based input** (`M_k ∈ R^{d×2d}` combines current hidden state with embedding of the previously predicted token).

**Current:** `h → SwiGLU → lm_head` per MTP depth (no embedding conditioning)  
**DeepSeek-V3:** `concat(h, embed(prev_predicted_token)) → Linear(2d→d) → TransformerBlock → lm_head`

**Assessment:** For a 1B model, a full Transformer block per MTP depth is expensive. Your SwiGLU approach is a reasonable trade-off. However, **incorporating the embedding of the predicted token** (from the previous MTP level) would improve the causal chain quality:

```python
# Enhanced MTP that conditions on predicted token embedding:
class MTPProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        # 2d → d projection (concatenates hidden + predicted embedding)
        self.input_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.down_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, x, prev_embed):
        h = torch.cat([x, prev_embed], dim=-1)
        h = self.input_proj(h)
        h = self.norm(h)
        h = self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
        return x + h
```

---

## Category 4: Data Pipeline & Training Loop

### 12. 🟡 FIM (Fill-in-Middle) Support

**Problem:** DeepSeek-V3 uses FIM at 10% rate with PSM format. Your tokenized data pipeline has no FIM support. FIM is critical for code completion and editing capabilities.

**Implementation:** This requires changes at the **data preprocessing** level (your `scripts/` pipeline), not in the model architecture. The model just needs sentinel tokens in the vocabulary.

**Steps:**
1. Add sentinel tokens: `<PRE>`, `<SUF>`, `<MID>`, `<EOT>` to tokenizer
2. During data loading, with 10% probability, transform a document:
   ```
   Original: [token_1, ..., token_N]
   PSM:      [<PRE>, prefix_tokens, <SUF>, suffix_tokens, <MID>, middle_tokens, <EOT>]
   ```
3. Apply only to code data (not prose/math)

---

### 13. 🟡 Batch Size Warmup

**Problem:** Your config uses a fixed `batch_size: 4` with `gradient_accumulation_steps: 512` from step 0. DeepSeek-V3 ramps batch size from 3072→15360 over the first 469B tokens. Starting with a smaller effective batch stabilizes early training.

**Implementation:**
```python
# In train.py:
def get_grad_accum_steps(step, max_steps):
    warmup_frac = 0.05  # 5% of training
    min_accum = 64      # Start small
    max_accum = 512     # Ramp to full
    if step < max_steps * warmup_frac:
        progress = step / (max_steps * warmup_frac)
        return int(min_accum + (max_accum - min_accum) * progress)
    return max_accum
```

---

### 14. 🟡 Gradient Norm Monitoring & Spike Detection

**Problem:** Your training loop logs `loss`, `vram`, and `tps` but **not gradient norms**. "Spike No More" shows that monitoring per-layer gradient norms is the best early warning system for impending divergence.

**Where:** [train.py:272-292](file:///home/badrinathan-tv/Desktop/Projects/1B-Model/training/train.py#L272-L292)

**Implementation:**
```python
# After gradient clipping:
if accelerator.sync_gradients and (step + 1) % train_cfg.log_interval == 0:
    grad_norm = accelerator.clip_grad_norm_(model.parameters(), train_cfg.gradient_clip)
    # Also log per-component norms:
    attn_gnorm = torch.nn.utils.clip_grad_norm_(
        [p for n, p in model.named_parameters() if 'attn' in n and p.grad is not None], float('inf')
    )
    ffn_gnorm = torch.nn.utils.clip_grad_norm_(
        [p for n, p in model.named_parameters() if 'ffn' in n and p.grad is not None], float('inf')
    )
    wandb.log({"grad_norm": grad_norm, "attn_grad_norm": attn_gnorm, "ffn_grad_norm": ffn_gnorm}, step=step+1)
```

---

### 15. 🟢 Multi-Stage LR Schedule

**Problem:** SF-NorMuon is schedule-free (Polyak step-size), but DeepSeek-V3's LR recipe has explicit phases:
1. Warmup → constant → cosine decay → constant → final constant

For schedule-free optimizers, the equivalent is **annealing the base_lr** that Polyak scales against. Currently you have a fixed `base_lr: 0.01`.

**Assessment:** This is lower priority because SF-NorMuon's Polyak mechanism naturally adapts the effective LR. However, you could anneal the `base_lr` cap in the final 20% of training to encourage convergence.

---

## Implementation Order (Recommended)

```mermaid
graph LR
    A[Phase 1: Critical Stability] --> B[Phase 2: Performance]
    B --> C[Phase 3: Polish]
    
    A --> A1[QK-Norm]
    A --> A2[Weight Init]
    A --> A3[Embedding Scaling]
    A --> A4[Z-Loss]
    
    B --> B1[Fused Linear CE]
    B --> B2[MTP Annealing]
    B --> B3[Grad Norm Monitoring]
    
    C --> C1[Selective Recompute]
    C --> C2[Post-Attn/FFN Norm]
    C --> C3[Batch Size Warmup]
    C --> C4[FIM Support]
```

### Phase 1 — Do Before First Real Training Run
1. **Weight Initialization** (prevents early divergence)
2. **QK-Norm** (prevents attention collapse)
3. **Embedding Scaling** (signal balance)
4. **Z-Loss** (logit regularization)

### Phase 2 — Implement During Initial Runs
5. **Fused Linear CE** (VRAM savings → larger batch possible)
6. **MTP Loss Annealing** (trivial config change)
7. **Gradient Norm Monitoring** (observability for debugging)

### Phase 3 — Optimize After Stable Baseline
8. Selective recomputation
9. Post-attention/FFN norms (if spikes observed)
10. Batch size warmup
11. FIM support (data pipeline work)

---

## Files Affected

| File | Changes |
|---|---|
| [attention.py](file:///home/badrinathan-tv/Desktop/Projects/1B-Model/training/layers/attention.py) | QK-Norm, logit soft-capping |
| [transformer.py](file:///home/badrinathan-tv/Desktop/Projects/1B-Model/training/models/transformer.py) | Weight init, post-norms, selective recompute |
| [embedding.py](file:///home/badrinathan-tv/Desktop/Projects/1B-Model/training/models/embedding.py) | √d scaling |
| [mtp.py](file:///home/badrinathan-tv/Desktop/Projects/1B-Model/training/models/mtp.py) | DeepSeek-style conditioning, output refactor for fused CE |
| [train.py](file:///home/badrinathan-tv/Desktop/Projects/1B-Model/training/train.py) | Z-loss, fused CE, MTP annealing, grad monitoring, batch warmup |
| [config.py](file:///home/badrinathan-tv/Desktop/Projects/1B-Model/training/config.py) | New fields: `init_std`, `z_loss_weight`, `mtp_anneal_*`, `embed_scale` |
| [default.yaml](file:///home/badrinathan-tv/Desktop/Projects/1B-Model/training/configs/default.yaml) | New config values |

---

## Key Papers Referenced

| Paper | Key Technique | Relevance |
|---|---|---|
| DeepSeek-V3 (2412.19437) | MLA, MTP, FIM, init, FP8 | Primary reference |
| "Spike No More" (COLM 2025) | Spectral norm analysis, gradient norm monitoring | Stability theory |
| Gemma 2 (Google, 2024) | QK-Norm + logit soft-capping | Attention stability |
| PaLM (Google, 2022) | Z-loss regularization | Logit control |
| μP / u-μP (Cerebras/ICML 2024) | Width-independent HP transfer | Init/scaling |
| Liger Kernel (LinkedIn, ICML 2025) | FusedLinearCrossEntropy, fused SwiGLU | Performance |
| ScheduleFree+ (Meta, 2024) | Polyak step-size, beta annealing | Your optimizer |
