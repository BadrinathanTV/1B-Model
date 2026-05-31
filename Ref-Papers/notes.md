# ScheduleFree+ Paper Notes

**Core insights for LLM Scaling (Large Batch / 1000 TPP)**

1. **Inner Momentum is Mandatory**: For large batch sizes (like 4M tokens), Schedule-Free without inner momentum diverges. We must reintroduce inner momentum (`beta1 = 0.75` to `0.9`) to handle the larger step sizes associated with large batch training.
2. **Weight Decay vs Gradient Norm Drift**: Standard weight decay in Schedule-Free causes shrinking weight norms, which in turn causes gradient norms (and thus the effective learning rate) to increase, destabilizing training.
    - *Fix 1*: Use small weight decay (`0.002`).
    - *Fix 2 (Better)*: Use an inverse-gradient L1 norm weighting for the learning rate.
3. **Polyak Step Size**: By using the Polyak step-size rule, we get a *learning-rate-free* optimizer that natively incorporates the inverse-gradient L1 norm scaling. This requires fully-decoupled weight decay (AdamC-style) because changing learning rates otherwise breaks the weight decay dynamics.
4. **Beta Annealing**: Interpolating the outer momentum `beta` from `0.8` (or `0.9`) to `0.965` over the training run gives the best combination of fast early convergence and stable final convergence.
5. **r=1 Weighting**: For long duration runs (>30B tokens), setting the averaging weight parameter `r=1` is significantly better than the default `r=0`.
6. **Warm-starting**: Do not apply iterate averaging (`c_t = 1`) for an initial duration (roughly 2x the learning rate warmup steps). This gives much faster early convergence.

---

# Anytime Training with Schedule-Free Spectral Optimization Notes

**Core insights for SF-NorMuon**

1. **Weight Decay on Z**: For Schedule-Free Spectral Optimizers, applying weight decay at the interpolation point `Y` eventually leads to divergence. Weight decay MUST be applied directly to the fast iterate `Z` (`Z = Z - lr * wd * Z - update`). This ensures the `Z` sequence remains bounded.
2. **Explicit Momentum Before Polar**: The algorithm smooths the gradient with an explicit momentum buffer (`M_t = \mu M_{t-1} + (1-\mu) G_t`) BEFORE computing the polar factor. Ablating this momentum leads to a significant performance drop.
3. **Row-wise Adaptive Normalization**: Uses an EMA of row-wise squared norms of the polar factor to adapt per-neuron step sizes, similar to Adam but acting on rows.
4. **Step Size Scaling**: The update is scaled by `0.2 * lr * max(m, n) / ||P_normalized||_F` to keep it comparable to Adam's RMS scaling.
