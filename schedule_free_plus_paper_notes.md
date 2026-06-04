# ScheduleFree+ Paper Summary & Hyperparameters Reference

This document summarizes the core components, optimizer mechanisms, and recommended hyperparameters from the research paper **"ScheduleFree+: Scaling Learning-Rate-Free & Schedule-Free Learning to Large Language Models"** (arXiv:2605.19095, Meta FAIR, May 2026).

---

## 1. The Core Optimizer: AdamC + Schedule-Free + Polyak

**ScheduleFree+** is designed to eliminate the need for manual learning rate (LR) schedule tuning (like Cosine or Warmup-Stable-Decay) at the scale of modern Large Language Models (LLMs). The paper introduces the **`AdamCScheduleFreePlusPaper`** optimizer, which integrates three key ideas:

### A. AdamC (Corrected/Fully-Decoupled Weight Decay)
* **The Problem:** In standard AdamW, standard decoupled weight decay can cause unstable training or gradient norm drift (where gradient magnitudes increase unexpectedly toward the end of training) when paired with adaptive learning rate updates.
* **The Fix:** AdamC decouples weight decay by scaling it by the **square of the learning rate** ($\gamma_t^2$) and applying it directly on the unaveraged iterate $z$ at the query point $y$. This prevents weight decay strength from changing dynamically during LR adaptation.
* **Update Rule:** 
  $$z_{t+1} = z_t - \eta \lambda \gamma_t^2 y_t$$
  Where $\lambda$ is weight decay, $\gamma_t$ is the step size, and $y_t$ is the gradient evaluation point.

### B. Schedule-Free Optimization
Instead of relying on momentum, the optimizer maintains three sequences:
1. **$z_t$:** The unaveraged base optimizer (AdamW/AdamC) iterate.
2. **$x_t$:** The averaged/evaluation iterate (equivalent to the model weights used for validation/inference).
3. **$y_t$:** The gradient evaluation point (where gradients are computed during the forward pass).

It continuously interpolates between these sequences to eliminate the need for final annealing schedules, keeping the model in an "anytime evaluation" state.

### C. Polyak Step-size Adaptivity
* The optimizer computes online step sizes using a practical variant of the Polyak step-size rule based on rank-local function values and gradient norm information.

---

## 2. Key Hyperparameter Suggestions

Because the weight decay is fully decoupled in AdamC, the hyperparameters **do not correspond to standard AdamW settings**. Using standard values (like `0.1`) will result in effectively no decay.

| Hyperparameter | Recommended Value / Range | Description |
| :--- | :--- | :--- |
| **Weight Decay ($\lambda$)** | **`5.0` to `50.0`** (e.g., `20.0` or `50.0`) | **Crucial Difference:** Since weight decay is scaled by $\text{lr}^2$, the absolute values must be orders of magnitude larger than standard AdamW ($0.1$). |
| **Betas ($\beta_1, \beta_2$)** | **`(0.9, 0.95)`** | Standard adaptive moving average coefficients for gradient and its square. |
| **`sf_beta1`** | **`0.9`** | Schedule-Free outer momentum parameter. |
| **`sf_beta1_max`** | **`0.965`** | The maximum value of `sf_beta1` as it is annealed during training. |
| **Polynomial Power ($r$)** | **`2.0`** (or default `0.0`) | Weighting power of the averages. The paper notes that $r=2$ yields better validation performance for long-duration runs. |
| **`weight_lr_power`** | **`2.0`** | Dictates the power relationship of the learning rate inside the averaging weight computations. |
| **`c_warmup`** | **Step-based value** (e.g., `2500`) | The number of steps where the extrapolation factor $c_t$ remains constant at $1.0$ (warm-starting the averaging process). |

---

## 3. Important Operational Notes

1. **Train/Eval Modes (Critical):** Because the optimizer swaps between the evaluation iterate ($x$) and the gradient query point ($y$), you must call **`optimizer.train()`** at the start of training iterations and **`optimizer.eval()`** before validation loops or checkpoint saving. Failing to call `eval()` prior to saving will write the wrong parameter sequence to the checkpoint.
2. **Predictability:** The paper demonstrates that with ScheduleFree+, you can predict the final convergence and validation loss curve within the first $5\%$ to $15\%$ of training steps, vastly simplifying hyperparameter verification.
3. **Memory footprint:** The optimizer does not increase VRAM usage over standard AdamW.
