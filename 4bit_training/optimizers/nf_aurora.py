"""
NF-Aurora: Schedule-Free Leverage-Aware Spectral Optimizer
==========================================================

Combines Schedule-Free dynamics with Aurora's leverage-uniform polar factor.
"""

import math
import torch

from .polar import polar_compiled as _polar_ns


@torch.no_grad()
def _aurora_polar(update, pp_iterations=2, pp_beta=0.5, eps=1e-7):
    """Compute leverage-uniform polar factor (Aurora's core innovation)."""
    m, n = update.shape
    if m == n:
        return _polar_ns(update)

    transposed = m < n
    if transposed:
        update = update.mT
        m, n = n, m

    G32 = update.to(torch.float32)
    target_row_sq = n / m
    row_norm = G32.norm(dim=-1, keepdim=True).clamp_(min=eps)
    D = 1.0 / row_norm

    for k in range(pp_iterations):
        U = _polar_ns(D * G32)
        if k < pp_iterations - 1:
            row_sq = U.to(torch.float32).pow(2).sum(dim=-1, keepdim=True).clamp_(min=eps * eps)
            D = D * (target_row_sq / row_sq).pow(pp_beta)

    return U.mT if transposed else U


class NFAurora(torch.optim.Optimizer):
    """Schedule-Free Aurora: horizon-free leverage-aware spectral optimizer.

    Designed for 2D matrix weights in transformers. For 1D parameters,
    falls back to an internal Schedule-Free AdamW.
    """

    def __init__(self, params, lr=0.01, beta=0.9, momentum=0.95,
                 weight_decay=0.05, warmup_steps=2000, eta_scale=0.2,
                 pp_iterations=2, pp_beta=0.5, nesterov=True, eps=1e-7):
        defaults = dict(
            lr=lr, beta=beta, momentum=momentum, weight_decay=weight_decay,
            warmup_steps=warmup_steps, eta_scale=eta_scale,
            pp_iterations=pp_iterations, pp_beta=pp_beta,
            nesterov=nesterov, eps=eps,
            k=0, train_mode=True, weight_sum=0.0,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def eval(self):
        """Switch to evaluation mode: set live weights to X_t (averaged)."""
        for group in self.param_groups:
            if group["train_mode"]:
                beta = group["beta"]
                for p in group["params"]:
                    state = self.state[p]
                    if "z" in state:
                        p.lerp_(end=state["z"], weight=1.0 - 1.0 / beta)
                group["train_mode"] = False

    @torch.no_grad()
    def train(self):
        """Switch to training mode: restore live weights to Y_t."""
        for group in self.param_groups:
            if not group["train_mode"]:
                beta = group["beta"]
                for p in group["params"]:
                    state = self.state[p]
                    if "z" in state:
                        p.lerp_(end=state["z"], weight=1.0 - beta)
                group["train_mode"] = True

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure else None

        for group in self.param_groups:
            beta = group["beta"]
            mu = group["momentum"]
            eta_scale = group["eta_scale"]
            decay = group["weight_decay"]
            nesterov = group["nesterov"]
            pp_iters = group["pp_iterations"]
            pp_b = group["pp_beta"]
            eps = group["eps"]
            k = group["k"]
            warmup = group["warmup_steps"]

            sched = min(1.0, (k + 1) / warmup) if warmup > 0 else 1.0
            lr = group["lr"] * sched
            weight = lr * lr
            weight_sum = group["weight_sum"] = group["weight_sum"] + weight
            ckp1 = weight / weight_sum

            use_aurora = group.get("use_aurora", True)
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if "z" not in state:
                    state["z"] = p.clone()
                    state["mom"] = torch.zeros_like(p)

                if p.ndim == 2 and use_aurora:
                    if "z" not in state:
                        state["z"] = p.clone()
                        state["mom"] = torch.zeros_like(p)
                    z = state["z"]
                    mom = state["mom"]
                    self._step_2d(p, grad, z, mom, mu, nesterov, pp_iters,
                                  pp_b, eps, eta_scale, lr, beta, decay, ckp1)
                else:
                    if "z" not in state:
                        state["z"] = p.clone()
                    z = state["z"]
                    self._step_1d(p, grad, z, state, lr, beta, decay, ckp1)

            group["k"] = k + 1
        return loss

    def _step_2d(self, p, grad, z, mom, mu, nesterov, pp_iters,
                 pp_b, eps, eta_scale, lr, beta, decay, ckp1):
        """2D matrix: Aurora + Schedule-Free."""
        mom.lerp_(grad, 1.0 - mu)
        update = grad.lerp(mom, mu) if nesterov else mom.clone()

        P = _aurora_polar(update, pp_iterations=pp_iters,
                          pp_beta=pp_b, eps=eps).to(p.dtype)

        m, n = p.shape
        P_norm = P.float().norm().item()
        eta_hat = eta_scale * lr * math.sqrt(m * n) / max(1e-12, P_norm)

        x_t = (p.data - (1.0 - beta) * z) / beta
        if decay != 0.0:
            z.sub_(z, alpha=lr * decay)
        z.sub_(P, alpha=eta_hat)
        x_tp1 = (1.0 - ckp1) * x_t + ckp1 * z
        p.data.copy_((1.0 - beta) * z + beta * x_tp1)

    def _step_1d(self, p, grad, z, state, lr, beta, decay, ckp1):
        """1D parameters: 4-bit quantized Schedule-Free AdamW fallback."""
        if "q_exp_avg_sq" not in state:
            state["q_exp_avg_sq"] = torch.zeros_like(p, dtype=torch.uint8)
            
            if p.ndim >= 2:
                scale_shape = (p.shape[0],) + (1,) * (p.ndim - 1)
            elif p.ndim == 1:
                scale_shape = (1,)
            else:
                scale_shape = ()
            state["scale_exp_avg_sq"] = torch.zeros(scale_shape, device=p.device, dtype=torch.float32)

        # Dequantize second moment
        v = state["q_exp_avg_sq"].to(torch.float32) * state["scale_exp_avg_sq"]

        # SF-AdamW uses raw gradient and no first moment EMA
        beta2 = 0.99
        v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

        # Unsigned 4-bit quantization of v to [0, 15]
        if p.ndim >= 2:
            max_v = torch.amax(v, dim=-1, keepdim=True)
        elif p.ndim == 1:
            max_v = torch.max(v).unsqueeze(0)
        else:
            max_v = v.clone()
        scale_v = (max_v / 15.0).clamp_(min=1e-12)
        q_v = torch.clamp(torch.round(v / scale_v), 0, 15).to(torch.uint8)

        state["q_exp_avg_sq"] = q_v
        state["scale_exp_avg_sq"] = scale_v

        # Dequantize again for the step update
        v = q_v.to(torch.float32) * scale_v

        # Safety bound against 4-bit quantization division-by-zero
        min_v = (1.0 - beta2) * grad.pow(2)
        v = torch.max(v, min_v)

        # SF-AdamW effective learning rate scaling
        eta_t = lr * math.sqrt(1.0 - beta2)
        update_1d = grad / (v.sqrt() + 1e-8)

        x_t = (p.data - (1.0 - beta) * z) / beta
        if decay != 0.0:
            z.sub_(z, alpha=eta_t * decay)
        z.sub_(update_1d, alpha=eta_t)
        x_tp1 = (1.0 - ckp1) * x_t + ckp1 * z
        p.data.copy_((1.0 - beta) * z + beta * x_tp1)
