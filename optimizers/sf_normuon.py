"""
SF-NorMuon: Schedule-Free NorMuon (Baseline)
=============================================

Baseline spectral optimizer for benchmarking against NF-Aurora.
Implements Algorithm 1 from "Anytime Training with Schedule-Free
Spectral Optimization" (Apte et al. 2026).
"""

import math
import torch

from .polar import polar_compiled as _polar_ns


class SFNorMuon(torch.optim.Optimizer):
    """Schedule-Free NorMuon: baseline spectral optimizer for benchmarking."""

    def __init__(self, params, lr=0.01, betas=(0.9, 0.95), momentum=0.8,
                 weight_decay=0.05, warmup_steps=2000, eta_scale=0.2, eps=1e-8):
        defaults = dict(
            lr=lr, betas=betas, momentum=momentum, weight_decay=weight_decay,
            warmup_steps=warmup_steps, eta_scale=eta_scale, eps=eps,
            k=0, train_mode=True, weight_sum=0.0,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            if group["train_mode"]:
                beta = group["betas"][0]
                for p in group["params"]:
                    state = self.state[p]
                    if "z" in state:
                        p.lerp_(end=state["z"], weight=1.0 - 1.0 / beta)
                group["train_mode"] = False

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            if not group["train_mode"]:
                beta = group["betas"][0]
                for p in group["params"]:
                    state = self.state[p]
                    if "z" in state:
                        p.lerp_(end=state["z"], weight=1.0 - beta)
                group["train_mode"] = True

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure else None

        for group in self.param_groups:
            beta, beta2 = group["betas"]
            mu = group["momentum"]
            eta_scale = group["eta_scale"]
            decay = group["weight_decay"]
            eps = group["eps"]
            k = group["k"]
            warmup = group["warmup_steps"]

            sched = min(1.0, (k + 1) / warmup) if warmup > 0 else 1.0
            lr = group["lr"] * sched
            weight = lr * lr
            weight_sum = group["weight_sum"] = group["weight_sum"] + weight
            ckp1 = weight / weight_sum

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if "z" not in state:
                    state["z"] = p.clone()
                    state["mom"] = torch.zeros_like(p)
                    if p.ndim == 2:
                        state["v"] = torch.zeros(p.shape[0], device=p.device, dtype=torch.float32)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(p)

                z = state["z"]
                mom = state["mom"]

                if p.ndim == 2:
                    mom.mul_(mu).add_(grad, alpha=1.0 - mu)
                    P = _polar_ns(mom).to(p.dtype)

                    v = state["v"]
                    row_ms = (P * P).mean(dim=1).float()
                    v.mul_(beta2).add_(row_ms, alpha=1.0 - beta2)
                    Phat = P / (v.sqrt() + eps).to(P.dtype).unsqueeze(1)

                    m, n = p.shape
                    P_norm = Phat.float().norm().item()
                    eta_hat = eta_scale * lr * math.sqrt(m * n) / max(1e-12, P_norm)

                    x_t = (p.data - (1.0 - beta) * z) / beta
                    if decay != 0.0:
                        z.sub_(z, alpha=lr * decay)
                    z.sub_(Phat, alpha=eta_hat)
                    x_tp1 = (1.0 - ckp1) * x_t + ckp1 * z
                    p.data.copy_((1.0 - beta) * z + beta * x_tp1)
                else:
                    v = state["exp_avg_sq"]
                    mom.lerp_(grad, 1.0 - 0.9)
                    v.mul_(0.99).addcmul_(grad, grad, value=0.01)
                    update_1d = mom / (v.sqrt() + 1e-8)
                    x_t = (p.data - (1.0 - beta) * z) / beta
                    if decay != 0.0:
                        z.sub_(z, alpha=lr * decay)
                    z.sub_(update_1d, alpha=lr)
                    x_tp1 = (1.0 - ckp1) * x_t + ckp1 * z
                    p.data.copy_((1.0 - beta) * z + beta * x_tp1)

            group["k"] = k + 1
        return loss
