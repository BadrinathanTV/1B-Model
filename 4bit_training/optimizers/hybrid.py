"""
Hybrid SLM Optimizer
=====================

Routes 2D matrix weights through Riemannian Aurora and 1D/embedding
parameters through 4-bit quantized AdamW.
"""

import math
import torch

from .aurora import aurora, riemannian_aurora


class HybridSLMOptimizer(torch.optim.Optimizer):
    """Hybrid optimizer: Aurora for 2D weights, 4-bit AdamW for 1D/embeddings."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-10,
                 weight_decay=0.1, momentum=0.95, use_riemannian=True,
                 warmup_steps=2000):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        momentum=momentum, use_riemannian=use_riemannian,
                        warmup_steps=warmup_steps, k=0)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            k = group.get("k", 0)
            warmup = group.get("warmup_steps", 2000)
            sched = min(1.0, (k + 1) / warmup) if warmup > 0 else 1.0
            lr = group["lr"] * sched

            for p in group["params"]:
                if p.grad is None:
                    continue
                if group.get("use_aurora", False) and p.ndim == 2:
                    self._step_aurora(p, group, lr)
                else:
                    self._step_adamw_4bit(p, group, lr)

            group["k"] = k + 1
        return loss

    def _step_aurora(self, p, group, lr):
        """Apply Riemannian Aurora update to a 2D matrix parameter."""
        state = self.state[p]
        if len(state) == 0:
            state["momentum_buffer"] = torch.zeros_like(p)
        step_fn = riemannian_aurora if group["use_riemannian"] else aurora
        step_fn(p.data, p.grad.data, state["momentum_buffer"],
                eta=lr, weight_decay=group["weight_decay"],
                mu=group["momentum"])

    def _step_adamw_4bit(self, p, group, lr):
        """Apply 4-bit quantized AdamW update to a 1D/embedding parameter."""
        state = self.state[p]
        if len(state) == 0:
            state["step"] = 0
            state["q_exp_avg"] = torch.zeros_like(p, dtype=torch.int8)
            state["q_exp_avg_sq"] = torch.zeros_like(p, dtype=torch.uint8)
            # Scale shape: per-row for 2D+, per-tensor scalar for 1D, () for 0D
            if p.ndim >= 2:
                scale_shape = (p.shape[0],) + (1,) * (p.ndim - 1)
            elif p.ndim == 1:
                scale_shape = (1,)
            else:
                # 0D scalar parameter — scale must also be scalar
                scale_shape = ()
            state["scale_exp_avg"] = torch.zeros(scale_shape, device=p.device, dtype=torch.float32)
            state["scale_exp_avg_sq"] = torch.zeros(scale_shape, device=p.device, dtype=torch.float32)

        state["step"] += 1

        # Dequantize states to float32
        exp_avg = state["q_exp_avg"].to(torch.float32) * state["scale_exp_avg"]
        exp_avg_sq = state["q_exp_avg_sq"].to(torch.float32) * state["scale_exp_avg_sq"]

        beta1, beta2 = group["betas"]
        exp_avg.mul_(beta1).add_(p.grad.data, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(p.grad.data, p.grad.data, value=1 - beta2)

        bias_correction1 = 1 - beta1 ** state["step"]
        bias_correction2 = 1 - beta2 ** state["step"]
        
        # Enforce second moment constraint to prevent division-by-zero from quantization rounding
        min_exp_avg_sq = bias_correction2 * (exp_avg / bias_correction1).pow(2)
        exp_avg_sq = torch.max(exp_avg_sq, min_exp_avg_sq)
        
        step_size = lr / bias_correction1
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])

        # Symmetric 4-bit quantization of exp_avg to [-7, 7]
        if p.ndim >= 2:
            max_avg = torch.amax(torch.abs(exp_avg), dim=-1, keepdim=True)
        elif p.ndim == 1:
            max_avg = torch.max(torch.abs(exp_avg)).unsqueeze(0)
        else:
            # 0D scalar — max is the value itself
            max_avg = torch.abs(exp_avg)
        scale_avg = (max_avg / 7.0).clamp_(min=1e-12)
        q_avg = torch.clamp(torch.round(exp_avg / scale_avg), -7, 7).to(torch.int8)

        # Unsigned 4-bit quantization of exp_avg_sq to [0, 15]
        if p.ndim >= 2:
            max_avg_sq = torch.amax(exp_avg_sq, dim=-1, keepdim=True)
        elif p.ndim == 1:
            max_avg_sq = torch.max(exp_avg_sq).unsqueeze(0)
        else:
            # 0D scalar
            max_avg_sq = exp_avg_sq.clone()
        scale_avg_sq = (max_avg_sq / 15.0).clamp_(min=1e-12)
        q_avg_sq = torch.clamp(torch.round(exp_avg_sq / scale_avg_sq), 0, 15).to(torch.uint8)

        state["q_exp_avg"] = q_avg
        state["scale_exp_avg"] = scale_avg
        state["q_exp_avg_sq"] = q_avg_sq
        state["scale_exp_avg_sq"] = scale_avg_sq


        p.data.mul_(1 - lr * group["weight_decay"])
        p.data.addcdiv_(exp_avg, denom, value=-step_size)
