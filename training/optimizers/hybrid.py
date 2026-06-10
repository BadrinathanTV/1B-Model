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
                 weight_decay=0.1, momentum=0.95, use_riemannian=True):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        momentum=momentum, use_riemannian=use_riemannian)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if group.get("use_aurora", False) and p.ndim == 2:
                    self._step_aurora(p, group, lr)
                else:
                    self._step_adamw_4bit(p, group, lr)

        return loss

    def _step_aurora(self, p, group, lr):
        """Apply Riemannian Aurora update to a 2D matrix parameter."""
        state = self.state[p]
        if len(state) == 0:
            state["momentum_buffer"] = torch.zeros_like(p)
        kwargs = {
            "eta": lr,
            "weight_decay": group["weight_decay"],
            "mu": group["momentum"],
            "update_clip": group.get("update_clip", 0.0)
        }
        step_fn = riemannian_aurora if group.get("use_riemannian", True) else aurora
        if not group.get("use_riemannian", True):
            kwargs["pp_iterations"] = group.get("pp_iterations", 2)
            kwargs["pp_beta"] = group.get("pp_beta", 0.5)
            
        step_fn(p.data, p.grad.data, state["momentum_buffer"], **kwargs)

    def _step_adamw_4bit(self, p, group, lr):
        """Apply 4-bit quantized AdamW update to a 1D/embedding parameter.
        
        Uses chunking for massive DeepSpeed-flattened 1D tensors to prevent OOM.
        """
        CHUNK_SIZE = 10_000_000
        state = self.state[p]
        
        if len(state) == 0:
            state["step"] = 0
            state["q_exp_avg"] = torch.zeros_like(p, dtype=torch.int8)
            state["q_exp_avg_sq"] = torch.zeros_like(p, dtype=torch.uint8)
            is_massive = (p.numel() > CHUNK_SIZE)
            if is_massive:
                num_chunks = (p.numel() + CHUNK_SIZE - 1) // CHUNK_SIZE
                scale_shape = (num_chunks,)
            elif p.ndim >= 2:
                scale_shape = (p.shape[0],) + (1,) * (p.ndim - 1)
            elif p.ndim == 1:
                scale_shape = (1,)
            else:
                scale_shape = ()
            state["scale_exp_avg"] = torch.zeros(scale_shape, device=p.device, dtype=torch.bfloat16)
            state["scale_exp_avg_sq"] = torch.zeros(scale_shape, device=p.device, dtype=torch.bfloat16)

        state["step"] += 1
        beta1, beta2 = group["betas"]
        bias_correction1 = 1 - beta1 ** state["step"]
        bias_correction2 = 1 - beta2 ** state["step"]
        step_size = lr / bias_correction1

        is_massive = (p.numel() > CHUNK_SIZE)

        if is_massive:
            # Process in chunks to cap VRAM spikes
            num_chunks = (p.numel() + CHUNK_SIZE - 1) // CHUNK_SIZE
            p_flat = p.data.view(-1)
            g_flat = p.grad.data.view(-1)
            qa_flat = state["q_exp_avg"].view(-1)
            qs_flat = state["q_exp_avg_sq"].view(-1)

            for i in range(num_chunks):
                s, e = i * CHUNK_SIZE, min((i + 1) * CHUNK_SIZE, p.numel())
                p_c, g_c = p_flat[s:e], g_flat[s:e]
                qa_c, qs_c = qa_flat[s:e], qs_flat[s:e]
                sa_c = state["scale_exp_avg"][i:i+1]
                ss_c = state["scale_exp_avg_sq"][i:i+1]

                # Dequantize
                exp_avg = qa_c.to(torch.bfloat16) * sa_c
                exp_avg_sq = qs_c.to(torch.bfloat16) * ss_c

                # Update EMA
                exp_avg.mul_(beta1).add_(g_c, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_c, g_c, value=1 - beta2)

                # Second-moment safety bound
                min_sq = bias_correction2 * (exp_avg / bias_correction1).pow(2)
                exp_avg_sq = torch.max(exp_avg_sq, min_sq)
                del min_sq

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])

                # Quantize exp_avg [-7, 7]
                max_a = torch.max(torch.abs(exp_avg)).unsqueeze(0)
                scale_a = (max_a / 7.0).clamp_(min=1e-12)
                qa_c.copy_(torch.clamp(torch.round(exp_avg / scale_a), -7, 7).to(torch.int8))
                sa_c.copy_(scale_a)

                # Quantize exp_avg_sq [0, 15]
                max_s = torch.max(exp_avg_sq).unsqueeze(0)
                scale_s = (max_s / 15.0).clamp_(min=1e-12)
                qs_c.copy_(torch.clamp(torch.round(exp_avg_sq / scale_s), 0, 15).to(torch.uint8))
                ss_c.copy_(scale_s)

                # Apply update
                p_c.mul_(1 - lr * group["weight_decay"])
                p_c.addcdiv_(exp_avg, denom, value=-step_size)
            return

        # Non-massive: original path
        exp_avg = state["q_exp_avg"].to(torch.bfloat16) * state["scale_exp_avg"]
        exp_avg_sq = state["q_exp_avg_sq"].to(torch.bfloat16) * state["scale_exp_avg_sq"]

        exp_avg.mul_(beta1).add_(p.grad.data, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(p.grad.data, p.grad.data, value=1 - beta2)

        min_exp_avg_sq = bias_correction2 * (exp_avg / bias_correction1).pow(2)
        exp_avg_sq = torch.max(exp_avg_sq, min_exp_avg_sq)

        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])

        # Quantize exp_avg
        if p.ndim >= 2:
            max_avg = torch.amax(torch.abs(exp_avg), dim=-1, keepdim=True)
        elif p.ndim == 1:
            max_avg = torch.max(torch.abs(exp_avg)).unsqueeze(0)
        else:
            max_avg = torch.abs(exp_avg)
        scale_avg = (max_avg / 7.0).clamp_(min=1e-12)
        state["q_exp_avg"] = torch.clamp(torch.round(exp_avg / scale_avg), -7, 7).to(torch.int8)
        state["scale_exp_avg"] = scale_avg

        # Quantize exp_avg_sq
        if p.ndim >= 2:
            max_avg_sq = torch.amax(exp_avg_sq, dim=-1, keepdim=True)
        elif p.ndim == 1:
            max_avg_sq = torch.max(exp_avg_sq).unsqueeze(0)
        else:
            max_avg_sq = exp_avg_sq.clone()
        scale_avg_sq = (max_avg_sq / 15.0).clamp_(min=1e-12)
        state["q_exp_avg_sq"] = torch.clamp(torch.round(exp_avg_sq / scale_avg_sq), 0, 15).to(torch.uint8)
        state["scale_exp_avg_sq"] = scale_avg_sq

        p.data.mul_(1 - lr * group["weight_decay"])
        p.data.addcdiv_(exp_avg, denom, value=-step_size)

