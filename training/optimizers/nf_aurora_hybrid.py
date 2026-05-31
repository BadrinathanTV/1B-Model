import math
import torch
from .nf_aurora import _aurora_polar
from .kernels.sf_adamw_4bit import step_sf_adamw_4bit

class NFAuroraHybrid(torch.optim.Optimizer):
    """NF-Aurora Hybrid: Leverage-Aware Aurora for 2D params + True NF-AdamW (4-bit Triton) for 1D params."""
    def __init__(self, params, lr=0.01, betas=(0.9, 0.99), momentum=0.95,
                 weight_decay=0.05, warmup_steps=2000, eta_scale=0.2,
                 pp_iterations=2, pp_beta=0.5, nesterov=True, eps=1e-7):
        defaults = dict(
            lr=lr, betas=betas, momentum=momentum, weight_decay=weight_decay,
            warmup_steps=warmup_steps, eta_scale=eta_scale,
            pp_iterations=pp_iterations, pp_beta=pp_beta,
            nesterov=nesterov, eps=eps,
            k=0, train_mode=True, weight_sum=0.0,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def eval(self):
        """Switch to evaluation mode for Schedule-Free parameters."""
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
        """Switch back to training mode for Schedule-Free parameters."""
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
            beta = group["betas"][0]
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

                if p.ndim == 2 and use_aurora:
                    self._step_2d(p, p.grad, mu, nesterov, pp_iters, pp_b, eps, eta_scale, lr, beta, decay, ckp1)
                else:
                    self._step_1d(p, p.grad, group, lr, beta, decay, ckp1)

            group["k"] = k + 1
        return loss

    def _step_2d(self, p, grad, mu, nesterov, pp_iters, pp_b, eps, eta_scale, lr, beta, decay, ckp1):
        """2D matrix: Leverage-Aware Aurora + Schedule-Free."""
        state = self.state[p]
        if "z" not in state:
            state["z"] = p.clone()
            state["mom"] = torch.zeros_like(p)
            
        z = state["z"]
        mom = state["mom"]
        
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

    def _step_1d(self, p, grad, group, lr, beta, decay, ckp1):
        """1D parameter: True Schedule-Free AdamW via 4-bit Triton kernel."""
        state = self.state[p]
        if "z" not in state:
            state["z"] = p.clone()
            
        # The Triton kernel expects `z` and applies the Schedule-Free extrapolation internally.
        step_sf_adamw_4bit(p, grad, state["z"], state, lr, beta, decay, ckp1)
