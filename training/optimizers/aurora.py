"""
Aurora Optimizers
==================

Vanilla Aurora and Riemannian Aurora leverage-aware spectral optimizers
for rectangular matrix parameters.
"""

import math
import torch

from .polar import polar


# ─── Helper: Balanced Stiefel solver ─────────────────────────────────────────

@torch.no_grad()
def _solve_row_norm_multipliers(
    U: torch.Tensor, r: float, b: torch.Tensor,
    max_iter: int = 20, eps: float = 1e-7,
) -> torch.Tensor:
    """Approximately solve (r I − P ∘ P) λ = b, where P = U U^T."""
    h_sq = U.pow(2).sum(dim=-1).pow(2)
    reg = (h_sq.max() - r + 1e-3).clamp_min(0.0).item()
    r_eff = r + reg

    def matvec(v):
        T = U.mT @ (v.unsqueeze(-1) * U)
        return r_eff * v - (U @ T * U).sum(dim=-1)

    x = torch.zeros_like(b)
    res = b.clone()
    p = res.clone()
    rs_old = (res * res).sum()
    b_norm = b.norm().clamp_min(1e-12)

    for _ in range(max_iter):
        Ap = matvec(p)
        denom = (p * Ap).sum()
        if denom < 1e-30:
            break

        alpha = rs_old / denom
        x = x + alpha * p
        res = res - alpha * Ap

        rs_new = (res * res).sum()
        if not rs_new.isfinite():
            break
        if rs_new.sqrt() < 1e-8 * b_norm:
            break

        p = res + (rs_new / rs_old.clamp_min(1e-30)) * p
        rs_old = rs_new

    return x if x.isfinite().all() else torch.zeros_like(b)


@torch.no_grad()
def _riemannian_balanced_polar(
    G: torch.Tensor,
    outer_steps: int = 3,
    cg_steps: int = 20,
    riemannian_eta: float = 0.1,
    retraction_steps: int = 2,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Approximate balanced-Stiefel update aligned with G."""
    if G.ndim != 2:
        raise ValueError(f"expected 2D matrix, got shape {tuple(G.shape)}")

    transposed = G.size(-2) < G.size(-1)
    if transposed:
        G = G.mT.contiguous()

    G32 = G.to(torch.float32)
    m, n = G32.shape
    r = n / m
    target_row_norm = math.sqrt(r)

    # Initial point: polar update.
    U = polar(G32).to(torch.float32)

    for _ in range(outer_steps):
        # Stiefel correction.
        UtG = U.mT @ G32
        B = 0.5 * (UtG + UtG.mT)

        # Row-norm correction RHS.
        q = (G32 * U).sum(dim=-1) - (U @ B * U).sum(dim=-1)
        q = q - q.mean()

        # Solve for row-norm Lagrange multipliers.
        lam = _solve_row_norm_multipliers(U, r, q, max_iter=cg_steps, eps=eps)
        lam = lam - lam.mean()

        # Tangent projection: Z = G − U S − D U.
        S = B - U.mT @ (lam.unsqueeze(-1) * U)
        Z = G32 - U @ S - lam.unsqueeze(-1) * U

        if not Z.isfinite().all():
            break

        # Riemannian ascent step.
        Y = U + riemannian_eta * Z

        # Approximate retraction by alternating row normalization and polar.
        for _ in range(retraction_steps):
            row_norm = Y.norm(dim=-1, keepdim=True).clamp_min(eps)
            Y = Y * (target_row_norm / row_norm)
            Y = polar(Y).to(torch.float32)

        U = Y

    return (U.mT.contiguous() if transposed else U).to(G.dtype)


# ─── Public API ──────────────────────────────────────────────────────────────

@torch.no_grad()
def aurora(
    W: torch.Tensor,
    G: torch.Tensor,
    momentum: torch.Tensor,
    eta: float = 0.05,
    weight_decay: float = 0.025,
    mu: float = 0.95,
    nesterov: bool = True,
    pp_iterations: int = 2,
    pp_beta: float = 0.5,
    eps: float = 1e-7,
    update_clip: float = 0.0,
) -> torch.Tensor:
    """Vanilla Aurora leverage-aware optimizer for rectangular matrices.

    Args:
        W: Weight parameter tensor (2D, modified in-place).
        G: Gradient tensor (same shape as W).
        momentum: Momentum buffer (same shape as W, modified in-place).
        eta: Learning rate.
        weight_decay: Decoupled weight decay coefficient.
        mu: Momentum coefficient (must be in (0, 1)).
        nesterov: Whether to use Nesterov look-ahead.
        pp_iterations: Number of diagonal refinement steps.
        pp_beta: Damping exponent for row-norm correction.
        eps: Numerical stability epsilon.

    Returns:
        Updated W (same tensor, modified in-place).
    """
    if W.ndim != 2:
        raise ValueError(f"aurora expects 2D weight tensors, got shape {tuple(W.shape)}")
    if G.shape != W.shape:
        raise ValueError(f"G shape {tuple(G.shape)} must match W shape {tuple(W.shape)}")
    if momentum.shape != W.shape:
        raise ValueError(f"momentum shape {tuple(momentum.shape)} must match W shape {tuple(W.shape)}")
    if not (0.0 < mu < 1.0):
        raise ValueError(f"mu must be in (0, 1), got {mu}")
    if eta <= 0.0:
        raise ValueError(f"eta must be positive, got {eta}")
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")
    if pp_iterations < 1:
        raise ValueError(f"pp_iterations must be >= 1, got {pp_iterations}")
    if pp_beta <= 0.0:
        raise ValueError(f"pp_beta must be positive, got {pp_beta}")

    # SGD-momentum (Nesterov by default).
    momentum.lerp_(G, 1 - mu)
    # Clone before using in-place lerp_ to avoid corrupting G (which is p.grad).
    update = G.clone().lerp_(momentum, mu) if nesterov else momentum.clone()

    # Aurora's leverage-uniform polar via diagonal preconditioning.
    m, n = update.size(-2), update.size(-1)
    if m == n:
        # Square: standard polar (no leverage freedom to exploit).
        update = polar(update)
    else:
        # For wide G, transpose to tall, apply, transpose back.
        transposed = m < n
        if transposed:
            update = update.mT
            m, n = n, m
        G32 = update.to(torch.float32)
        target_row_sq = n / m
        row_norm = G32.norm(dim=-1, keepdim=True).clamp_(min=eps)
        D = 1.0 / row_norm
        for k in range(pp_iterations):
            U = polar(D * G32)
            if k < pp_iterations - 1:
                row_sq = U.to(torch.float32).pow(2).sum(dim=-1, keepdim=True).clamp_(min=eps * eps)
                D = D * (target_row_sq / row_sq).pow(pp_beta)
        update = U.mT if transposed else U

    # Spectral aspect-ratio scaling (Muon convention).
    update *= max(1, G.size(-2) / G.size(-1)) ** 0.5
    
    if update_clip > 0.0:
        update.clamp_(-update_clip, update_clip)

    if not update.isfinite().all():
        raise RuntimeError(
            f"aurora produced non-finite update for parameter of shape {tuple(W.shape)}. "
            "Check for NaN/Inf in gradients or an ill-conditioned weight matrix."
        )
    # Decoupled weight decay then apply.
    W.mul_(1 - eta * weight_decay)
    W.add_(update, alpha=-eta)
    return W


@torch.no_grad()
def riemannian_aurora(
    W: torch.Tensor,
    G: torch.Tensor,
    momentum: torch.Tensor,
    eta: float = 0.05,
    weight_decay: float = 0.025,
    mu: float = 0.95,
    nesterov: bool = True,
    outer_steps: int = 3,
    cg_steps: int = 20,
    riemannian_eta: float = 0.1,
    retraction_steps: int = 2,
    eps: float = 1e-7,
    update_clip: float = 0.0,
) -> torch.Tensor:
    """Riemannian leverage-aware polar update optimizer for rectangular matrices.

    Like aurora() but uses the Riemannian balanced-Stiefel manifold for
    higher-quality leverage-uniform updates at extra compute cost.

    Args:
        W: Weight parameter tensor (2D, modified in-place).
        G: Gradient tensor (same shape as W).
        momentum: Momentum buffer (same shape as W, modified in-place).
        eta: Learning rate.
        weight_decay: Decoupled weight decay coefficient.
        mu: Momentum coefficient.
        nesterov: Whether to use Nesterov look-ahead.
        outer_steps: Number of Riemannian outer iterations.
        cg_steps: Max CG steps for Lagrange multiplier solve.
        riemannian_eta: Step size for tangent-space ascent.
        retraction_steps: Number of retraction (normalize + polar) steps.
        eps: Numerical stability epsilon.

    Returns:
        Updated W (same tensor, modified in-place).
    """
    # SGD-momentum (Nesterov by default).
    momentum.lerp_(G, 1 - mu)
    # Clone before using in-place lerp_ to avoid corrupting G (which is p.grad)
    # and clone momentum when not using Nesterov to prevent aliasing issues.
    update = G.clone().lerp_(momentum, mu) if nesterov else momentum.clone()

    # Riemannian-Aurora balanced polar update.
    m, n = update.size(-2), update.size(-1)
    if m == n:
        # Square: no leverage freedom to exploit.
        update = polar(update)
    else:
        update = _riemannian_balanced_polar(
            update,
            outer_steps=outer_steps,
            cg_steps=cg_steps,
            riemannian_eta=riemannian_eta,
            retraction_steps=retraction_steps,
            eps=eps,
        )

    # Spectral aspect-ratio scaling (Muon convention).
    update *= max(1, G.size(-2) / G.size(-1)) ** 0.5

    if update_clip > 0.0:
        update.clamp_(-update_clip, update_clip)

    # Decoupled weight decay then apply.
    W.mul_(1 - eta * weight_decay)
    W.add_(update, alpha=-eta)
    return W
