"""
Polar Factor
=============

Newton-Schulz polar decomposition used by Aurora and Riemannian Aurora.
"""

import torch

@torch.no_grad()
def polar(G: torch.Tensor, steps: int = 12, eps: float = 1e-7) -> torch.Tensor:
    """Polar factor via 12-step simple-quintic Newton-Schulz.

    Maps all non-zero singular values of G to 1. Uses the simple-quintic
    iteration p(σ) = 2σ − 1.5σ³ + 0.5σ⁵ with cubic convergence rate.
    12 iterations suffice for bf16 precision.

    Args:
        G: Input matrix of shape [..., m, n].
        steps: Number of Newton-Schulz iterations (default 12).
        eps: Numerical stability epsilon for spectral norm.

    Returns:
        polar(G) of the same shape, in bfloat16.
    """
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm <= 1 so the iteration converges to polar.
    # Guard against zero-gradient or underflowing input to avoid NaN/Inf updates.
    norm = X.norm(dim=(-2, -1), keepdim=True)
    X = torch.where(norm < eps, torch.zeros_like(X), X / (norm + eps))

    # Simple-quintic coefficients: p(σ) = aσ + bσ³ + cσ⁵ with σ=1 super-attracting.
    a, b, c = 2.0, -1.5, 0.5
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


