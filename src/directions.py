from __future__ import annotations

import torch


def difference_in_means(pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    """pos/neg: [n, d] or [d]. Returns unit vector pos_mean - neg_mean."""
    if pos.ndim == 1:
        pos = pos.unsqueeze(0)
    if neg.ndim == 1:
        neg = neg.unsqueeze(0)
    v = pos.float().mean(0) - neg.float().mean(0)
    return v / (v.norm() + 1e-8)


def random_unit(dim: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(dim, generator=g)
    return v / (v.norm() + 1e-8)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    return float((a @ b) / (a.norm() * b.norm() + 1e-8))
