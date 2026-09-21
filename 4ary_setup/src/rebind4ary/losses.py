from __future__ import annotations

import torch
import torch.nn.functional as F


def triplet_regression_loss(
    pred_ap: torch.Tensor,
    pred_an: torch.Tensor,
    pred_pn: torch.Tensor,
    d_ap: torch.Tensor,
    d_an: torch.Tensor,
    d_pn: torch.Tensor,
    rank_margin: float = 0.2,
    lambda_rank: float = 0.2,
):
    reg = (
        F.mse_loss(pred_ap, d_ap)
        + F.mse_loss(pred_an, d_an)
        + F.mse_loss(pred_pn, d_pn)
    ) / 3.0
    rank = F.relu(float(rank_margin) + pred_ap - pred_an).mean()
    return reg + float(lambda_rank) * rank, reg, rank


def pairwise_rank_loss(pred: torch.Tensor, target: torch.Tensor, margin: float = 0.05) -> torch.Tensor:
    if pred.numel() < 2:
        return pred.new_tensor(0.0)
    p1 = pred[:-1]
    p2 = pred[1:]
    t1 = target[:-1]
    t2 = target[1:]
    sign = torch.sign(t2 - t1)
    keep = sign.ne(0)
    if not keep.any():
        return pred.new_tensor(0.0)
    return F.relu(float(margin) - sign[keep] * (p2[keep] - p1[keep])).mean()


def offdiag_correlation_loss(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x - x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    y = x / std
    c = (y.T @ y) / max(1, y.size(0))
    off = c - torch.diag(torch.diag(c))
    return off.square().mean()
