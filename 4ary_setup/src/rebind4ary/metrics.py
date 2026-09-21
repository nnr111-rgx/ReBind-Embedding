from __future__ import annotations

import numpy as np


def pearson(x, y) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(x):
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def spearman(x, y) -> float:
    return pearson(_rankdata(x), _rankdata(y))


def fit_linear(pred, target):
    x = np.asarray(pred, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    A = np.stack([x, np.ones_like(x)], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[0]), float(coef[1])


def distance_metrics(pred, target):
    p = np.asarray(pred, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    e = p - y
    return {
        "n": int(len(y)),
        "rmse": float(np.sqrt(np.mean(e**2))),
        "mae": float(np.mean(np.abs(e))),
        "pearson": pearson(p, y),
        "spearman": spearman(p, y),
    }


def binary_metrics_from_llr(llr, bits, n_bins: int = 15):
    llr = np.asarray(llr, dtype=np.float64)
    bits = np.asarray(bits, dtype=np.float64)
    hard = (llr < 0).astype(np.float64)
    ber = float(np.mean(hard != bits))
    prob1 = 1.0 / (1.0 + np.exp(np.clip(llr, -60, 60)))
    eps = 1e-12
    bce = float(np.mean(-(bits * np.log(prob1 + eps) + (1 - bits) * np.log(1 - prob1 + eps))))
    brier = float(np.mean((prob1 - bits) ** 2))
    conf = np.maximum(prob1, 1.0 - prob1)
    correct = (hard == bits).astype(np.float64)
    bins = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            m = (conf >= bins[i]) & (conf <= bins[i + 1])
        else:
            m = (conf >= bins[i]) & (conf < bins[i + 1])
        if np.any(m):
            ece += float(np.mean(m)) * abs(float(np.mean(correct[m])) - float(np.mean(conf[m])))
    frame_error = float(np.mean(np.any(hard != bits, axis=1))) if bits.ndim == 2 else float("nan")
    return {
        "ber": ber,
        "bit_accuracy": 1.0 - ber,
        "frame_error_rate": frame_error,
        "bce": bce,
        "brier": brier,
        "ece": float(ece),
        "mean_abs_llr": float(np.mean(np.abs(llr))),
    }


def correlation_summary(x):
    x = np.asarray(x, dtype=np.float64)
    c = np.corrcoef(x, rowvar=False)
    d = c.shape[0]
    off = c[~np.eye(d, dtype=bool)]
    eig = np.linalg.eigvalsh(np.cov(x, rowvar=False))
    pos = eig[eig > 1e-12]
    return {
        "mean_abs_offdiag_correlation": float(np.mean(np.abs(off))),
        "max_abs_offdiag_correlation": float(np.max(np.abs(off))),
        "covariance_condition_number": float(np.max(pos) / np.min(pos)) if len(pos) else float("inf"),
    }
