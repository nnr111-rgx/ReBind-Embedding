from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Whitener:
    mean: torch.Tensor
    matrix: torch.Tensor
    shrinkage: float
    eps_relative: float

    def to(self, device) -> "Whitener":
        return Whitener(self.mean.to(device), self.matrix.to(device), self.shrinkage, self.eps_relative)

    def transform(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.mean) @ self.matrix.T

    def state_dict(self) -> dict:
        return {
            "mean": self.mean.detach().cpu(),
            "matrix": self.matrix.detach().cpu(),
            "shrinkage": float(self.shrinkage),
            "eps_relative": float(self.eps_relative),
        }


def fit_whitener(residuals, shrinkage: float = 0.01, eps_relative: float = 1e-4) -> Whitener:
    x = np.asarray(residuals, dtype=np.float64)
    mean = x.mean(axis=0)
    xc = x - mean
    cov = (xc.T @ xc) / max(1, len(xc) - 1)
    diag_mean = float(np.trace(cov) / cov.shape[0])
    shr = (1.0 - shrinkage) * cov + shrinkage * diag_mean * np.eye(cov.shape[0])
    eps = max(1e-12, eps_relative * max(diag_mean, 1e-12))
    vals, vecs = np.linalg.eigh(shr + eps * np.eye(shr.shape[0]))
    vals = np.maximum(vals, eps)
    matrix = (vecs * (1.0 / np.sqrt(vals))[None, :]) @ vecs.T
    return Whitener(
        mean=torch.tensor(mean, dtype=torch.float32),
        matrix=torch.tensor(matrix, dtype=torch.float32),
        shrinkage=float(shrinkage),
        eps_relative=float(eps_relative),
    )


def save_whitener(path, whitener: Whitener, metadata: dict | None = None) -> None:
    torch.save({"whitener": whitener.state_dict(), "metadata": metadata or {}}, path)


def load_whitener(path, device="cpu") -> Whitener:
    obj = torch.load(path, map_location=device, weights_only=False)
    s = obj["whitener"] if "whitener" in obj else obj
    return Whitener(
        mean=s["mean"].to(device),
        matrix=s["matrix"].to(device),
        shrinkage=float(s.get("shrinkage", 0.0)),
        eps_relative=float(s.get("eps_relative", 0.0)),
    )
