from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class LogisticCalibrator:
    weight: torch.Tensor
    bias: torch.Tensor

    def to(self, device) -> "LogisticCalibrator":
        return LogisticCalibrator(self.weight.to(device), self.bias.to(device))

    def bit1_logit(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight + self.bias

    def llr_bit0_over_bit1(self, x: torch.Tensor) -> torch.Tensor:
        return -self.bit1_logit(x)

    def state_dict(self):
        return {"weight": self.weight.detach().cpu(), "bias": self.bias.detach().cpu()}


def fit_logistic_calibrator(observations, bits, steps: int = 30, l2: float = 1e-6) -> LogisticCalibrator:
    x = np.asarray(observations, dtype=np.float64)
    y = np.asarray(bits, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"observation shape {x.shape} != bit shape {y.shape}")
    d = x.shape[1]
    w = np.zeros(d, dtype=np.float64)
    p0 = np.clip(y.mean(axis=0), 1e-5, 1 - 1e-5)
    b = np.log(p0 / (1 - p0))
    for _ in range(int(steps)):
        eta = np.clip(x * w[None, :] + b[None, :], -40, 40)
        p = 1.0 / (1.0 + np.exp(-eta))
        r = p - y
        q = np.maximum(p * (1 - p), 1e-8)
        gw = np.sum(r * x, axis=0) + l2 * w
        gb = np.sum(r, axis=0) + l2 * b
        hww = np.sum(q * x * x, axis=0) + l2
        hbb = np.sum(q, axis=0) + l2
        hwb = np.sum(q * x, axis=0)
        det = np.maximum(hww * hbb - hwb * hwb, 1e-12)
        dw = (hbb * gw - hwb * gb) / det
        db = (-hwb * gw + hww * gb) / det
        w -= dw
        b -= db
        if max(float(np.max(np.abs(dw))), float(np.max(np.abs(db)))) < 1e-7:
            break
    return LogisticCalibrator(torch.tensor(w, dtype=torch.float32), torch.tensor(b, dtype=torch.float32))


def save_calibrator(path, calibrator: LogisticCalibrator, metadata: dict | None = None) -> None:
    torch.save({"calibrator": calibrator.state_dict(), "metadata": metadata or {}}, path)


def load_calibrator(path, device="cpu") -> LogisticCalibrator:
    obj = torch.load(path, map_location=device, weights_only=False)
    s = obj["calibrator"] if "calibrator" in obj else obj
    return LogisticCalibrator(s["weight"].to(device), s["bias"].to(device))
