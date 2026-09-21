from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import binary_metrics_from_llr, correlation_summary, distance_metrics, fit_linear


def evaluate_geometry(encoded, train_calibration: tuple[float, float] | None = None):
    d = np.linalg.norm(encoded["z_noisy"] - encoded["z_clean"], axis=1)
    target = encoded["edit_distance"]
    if train_calibration is None:
        a, b = fit_linear(d, target)
    else:
        a, b = train_calibration
    pred = a * d + b
    out = distance_metrics(pred, target)
    out.update({"calibration_slope": float(a), "calibration_intercept": float(b)})
    return out, pred, (a, b)


def evaluate_detector(encoded, whitener, calibrator, llr_clip: float = 20.0):
    zc = encoded["z_clean"]
    zn = encoded["z_noisy"]
    mean = whitener.mean.detach().cpu().numpy()
    W = whitener.matrix.detach().cpu().numpy()
    wc = (zc - mean) @ W.T
    wn = (zn - mean) @ W.T
    w = calibrator.weight.detach().cpu().numpy()
    b = calibrator.bias.detach().cpu().numpy()
    llr = -(wn * w + b)
    llr = np.clip(llr, -llr_clip, llr_clip)
    overall = binary_metrics_from_llr(llr, encoded["bits"])
    overall.update({"raw_residual": correlation_summary(zn - zc), "white_residual": correlation_summary(wn - wc)})
    rows = []
    for cid in np.unique(encoded["condition_id"]):
        m = encoded["condition_id"] == cid
        met = binary_metrics_from_llr(llr[m], encoded["bits"][m])
        p = encoded["params"][m][0]
        rows.append({"condition_id": int(cid), "p_ins": float(p[0]), "p_del": float(p[1]), "p_sub": float(p[2]), "n": int(m.sum()), **met})
    return overall, pd.DataFrame(rows), llr, wc, wn
