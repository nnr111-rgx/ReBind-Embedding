import numpy as np
from scipy.stats import pearsonr, spearmanr


def fit_linear(raw, true):
    x = np.asarray(raw, dtype=np.float64)
    y = np.asarray(true, dtype=np.float64)

    design = np.stack(
        [x, np.ones_like(x)],
        axis=1,
    )

    coef, _, _, _ = np.linalg.lstsq(
        design,
        y,
        rcond=None,
    )

    return float(coef[0]), float(coef[1])


def apply_linear(raw, coef):
    scale, bias = coef
    return scale * raw + bias


def distance_metrics(true, pred):
    y = np.asarray(true, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)

    err = p - y

    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "pearson": float(pearsonr(y, p).statistic),
        "spearman": float(spearmanr(y, p).statistic),
    }
