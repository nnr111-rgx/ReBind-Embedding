from __future__ import annotations

import numpy as np

from .config import NP, T
from .marker import build_marker_patterns

def rfz(mp1: np.ndarray, mp2: np.ndarray, ps: float, T: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rho = []

    for k in range(T):
        a = int(mp1[k])
        b = int(mp2[k])

        if a == 0 and b == 0:
            rhok = [1.0, 0.0, 0.0, 0.0]
        elif a == 0 and b == 1:
            rhok = [0.0, 1.0, 0.0, 0.0]
        elif a == 1 and b == 0:
            rhok = [0.0, 0.0, 1.0, 0.0]
        elif a == 1 and b == 1:
            rhok = [0.0, 0.0, 0.0, 1.0]
        elif a == 0 and b == -1:
            rhok = [0.5, 0.5, 0.0, 0.0]
        elif a == 1 and b == -1:
            rhok = [0.0, 0.0, 0.5, 0.5]
        elif a == -1 and b == 0:
            rhok = [0.5, 0.0, 0.5, 0.0]
        elif a == -1 and b == 1:
            rhok = [0.0, 0.5, 0.0, 0.5]
        else:
            rhok = [0.25, 0.25, 0.25, 0.25]

        rho.append(rhok)

    rho = np.asarray(rho, dtype=np.float64)

    q = 4
    f = (ps / (q - 1)) * np.ones((q, q), dtype=np.float64)
    for i in range(q):
        f[i, i] = 1.0 - ps

    zeta = np.zeros((T, q), dtype=np.float64)
    for j in range(T):
        for i in range(q):
            for iprim in range(q):
                zeta[j, i] += rho[j, iprim] * f[iprim, i]

    return rho, f, zeta


def log_add(cur: float, cand: float, log_map_vec: np.ndarray, delta_step: float) -> float:
    t = abs(cand - cur)
    loc = min(len(log_map_vec) - 1, int(np.floor(t / delta_step)))
    return max(cur, cand) + float(log_map_vec[loc])


def FB_decode(
    y: np.ndarray,
    T: int,
    mu: np.ndarray,
    rho: np.ndarray,
    f: np.ndarray,
    zeta: np.ndarray,
    mp1: np.ndarray,
    mp2: np.ndarray,
    log_map_vec: np.ndarray,
    delta_step: float,
    l_max: int,
) -> np.ndarray:
    Mval = 1e7
    R = len(y)
    q = 4
    no = 2 + l_max

    log_alf = -Mval * np.ones((T + 1, R + 2 * l_max + 3), dtype=np.float64)
    log_bet = -Mval * np.ones((T + 1, R + 2 * l_max + 3), dtype=np.float64)

    mu00 = mu[0, 0]

    log_alf[0, no] = 0.0
    if mu00 == 0:
        log_alf[1 : T + 1, no] = -Mval
    else:
        log_alf[1 : T + 1, no] = np.log(mu00) * np.arange(1, T + 1)

    log_bet[T, R + no] = 0.0
    if mu00 > 0:
        log_bet[0:T, R + no] = 0.0
    else:
        log_bet[0:T, R + no] = -Mval

    for k in range(1, T + 1):
        for n in range(1, R + 1):
            for l in range(l_max + 1):
                for b in (0, 1):
                    prev_col = n - l - b + no
                    coef1 = mu[l, b] * (q ** (-l)) * (zeta[k - 1, y[n - 1]] ** b)
                    if coef1 == 0:
                        calf = -Mval
                    else:
                        calf = np.log(coef1) + log_alf[k - 1, prev_col]
                    log_alf[k, n + no] = log_add(log_alf[k, n + no], calf, log_map_vec, delta_step)

    for k in range(T - 1, -1, -1):
        for n in range(R - 1, -1, -1):
            for l in range(l_max + 1):
                for b in (0, 1):
                    zky = zeta[k, y[min(R - 1, n + l)]]
                    coef1 = mu[l, b] * (q ** (-l)) * (zky ** b)
                    if coef1 == 0:
                        cbet = -Mval
                    else:
                        cbet = np.log(coef1) + log_bet[k + 1, n + l + b + no]
                    log_bet[k, n + no] = log_add(log_bet[k, n + no], cbet, log_map_vec, delta_step)

    log_ap_prob = -Mval * np.ones((T, q), dtype=np.float64)

    for k in range(1, T + 1):
        for a in range(q):
            for l in range(l_max + 1):
                for b in (0, 1):
                    Upper_val = min(R, (l_max + 1) * (k - 1))
                    for n in range(Upper_val + 1):
                        fbb = f[a, y[min(R - 1, n + l)]] ** b
                        coef1 = mu[l, b] * (q ** (-l)) * fbb
                        if coef1 == 0:
                            log_add_term = -Mval
                        else:
                            log_add_term = np.log(coef1) + log_alf[k - 1, n + no] + log_bet[k, n + l + b + no]
                        log_ap_prob[k - 1, a] = log_add(log_ap_prob[k - 1, a], log_add_term, log_map_vec, delta_step)

    mmap = np.max(log_ap_prob)
    joint_prob = rho * np.exp(log_ap_prob - mmap)

    bit_prob1 = np.zeros((2, T), dtype=np.float64)
    bit_prob2 = np.zeros((2, T), dtype=np.float64)

    for k in range(T):
        bit_prob1[0, k] = joint_prob[k, 0] + joint_prob[k, 1]
        bit_prob1[1, k] = joint_prob[k, 2] + joint_prob[k, 3]
        bit_prob2[0, k] = joint_prob[k, 0] + joint_prob[k, 2]
        bit_prob2[1, k] = joint_prob[k, 1] + joint_prob[k, 3]

    j1 = np.where(mp1 == -1)[0]
    j2 = np.where(mp2 == -1)[0]

    eps = 1e-300
    bit_llrs = np.concatenate([
        np.log(np.maximum(bit_prob1[1, j1], eps) / np.maximum(bit_prob1[0, j1], eps)),
        np.log(np.maximum(bit_prob2[1, j2], eps) / np.maximum(bit_prob2[0, j2], eps)),
    ])

    p_ub_1 = np.exp(bit_llrs) / (1.0 + np.exp(bit_llrs))
    return p_ub_1

def oracle_bit_llr(
    y,
    p_ins: float,
    p_del: float,
    p_sub: float,
    T_value: int = T,
    Np: int = NP,
    l_max: int = 2,
    clip: float | None = 100.0,
) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    if p_ins == 0:
        l_max = 0
    mu = np.array(
        [[p_del, 1.0 - p_del - p_ins], [0.0, 0.0], [p_ins, 0.0]],
        dtype=np.float64,
    )
    mu = mu / np.sum(mu)
    mp1, mp2 = build_marker_patterns(T_value, Np)
    rho, f, zeta = rfz(mp1, mp2, p_sub, T_value)
    delta_step = 0.01
    xx = np.arange(0.0, 10.0 + 1e-12, delta_step)
    log_map_vec = np.log1p(np.exp(-xx))
    p1 = FB_decode(
        y=y,
        T=T_value,
        mu=mu,
        rho=rho,
        f=f,
        zeta=zeta,
        mp1=mp1,
        mp2=mp2,
        log_map_vec=log_map_vec,
        delta_step=delta_step,
        l_max=l_max,
    )
    eps = 1e-300
    llr = -np.log(np.maximum(p1, eps) / np.maximum(1.0 - p1, eps))
    if clip is not None:
        llr = np.clip(llr, -float(clip), float(clip))
    return llr.astype(np.float32)
