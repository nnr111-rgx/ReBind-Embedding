from __future__ import annotations

import numpy as np


def ids_channel_matlab(
    symbols,
    p_ins: float,
    p_del: float,
    p_sub: float,
    rng: np.random.Generator | None = None,
    l_max: int = 2,
) -> np.ndarray:
    x = np.asarray(symbols, dtype=np.int64).reshape(-1)
    rng = rng or np.random.default_rng()
    if p_ins == 0:
        l_max = 0
    out: list[int] = []
    q = 4
    for s in x:
        ins_num = 0
        rpi = int(rng.random() < p_ins)
        while ins_num < l_max and rpi == 1:
            out.append(int(np.ceil(rng.random() * q) - 1))
            ins_num += 1
        rpd = int(rng.random() < p_del)
        rps = int(rng.random() < p_sub)
        if rpd == 1:
            continue
        if rps == 0:
            out.append(int(s))
        else:
            sn = int(np.ceil(q * rng.random()) - 1)
            out.append(int((int(s) + sn) % q))
    return np.asarray(out, dtype=np.int64)


def ids_channel_standard(
    symbols,
    p_ins: float,
    p_del: float,
    p_sub: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    x = np.asarray(symbols, dtype=np.int64).reshape(-1)
    rng = rng or np.random.default_rng()
    out: list[int] = []
    for s in x:
        if rng.random() < p_del:
            if rng.random() < p_ins:
                out.append(int(rng.integers(0, 4)))
            continue
        cur = int(s)
        if rng.random() < p_sub:
            cand = [v for v in range(4) if v != cur]
            cur = int(rng.choice(cand))
        out.append(cur)
        if rng.random() < p_ins:
            out.append(int(rng.integers(0, 4)))
    return np.asarray(out, dtype=np.int64)


def apply_ids(symbols, p_ins, p_del, p_sub, rng=None, mode: str = "matlab", l_max: int = 2):
    if mode == "matlab":
        return ids_channel_matlab(symbols, p_ins, p_del, p_sub, rng=rng, l_max=l_max)
    if mode == "standard":
        return ids_channel_standard(symbols, p_ins, p_del, p_sub, rng=rng)
    raise ValueError(f"unknown channel mode: {mode}")
