from __future__ import annotations

import numpy as np


def ids_channel(symbols, p_ins, p_del, p_sub, vocab=4, rng=None, l_max=2):
    if vocab != 4:
        raise ValueError("Expected vocab=4")
    rng = rng or np.random.default_rng()
    symbols = np.asarray(symbols, dtype=np.int64).reshape(-1)
    if p_ins == 0:
        l_max = 0
    out = []
    for s in symbols.tolist():
        ins_num = 0
        rpi = int(rng.random() < p_ins)
        while ins_num < l_max and rpi == 1:
            out.append(int(rng.integers(0, vocab)))
            ins_num += 1
            rpi = int(rng.random() < p_ins)
        rpd = int(rng.random() < p_del)
        rps = int(rng.random() < p_sub)
        if rpd == 1:
            continue
        if rps == 0:
            out.append(int(s))
        else:
            sn = int(rng.integers(0, vocab))
            out.append(int((int(s) + sn) % vocab))
    return np.asarray(out, dtype=np.int64)
