from __future__ import annotations

import numpy as np

from .config import CC_G, CC_K, MSG_LEN, NP, T


def random_message(msg_len=MSG_LEN, rng=None):
    rng = rng or np.random.default_rng()
    return rng.integers(0, 2, size=msg_len, dtype=np.int64)


def conv_encode_bits(msg_bits, g=CC_G, K=CC_K, terminate=True):
    msg_bits = np.asarray(msg_bits, dtype=np.int64).reshape(-1)
    seq = msg_bits.tolist() + ([0] * (K - 1) if terminate else [])
    state = [0] * (K - 1)
    out = []
    for bit in seq:
        reg = [int(bit)] + state
        for poly in g:
            taps = [(poly >> i) & 1 for i in range(K - 1, -1, -1)]
            acc = 0
            for r, t in zip(reg, taps):
                if t:
                    acc ^= r
            out.append(acc)
        state = [int(bit)] + state[:-1]
    return np.asarray(out, dtype=np.int64)


def build_marker_patterns(T=T, Np=NP):
    mp1 = -np.ones(T, dtype=np.int64)
    mp2 = -np.ones(T, dtype=np.int64)
    mp1[Np:T:Np] = 1
    mp1[Np + 1:T:Np] = 0
    mp2[Np:T:Np] = 1
    mp2[Np + 1:T:Np] = 0
    return mp1, mp2


def marcode(coded_bits, mp1, mp2):
    coded_bits = np.asarray(coded_bits, dtype=np.int64).reshape(-1)
    j1 = np.where(mp1 == -1)[0]
    j2 = np.where(mp2 == -1)[0]
    if coded_bits.size != j1.size + j2.size:
        raise ValueError(f"coded-bit length mismatch: {coded_bits.size} vs {j1.size + j2.size}")
    xb1 = np.asarray(mp1, dtype=np.int64).copy()
    xb2 = np.asarray(mp2, dtype=np.int64).copy()
    xb1[j1] = coded_bits[:j1.size]
    xb2[j2] = coded_bits[j1.size:]
    return (2 * xb1 + xb2).astype(np.int64)


def qary_to_binary(symbols):
    symbols = np.asarray(symbols, dtype=np.int64).reshape(-1)
    if symbols.size and (symbols.min() < 0 or symbols.max() > 3):
        raise ValueError("4-ary symbols must be in {0,1,2,3}")
    out = np.empty(2 * symbols.size, dtype=np.int64)
    out[0::2] = (symbols >> 1) & 1
    out[1::2] = symbols & 1
    return out
