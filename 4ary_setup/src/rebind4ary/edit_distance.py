from __future__ import annotations

import numpy as np

try:
    from rapidfuzz.distance import Levenshtein
except Exception:
    Levenshtein = None


def _to_bytes(x) -> bytes:
    a = np.asarray(x, dtype=np.uint8).reshape(-1)
    return bytes(a.tolist())


def edit_distance(a, b) -> int:
    if Levenshtein is not None:
        return int(Levenshtein.distance(_to_bytes(a), _to_bytes(b)))
    aa = list(map(int, np.asarray(a).reshape(-1)))
    bb = list(map(int, np.asarray(b).reshape(-1)))
    prev = list(range(len(bb) + 1))
    for i, ai in enumerate(aa, start=1):
        cur = [i] + [0] * len(bb)
        for j, bj in enumerate(bb, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ai != bj))
        prev = cur
    return int(prev[-1])
