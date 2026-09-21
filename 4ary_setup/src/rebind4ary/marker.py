from __future__ import annotations

import numpy as np

from .config import N_BITS, NP, T


def build_marker_patterns(T_value: int = T, Np: int = NP) -> tuple[np.ndarray, np.ndarray]:
    mp1 = -np.ones(T_value, dtype=np.int64)
    mp2 = -np.ones(T_value, dtype=np.int64)
    mp1[Np::Np] = 1
    mp1[Np + 1 :: Np] = 0
    mp2[Np::Np] = 1
    mp2[Np + 1 :: Np] = 0
    return mp1, mp2


def payload_positions(T_value: int = T, Np: int = NP) -> tuple[np.ndarray, np.ndarray]:
    mp1, mp2 = build_marker_patterns(T_value, Np)
    return np.where(mp1 == -1)[0], np.where(mp2 == -1)[0]


def marker_encode(bits, T_value: int = T, Np: int = NP) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.int64).reshape(-1)
    mp1, mp2 = build_marker_patterns(T_value, Np)
    j1 = np.where(mp1 == -1)[0]
    j2 = np.where(mp2 == -1)[0]
    need = len(j1) + len(j2)
    if len(bits) != need:
        raise ValueError(f"marker encoder requires {need} bits, got {len(bits)}")
    xb1 = mp1.copy()
    xb2 = mp2.copy()
    xb1[j1] = bits[: len(j1)]
    xb2[j2] = bits[len(j1) :]
    return (2 * xb1 + xb2).astype(np.int64)


def marker_decode_payload_symbols(symbols, T_value: int = T, Np: int = NP) -> np.ndarray:
    symbols = np.asarray(symbols, dtype=np.int64).reshape(-1)
    if len(symbols) != T_value:
        raise ValueError(f"expected {T_value} clean marker symbols, got {len(symbols)}")
    b1 = symbols // 2
    b2 = symbols % 2
    j1, j2 = payload_positions(T_value, Np)
    return np.concatenate([b1[j1], b2[j2]]).astype(np.int64)


def validate_default_marker() -> None:
    j1, j2 = payload_positions()
    if len(j1) + len(j2) != N_BITS:
        raise RuntimeError("default T/Np marker does not expose 204 payload bits")
