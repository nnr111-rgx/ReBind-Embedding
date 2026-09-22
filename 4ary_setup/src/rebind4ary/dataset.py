from __future__ import annotations

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .coding import qary_to_binary
from .config import REBIND_MAX_LEN


class StageDataset(Dataset):
    def __init__(self, path, split, stage, max_len=REBIND_MAX_LEN):
        self.path = str(path)
        self.split = str(split)
        self.stage = str(stage)
        self.max_len = int(max_len)
        self._h5 = None
        with h5py.File(self.path, "r") as f:
            self.n = int(f[self.split]["coded_bits"].shape[0])

    def __len__(self):
        return self.n

    def _file(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r")
        return self._h5

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_h5"] = None
        return d

    def _pad_binary(self, qary, qary_len):
        bits = qary_to_binary(np.asarray(qary[:qary_len], dtype=np.int64))
        if bits.size > self.max_len:
            raise ValueError(f"binary-expanded sequence length {bits.size} exceeds max_len={self.max_len}")
        out = np.full(self.max_len, -1, dtype=np.int64)
        out[:bits.size] = bits
        return out

    def __getitem__(self, idx):
        g = self._file()[self.split]
        clean = np.asarray(g["clean_qary"][idx], dtype=np.int64)
        noisy = np.asarray(g[f"{self.stage}_noisy_qary"][idx], dtype=np.int64)
        noisy_len = int(g[f"{self.stage}_noisy_len"][idx])
        coded = np.asarray(g["coded_bits"][idx], dtype=np.float32)
        return (
            torch.from_numpy(self._pad_binary(clean, clean.size)),
            torch.from_numpy(self._pad_binary(noisy, noisy_len)),
            torch.from_numpy(coded),
        )
