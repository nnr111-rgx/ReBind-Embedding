from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import PAD


def pad_batch(seqs, pad_value: int = PAD):
    lengths = torch.tensor([len(x) for x in seqs], dtype=torch.long)
    max_len = int(lengths.max().item()) if len(seqs) else 0
    out = torch.full((len(seqs), max_len), int(pad_value), dtype=torch.long)
    for i, x in enumerate(seqs):
        t = torch.as_tensor(np.asarray(x), dtype=torch.long)
        out[i, : len(t)] = t
    return out, lengths


class TripletH5Dataset(Dataset):
    def __init__(self, path: str | Path, split: str):
        self.path = str(path)
        self.split = split
        self._h5 = None
        with h5py.File(self.path, "r") as f:
            if split not in f:
                raise KeyError(f"split {split!r} not found in {self.path}")
            self.n = int(f[split].attrs["n"])

    def _file(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r")
        return self._h5

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        g = self._file()[self.split]
        return {
            "anchor": np.asarray(g["anchor"][idx], dtype=np.int64),
            "positive": np.asarray(g["positive"][idx], dtype=np.int64),
            "negative": np.asarray(g["negative"][idx], dtype=np.int64),
            "d_ap": float(g["d_ap"][idx]),
            "d_an": float(g["d_an"][idx]),
            "d_pn": float(g["d_pn"][idx]),
        }


class IDSH5Dataset(Dataset):
    def __init__(self, path: str | Path, split: str, teacher_path: str | Path | None = None):
        self.path = str(path)
        self.split = split
        self.teacher_path = str(teacher_path) if teacher_path is not None else None
        self._h5 = None
        self._teacher = None
        with h5py.File(self.path, "r") as f:
            if split not in f:
                raise KeyError(f"split {split!r} not found in {self.path}")
            self.n = int(f[split].attrs["n"])
        if self.teacher_path is not None:
            with h5py.File(self.teacher_path, "r") as f:
                if split not in f:
                    raise KeyError(f"teacher split {split!r} not found")
                if int(f[split].attrs["n"]) != self.n:
                    raise ValueError("teacher and IDS split lengths differ")

    def _file(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r")
        return self._h5

    def _teacher_file(self):
        if self.teacher_path is None:
            return None
        if self._teacher is None:
            self._teacher = h5py.File(self.teacher_path, "r")
        return self._teacher

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        g = self._file()[self.split]
        item = {
            "sample_id": int(g["sample_id"][idx]),
            "bits": np.asarray(g["bits"][idx], dtype=np.int64),
            "clean": np.asarray(g["clean"][idx], dtype=np.int64),
            "noisy": np.asarray(g["noisy"][idx], dtype=np.int64),
            "edit_distance": float(g["edit_distance"][idx]),
            "condition_id": int(g["condition_id"][idx]),
            "p_ins": float(g["p_ins"][idx]),
            "p_del": float(g["p_del"][idx]),
            "p_sub": float(g["p_sub"][idx]),
        }
        tf = self._teacher_file()
        if tf is not None:
            item["teacher_llr"] = np.asarray(tf[self.split]["teacher_llr"][idx], dtype=np.float32)
        return item


def collate_triplet(batch):
    a, al = pad_batch([b["anchor"] for b in batch])
    p, pl = pad_batch([b["positive"] for b in batch])
    n, nl = pad_batch([b["negative"] for b in batch])
    return {
        "anchor": a,
        "anchor_len": al,
        "positive": p,
        "positive_len": pl,
        "negative": n,
        "negative_len": nl,
        "d_ap": torch.tensor([b["d_ap"] for b in batch], dtype=torch.float32),
        "d_an": torch.tensor([b["d_an"] for b in batch], dtype=torch.float32),
        "d_pn": torch.tensor([b["d_pn"] for b in batch], dtype=torch.float32),
    }


def collate_ids(batch):
    clean, clean_len = pad_batch([b["clean"] for b in batch])
    noisy, noisy_len = pad_batch([b["noisy"] for b in batch])
    out = {
        "sample_id": torch.tensor([b["sample_id"] for b in batch], dtype=torch.long),
        "bits": torch.tensor(np.stack([b["bits"] for b in batch]), dtype=torch.float32),
        "clean": clean,
        "clean_len": clean_len,
        "noisy": noisy,
        "noisy_len": noisy_len,
        "edit_distance": torch.tensor([b["edit_distance"] for b in batch], dtype=torch.float32),
        "condition_id": torch.tensor([b["condition_id"] for b in batch], dtype=torch.long),
        "p_ins": torch.tensor([b["p_ins"] for b in batch], dtype=torch.float32),
        "p_del": torch.tensor([b["p_del"] for b in batch], dtype=torch.float32),
        "p_sub": torch.tensor([b["p_sub"] for b in batch], dtype=torch.float32),
    }
    if "teacher_llr" in batch[0]:
        out["teacher_llr"] = torch.tensor(np.stack([b["teacher_llr"] for b in batch]), dtype=torch.float32)
    return out


def write_vlen_dataset(group, name, arrays, dtype=np.int16):
    vlen = h5py.vlen_dtype(np.dtype(dtype))
    ds = group.create_dataset(name, shape=(len(arrays),), dtype=vlen)
    for i, x in enumerate(arrays):
        ds[i] = np.asarray(x, dtype=dtype)
    return ds
