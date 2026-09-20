from pathlib import Path
import json
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class TripletStore:
    def __init__(self, path):
        self.path = Path(path)

        if not self.path.exists():
            raise FileNotFoundError(self.path)

        self.anchor = []
        self.positive = []
        self.negative = []
        self.d_ap = []
        self.d_an = []
        self.d_pn = []

        with h5py.File(self.path, "r") as f:
            keys = list(f.keys())

            def key_fn(x):
                return int(x) if str(x).isdigit() else str(x)

            keys = sorted(keys, key=key_fn)

            for key in keys:
                g = f[key]

                a = np.asarray(g["anchor"], dtype=np.int64)
                p = np.asarray(g["positive"], dtype=np.int64)
                n = np.asarray(g["negative"], dtype=np.int64)

                for seq in (a, p, n):
                    values = set(np.unique(seq).tolist())
                    if not values.issubset({0, 1}):
                        raise ValueError(f"Non-binary sequence in group {key}")

                self.anchor.append(a)
                self.positive.append(p)
                self.negative.append(n)
                self.d_ap.append(float(g.attrs["d_anchor_positive"]))
                self.d_an.append(float(g.attrs["d_anchor_negative"]))
                self.d_pn.append(float(g.attrs["d_positive_negative"]))

        self.max_len = max(
            max(len(x) for x in self.anchor),
            max(len(x) for x in self.positive),
            max(len(x) for x in self.negative),
        )

    def __len__(self):
        return len(self.anchor)


def pad_sequence(seq, max_len):
    out = np.full(max_len, -1, dtype=np.int64)
    out[: len(seq)] = seq
    return out


def augment_triplet(a, p, n):
    if np.random.rand() < 0.5:
        a = 1 - a
        p = 1 - p
        n = 1 - n

    if np.random.rand() < 0.5:
        a = a[::-1].copy()
        p = p[::-1].copy()
        n = n[::-1].copy()

    return a, p, n


class TripletDataset(Dataset):
    def __init__(self, store, indices, max_len=None, augment=False):
        self.store = store
        self.indices = np.asarray(indices, dtype=np.int64)
        self.max_len = int(max_len or store.max_len)
        self.augment = bool(augment)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])

        a = self.store.anchor[idx].copy()
        p = self.store.positive[idx].copy()
        n = self.store.negative[idx].copy()

        if self.augment:
            a, p, n = augment_triplet(a, p, n)

        return (
            torch.from_numpy(pad_sequence(a, self.max_len)),
            torch.from_numpy(pad_sequence(p, self.max_len)),
            torch.from_numpy(pad_sequence(n, self.max_len)),
            torch.tensor(self.store.d_ap[idx], dtype=torch.float32),
            torch.tensor(self.store.d_an[idx], dtype=torch.float32),
            torch.tensor(self.store.d_pn[idx], dtype=torch.float32),
        )


def load_split(path, n_total):
    obj = json.loads(Path(path).read_text())

    train = np.asarray(obj["train"], dtype=np.int64)
    val = np.asarray(obj["val"], dtype=np.int64)
    test = np.asarray(obj["test"], dtype=np.int64)

    all_idx = np.concatenate([train, val, test])

    if len(np.unique(all_idx)) != len(all_idx):
        raise ValueError("Split contains duplicate indices")

    if np.any(all_idx < 0) or np.any(all_idx >= n_total):
        raise ValueError("Split index out of range")

    return train, val, test
