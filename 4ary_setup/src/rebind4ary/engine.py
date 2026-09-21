from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import collate_ids, collate_triplet
from .models import l2
from .utils import worker_seed


def make_loader(dataset, batch_size, shuffle, workers, seed, kind="ids"):
    gen = torch.Generator().manual_seed(int(seed))
    collate = collate_ids if kind == "ids" else collate_triplet
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(workers),
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_seed,
        generator=gen,
        persistent_workers=int(workers) > 0,
    )


def move_batch(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def encode_ids_dataset(encoder, loader, device):
    encoder.eval()
    clean_all = []
    noisy_all = []
    bits_all = []
    ed_all = []
    cond_all = []
    params = []
    ids = []
    teacher = []
    for batch in loader:
        batch = move_batch(batch, device)
        zc = encoder(batch["clean"], batch["clean_len"])
        zn = encoder(batch["noisy"], batch["noisy_len"])
        clean_all.append(zc.cpu().numpy())
        noisy_all.append(zn.cpu().numpy())
        bits_all.append(batch["bits"].cpu().numpy())
        ed_all.append(batch["edit_distance"].cpu().numpy())
        cond_all.append(batch["condition_id"].cpu().numpy())
        ids.append(batch["sample_id"].cpu().numpy())
        params.append(np.stack([
            batch["p_ins"].cpu().numpy(),
            batch["p_del"].cpu().numpy(),
            batch["p_sub"].cpu().numpy(),
        ], axis=1))
        if "teacher_llr" in batch:
            teacher.append(batch["teacher_llr"].cpu().numpy())
    out = {
        "z_clean": np.concatenate(clean_all),
        "z_noisy": np.concatenate(noisy_all),
        "bits": np.concatenate(bits_all),
        "edit_distance": np.concatenate(ed_all),
        "condition_id": np.concatenate(cond_all),
        "sample_id": np.concatenate(ids),
        "params": np.concatenate(params),
    }
    if teacher:
        out["teacher_llr"] = np.concatenate(teacher)
    return out


def append_history(path, row: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def embedding_distance(encoder, clean, clean_len, noisy, noisy_len):
    zc = encoder(clean, clean_len)
    zn = encoder(noisy, noisy_len)
    return zc, zn, l2(zc, zn)
