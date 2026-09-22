from __future__ import annotations

import csv
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .losses import bit_bce, info_nce
from .transfer import save_stage_checkpoint


def make_loader(dataset, batch_size, shuffle, workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _run_epoch(encoder, bit_head, loader, device, optimizer, lambda_bit, temperature):
    train = optimizer is not None
    encoder.train(train)
    bit_head.train(train)
    sums = {"loss": 0.0, "contrastive": 0.0, "bit": 0.0, "samples": 0}
    for clean, noisy, coded in loader:
        clean = clean.to(device, non_blocking=True)
        noisy = noisy.to(device, non_blocking=True)
        coded = coded.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        zc = encoder(clean)
        zn = encoder(noisy)
        l_global = info_nce(zc, zn, temperature=temperature)
        l_bit, _, _ = bit_bce(bit_head, zc, zn, coded)
        loss = l_global + float(lambda_bit) * l_bit
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(bit_head.parameters()), 1.0)
            optimizer.step()
        b = clean.size(0)
        sums["loss"] += float(loss.item()) * b
        sums["contrastive"] += float(l_global.item()) * b
        sums["bit"] += float(l_bit.item()) * b
        sums["samples"] += b
    n = max(sums["samples"], 1)
    return {k: sums[k] / n for k in ("loss", "contrastive", "bit")}


def train_stage(stage, encoder, bit_head, source_checkpoint, train_loader, val_loader, device, output_dir, epochs, lr, weight_decay, lambda_bit, temperature, patience, min_epochs, config):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(bit_head.parameters()), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3, min_lr=1e-7)
    best = math.inf
    bad = 0
    rows = []
    best_path = out / "best.pt"
    for epoch in range(1, epochs + 1):
        tr = _run_epoch(encoder, bit_head, train_loader, device, opt, lambda_bit, temperature)
        with torch.no_grad():
            va = _run_epoch(encoder, bit_head, val_loader, device, None, lambda_bit, temperature)
        sched.step(va["loss"])
        row = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "train_contrastive": tr["contrastive"],
            "train_bit": tr["bit"],
            "val_loss": va["loss"],
            "val_contrastive": va["contrastive"],
            "val_bit": va["bit"],
            "lr": opt.param_groups[0]["lr"],
        }
        rows.append(row)
        print(
            f"[{stage}] epoch={epoch:03d} train={tr['loss']:.6f} "
            f"val={va['loss']:.6f} global={va['contrastive']:.6f} bit={va['bit']:.6f} "
            f"lr={opt.param_groups[0]['lr']:.3e}"
        )
        if va["loss"] < best:
            best = va["loss"]
            bad = 0
            save_stage_checkpoint(best_path, encoder, bit_head, source_checkpoint, stage, epoch, best, config)
        else:
            bad += 1
        if epoch >= min_epochs and bad >= patience:
            break
    with (out / "history.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return best_path
