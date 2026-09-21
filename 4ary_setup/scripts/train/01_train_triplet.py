#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from tqdm import tqdm

from rebind4ary.checkpoint import save_encoder_checkpoint
from rebind4ary.data import TripletH5Dataset
from rebind4ary.engine import append_history, make_loader, move_batch
from rebind4ary.losses import triplet_regression_loss
from rebind4ary.models import AffineDistance, FourAryReBindEncoder, l2
from rebind4ary.utils import auto_device, set_seed


def run_epoch(encoder, calib, loader, device, rank_margin, lambda_rank, optimizer=None):
    train = optimizer is not None
    encoder.train(train)
    calib.train(train)
    total = reg_total = rank_total = 0.0
    n = 0
    for batch in tqdm(loader, leave=False):
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(train):
            za = encoder(batch["anchor"], batch["anchor_len"])
            zp = encoder(batch["positive"], batch["positive_len"])
            zn = encoder(batch["negative"], batch["negative_len"])
            pap = calib(l2(za, zp))
            pan = calib(l2(za, zn))
            ppn = calib(l2(zp, zn))
            loss, reg, rank = triplet_regression_loss(
                pap, pan, ppn,
                batch["d_ap"], batch["d_an"], batch["d_pn"],
                rank_margin=rank_margin,
                lambda_rank=lambda_rank,
            )
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(calib.parameters()), 1.0)
                optimizer.step()
        bs = len(batch["d_ap"])
        total += float(loss.detach()) * bs
        reg_total += float(reg.detach()) * bs
        rank_total += float(rank.detach()) * bs
        n += bs
    return {"loss": total/n, "reg": reg_total/n, "rank": rank_total/n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--embed-dim", type=int, default=204)
    ap.add_argument("--symbol-emb-dim", type=int, default=64)
    ap.add_argument("--hidden-size", type=int, default=256)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--lambda-rank", type=float, default=0.2)
    ap.add_argument("--rank-margin", type=float, default=0.2)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--min-epochs", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=4301)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    set_seed(args.seed)
    device = auto_device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    train_ds = TripletH5Dataset(args.data, "train")
    val_ds = TripletH5Dataset(args.data, "validation")
    train_loader = make_loader(train_ds, args.batch_size, True, args.workers, args.seed, kind="triplet")
    val_loader = make_loader(val_ds, args.batch_size*2, False, args.workers, args.seed+1, kind="triplet")
    encoder = FourAryReBindEncoder(args.embed_dim, args.symbol_emb_dim, args.hidden_size, args.num_layers, args.dropout).to(device)
    calib = AffineDistance().to(device)
    opt = AdamW(list(encoder.parameters()) + list(calib.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    best = float("inf")
    stale = 0
    hist = save_dir / "history.csv"
    for epoch in range(1, args.epochs+1):
        tr = run_epoch(encoder, calib, train_loader, device, args.rank_margin, args.lambda_rank, opt)
        va = run_epoch(encoder, calib, val_loader, device, args.rank_margin, args.lambda_rank, None)
        row = {"epoch": epoch, "train_loss": tr["loss"], "val_loss": va["loss"], "val_reg": va["reg"], "val_rank": va["rank"]}
        append_history(hist, row)
        print(row)
        if va["loss"] < best:
            best = va["loss"]
            stale = 0
            save_encoder_checkpoint(save_dir/"best.pt", encoder, calib, opt, epoch, va, {"stage": "triplet_pretrain"})
        else:
            stale += 1
        if epoch >= args.min_epochs and stale >= args.patience:
            break
    save_encoder_checkpoint(save_dir/"last.pt", encoder, calib, opt, epoch, va, {"stage": "triplet_pretrain"})
    print({"best": str(save_dir/"best.pt"), "best_val_loss": best})


if __name__ == "__main__":
    main()
