from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from rebind4ary.dataset import StageDataset
from rebind4ary.engine import make_loader, train_stage
from rebind4ary.transfer import build_rebind_from_checkpoint, load_stage_checkpoint


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--stage", choices=["stage1", "stage2", "stage3"], required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pretrained-rebind")
    src.add_argument("--init-stage-checkpoint")
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--lambda-bit", type=float, default=0.35)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--min-epochs", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    if args.pretrained_rebind:
        encoder, bit_head, obj = build_rebind_from_checkpoint(args.pretrained_rebind, device)
        source = args.pretrained_rebind
        source_meta = obj
    else:
        encoder, bit_head, obj = load_stage_checkpoint(args.init_stage_checkpoint, device)
        source = args.init_stage_checkpoint
        source_meta = obj
    train_ds = StageDataset(args.data, "train", args.stage, max_len=encoder.max_len)
    val_ds = StageDataset(args.data, "validation", args.stage, max_len=encoder.max_len)
    train_loader = make_loader(train_ds, args.batch_size, True, args.workers)
    val_loader = make_loader(val_ds, args.batch_size, False, args.workers)
    cfg = {
        "embed_dim": encoder.embed_dim,
        "symbol_emb_dim": encoder.symbol_embedding.embedding_dim,
        "hidden_size": encoder.bigru.hidden_size,
        "num_layers": encoder.bigru.num_layers,
        "position_branch_dim": encoder.position_branch[3].out_features,
        "dropout": float(encoder.projection[3].p),
        "source_stage": str(source_meta.get("stage", "pretrained_rebind")),
    }
    train_stage(
        args.stage,
        encoder,
        bit_head,
        source,
        train_loader,
        val_loader,
        device,
        args.save_dir,
        args.epochs,
        args.lr,
        args.weight_decay,
        args.lambda_bit,
        args.temperature,
        args.patience,
        args.min_epochs,
        cfg,
    )


if __name__ == "__main__":
    main()
