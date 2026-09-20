import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from rebind.data import TripletStore, TripletDataset, load_split
from rebind.build import build_models
from rebind.engine import make_loader, run_stage
from rebind.checkpoint import load_checkpoint


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name):
    if name == "auto":
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    return torch.device(name)


def add_common_args(p):
    p.add_argument("--h5", type=Path, required=True)
    p.add_argument("--split-json", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)

    p.add_argument("--embed-dim", type=int, default=300)
    p.add_argument("--symbol-emb-dim", type=int, default=16)
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--position-branch-dim", type=int, default=256)
    p.add_argument("--encoder-dropout", type=float, default=0.0)

    p.add_argument("--decoder-token-emb", type=int, default=64)
    p.add_argument("--decoder-context-dim", type=int, default=256)
    p.add_argument("--decoder-hidden", type=int, default=512)
    p.add_argument("--decoder-layers", type=int, default=2)
    p.add_argument("--decoder-dropout", type=float, default=0.1)

    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=2)

    p.add_argument("--lr-encoder", type=float, required=True)
    p.add_argument("--lr-decoder", type=float, required=True)

    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--rank-margin", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--min-epochs", type=int, default=20)

    p.add_argument("--seed", type=int, default=4100)

    p.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )

    return p


def prepare(args):
    seed_all(args.seed)

    device = resolve_device(
        args.device
    )

    store = TripletStore(
        args.h5
    )

    train_idx, val_idx, test_idx = load_split(
        args.split_json,
        len(store),
    )

    del test_idx

    anchor_lengths = {
        len(x)
        for x in store.anchor
    }

    if len(anchor_lengths) != 1:
        raise ValueError(
            "Expected fixed clean-anchor length"
        )

    anchor_len = int(
        next(
            iter(
                anchor_lengths
            )
        )
    )

    max_len = int(
        store.max_len
    )

    train_ds = TripletDataset(
        store,
        train_idx,
        max_len=max_len,
        augment=True,
    )

    val_ds = TripletDataset(
        store,
        val_idx,
        max_len=max_len,
        augment=False,
    )

    train_loader = make_loader(
        train_ds,
        args.batch_size,
        True,
        args.num_workers,
    )

    val_loader = make_loader(
        val_ds,
        args.eval_batch_size,
        False,
        args.num_workers,
    )

    config = vars(args).copy()

    encoder, decoder, bit_head, calibration = build_models(
        config,
        max_len,
        anchor_len,
        device,
    )

    return (
        device,
        store,
        anchor_len,
        max_len,
        train_loader,
        val_loader,
        config,
        encoder,
        decoder,
        bit_head,
        calibration,
    )
