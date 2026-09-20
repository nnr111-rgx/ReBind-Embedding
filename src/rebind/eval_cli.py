import argparse
from pathlib import Path

import torch

from rebind.data import TripletStore, TripletDataset, load_split
from rebind.build import build_models
from rebind.engine import make_loader
from rebind.checkpoint import load_checkpoint


def resolve_device(name):
    if name == "auto":
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    return torch.device(name)


def add_eval_args(p):
    p.add_argument("--h5", type=Path, required=True)
    p.add_argument("--split-json", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)

    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=2)

    p.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )

    return p


def prepare_eval(args):
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

    del val_idx

    obj = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    config = obj["config"]

    max_len = int(
        obj["max_len"]
    )

    anchor_len = int(
        obj["anchor_len"]
    )

    encoder, decoder, bit_head, calibration = build_models(
        config,
        max_len,
        anchor_len,
        device,
    )

    load_checkpoint(
        args.checkpoint,
        encoder,
        decoder,
        bit_head,
        calibration,
        device,
    )

    train_loader = make_loader(
        TripletDataset(
            store,
            train_idx,
            max_len=max_len,
            augment=False,
        ),
        args.batch_size,
        False,
        args.num_workers,
    )

    test_loader = make_loader(
        TripletDataset(
            store,
            test_idx,
            max_len=max_len,
            augment=False,
        ),
        args.batch_size,
        False,
        args.num_workers,
    )

    return (
        device,
        max_len,
        anchor_len,
        encoder,
        decoder,
        bit_head,
        calibration,
        train_loader,
        test_loader,
    )
