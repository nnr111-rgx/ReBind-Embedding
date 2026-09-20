import argparse
from pathlib import Path

from rebind.train_cli import add_common_args, prepare
from rebind.engine import run_stage


def main():
    p = add_common_args(
        argparse.ArgumentParser()
    )

    p.add_argument("--epochs", type=int, default=80)

    args = p.parse_args()

    (
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
    ) = prepare(args)

    del store
    del max_len

    path = run_stage(
        stage_name="stage1_anchor_reconstruction",
        encoder=encoder,
        decoder=decoder,
        bit_head=bit_head,
        calibration=calibration,
        train_loader=train_loader,
        val_loader=val_loader,
        anchor_len=anchor_len,
        device=device,
        output_dir=args.output_dir,
        config=config,
        epochs=args.epochs,
        mode="anchor",
        lr_encoder=args.lr_encoder,
        lr_decoder=args.lr_decoder,
        bit_weight=1.0,
        ed_start=0.0,
        ed_end=0.0,
        rank_start=0.0,
        rank_end=0.0,
        rank_margin=args.rank_margin,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        patience=args.patience,
        min_epochs=args.min_epochs,
    )

    print(path)


if __name__ == "__main__":
    main()
