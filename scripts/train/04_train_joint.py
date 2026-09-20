import argparse
import shutil
from pathlib import Path

from rebind.train_cli import add_common_args, prepare
from rebind.checkpoint import load_checkpoint
from rebind.engine import run_stage


def main():
    p = add_common_args(
        argparse.ArgumentParser()
    )

    p.add_argument("--input-checkpoint", required=True)
    p.add_argument("--epochs", type=int, default=180)

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

    load_checkpoint(
        args.input_checkpoint,
        encoder,
        decoder,
        bit_head,
        calibration,
        device,
    )

    path = run_stage(
        stage_name="stage4_joint",
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
        mode="all",
        lr_encoder=args.lr_encoder,
        lr_decoder=args.lr_decoder,
        bit_weight=0.25,
        ed_start=1.0,
        ed_end=1.0,
        rank_start=0.5,
        rank_end=0.5,
        rank_margin=args.rank_margin,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        patience=args.patience,
        min_epochs=args.min_epochs,
    )

    final_path = Path(args.output_dir) / "best.pt"

    shutil.copy2(
        path,
        final_path,
    )

    print(final_path)


if __name__ == "__main__":
    main()
