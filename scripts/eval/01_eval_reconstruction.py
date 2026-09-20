import argparse
import json

from rebind.eval_cli import add_eval_args, prepare_eval
from rebind.evaluation import evaluate_reconstruction


def main():
    p = add_eval_args(
        argparse.ArgumentParser()
    )

    args = p.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        device,
        max_len,
        anchor_len,
        encoder,
        decoder,
        bit_head,
        calibration,
        train_loader,
        test_loader,
    ) = prepare_eval(args)

    del calibration
    del train_loader

    result = evaluate_reconstruction(
        encoder,
        decoder,
        bit_head,
        test_loader,
        anchor_len,
        max_len,
        device,
    )

    path = args.output_dir / "reconstruction_metrics.json"

    path.write_text(
        json.dumps(
            result,
            indent=2,
        )
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
