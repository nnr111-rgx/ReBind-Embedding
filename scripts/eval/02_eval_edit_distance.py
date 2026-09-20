import argparse
import json

import pandas as pd

from rebind.eval_cli import add_eval_args, prepare_eval
from rebind.evaluation import evaluate_edit_distance


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

    del max_len
    del anchor_len
    del decoder
    del bit_head
    del calibration

    metrics, arrays = evaluate_edit_distance(
        encoder,
        train_loader,
        test_loader,
        device,
    )

    metrics_path = args.output_dir / "edit_distance_metrics.json"

    metrics_path.write_text(
        json.dumps(
            metrics,
            indent=2,
        )
    )

    frame = pd.DataFrame(
        arrays
    )

    frame.to_csv(
        args.output_dir / "ed_scatter.csv",
        index=False,
    )

    print(
        json.dumps(
            metrics,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
