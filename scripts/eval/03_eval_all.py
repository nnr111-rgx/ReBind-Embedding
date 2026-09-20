import argparse
import json

import pandas as pd

from rebind.eval_cli import add_eval_args, prepare_eval
from rebind.evaluation import evaluate_edit_distance, evaluate_reconstruction


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

    ed_metrics, arrays = evaluate_edit_distance(
        encoder,
        train_loader,
        test_loader,
        device,
    )

    rec_metrics = evaluate_reconstruction(
        encoder,
        decoder,
        bit_head,
        test_loader,
        anchor_len,
        max_len,
        device,
    )

    result = {
        **ed_metrics,
        **rec_metrics,
    }

    (args.output_dir / "all_metrics.json").write_text(
        json.dumps(
            result,
            indent=2,
        )
    )

    pd.DataFrame(
        arrays
    ).to_csv(
        args.output_dir / "ed_scatter.csv",
        index=False,
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
