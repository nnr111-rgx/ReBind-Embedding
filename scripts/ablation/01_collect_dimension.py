import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--dims", nargs="+", type=int, default=[64, 128, 300])
    p.add_argument("--output", type=Path, required=True)

    args = p.parse_args()

    rows = []

    for dim in args.dims:
        path = (
            args.root
            / f"D{dim}"
            / "eval"
            / "all_metrics.json"
        )

        result = json.loads(
            path.read_text()
        )

        rows.append(
            {
                "dimension": dim,
                **result,
            }
        )

    frame = pd.DataFrame(
        rows
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        args.output,
        index=False,
    )

    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
