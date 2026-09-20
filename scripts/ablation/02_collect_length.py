import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--lengths", nargs="+", type=int, default=[50, 100, 200])
    p.add_argument("--output", type=Path, required=True)

    args = p.parse_args()

    rows = []

    for length in args.lengths:
        path = (
            args.root
            / f"L{length}"
            / "eval"
            / "all_metrics.json"
        )

        result = json.loads(
            path.read_text()
        )

        rows.append(
            {
                "length": length,
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
