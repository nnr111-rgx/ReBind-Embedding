import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-points", type=int, default=3000)

    args = p.parse_args()

    frame = pd.read_csv(
        args.csv
    )

    if len(frame) > args.max_points:
        frame = frame.iloc[
            :: max(
                1,
                len(frame) // args.max_points,
            )
        ]

    fig, ax = plt.subplots(
        figsize=(5.0, 5.0)
    )

    ax.scatter(
        frame["true_ed"],
        frame["estimated_ed"],
        s=8,
        alpha=0.45,
    )

    low = min(
        frame["true_ed"].min(),
        frame["estimated_ed"].min(),
    )

    high = max(
        frame["true_ed"].max(),
        frame["estimated_ed"].max(),
    )

    ax.plot(
        [low, high],
        [low, high],
        linestyle="--",
    )

    ax.set_xlabel(
        "Ground-truth edit distance"
    )

    ax.set_ylabel(
        "Estimated edit distance"
    )

    ax.set_title(
        "REBIND edit-distance embedding"
    )

    fig.tight_layout()

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        args.output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


if __name__ == "__main__":
    main()
