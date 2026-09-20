import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--x", required=True)
    p.add_argument("--metric", default="rmse")
    p.add_argument("--output", type=Path, required=True)

    args = p.parse_args()

    frame = pd.read_csv(
        args.csv
    )

    fig, ax = plt.subplots(
        figsize=(5.5, 4.0)
    )

    ax.plot(
        frame[args.x],
        frame[args.metric],
        marker="o",
    )

    ax.set_xlabel(
        args.x
    )

    ax.set_ylabel(
        args.metric
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
