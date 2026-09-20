import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)

    args = p.parse_args()

    frame = pd.read_csv(
        args.csv
    )

    fig, ax = plt.subplots(
        figsize=(6.0, 4.0)
    )

    ax.plot(
        frame["epoch"],
        frame["train_loss"],
        label="Train loss",
    )

    ax.plot(
        frame["epoch"],
        frame["val_rec"],
        label="Validation reconstruction",
    )

    if frame["val_rmse"].notna().any():
        ax.plot(
            frame["epoch"],
            frame["val_rmse"],
            label="Validation ED RMSE",
        )

    ax.set_xlabel(
        "Epoch"
    )

    ax.set_ylabel(
        "Value"
    )

    ax.legend()

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
