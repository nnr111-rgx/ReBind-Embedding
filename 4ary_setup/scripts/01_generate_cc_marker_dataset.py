from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from rebind4ary.channel import ids_channel
from rebind4ary.coding import build_marker_patterns, conv_encode_bits, marcode, random_message
from rebind4ary.config import L_MAX, PAD_QARY, P_SUB_MAX, P_SUB_MIN, QARY_MAX_LEN, STAGES, T


def write_split(group, n, seed):
    mp1, mp2 = build_marker_patterns()
    msg = group.create_dataset("message_bits", (n, 100), dtype="u1")
    coded = group.create_dataset("coded_bits", (n, 204), dtype="u1")
    clean = group.create_dataset("clean_qary", (n, T), dtype="u1")
    stage_ds = {}
    for stage in STAGES:
        stage_ds[stage] = {
            "noisy": group.create_dataset(f"{stage}_noisy_qary", (n, QARY_MAX_LEN), dtype="u1", fillvalue=PAD_QARY),
            "len": group.create_dataset(f"{stage}_noisy_len", (n,), dtype="u2"),
            "p_sub": group.create_dataset(f"{stage}_p_sub", (n,), dtype="f4"),
        }
    for i in range(n):
        rng_msg = np.random.default_rng(seed + i)
        m = random_message(rng=rng_msg)
        c = conv_encode_bits(m)
        x = marcode(c, mp1, mp2)
        msg[i] = m
        coded[i] = c
        clean[i] = x
        for si, (stage, cond) in enumerate(STAGES.items()):
            rng = np.random.default_rng(seed + 10_000_000 * (si + 1) + i)
            p_sub = float(rng.uniform(P_SUB_MIN, P_SUB_MAX))
            y = ids_channel(x, cond["p_ins"], cond["p_del"], p_sub, rng=rng, l_max=L_MAX)
            if y.size > QARY_MAX_LEN:
                raise RuntimeError("Unexpected IDS output overflow")
            arr = np.full(QARY_MAX_LEN, PAD_QARY, dtype=np.uint8)
            arr[:y.size] = y.astype(np.uint8)
            stage_ds[stage]["noisy"][i] = arr
            stage_ds[stage]["len"][i] = y.size
            stage_ds[stage]["p_sub"][i] = p_sub
        if (i + 1) % 10000 == 0 or i + 1 == n:
            print(f"generated {group.name}: {i + 1}/{n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--train-samples", type=int, default=120000)
    ap.add_argument("--val-samples", type=int, default=12000)
    ap.add_argument("--test-samples", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=202607)
    args = ap.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    mp1, mp2 = build_marker_patterns()
    with h5py.File(path, "w") as f:
        f.attrs["msg_len"] = 100
        f.attrs["code_bits"] = 204
        f.attrs["cc_generators_octal"] = "5,7"
        f.attrs["cc_constraint_length"] = 3
        f.attrs["T"] = 142
        f.attrs["Np"] = 7
        f.attrs["alphabet_size"] = 4
        f.attrs["qary_max_len"] = QARY_MAX_LEN
        f.attrs["binary_expanded_max_len"] = 2 * QARY_MAX_LEN
        f.create_dataset("marker_mp1", data=mp1)
        f.create_dataset("marker_mp2", data=mp2)
        write_split(f.create_group("train"), args.train_samples, args.seed)
        write_split(f.create_group("validation"), args.val_samples, args.seed + 1_000_000)
        write_split(f.create_group("test"), args.test_samples, args.seed + 2_000_000)
    print(path)


if __name__ == "__main__":
    main()
