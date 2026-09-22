from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from rebind4ary.channel import ids_channel
from rebind4ary.coding import qary_to_binary
from rebind4ary.config import REBIND_MAX_LEN
from rebind4ary.transfer import load_stage_checkpoint


def pad_binary(qary, max_len):
    bits = qary_to_binary(qary)
    if bits.size > max_len:
        raise ValueError("Input exceeds REBIND max length")
    out = np.full(max_len, -1, dtype=np.int64)
    out[:bits.size] = bits
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--n-test", type=int, default=12000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=9001)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    encoder, bit_head, _ = load_stage_checkpoint(args.checkpoint, device)
    encoder.eval()
    bit_head.eval()
    with h5py.File(args.data, "r") as f:
        clean_all = np.asarray(f["test"]["clean_qary"][: min(args.n_test, f["test"]["clean_qary"].shape[0])], dtype=np.int64)
        coded_all = np.asarray(f["test"]["coded_bits"][:clean_all.shape[0]], dtype=np.float32)
    rows = []
    p_sync_list = [0.01, 0.02, 0.03]
    p_sub_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    for ci, p_sync in enumerate(p_sync_list):
        for cj, p_sub in enumerate(p_sub_list):
            cosine_sum = 0.0
            l2_sum = 0.0
            clean_err = 0.0
            noisy_err = 0.0
            retrieval = 0.0
            seen = 0
            for start in range(0, clean_all.shape[0], args.batch_size):
                end = min(start + args.batch_size, clean_all.shape[0])
                clean_np = []
                noisy_np = []
                for k in range(start, end):
                    rng = np.random.default_rng(args.seed + ci * 10_000_000 + cj * 1_000_000 + k)
                    y = ids_channel(clean_all[k], p_sync, p_sync, p_sub, rng=rng, l_max=2)
                    clean_np.append(pad_binary(clean_all[k], encoder.max_len))
                    noisy_np.append(pad_binary(y, encoder.max_len))
                clean = torch.as_tensor(np.stack(clean_np), dtype=torch.long, device=device)
                noisy = torch.as_tensor(np.stack(noisy_np), dtype=torch.long, device=device)
                coded = torch.as_tensor(coded_all[start:end], dtype=torch.float32, device=device)
                with torch.no_grad():
                    zc = encoder(clean)
                    zn = encoder(noisy)
                    lc = bit_head(zc)
                    ln = bit_head(zn)
                cosine_sum += float(F.cosine_similarity(zc, zn, dim=-1).sum().item())
                l2_sum += float(torch.norm(zc - zn, dim=-1).sum().item())
                clean_err += float((lc.ge(0).float() != coded).float().sum().item())
                noisy_err += float((ln.ge(0).float() != coded).float().sum().item())
                sim = F.normalize(zc, dim=-1) @ F.normalize(zn, dim=-1).t()
                retrieval += float(sim.argmax(dim=1).eq(torch.arange(sim.size(0), device=device)).sum().item())
                seen += end - start
            rows.append(
                {
                    "p_ins": p_sync,
                    "p_del": p_sync,
                    "p_sub": p_sub,
                    "n": seen,
                    "cosine_clean_noisy": cosine_sum / seen,
                    "embedding_l2_drift": l2_sum / seen,
                    "clean_code_ber": clean_err / (seen * coded_all.shape[1]),
                    "noisy_code_ber": noisy_err / (seen * coded_all.shape[1]),
                    "batch_retrieval_top1": retrieval / seen,
                }
            )
            print(rows[-1])
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "metrics_by_condition.csv", index=False)
    summary = {
        "conditions": len(rows),
        "mean_cosine_clean_noisy": float(df["cosine_clean_noisy"].mean()),
        "mean_embedding_l2_drift": float(df["embedding_l2_drift"].mean()),
        "mean_clean_code_ber": float(df["clean_code_ber"].mean()),
        "mean_noisy_code_ber": float(df["noisy_code_ber"].mean()),
        "mean_batch_retrieval_top1": float(df["batch_retrieval_top1"].mean()),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(summary)


if __name__ == "__main__":
    main()
