import numpy as np
import torch

from .edit_distance import edit_distance
from .losses import l2
from .metrics import fit_linear, apply_linear, distance_metrics


@torch.no_grad()
def collect_distances(encoder, loader, device):
    raw = []
    true = []
    pair = []
    triplet = []

    for a, p, n, dap, dan, dpn in loader:
        a = a.to(device)
        p = p.to(device)
        n = n.to(device)

        za = encoder(a)
        zp = encoder(p)
        zn = encoder(n)

        rap = l2(za, zp).cpu().numpy()
        ran = l2(za, zn).cpu().numpy()
        rpn = l2(zp, zn).cpu().numpy()

        raw.extend(rap.tolist())
        true.extend(dap.numpy().tolist())
        pair.extend(["ap"] * len(rap))

        raw.extend(ran.tolist())
        true.extend(dan.numpy().tolist())
        pair.extend(["an"] * len(ran))

        raw.extend(rpn.tolist())
        true.extend(dpn.numpy().tolist())
        pair.extend(["pn"] * len(rpn))

        triplet.extend((rap < ran).astype(np.float64).tolist())

    return (
        np.asarray(raw, dtype=np.float64),
        np.asarray(true, dtype=np.float64),
        np.asarray(pair),
        np.asarray(triplet, dtype=np.float64),
    )


@torch.no_grad()
def evaluate_edit_distance(
    encoder,
    train_loader,
    test_loader,
    device,
):
    train_raw, train_true, _, _ = collect_distances(
        encoder,
        train_loader,
        device,
    )

    coef = fit_linear(
        train_raw,
        train_true,
    )

    test_raw, test_true, test_pair, triplet = collect_distances(
        encoder,
        test_loader,
        device,
    )

    test_pred = apply_linear(
        test_raw,
        coef,
    )

    metrics = distance_metrics(
        test_true,
        test_pred,
    )

    metrics["triplet"] = float(triplet.mean())

    return metrics, {
        "pair": test_pair,
        "true_ed": test_true,
        "raw_distance": test_raw,
        "estimated_ed": test_pred,
    }


@torch.no_grad()
def evaluate_reconstruction(
    encoder,
    decoder,
    bit_head,
    loader,
    anchor_len,
    max_len,
    device,
):
    anchor_wrong_ar = 0
    anchor_wrong_aux = 0
    anchor_total = 0

    full_wrong = 0
    full_total = 0

    length_correct = 0
    exact_correct = 0

    edit_sum = 0.0
    normalized_edit_sum = 0.0
    n_sequences = 0

    for a, p, n, dap, dan, dpn in loader:
        a = a.to(device)
        p = p.to(device)
        n = n.to(device)

        za = encoder(a)
        zp = encoder(p)
        zn = encoder(n)

        anchor_bits, _ = decoder.generate(
            za,
            max_len=max_len,
            fixed_length=anchor_len,
        )

        anchor_true = a[:, :anchor_len].long()

        anchor_wrong_ar += int(
            anchor_bits.ne(anchor_true).sum().item()
        )

        anchor_wrong_aux += int(
            bit_head(za)
            .ge(0)
            .long()
            .ne(anchor_true)
            .sum()
            .item()
        )

        anchor_total += int(anchor_true.numel())

        z_all = torch.cat(
            [za, zp, zn],
            dim=0,
        )

        x_all = torch.cat(
            [a, p, n],
            dim=0,
        )

        true_lengths = x_all.ge(0).sum(1).long()

        fixed_bits, _ = decoder.generate(
            z_all,
            max_len=max_len,
            fixed_length=max_len,
        )

        for i in range(x_all.size(0)):
            length = int(true_lengths[i].item())

            full_wrong += int(
                fixed_bits[i, :length]
                .ne(x_all[i, :length].long())
                .sum()
                .item()
            )

            full_total += length

        pred_bits, pred_lengths = decoder.generate(
            z_all,
            max_len=max_len,
            fixed_length=None,
        )

        length_correct += int(
            pred_lengths.eq(true_lengths).sum().item()
        )

        x_np = x_all.cpu().numpy()
        pred_np = pred_bits.cpu().numpy()
        true_len_np = true_lengths.cpu().numpy()
        pred_len_np = pred_lengths.cpu().numpy()

        for i in range(len(x_np)):
            true_len = int(true_len_np[i])
            pred_len = int(pred_len_np[i])

            true_seq = x_np[i, :true_len].astype(np.int64).tolist()
            pred_seq = pred_np[i, :pred_len].astype(np.int64).tolist()

            exact_correct += int(
                true_len == pred_len
                and true_seq == pred_seq
            )

            ed = edit_distance(
                true_seq,
                pred_seq,
            )

            edit_sum += ed
            normalized_edit_sum += ed / max(true_len, 1)
            n_sequences += 1

    return {
        "anchor_hamming_ar": anchor_wrong_ar / max(anchor_total, 1),
        "anchor_hamming_aux": anchor_wrong_aux / max(anchor_total, 1),
        "full_hamming_at_true_length": full_wrong / max(full_total, 1),
        "length_accuracy": length_correct / max(n_sequences, 1),
        "exact_reconstruction": exact_correct / max(n_sequences, 1),
        "reconstruction_mean_edit_distance": edit_sum / max(n_sequences, 1),
        "reconstruction_normalized_edit_distance": normalized_edit_sum / max(n_sequences, 1),
    }
