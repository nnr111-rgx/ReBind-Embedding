from pathlib import Path
import csv
import math

import torch
from torch.utils.data import DataLoader

from .losses import l2, regression_loss, ranking_loss
from .reconstruction import autoregressive_ce, anchor_bit_bce
from .checkpoint import save_checkpoint, load_checkpoint


def make_loader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def linear_ramp(epoch, epochs, start, end):
    if epochs <= 1:
        return float(end)

    alpha = (epoch - 1) / float(epochs - 1)

    return float(
        start
        + alpha * (end - start)
    )


@torch.no_grad()
def validate(
    encoder,
    decoder,
    bit_head,
    calibration,
    loader,
    anchor_len,
    device,
    mode,
    use_ed,
):
    encoder.eval()
    decoder.eval()
    bit_head.eval()
    calibration.eval()

    rec_sum = 0.0
    bit_sum = 0.0
    seen = 0

    squared_errors = []
    triplet_terms = []

    for a, p, n, dap, dan, dpn in loader:
        a = a.to(device)
        p = p.to(device)
        n = n.to(device)

        dap = dap.to(device)
        dan = dan.to(device)
        dpn = dpn.to(device)

        za = encoder(a)

        bit = anchor_bit_bce(
            bit_head,
            za,
            a,
            anchor_len,
        )

        if mode == "anchor":
            rec = autoregressive_ce(
                decoder,
                za,
                a,
            )
        else:
            zp = encoder(p)
            zn = encoder(n)

            rec = autoregressive_ce(
                decoder,
                torch.cat([za, zp, zn], dim=0),
                torch.cat([a, p, n], dim=0),
            )

        batch = a.size(0)
        seen += batch

        rec_sum += float(rec.item()) * batch
        bit_sum += float(bit.item()) * batch

        if use_ed:
            if mode == "anchor":
                zp = encoder(p)
                zn = encoder(n)

            pred_ap = calibration(l2(za, zp))
            pred_an = calibration(l2(za, zn))
            pred_pn = calibration(l2(zp, zn))

            pred = torch.stack(
                [pred_ap, pred_an, pred_pn],
                dim=1,
            )

            true = torch.stack(
                [dap, dan, dpn],
                dim=1,
            )

            squared_errors.append(
                (pred - true).pow(2).reshape(-1).cpu()
            )

            triplet_terms.append(
                pred_ap.lt(pred_an).float().cpu()
            )

    out = {
        "rec_ce": rec_sum / max(seen, 1),
        "bit_bce": bit_sum / max(seen, 1),
    }

    if use_ed:
        out["rmse"] = float(
            torch.cat(squared_errors).mean().sqrt().item()
        )

        out["triplet"] = float(
            torch.cat(triplet_terms).mean().item()
        )

    return out


def run_stage(
    stage_name,
    encoder,
    decoder,
    bit_head,
    calibration,
    train_loader,
    val_loader,
    anchor_len,
    device,
    output_dir,
    config,
    epochs,
    mode,
    lr_encoder,
    lr_decoder,
    bit_weight,
    ed_start,
    ed_end,
    rank_start,
    rank_end,
    rank_margin,
    weight_decay,
    grad_clip,
    patience,
    min_epochs,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": encoder.parameters(),
                "lr": lr_encoder,
            },
            {
                "params": list(decoder.parameters()) + list(bit_head.parameters()),
                "lr": lr_decoder,
            },
            {
                "params": calibration.parameters(),
                "lr": lr_encoder,
            },
        ],
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=6,
        min_lr=1e-7,
    )

    best_score = math.inf
    bad_epochs = 0

    best_path = output_dir / f"{stage_name}_best.pt"
    history_path = output_dir / f"{stage_name}_history.csv"

    rows = []

    for epoch in range(1, epochs + 1):
        encoder.train()
        decoder.train()
        bit_head.train()
        calibration.train()

        lambda_ed = linear_ramp(
            epoch,
            epochs,
            ed_start,
            ed_end,
        )

        lambda_rank = linear_ramp(
            epoch,
            epochs,
            rank_start,
            rank_end,
        )

        sums = {
            "loss": 0.0,
            "rec": 0.0,
            "bit": 0.0,
            "ed": 0.0,
            "rank": 0.0,
        }

        seen = 0

        for a, p, n, dap, dan, dpn in train_loader:
            a = a.to(device)
            p = p.to(device)
            n = n.to(device)

            dap = dap.to(device)
            dan = dan.to(device)
            dpn = dpn.to(device)

            optimizer.zero_grad(set_to_none=True)

            za = encoder(a)

            bit = anchor_bit_bce(
                bit_head,
                za,
                a,
                anchor_len,
            )

            if mode == "anchor":
                rec = autoregressive_ce(
                    decoder,
                    za,
                    a,
                )

                ed = rec.new_zeros(())
                rank = rec.new_zeros(())

            else:
                zp = encoder(p)
                zn = encoder(n)

                rec = autoregressive_ce(
                    decoder,
                    torch.cat([za, zp, zn], dim=0),
                    torch.cat([a, p, n], dim=0),
                )

                if lambda_ed > 0.0 or lambda_rank > 0.0:
                    pred_ap = calibration(l2(za, zp))
                    pred_an = calibration(l2(za, zn))
                    pred_pn = calibration(l2(zp, zn))

                    ed = regression_loss(
                        pred_ap,
                        pred_an,
                        pred_pn,
                        dap,
                        dan,
                        dpn,
                    )

                    rank = ranking_loss(
                        pred_ap,
                        pred_an,
                        pred_pn,
                        dap,
                        dan,
                        dpn,
                        margin=rank_margin,
                    )

                else:
                    ed = rec.new_zeros(())
                    rank = rec.new_zeros(())

            loss = (
                rec
                + bit_weight * bit
                + lambda_ed * ed
                + lambda_rank * rank
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters())
                + list(decoder.parameters())
                + list(bit_head.parameters())
                + list(calibration.parameters()),
                grad_clip,
            )

            optimizer.step()

            batch = a.size(0)
            seen += batch

            sums["loss"] += float(loss.item()) * batch
            sums["rec"] += float(rec.item()) * batch
            sums["bit"] += float(bit.item()) * batch
            sums["ed"] += float(ed.item()) * batch
            sums["rank"] += float(rank.item()) * batch

        use_ed = ed_end > 0.0 or rank_end > 0.0

        val = validate(
            encoder,
            decoder,
            bit_head,
            calibration,
            val_loader,
            anchor_len,
            device,
            mode,
            use_ed,
        )

        if use_ed:
            score = (
                val["rec_ce"]
                + bit_weight * val["bit_bce"]
                + 0.15 * val["rmse"]
                + 0.20 * (1.0 - val["triplet"])
            )
        else:
            score = (
                val["rec_ce"]
                + bit_weight * val["bit_bce"]
            )

        scheduler.step(score)

        improved = score < best_score - 1e-6

        if improved:
            best_score = score
            bad_epochs = 0

            save_checkpoint(
                best_path,
                encoder,
                decoder,
                bit_head,
                calibration,
                config,
                encoder.max_len,
                anchor_len,
                stage_name,
                epoch,
                score,
            )
        else:
            bad_epochs += 1

        row = {
            "epoch": epoch,
            "train_loss": sums["loss"] / max(seen, 1),
            "train_rec": sums["rec"] / max(seen, 1),
            "train_bit": sums["bit"] / max(seen, 1),
            "train_ed": sums["ed"] / max(seen, 1),
            "train_rank": sums["rank"] / max(seen, 1),
            "lambda_ed": lambda_ed,
            "lambda_rank": lambda_rank,
            "val_rec": val["rec_ce"],
            "val_bit": val["bit_bce"],
            "val_rmse": val.get("rmse", float("nan")),
            "val_triplet": val.get("triplet", float("nan")),
            "score": score,
            "lr_encoder": optimizer.param_groups[0]["lr"],
            "lr_decoder": optimizer.param_groups[1]["lr"],
            "best": int(improved),
        }

        rows.append(row)

        print(
            f"{stage_name} ep={epoch:03d} "
            f"L={row['train_loss']:.4f} "
            f"Rec={row['train_rec']:.4f} "
            f"Bit={row['train_bit']:.4f} "
            f"ED={row['train_ed']:.4f} "
            f"Rank={row['train_rank']:.4f} "
            f"ValRec={row['val_rec']:.4f} "
            f"ValRMSE={row['val_rmse']:.4f} "
            f"Trip={row['val_triplet']:.4f}"
            + (" [BEST]" if improved else "")
        )

        if epoch >= min_epochs and bad_epochs >= patience:
            print(f"early stop {stage_name} at epoch {epoch}")
            break

    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    load_checkpoint(
        best_path,
        encoder,
        decoder,
        bit_head,
        calibration,
        device,
    )

    return best_path
