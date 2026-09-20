import torch
import torch.nn.functional as F


def l2(z1, z2):
    return torch.linalg.vector_norm(
        z1 - z2,
        ord=2,
        dim=-1,
    )


def regression_loss(
    pred_ap,
    pred_an,
    pred_pn,
    true_ap,
    true_an,
    true_pn,
):
    return (
        F.mse_loss(pred_ap, true_ap)
        + F.mse_loss(pred_an, true_an)
        + F.mse_loss(pred_pn, true_pn)
    ) / 3.0


def ranking_loss(
    pred_ap,
    pred_an,
    pred_pn,
    true_ap,
    true_an,
    true_pn,
    margin=0.2,
):
    pred = torch.stack(
        [pred_ap, pred_an, pred_pn],
        dim=1,
    )

    true = torch.stack(
        [true_ap, true_an, true_pn],
        dim=1,
    )

    terms = []

    for i in range(3):
        for j in range(i + 1, 3):
            diff = true[:, i] - true[:, j]

            lt = diff < 0
            gt = diff > 0

            if lt.any():
                terms.append(
                    F.relu(
                        pred[lt, i]
                        - pred[lt, j]
                        + margin
                    )
                )

            if gt.any():
                terms.append(
                    F.relu(
                        pred[gt, j]
                        - pred[gt, i]
                        + margin
                    )
                )

    if not terms:
        return pred.new_zeros(())

    return torch.cat(terms).mean()
