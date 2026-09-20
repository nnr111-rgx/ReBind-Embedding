import torch
import torch.nn.functional as F


def build_ar_io(x):
    batch, max_len = x.shape

    valid = x.ge(0)
    lengths = valid.sum(1).long()

    inp = torch.zeros(
        batch,
        max_len + 1,
        dtype=torch.long,
        device=x.device,
    )

    inp[:, 0] = 2
    inp[:, 1:][valid] = x[valid].long()

    target = torch.full(
        (batch, max_len + 1),
        -100,
        dtype=torch.long,
        device=x.device,
    )

    target[:, :max_len][valid] = x[valid].long()

    target[
        torch.arange(batch, device=x.device),
        lengths,
    ] = 2

    return inp, target


def autoregressive_ce(decoder, z, x):
    inp, target = build_ar_io(x)

    logits = decoder(
        z,
        inp,
    )

    return F.cross_entropy(
        logits.reshape(-1, 3),
        target.reshape(-1),
        ignore_index=-100,
    )


def anchor_bit_bce(bit_head, z, anchor, anchor_len):
    target = anchor[:, :anchor_len].float()

    if torch.any(target.lt(0)):
        raise ValueError("Anchor shorter than anchor_len")

    return F.binary_cross_entropy_with_logits(
        bit_head(z),
        target,
    )
