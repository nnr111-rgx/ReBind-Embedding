from __future__ import annotations

import torch
import torch.nn.functional as F


def info_nce(z_clean, z_noisy, temperature=0.1):
    zc = F.normalize(z_clean, dim=-1)
    zn = F.normalize(z_noisy, dim=-1)
    logits = zc @ zn.t() / float(temperature)
    target = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))


def bit_bce(bit_head, z_clean, z_noisy, coded_bits):
    lc = bit_head(z_clean)
    ln = bit_head(z_noisy)
    loss = 0.5 * (
        F.binary_cross_entropy_with_logits(lc, coded_bits)
        + F.binary_cross_entropy_with_logits(ln, coded_bits)
    )
    return loss, lc, ln
