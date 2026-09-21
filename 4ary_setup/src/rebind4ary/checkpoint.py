from __future__ import annotations

from pathlib import Path

import torch

from .models import AffineDistance, build_encoder


def save_encoder_checkpoint(path, encoder, calibration, optimizer=None, epoch=0, metrics=None, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "encoder": encoder.state_dict(),
        "calibration": calibration.state_dict() if calibration is not None else None,
        "encoder_config": encoder.config(),
        "epoch": int(epoch),
        "metrics": metrics or {},
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


def load_encoder_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    encoder = build_encoder(ckpt.get("encoder_config", {})).to(device)
    encoder.load_state_dict(ckpt["encoder"], strict=True)
    calibration = AffineDistance().to(device)
    state = ckpt.get("calibration")
    if state is not None:
        calibration.load_state_dict(state, strict=True)
    return encoder, calibration, ckpt
