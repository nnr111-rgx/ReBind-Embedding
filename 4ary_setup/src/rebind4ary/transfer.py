from __future__ import annotations

from pathlib import Path

import torch

from rebind.models import REBINDEncoder, AnchorBitHead

from .config import CODE_BITS, REBIND_MAX_LEN


def _config_from_checkpoint(obj):
    cfg = dict(obj.get("config", {}))
    return {
        "embed_dim": int(cfg.get("embed_dim", 300)),
        "symbol_emb_dim": int(cfg.get("symbol_emb_dim", 16)),
        "hidden_size": int(cfg.get("hidden_size", 256)),
        "num_layers": int(cfg.get("num_layers", 2)),
        "position_branch_dim": int(cfg.get("position_branch_dim", 256)),
        "dropout": float(cfg.get("encoder_dropout", 0.0)),
    }


def _adapt_position_weight(old_weight, old_max_len, new_max_len):
    if old_weight.ndim != 2 or old_weight.size(1) != 2 * old_max_len:
        raise ValueError("Unexpected position-branch first-layer shape")
    new_weight = old_weight.new_zeros(old_weight.size(0), 2 * new_max_len)
    copy_len = min(old_max_len, new_max_len)
    new_weight[:, :copy_len] = old_weight[:, :copy_len]
    new_weight[:, new_max_len:new_max_len + copy_len] = old_weight[:, old_max_len:old_max_len + copy_len]
    return new_weight


def build_rebind_from_checkpoint(path, device, new_max_len=REBIND_MAX_LEN):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if "encoder" not in obj:
        raise KeyError("Expected a REBIND checkpoint containing key 'encoder'")
    old_max_len = int(obj.get("max_len", 0))
    if old_max_len <= 0:
        raise ValueError("Checkpoint must contain max_len")
    cfg = _config_from_checkpoint(obj)
    encoder = REBINDEncoder(max_len=int(new_max_len), **cfg).to(device)
    src = obj["encoder"]
    dst = encoder.state_dict()
    adapted = {}
    for key, value in src.items():
        if key == "position_branch.0.weight" and value.shape != dst[key].shape:
            adapted[key] = _adapt_position_weight(value, old_max_len, int(new_max_len))
        elif key in dst and value.shape == dst[key].shape:
            adapted[key] = value
        else:
            raise ValueError(f"Cannot transfer encoder tensor {key}: {tuple(value.shape)} -> {tuple(dst[key].shape)}")
    encoder.load_state_dict(adapted, strict=True)
    bit_head = AnchorBitHead(embed_dim=cfg["embed_dim"], anchor_len=CODE_BITS).to(device)
    return encoder, bit_head, obj


def save_stage_checkpoint(path, encoder, bit_head, source_checkpoint, stage, epoch, val_loss, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "bit_head": bit_head.state_dict(),
            "source_checkpoint": str(source_checkpoint),
            "stage": str(stage),
            "epoch": int(epoch),
            "val_loss": float(val_loss),
            "config": dict(config),
            "max_len": int(encoder.max_len),
        },
        path,
    )


def load_stage_checkpoint(path, device):
    obj = torch.load(path, map_location=device, weights_only=False)
    cfg = dict(obj["config"])
    encoder = REBINDEncoder(
        max_len=int(obj["max_len"]),
        embed_dim=int(cfg["embed_dim"]),
        symbol_emb_dim=int(cfg["symbol_emb_dim"]),
        hidden_size=int(cfg["hidden_size"]),
        num_layers=int(cfg["num_layers"]),
        position_branch_dim=int(cfg["position_branch_dim"]),
        dropout=float(cfg["dropout"]),
    ).to(device)
    bit_head = AnchorBitHead(embed_dim=int(cfg["embed_dim"]), anchor_len=CODE_BITS).to(device)
    encoder.load_state_dict(obj["encoder"], strict=True)
    bit_head.load_state_dict(obj["bit_head"], strict=True)
    return encoder, bit_head, obj
