from pathlib import Path
import torch


def save_checkpoint(
    path,
    encoder,
    decoder,
    bit_head,
    calibration,
    config,
    max_len,
    anchor_len,
    stage,
    epoch,
    score,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "bit_head": bit_head.state_dict(),
            "calibration": calibration.state_dict(),
            "config": config,
            "max_len": int(max_len),
            "anchor_len": int(anchor_len),
            "stage": str(stage),
            "epoch": int(epoch),
            "score": float(score),
        },
        path,
    )


def load_checkpoint(
    path,
    encoder,
    decoder,
    bit_head,
    calibration,
    device,
):
    obj = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    encoder.load_state_dict(obj["encoder"])
    decoder.load_state_dict(obj["decoder"])
    bit_head.load_state_dict(obj["bit_head"])
    calibration.load_state_dict(obj["calibration"])

    return obj
