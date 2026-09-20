import torch

from .models import REBINDEncoder, ARGRUDecoder, AnchorBitHead, AffineDistance


def build_models(config, max_len, anchor_len, device):
    encoder = REBINDEncoder(
        max_len=max_len,
        embed_dim=int(config.get("embed_dim", 300)),
        symbol_emb_dim=int(config.get("symbol_emb_dim", 16)),
        hidden_size=int(config.get("hidden_size", 256)),
        num_layers=int(config.get("num_layers", 2)),
        position_branch_dim=int(config.get("position_branch_dim", 256)),
        dropout=float(config.get("encoder_dropout", 0.0)),
    ).to(device)

    decoder = ARGRUDecoder(
        embed_dim=int(config.get("embed_dim", 300)),
        token_emb_dim=int(config.get("decoder_token_emb", 64)),
        context_dim=int(config.get("decoder_context_dim", 256)),
        hidden_size=int(config.get("decoder_hidden", 512)),
        num_layers=int(config.get("decoder_layers", 2)),
        dropout=float(config.get("decoder_dropout", 0.1)),
    ).to(device)

    bit_head = AnchorBitHead(
        embed_dim=int(config.get("embed_dim", 300)),
        anchor_len=anchor_len,
    ).to(device)

    calibration = AffineDistance().to(device)

    return encoder, decoder, bit_head, calibration
