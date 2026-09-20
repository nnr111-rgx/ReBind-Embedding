import torch

from rebind.models import REBINDEncoder, ARGRUDecoder, AnchorBitHead


def test_shapes():
    batch = 4
    max_len = 107
    anchor_len = 100

    x = torch.randint(
        0,
        2,
        (batch, max_len),
    )

    encoder = REBINDEncoder(
        max_len=max_len,
        embed_dim=300,
    )

    decoder = ARGRUDecoder(
        embed_dim=300,
    )

    bit_head = AnchorBitHead(
        300,
        anchor_len,
    )

    z = encoder(x)

    assert z.shape == (
        batch,
        300,
    )

    bits, lengths = decoder.generate(
        z,
        max_len=max_len,
        fixed_length=anchor_len,
    )

    assert bits.shape == (
        batch,
        anchor_len,
    )

    assert lengths.shape == (
        batch,
    )

    assert bit_head(z).shape == (
        batch,
        anchor_len,
    )
