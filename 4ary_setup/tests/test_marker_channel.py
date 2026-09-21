import numpy as np

from rebind4ary.ids_channel import ids_channel_matlab
from rebind4ary.marker import marker_decode_payload_symbols, marker_encode, payload_positions


def test_marker_roundtrip_204_bits():
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, size=204, dtype=np.int64)
    x = marker_encode(bits)
    assert x.shape == (142,)
    assert set(np.unique(x)).issubset({0, 1, 2, 3})
    assert np.array_equal(marker_decode_payload_symbols(x), bits)
    j1, j2 = payload_positions()
    assert len(j1) == 102
    assert len(j2) == 102


def test_zero_noise_channel_is_identity():
    x = np.arange(142, dtype=np.int64) % 4
    y = ids_channel_matlab(x, 0.0, 0.0, 0.0, rng=np.random.default_rng(2))
    assert np.array_equal(x, y)
