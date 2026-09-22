import numpy as np

from rebind4ary.channel import ids_channel
from rebind4ary.coding import build_marker_patterns, conv_encode_bits, marcode, qary_to_binary


def test_cc_marker_lengths():
    msg = np.zeros(100, dtype=np.int64)
    c = conv_encode_bits(msg)
    mp1, mp2 = build_marker_patterns()
    x = marcode(c, mp1, mp2)
    assert c.size == 204
    assert x.size == 142
    assert set(np.unique(x)).issubset({0, 1, 2, 3})


def test_qary_binary_round_length():
    x = np.asarray([0, 1, 2, 3], dtype=np.int64)
    b = qary_to_binary(x)
    assert b.tolist() == [0, 0, 0, 1, 1, 0, 1, 1]


def test_channel_bounds():
    x = np.arange(142, dtype=np.int64) % 4
    y = ids_channel(x, 0.03, 0.03, 0.05, rng=np.random.default_rng(1), l_max=2)
    assert 0 <= y.size <= 426
    assert set(np.unique(y)).issubset({0, 1, 2, 3})
