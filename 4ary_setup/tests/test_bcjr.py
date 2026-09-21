import numpy as np

from rebind4ary.bcjr import oracle_bit_llr
from rebind4ary.marker import marker_encode


def test_bcjr_output_shape_and_finite():
    rng = np.random.default_rng(4)
    bits = rng.integers(0, 2, size=204, dtype=np.int64)
    clean = marker_encode(bits)
    llr = oracle_bit_llr(clean, 0.0, 0.0, 0.01, clip=100.0)
    assert llr.shape == (204,)
    assert np.isfinite(llr).all()
