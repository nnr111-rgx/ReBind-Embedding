import numpy as np
import torch

from rebind4ary.models import FourAryReBindEncoder
from rebind4ary.whitening import fit_whitener


def test_encoder_shape():
    model = FourAryReBindEncoder(embed_dim=204, symbol_emb_dim=16, hidden_size=16, num_layers=1, dropout=0.0)
    x = torch.full((3, 12), 4, dtype=torch.long)
    x[0, :10] = torch.randint(0, 4, (10,))
    x[1, :7] = torch.randint(0, 4, (7,))
    x[2, :12] = torch.randint(0, 4, (12,))
    lengths = torch.tensor([10, 7, 12])
    z = model(x, lengths)
    assert z.shape == (3, 204)
    assert torch.isfinite(z).all()


def test_whitener_reduces_covariance_scale_mismatch():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(5000, 4)) @ np.array([[2.0, 0.5, 0.0, 0.0], [0.0, 0.5, 0.2, 0.0], [0.0, 0.0, 1.5, 0.3], [0.1, 0.0, 0.0, 0.8]])
    w = fit_whitener(x, shrinkage=0.0, eps_relative=1e-6)
    y = (x - w.mean.numpy()) @ w.matrix.numpy().T
    cov = np.cov(y, rowvar=False)
    assert np.max(np.abs(cov - np.eye(4))) < 0.08
