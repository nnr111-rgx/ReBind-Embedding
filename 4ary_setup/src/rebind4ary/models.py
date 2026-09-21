from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .config import EMBED_DIM, PAD, VOCAB_SIZE


class MaskedAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.score = nn.Linear(dim, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.score(torch.tanh(self.fc(x))).squeeze(-1)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        w = torch.softmax(logits, dim=-1)
        w = w * mask.to(w.dtype)
        w = w / w.sum(-1, keepdim=True).clamp_min(1e-12)
        return torch.bmm(w.unsqueeze(1), x).squeeze(1)


class FourAryReBindEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        symbol_emb_dim: int = 64,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.symbol_emb_dim = int(symbol_emb_dim)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.symbol_embedding = nn.Embedding(VOCAB_SIZE, symbol_emb_dim, padding_idx=PAD)
        self.bigru = nn.GRU(
            symbol_emb_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = MaskedAttention(2 * hidden_size)
        summary_dim = 4 * hidden_size
        self.projection = nn.Sequential(
            nn.Linear(summary_dim, 2 * summary_dim),
            nn.GELU(),
            nn.LayerNorm(2 * summary_dim),
            nn.Dropout(dropout),
            nn.Linear(2 * summary_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        x = x.long()
        mask = x.ne(PAD)
        if lengths is None:
            lengths = mask.sum(1).long().clamp_min(1)
        emb = self.symbol_embedding(x)
        packed = pack_padded_sequence(
            emb,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, h = self.bigru(packed)
        out, _ = pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=x.size(1),
        )
        final_hidden = torch.cat([h[-2], h[-1]], dim=-1)
        attn = self.attention(out, mask)
        return self.projection(torch.cat([final_hidden, attn], dim=-1))

    def config(self) -> dict:
        return {
            "embed_dim": self.embed_dim,
            "symbol_emb_dim": self.symbol_emb_dim,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }


class AffineDistance(nn.Module):
    def __init__(self, init_scale: float = 1.0, init_bias: float = 0.0):
        super().__init__()
        inv = torch.log(torch.expm1(torch.tensor(float(init_scale)).clamp_min(1e-6)))
        self.raw_scale = nn.Parameter(inv)
        self.bias = nn.Parameter(torch.tensor(float(init_bias)))

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.raw_scale) * distance + self.bias

    def values(self) -> tuple[float, float]:
        return float(F.softplus(self.raw_scale).detach()), float(self.bias.detach())


def l2(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    return torch.norm(z1 - z2, p=2, dim=-1)


def build_encoder(config: dict | None = None) -> FourAryReBindEncoder:
    cfg = dict(config or {})
    return FourAryReBindEncoder(
        embed_dim=int(cfg.get("embed_dim", EMBED_DIM)),
        symbol_emb_dim=int(cfg.get("symbol_emb_dim", 64)),
        hidden_size=int(cfg.get("hidden_size", 256)),
        num_layers=int(cfg.get("num_layers", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
    )
