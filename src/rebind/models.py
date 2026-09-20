import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class MaskedAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1, bias=False),
        )

    def forward(self, h, mask):
        score = self.score(h).squeeze(-1)
        score = score.masked_fill(~mask, torch.finfo(score.dtype).min)

        weight = torch.softmax(score, dim=-1)
        weight = weight * mask.to(weight.dtype)
        weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        return torch.bmm(weight.unsqueeze(1), h).squeeze(1)


class REBINDEncoder(nn.Module):
    def __init__(
        self,
        max_len,
        embed_dim=300,
        symbol_emb_dim=16,
        hidden_size=256,
        num_layers=2,
        position_branch_dim=256,
        dropout=0.0,
    ):
        super().__init__()

        self.max_len = int(max_len)
        self.embed_dim = int(embed_dim)

        self.symbol_embedding = nn.Embedding(
            3,
            symbol_emb_dim,
            padding_idx=0,
        )

        self.bigru = nn.GRU(
            symbol_emb_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.attention = MaskedAttention(2 * hidden_size)

        self.position_branch = nn.Sequential(
            nn.Linear(2 * self.max_len, 2 * position_branch_dim),
            nn.GELU(),
            nn.LayerNorm(2 * position_branch_dim),
            nn.Linear(2 * position_branch_dim, position_branch_dim),
            nn.GELU(),
            nn.LayerNorm(position_branch_dim),
        )

        recurrent_dim = 4 * hidden_size
        joint_dim = recurrent_dim + position_branch_dim

        self.projection = nn.Sequential(
            nn.Linear(joint_dim, 2 * joint_dim),
            nn.GELU(),
            nn.LayerNorm(2 * joint_dim),
            nn.Dropout(dropout),
            nn.Linear(2 * joint_dim, embed_dim),
        )

    def forward(self, x):
        if x.ndim != 2:
            raise ValueError(f"Expected [B,L], got {tuple(x.shape)}")

        if x.size(1) != self.max_len:
            raise ValueError(
                f"Expected padded length {self.max_len}, got {x.size(1)}"
            )

        mask = x.ge(0)
        lengths = mask.sum(1).long().clamp_min(1)

        ids = torch.zeros_like(x, dtype=torch.long)
        ids[mask] = x[mask].long() + 1

        emb = self.symbol_embedding(ids)

        packed = pack_padded_sequence(
            emb,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        packed_out, hidden = self.bigru(packed)

        out, _ = pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=x.size(1),
        )

        final_hidden = torch.cat(
            [hidden[-2], hidden[-1]],
            dim=-1,
        )

        attention_summary = self.attention(out, mask)

        recurrent_summary = torch.cat(
            [final_hidden, attention_summary],
            dim=-1,
        )

        signed = torch.zeros_like(x, dtype=torch.float32)
        signed[mask] = 2.0 * x[mask].float() - 1.0

        position_input = torch.cat(
            [signed, mask.float()],
            dim=-1,
        )

        position_summary = self.position_branch(position_input)

        summary = torch.cat(
            [recurrent_summary, position_summary],
            dim=-1,
        )

        return self.projection(summary)


class ARGRUDecoder(nn.Module):
    def __init__(
        self,
        embed_dim=300,
        token_emb_dim=64,
        context_dim=256,
        hidden_size=512,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)

        self.token_embedding = nn.Embedding(3, token_emb_dim)

        self.context = nn.Sequential(
            nn.Linear(embed_dim, context_dim),
            nn.GELU(),
            nn.LayerNorm(context_dim),
        )

        self.hidden_init = nn.Sequential(
            nn.Linear(embed_dim, num_layers * hidden_size),
            nn.Tanh(),
        )

        self.gru = nn.GRU(
            token_emb_dim + context_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 3),
        )

    def initial_hidden(self, z):
        batch = z.size(0)

        return (
            self.hidden_init(z)
            .view(batch, self.num_layers, self.hidden_size)
            .transpose(0, 1)
            .contiguous()
        )

    def forward(self, z, input_ids):
        token = self.token_embedding(input_ids)

        context = self.context(z).unsqueeze(1).expand(
            -1,
            input_ids.size(1),
            -1,
        )

        decoder_input = torch.cat(
            [token, context],
            dim=-1,
        )

        out, _ = self.gru(
            decoder_input,
            self.initial_hidden(z),
        )

        return self.output(out)

    @torch.no_grad()
    def generate(self, z, max_len, fixed_length=None):
        batch = z.size(0)
        device = z.device

        context = self.context(z)
        hidden = self.initial_hidden(z)

        token = torch.full(
            (batch,),
            2,
            dtype=torch.long,
            device=device,
        )

        if fixed_length is not None:
            bits = []

            for _ in range(int(fixed_length)):
                emb = self.token_embedding(token).unsqueeze(1)

                out, hidden = self.gru(
                    torch.cat(
                        [emb, context.unsqueeze(1)],
                        dim=-1,
                    ),
                    hidden,
                )

                logits = self.output(out[:, 0, :])
                bit = logits[:, :2].argmax(dim=-1)

                bits.append(bit)
                token = bit

            return (
                torch.stack(bits, dim=1),
                torch.full(
                    (batch,),
                    int(fixed_length),
                    dtype=torch.long,
                    device=device,
                ),
            )

        generated = torch.zeros(
            batch,
            max_len,
            dtype=torch.long,
            device=device,
        )

        lengths = torch.full(
            (batch,),
            max_len,
            dtype=torch.long,
            device=device,
        )

        position = torch.zeros(
            batch,
            dtype=torch.long,
            device=device,
        )

        finished = torch.zeros(
            batch,
            dtype=torch.bool,
            device=device,
        )

        for _ in range(max_len + 1):
            emb = self.token_embedding(token).unsqueeze(1)

            out, hidden = self.gru(
                torch.cat(
                    [emb, context.unsqueeze(1)],
                    dim=-1,
                ),
                hidden,
            )

            pred = self.output(out[:, 0, :]).argmax(dim=-1)

            stop = (~finished) & pred.eq(2)
            lengths[stop] = position[stop]

            active = (
                (~finished)
                & pred.ne(2)
                & position.lt(max_len)
            )

            rows = torch.nonzero(
                active,
                as_tuple=False,
            ).squeeze(-1)

            if rows.numel() > 0:
                generated[
                    rows,
                    position[rows],
                ] = pred[rows]

                position[rows] += 1

            finished |= stop

            token = torch.where(
                pred.eq(2),
                torch.zeros_like(pred),
                pred,
            )

            if bool(finished.all()):
                break

        return generated, lengths


class AnchorBitHead(nn.Module):
    def __init__(self, embed_dim, anchor_len):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, anchor_len),
        )

    def forward(self, z):
        return self.net(z)


class AffineDistance(nn.Module):
    def __init__(self):
        super().__init__()

        self.scale_raw = nn.Parameter(torch.tensor(0.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, d):
        return F.softplus(self.scale_raw) * d + self.bias

    def values(self):
        return (
            float(F.softplus(self.scale_raw).detach().cpu()),
            float(self.bias.detach().cpu()),
        )
