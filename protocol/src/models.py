"""Faithful re-implementations of Guardian and TrustGuard, retrained under our protocol.

Why re-implement rather than run the released code:

  * Guardian's repo targets torch 1.8 and crashes on modern NumPy (`np.float`), and its
    `calculate_auc` returns a binary F1 in the slot the caller unpacks as MAE - so the
    numbers it prints are not the numbers it claims. Running it unchanged would compare
    our protocol against a broken meter.
  * TrustGuard's released `setup_layers` never passes `robust_aggr=True`, so the defence
    mechanism that is the paper's headline contribution is dead code in the artifact.
  * Both select their reported epoch on the TEST split. Reusing their training loop would
    import that leak into our comparison.

The layer algebra below is transcribed from the released `convolution.py` of each repo so
that the models are theirs; only the data pipeline, the epoch-selection rule and the
metrics are ours.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# Scatter ops (Guardian ships its own in mean.py; TrustGuard uses torch_scatter)
# --------------------------------------------------------------------------------------

def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros(dim_size, src.size(1), dtype=src.dtype, device=src.device)
    out.index_add_(0, index, src)
    cnt = torch.zeros(dim_size, 1, dtype=src.dtype, device=src.device)
    cnt.index_add_(0, index, torch.ones(src.size(0), 1, dtype=src.dtype, device=src.device))
    return out / cnt.clamp(min=1.0)


# --------------------------------------------------------------------------------------
# Shared convolution: the in/out role-separated aggregator both papers use
# --------------------------------------------------------------------------------------

class RoleSeparatedConv(nn.Module):
    """`ConvolutionBase_in_out` / `ConvolutionDeep_in_out`, identical in both repos.

    row = trustor, col = trustee. Four streams are aggregated and concatenated:
      out, opinion         - what u, acting as a trustor, points at and how it rates
      inn, inn_opinion     - who points at u as a trustee and how they rate it
    Note neither repo concatenates the node's own feature here: there is no self-loop in
    this branch, so a node's own embedding reaches it only through its neighbours.
    """

    def __init__(self, in_channels: int, out_channels: int, num_labels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        self.trans_weight = nn.Parameter(torch.empty(num_labels, in_channels // 4))
        self.bias = nn.Parameter(torch.empty(out_channels))
        stdv = 1.0 / np.sqrt(in_channels)
        nn.init.uniform_(self.weight, -stdv, stdv)
        nn.init.uniform_(self.bias, -stdv, stdv)
        stdv2 = 1.0 / np.sqrt(num_labels)
        nn.init.uniform_(self.trans_weight, -stdv2, stdv2)

    def forward(self, x, edge_index, edge_label):
        row, col = edge_index
        n = x.size(0)
        elt = edge_label @ self.trans_weight
        opinion = scatter_mean(elt, row, n)
        out = scatter_mean(x[col], row, n)
        inn_opinion = scatter_mean(elt, col, n)
        inn = scatter_mean(x[row], col, n)
        h = torch.cat((out, opinion, inn, inn_opinion), dim=1)
        return h @ self.weight + self.bias


# --------------------------------------------------------------------------------------
# Guardian (INFOCOM 2020)
# --------------------------------------------------------------------------------------

class Guardian(nn.Module):
    """Single trust-convolution layer with tanh, then a bilinear-free readout.

    Defaults follow `arg_parser.py`: layers=[64], both_degree=True, dropout=0.
    """

    def __init__(self, in_dim: int, num_labels: int, layers=(64,), dropout: float = 0.0):
        super().__init__()
        self.dropout = dropout
        dims = [in_dim] + list(layers)
        self.convs = nn.ModuleList([
            RoleSeparatedConv(dims[i] * 4, dims[i + 1], num_labels)
            for i in range(len(layers))
        ])
        self.regression_weights = nn.Parameter(torch.empty(layers[-1] * 2, num_labels))
        nn.init.xavier_normal_(self.regression_weights)

    def encode(self, x, edge_index, edge_label):
        h = F.dropout(x, self.dropout, training=self.training)
        for conv in self.convs:
            h = torch.tanh(conv(h, edge_index, edge_label))
            h = F.dropout(h, self.dropout, training=self.training)
        return h

    def score(self, z, pairs):
        return torch.cat((z[pairs[0]], z[pairs[1]]), dim=1) @ self.regression_weights


# --------------------------------------------------------------------------------------
# TrustGuard (IEEE TDSC 2024)
# --------------------------------------------------------------------------------------

class TemporalAttention(nn.Module):
    """Position-aware self-attention over snapshots (`AttentionNetwork`).

    The released mask opens only the final row, i.e. only the target timeslot attends over
    the whole history - matching Eq. 23-25, which define alpha^{<t_i, t_n>} alone. We keep
    that, including the paper-divergent sqrt(T) scaling the code actually uses.
    """

    def __init__(self, dim: int, n_heads: int, n_slots: int, attn_drop: float = 0.5):
        super().__init__()
        self.n_heads, self.n_slots = n_heads, n_slots
        self.pos = nn.Parameter(torch.empty(n_slots, dim))
        self.Q = nn.Parameter(torch.empty(dim, dim))
        self.K = nn.Parameter(torch.empty(dim, dim))
        self.V = nn.Parameter(torch.empty(dim, dim))
        for p in (self.pos, self.Q, self.K, self.V):
            nn.init.xavier_uniform_(p)
        self.lin = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(attn_drop)

    def forward(self, h):                      # h: [N, T, F]
        x = h + self.pos.unsqueeze(0)
        q, k, v = x @ self.Q, x @ self.K, x @ self.V
        s = q.shape[-1] // self.n_heads
        q_, k_, v_ = (torch.cat(torch.split(t, s, dim=2), dim=0) for t in (q, k, v))
        a = (q_ @ k_.transpose(1, 2)) / (self.n_slots ** 0.5)
        mask = torch.cat([torch.zeros(self.n_slots - 1, self.n_slots),
                          torch.ones(1, self.n_slots)], dim=0).to(h.device)
        a = torch.where(mask.unsqueeze(0) == 0, torch.full_like(a, -2 ** 31), a)
        a = self.drop(torch.softmax(a, dim=2)) if self.training else torch.softmax(a, dim=2)
        o = a @ v_
        o = torch.cat(torch.split(o, o.shape[0] // self.n_heads, dim=0), dim=2)
        return F.relu(self.lin(o)) + o + x      # feed-forward residual, then outer residual


class TrustGuard(nn.Module):
    """Spatial stack (relu, layers=[32,64,32]) shared across snapshots, then temporal
    attention. With a single snapshot the temporal layer is a no-op by construction, which
    is exactly how TrustGuard behaves on a dataset without timestamps.
    """

    def __init__(self, in_dim: int, num_labels: int, layers=(32, 64, 32),
                 n_slots: int = 1, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.dropout = dropout
        self.n_slots = n_slots
        dims = [in_dim] + list(layers)
        self.convs = nn.ModuleList([
            RoleSeparatedConv(dims[i] * 4, dims[i + 1], num_labels)
            for i in range(len(layers))
        ])
        self.temporal = (TemporalAttention(layers[-1], n_heads, n_slots)
                         if n_slots > 1 else None)
        self.regression_weights = nn.Parameter(torch.empty(layers[-1] * 2, num_labels))
        nn.init.xavier_normal_(self.regression_weights)

    def _spatial(self, x, edge_index, edge_label):
        h = F.dropout(x, self.dropout, training=self.training)
        for conv in self.convs:
            h = torch.relu(conv(h, edge_index, edge_label))
            h = F.dropout(h, self.dropout, training=self.training)
        return h

    def encode(self, x, snapshots):
        """snapshots: list of (edge_index, edge_label), chronologically ordered."""
        outs = [self._spatial(x, ei, el) for ei, el in snapshots]
        if self.temporal is None:
            return outs[-1]
        h = torch.stack(outs).permute(1, 0, 2)          # [N, T, F]
        return self.temporal(h)[:, -1, :]

    def score(self, z, pairs):
        return torch.cat((z[pairs[0]], z[pairs[1]]), dim=1) @ self.regression_weights
