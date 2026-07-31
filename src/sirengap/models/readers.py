"""Permutation-equivariant readers over L=2 sine weights (rung W11, Ch3.7).

Every rung of the S1 ladder so far changes the *feature map* and holds the reader fixed at a
plain MLP. That is what makes the decomposition interpretable, but it leaves the ladder with no
comparison against the thing the field actually builds: an equivariant weight-space architecture.
W11 supplies it, in two variants that answer two different questions.

**W11a — `RawGraphReader`.** The literature's construction: treat the network as a bipartite graph
(layer-1 neurons, layer-2 neurons, W2 as edges) and message-pass with permutation-equivariant
operations. Equivariant to S_{n1} x S_{n2}; **not** invariant to D_infinity, because raw w, b and u
move under sign flips and phase shifts. This is the DWSNets/NFN/GMN family's coverage for sine
networks, and measuring it answers: *does reader architecture substitute for frame choice?*

**W11b — `InvariantGraphReader`.** Ch3.7's construction: the node and edge features are exactly the
D_infinity-invariant quantities W10 already uses — per-neuron even scalars, and the sign-cancelling
matrices A, B, C built from the layer-2 Gram — but the pooling is *learned* and equivariant instead
of W10's sorted eigenvalue spectra. Invariant to the full product group by construction. Measuring
it answers OPEN_PROBLEMS #4's remaining half: *is W10's eigenvalue pooling the bottleneck, or are
the invariants themselves?*

Both readers consume pre-extracted feature dictionaries rather than `SirenParams`, so that
normalisation statistics can be fitted on the training split once and reused, exactly as the
matched-MLP decoder z-scores its own input.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from sirengap.canon.deep_invariants import layer2_features
from sirengap.models.params import SirenParams, outgoing


# ------------------------------------------------------------------ feature extraction


def raw_graph_features(params: SirenParams) -> dict[str, Tensor]:
    """Bipartite view of an L=2 network, no invariance imposed (W11a).

    x1 [B, n1, m+1]: each layer-1 neuron's incoming row and bias.
    x2 [B, n2, 1+c]: each layer-2 neuron's bias and outgoing column.
    e  [B, n2, n1]:  W2, the edges between them.
    """
    if params.n_layers != 2:
        raise ValueError(f"W11 readers are derived for L=2 (got L={params.n_layers})")
    w1, b1 = params.hidden[0]
    w2, b2 = params.hidden[1]
    u = params.w_out  # [B, c, n2]
    return {
        "x1": torch.cat([w1, b1[:, :, None]], dim=2),
        "x2": torch.cat([b2[:, :, None], u.transpose(1, 2)], dim=2),
        "e": w2,
    }


def invariant_graph_features(params: SirenParams) -> dict[str, Tensor]:
    """D_infinity-invariant view over the layer-1 neuron graph (W11b).

    Node features are W10's per-neuron even scalars; edge features are the three sign-cancelling
    matrices of Proposition (exact L=2 invariant encoding) *before* they are collapsed to
    eigenvalue spectra; the global vector carries the layer-2 block unchanged.
    """
    if params.n_layers != 2:
        raise ValueError(f"W11 readers are derived for L=2 (got L={params.n_layers})")
    w, b = params.hidden[0]
    out = outgoing(params, 0)  # W2, [B, n2, n1]
    gram = torch.einsum("bki,bkl->bil", out, out)  # [B, n1, n1], invariant under layer 2

    sin_b, cos_b, sin_2b, cos_2b = torch.sin(b), torch.cos(b), torch.sin(2 * b), torch.cos(2 * b)
    wb = cos_b[:, :, None] * w
    wc = sin_2b[:, :, None] * w

    a_mat = sin_b[:, :, None] * sin_b[:, None, :] * gram
    b_mat = torch.einsum("bim,blm->bil", wb, wb) * gram
    c_mat = torch.einsum("bim,blm->bil", wc, wc)

    nodes = torch.stack(
        [(w * w).sum(dim=2), cos_2b, torch.diagonal(gram, dim1=1, dim2=2)], dim=2
    )  # [B, n1, 3]
    edges = torch.stack([a_mat, b_mat, c_mat], dim=3)  # [B, n1, n1, 3]
    return {"x1": nodes, "e": edges, "g": layer2_features(params)}


def feature_stats(feats: dict[str, Tensor]) -> dict[str, tuple[Tensor, Tensor]]:
    """Per-channel mean/std over batch and index axes, for input standardisation."""
    stats = {}
    for k, v in feats.items():
        dims = tuple(range(v.ndim - 1))
        mean = v.mean(dim=dims, keepdim=True)
        std = v.std(dim=dims, keepdim=True).clamp_min(1e-6)
        stats[k] = (mean, std)
    return stats


def apply_stats(
    feats: dict[str, Tensor], stats: dict[str, tuple[Tensor, Tensor]]
) -> dict[str, Tensor]:
    return {k: (v - stats[k][0].to(v.device)) / stats[k][1].to(v.device) for k, v in feats.items()}


# ------------------------------------------------------------------ readers


def _mlp(sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


class RawGraphReader(nn.Module):
    """W11a: bipartite message passing on raw weights. S_n-equivariant, not D_inf-invariant."""

    def __init__(self, m: int, c: int, width: int = 128, rounds: int = 2, n_classes: int = 10):
        super().__init__()
        self.rounds = rounds
        self.enc1 = nn.Linear(m + 1, width)
        self.enc2 = nn.Linear(1 + c, width)
        self.up1 = nn.ModuleList([nn.Linear(2 * width, width) for _ in range(rounds)])
        self.up2 = nn.ModuleList([nn.Linear(2 * width, width) for _ in range(rounds)])
        self.head = _mlp([4 * width, 256, n_classes])

    def forward(self, f: dict[str, Tensor]) -> Tensor:
        h1 = self.enc1(f["x1"])  # [B, n1, d]
        h2 = self.enc2(f["x2"])  # [B, n2, d]
        e = f["e"]  # [B, n2, n1]
        n1, n2 = h1.shape[1], h2.shape[1]
        for r in range(self.rounds):
            # edge-weighted aggregation: covariant under independent permutations of both axes
            m1 = torch.einsum("bji,bjd->bid", e, h2) / n2
            m2 = torch.einsum("bji,bid->bjd", e, h1) / n1
            h1 = torch.nn.functional.gelu(self.up1[r](torch.cat([h1, m1], dim=2)))
            h2 = torch.nn.functional.gelu(self.up2[r](torch.cat([h2, m2], dim=2)))
        pooled = torch.cat(
            [h1.mean(1), h1.amax(1), h2.mean(1), h2.amax(1)], dim=1
        )  # invariant readout
        return self.head(pooled)


class InvariantGraphReader(nn.Module):
    """W11b: W10's invariants with learned equivariant pooling. Invariant to the full group.

    Messages are multi-relational rather than gated per feature channel: the three invariant edge
    matrices are mapped to `relations` continuous relation weights, and each relation carries its
    own linear map of the neighbour states. That keeps the edge tensor at [B, n, n, relations]
    instead of [B, n, n, width] — the difference between ~4 MB and ~800 MB at width 384 — while
    staying exactly as permutation-covariant.
    """

    def __init__(self, n_node: int, n_edge: int, n_global: int, width: int = 128,
                 rounds: int = 2, relations: int = 8, n_classes: int = 10):
        super().__init__()
        self.rounds, self.relations = rounds, relations
        self.enc = nn.Linear(n_node, width)
        self.edge = nn.ModuleList([_mlp([n_edge, 64, relations]) for _ in range(rounds)])
        self.val = nn.ModuleList(
            [nn.Linear(width, relations * width, bias=False) for _ in range(rounds)]
        )
        self.upd = nn.ModuleList([nn.Linear(2 * width, width) for _ in range(rounds)])
        self.head = _mlp([2 * width + n_global, 256, n_classes])

    def forward(self, f: dict[str, Tensor]) -> Tensor:
        h = self.enc(f["x1"])  # [B, n, d]
        e = f["e"]  # [B, n, n, k]
        b, n, d = h.shape
        for r in range(self.rounds):
            rel = self.edge[r](e)  # [B, n, n, R]
            vals = self.val[r](h).view(b, n, self.relations, d)  # [B, n, R, d]
            msg = torch.einsum("bilr,blrd->bid", rel, vals) / n
            h = torch.nn.functional.gelu(self.upd[r](torch.cat([h, msg], dim=2)))
        pooled = torch.cat([h.mean(1), h.amax(1), f["g"]], dim=1)
        return self.head(pooled)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
