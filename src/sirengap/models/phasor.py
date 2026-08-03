"""Phasor-graded equivariant reader for L=2 sine networks (rung W12).

The external review's Priority 4. W11b is $G$-invariant, but only because it consumes W10's
fixed invariant family, so it inherits whatever that family discards. This module is a reader
that quotients $D_\\infty$ on the **raw parameters**.

The construction rests on one observation. Under the per-neuron element
$g_{d,j}: (w, b, u) \\mapsto ((-1)^d w,\\ (-1)^d b + \\pi j,\\ (-1)^{d+j} u)$, the *phasor*
coordinates of the bias transform with $j$ only through its **parity**:

    cos b  ->  (-1)^j cos b            sin b  ->  (-1)^{d+j} sin b
    cos 2b ->  cos 2b                  sin 2b ->  (-1)^d sin 2b
    w      ->  (-1)^d w                u      ->  (-1)^{d+j} u

So replacing $b$ by $(\\cos b, \\sin b)$ turns the infinite group $\\Z \\rtimes \\Z_2$ into a
**finite** $\\Z_2 \\times \\Z_2$ acting by signs. Write a *character* $\\chi = (a,c)$ for a feature
that picks up $(-1)^{ad + cj}$. The four blocks are then

    (0,0)  ||w||^2, cos 2b, 1          (1,0)  w, sin 2b
    (0,1)  cos b                       (1,1)  sin b, u

and every layer below preserves the grading by construction, which is what makes the whole
reader exactly invariant rather than approximately so -- and invariant for *unbounded* windings,
since the phasor quotients the integer translation exactly (T16 tests |j| <= 40).

Two-layer bookkeeping. Each hidden layer carries its own $\\Z_2\\times\\Z_2$, and the second-layer
weight matrix $W^2$ sits between them: its entry $W^2_{ki}$ carries character $(1,1)$ for layer-1
neuron $i$ and $(1,0)$ for layer-2 neuron $k$. A message is legitimate only if it arrives
*neutral* in the source layer's grading, which leaves exactly two channels in each direction:

    layer 1 -> layer 2:   sum_i W^2_{ki} f_i^{(1,1)}   (lands in layer-2 character (1,0))
                          sum_i (W^2_{ki})^2 f_i^{(0,0)}   (lands in (0,0))
    layer 2 -> layer 1:   sum_k W^2_{ki} g_k^{(1,0)}   (lands in layer-1 character (1,1))
                          sum_k (W^2_{ki})^2 g_k^{(0,0)}   (lands in (0,0))

That is the Gram coupling of Proposition~\\ref{prop:deep} rediscovered as a message rule --- but
here it is neither pooled nor fixed: the network learns which couplings to use, and a graded
bilinear layer before each round lets characters that cannot pass alone (a $(1,0)$ and a $(0,1)$)
combine into one that can.

Permutation equivariance comes for free: every message is a sum over the other layer's index, and
the readout pools over neurons. The readout reads only $(0,0)$ channels, which is where invariance
is finally cashed in.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from sirengap.models.params import SirenParams, outgoing

Character = tuple[int, int]
CHARACTERS: tuple[Character, ...] = ((0, 0), (1, 0), (0, 1), (1, 1))

# characters compose additively mod 2
def _add(x: Character, y: Character) -> Character:
    return ((x[0] + y[0]) % 2, (x[1] + y[1]) % 2)


def phasor_features(params: SirenParams) -> dict[str, dict[Character, Tensor]]:
    """Graded node features and the coupling matrix: no learned parameters, no pooling.

    Returns {"l1": {char: [B, n, d]}, "l2": {char: [B, p, d]}, "edge": [B, p, n]}.
    """
    if params.n_layers != 2:
        raise ValueError(f"phasor_features is derived for L=2 (got L={params.n_layers})")

    w1, b1 = params.hidden[0]  # [B, n, m], [B, n]
    w2, b2 = params.hidden[1]  # [B, p, n], [B, p]
    e = outgoing(params, 0)  # [B, p, n] = W^2, the coupling matrix
    u = params.w_out.transpose(1, 2)  # [B, p, c]

    ones_n = torch.ones_like(b1)[:, :, None]
    ones_p = torch.ones_like(b2)[:, :, None]
    e2 = e * e

    l1: dict[Character, Tensor] = {
        (0, 0): torch.cat([(w1 * w1).sum(2, keepdim=True), torch.cos(2 * b1)[:, :, None],
                           e2.sum(1).unsqueeze(2), ones_n], dim=2),
        (1, 0): torch.cat([w1, torch.sin(2 * b1)[:, :, None]], dim=2),
        (0, 1): torch.cos(b1)[:, :, None],
        (1, 1): torch.sin(b1)[:, :, None],
    }
    l2: dict[Character, Tensor] = {
        (0, 0): torch.cat([e2.sum(2, keepdim=True), (u * u).sum(2, keepdim=True),
                           torch.cos(2 * b2)[:, :, None], ones_p], dim=2),
        (1, 0): torch.sin(2 * b2)[:, :, None],
        (0, 1): torch.cos(b2)[:, :, None],
        (1, 1): torch.cat([torch.sin(b2)[:, :, None], u], dim=2),
    }
    return {"l1": l1, "l2": l2, "edge": e}


def feature_scale(train: dict) -> dict:
    """Per-channel scales for standardization that is *character-safe*.

    Scale only, never shift. A shift would add a constant to a sign-covariant channel and
    destroy its character, breaking the invariance this whole rung exists to have; dividing by
    a per-channel scale commutes with the sign action. The statistics are themselves
    $G$-invariant, because $|x|$ is unchanged by a sign flip and the mean over neurons is
    unchanged by a permutation -- asserted in T16.
    """
    stats: dict = {"l1": {}, "l2": {}}
    for layer in ("l1", "l2"):
        for c in CHARACTERS:
            stats[layer][c] = train[layer][c].abs().mean(dim=(0, 1)).clamp_min(1e-6)
    stats["edge"] = train["edge"].abs().mean().clamp_min(1e-6)
    return stats


def apply_scale(feats: dict, stats: dict) -> dict:
    return {
        "l1": {c: feats["l1"][c] / stats["l1"][c] for c in CHARACTERS},
        "l2": {c: feats["l2"][c] / stats["l2"][c] for c in CHARACTERS},
        "edge": feats["edge"] / stats["edge"],
    }


class GradedLinear(nn.Module):
    """One learned linear map per character. Characters never mix, so the grading survives."""

    def __init__(self, dims_in: dict[Character, int], width: int) -> None:
        super().__init__()
        self.lin = nn.ModuleDict(
            {str(c): nn.Linear(dims_in[c], width, bias=(c == (0, 0))) for c in CHARACTERS}
        )

    def forward(self, x: dict[Character, Tensor]) -> dict[Character, Tensor]:
        return {c: self.lin[str(c)](x[c]) for c in CHARACTERS}


class GradedBilinear(nn.Module):
    """Products add characters. This is how a (1,0) and a (0,1) reach (1,1) and become passable."""

    def __init__(self, width: int) -> None:
        super().__init__()
        pairs = [(x, y) for i, x in enumerate(CHARACTERS) for y in CHARACTERS[i:]]
        self.pairs = pairs
        self.mix = nn.ModuleDict({f"{x}|{y}": nn.Linear(width, width, bias=False)
                                  for x, y in pairs})

    def forward(self, x: dict[Character, Tensor]) -> dict[Character, Tensor]:
        out = {c: x[c] for c in CHARACTERS}
        for cx, cy in self.pairs:
            out[_add(cx, cy)] = out[_add(cx, cy)] + self.mix[f"{cx}|{cy}"](x[cx] * x[cy])
        return out


class GradedAct(nn.Module):
    """GELU on the neutral block; an *odd* nonlinearity elsewhere, which preserves the sign."""

    def forward(self, x: dict[Character, Tensor]) -> dict[Character, Tensor]:
        return {c: (torch.nn.functional.gelu(x[c]) if c == (0, 0) else torch.tanh(x[c]))
                for c in CHARACTERS}


class GradedMessage(nn.Module):
    """One round of bipartite message passing that respects both layers' gradings.

    Only two channels per direction are legitimate; see the module docstring.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.up_odd = nn.Linear(width, width, bias=False)     # (1,1) via E   -> l2 (1,0)
        self.up_even = nn.Linear(width, width, bias=False)    # (0,0) via E^2 -> l2 (0,0)
        self.down_odd = nn.Linear(width, width, bias=False)   # (1,0) via E   -> l1 (1,1)
        self.down_even = nn.Linear(width, width, bias=False)  # (0,0) via E^2 -> l1 (0,0)
        self.scale = nn.Parameter(torch.zeros(4))

    def forward(self, l1: dict[Character, Tensor], l2: dict[Character, Tensor],
                e: Tensor) -> tuple[dict[Character, Tensor], dict[Character, Tensor]]:
        n, p = l1[(0, 0)].shape[1], l2[(0, 0)].shape[1]
        e2 = e * e
        up_odd = torch.einsum("bpn,bnd->bpd", e, self.up_odd(l1[(1, 1)])) / n
        up_even = torch.einsum("bpn,bnd->bpd", e2, self.up_even(l1[(0, 0)])) / n
        down_odd = torch.einsum("bpn,bpd->bnd", e, self.down_odd(l2[(1, 0)])) / p
        down_even = torch.einsum("bpn,bpd->bnd", e2, self.down_even(l2[(0, 0)])) / p

        s = self.scale
        new_l2 = dict(l2)
        new_l2[(1, 0)] = l2[(1, 0)] + s[0] * up_odd
        new_l2[(0, 0)] = l2[(0, 0)] + s[1] * up_even
        new_l1 = dict(l1)
        new_l1[(1, 1)] = l1[(1, 1)] + s[2] * down_odd
        new_l1[(0, 0)] = l1[(0, 0)] + s[3] * down_even
        return new_l1, new_l2


class PhasorGradedReader(nn.Module):
    """Exactly $G$-invariant reader on raw parameters (rung W12)."""

    def __init__(self, dims_l1: dict[Character, int], dims_l2: dict[Character, int],
                 width: int = 256, n_classes: int = 10, rounds: int = 2) -> None:
        super().__init__()
        self.embed_l1 = GradedLinear(dims_l1, width)
        self.embed_l2 = GradedLinear(dims_l2, width)
        self.bilinear = nn.ModuleList(GradedBilinear(width) for _ in range(rounds))
        self.messages = nn.ModuleList(GradedMessage(width) for _ in range(rounds))
        self.mix_l1 = nn.ModuleList(GradedLinear({c: width for c in CHARACTERS}, width)
                                    for _ in range(rounds))
        self.mix_l2 = nn.ModuleList(GradedLinear({c: width for c in CHARACTERS}, width)
                                    for _ in range(rounds))
        self.act = GradedAct()
        self.head = nn.Sequential(
            nn.Linear(4 * width, 2 * width), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(2 * width, width), nn.GELU(),
            nn.Linear(width, n_classes),
        )

    @classmethod
    def from_features(cls, feats: dict, width: int = 256, n_classes: int = 10,
                      rounds: int = 2) -> "PhasorGradedReader":
        dims_l1 = {c: feats["l1"][c].shape[2] for c in CHARACTERS}
        dims_l2 = {c: feats["l2"][c].shape[2] for c in CHARACTERS}
        return cls(dims_l1, dims_l2, width=width, n_classes=n_classes, rounds=rounds)

    def forward(self, feats: dict) -> Tensor:
        l1, l2, e = self.embed_l1(feats["l1"]), self.embed_l2(feats["l2"]), feats["edge"]
        for bil, msg, m1, m2 in zip(self.bilinear, self.messages, self.mix_l1, self.mix_l2):
            l1, l2 = msg(l1, l2, e)
            l1, l2 = self.act(m1(bil(l1))), self.act(m2(bil(l2)))
        # invariance is cashed in here: only the neutral block reaches the head
        pooled = torch.cat([l1[(0, 0)].mean(1), l1[(0, 0)].amax(1),
                            l2[(0, 0)].mean(1), l2[(0, 0)].amax(1)], dim=1)
        return self.head(pooled)
