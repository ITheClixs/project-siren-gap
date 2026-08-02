"""Function-query access: classify an INR by *querying* it, never by reading its weights (S5).

This is the baseline PO-6's corollary makes unavoidable. If a complete invariant of the weights
carries exactly the information of the realised function, then a model that simply evaluates the
function at well-chosen inputs is entitled to the same accuracy — and the field's case for weight
access has to be made in FLOPs rather than in accuracy.

The probe coordinates are **learned** end to end, which is the strong form of the baseline
(cf. the probe-generation line of work). Learning them can only move the function-access frontier
*up*, i.e. against this program's own weight-access thesis, which is the direction an honest
adjudication should err in.

The INR parameters are frozen throughout: gradients flow to the probe coordinates and the
classifier only. Nothing here fits or modifies a network.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from sirengap.models.params import SirenParams


def evaluate_at(params: SirenParams, coords: Tensor) -> Tensor:
    """f_theta(coords) for a batch of INRs at *shared*, possibly non-grid, coordinates.

    coords [P, in_dim] may carry gradient; `params` is treated as data. Returns [B, P, c].
    """
    w0, b0 = params.hidden[0]
    h = torch.sin(torch.einsum("bji,pi->bpj", w0, coords) + b0[:, None, :])
    for w, b in params.hidden[1:]:
        h = torch.sin(torch.einsum("bji,bpi->bpj", w, h) + b[:, None, :])
    return torch.einsum("bci,bpi->bpc", params.w_out, h) + params.b_out[:, None, :]


class ProbeReader(nn.Module):
    """Learned probe coordinates + the ladder's matched decoder on the queried outputs.

    Probes are initialised on a deterministic sub-grid of the fit domain so the model starts from
    the natural baseline (uniform sampling) and can only improve on it; `freeze_probes` recovers
    that baseline exactly, which is the ablation that separates "querying helps" from "learning
    *where* to query helps".
    """

    def __init__(self, n_probes: int, in_dim: int = 2, out_dim: int = 1,
                 n_classes: int = 10, dropout: float = 0.1, freeze_probes: bool = False):
        super().__init__()
        init = _sub_grid(n_probes, in_dim)
        self.probes = nn.Parameter(init, requires_grad=not freeze_probes)
        dims = [n_probes * out_dim, 1024, 512, 256]
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.GELU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(dims[-1], n_classes))
        self.head = nn.Sequential(*layers)
        self.register_buffer("_mu", torch.zeros(1))
        self.register_buffer("_sd", torch.ones(1))

    def features(self, params: SirenParams) -> Tensor:
        return evaluate_at(params, self.probes).flatten(1)

    def set_normalization(self, feats: Tensor) -> None:
        self._mu = feats.mean().detach().clone().reshape(1)
        self._sd = feats.std().clamp_min(1e-6).detach().clone().reshape(1)

    def forward(self, params: SirenParams) -> Tensor:
        x = (self.features(params) - self._mu) / self._sd
        return self.head(x)


def _sub_grid(n_probes: int, in_dim: int) -> Tensor:
    """Deterministic near-uniform points in [-1, 1]^in_dim, the natural query baseline."""
    if in_dim != 2:
        g = torch.linspace(-1.0, 1.0, n_probes)
        return g[:, None].repeat(1, in_dim)
    side = max(1, int(round(n_probes**0.5)))
    g = torch.linspace(-1.0, 1.0, side)
    pts = torch.stack(torch.meshgrid(g, g, indexing="ij"), dim=-1).reshape(-1, 2)
    if len(pts) >= n_probes:
        step = max(1, len(pts) // n_probes)
        return pts[::step][:n_probes].clone()
    extra = torch.rand(n_probes - len(pts), 2) * 2 - 1
    return torch.cat([pts, extra], dim=0)
