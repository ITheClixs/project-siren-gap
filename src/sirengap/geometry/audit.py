"""Strata audit (PO-3): empirical distance-to-degenerate-strata of fitted INRs.

Metrics per INR (per hidden layer): min ||w_i||, min ||u_i||, and the minimal
pairwise angular distance min_{i != j} angle(±w_i, w_j) — the parallel-frequency
stratum witness (phasor merge). "Generic" is measured, never assumed.
"""

from __future__ import annotations

import torch
from torch import Tensor

from sirengap.models.params import SirenParams, outgoing


def _min_pairwise_parallel_angle(w: Tensor) -> Tensor:
    """w: [B, n, m] -> [B]: min over pairs of angle between lines span(w_i), span(w_j)."""
    unit = w / (w.norm(dim=2, keepdim=True) + 1e-12)
    cos = torch.einsum("bnm,bkm->bnk", unit, unit).abs().clamp(max=1.0)
    n = w.shape[1]
    eye = torch.eye(n, dtype=torch.bool, device=w.device)
    cos = cos.masked_fill(eye[None, :, :], -1.0)
    return torch.arccos(cos.amax(dim=(1, 2)))


@torch.no_grad()
def strata_audit(params: SirenParams) -> dict[str, list[float]]:
    """Summary quantiles (q05/q50 of per-INR minima) per hidden layer."""
    out: dict[str, list[float]] = {}
    for layer in range(params.n_layers):
        w, _ = params.hidden[layer]
        u = outgoing(params, layer)  # [B, next, n]
        min_w = w.norm(dim=2).amin(dim=1)
        min_u = u.norm(dim=1).amin(dim=1)
        min_angle = _min_pairwise_parallel_angle(w)
        for name, t in (("min_w_norm", min_w), ("min_u_norm", min_u), ("min_parallel_angle_rad", min_angle)):
            q = torch.quantile(t.float().cpu(), torch.tensor([0.0, 0.05, 0.5]))
            out[f"layer{layer}.{name}.q0_q05_q50"] = [round(float(x), 6) for x in q]
    return out
