"""Phase-invariant encoding (PO-4 corrected parity classes; Ch3.6, rung W10, T6).

Per-neuron invariants of the D_inf action on (w, b, u), from the verified table
(docs/THINKING/G0-theory-scoping.md §2):
  (0,0): ||w||^2, w (x) w, ||u||^2, cos 2b, cos 2b * (w (x) w)
  (1,0): sin 2b * w
  (0,1): sin b * u
  (1,1): cos b * (w (x) u)
Scope: single-hidden-layer networks (L=1), where u is an output-layer column and
carries no further group action. For L >= 2 the u-features entangle with the
next layer's group (OPEN_PROBLEMS #4) — this module refuses deep inputs rather
than silently emitting non-invariant features.

Permutation invariance: sum-pool over neurons (`encode_pooled`).
"""

from __future__ import annotations

import torch
from torch import Tensor

from sirengap.models.params import SirenParams


def _upper_tri(mat: Tensor) -> Tensor:
    """Vectorize upper triangle (incl. diagonal) of [..., k, k] -> [..., k(k+1)/2]."""
    k = mat.shape[-1]
    idx = torch.triu_indices(k, k, device=mat.device)
    return mat[..., idx[0], idx[1]]


def encode_per_neuron(params: SirenParams) -> Tensor:
    """Per-neuron invariant features [B, n, F] (L=1 only; see module docstring)."""
    if params.n_layers != 1:
        raise ValueError(
            "phase-invariant encoding is only D_inf-invariant for L=1 "
            "(deep extension = OPEN_PROBLEMS #4)"
        )
    w, b = params.hidden[0]  # w: [B, n, m], b: [B, n]
    u = params.w_out.transpose(1, 2)  # [B, n, c] — column i of w_out is neuron i's u

    ww = torch.einsum("bni,bnj->bnij", w, w)
    wu = torch.einsum("bni,bnc->bnic", w, u)
    cos2b, sin2b = torch.cos(2 * b), torch.sin(2 * b)
    cosb, sinb = torch.cos(b), torch.sin(b)

    feats = [
        _upper_tri(ww),  # (0,0)
        (u * u).sum(dim=2, keepdim=True),  # (0,0) ||u||^2
        cos2b[:, :, None],  # (0,0)
        cos2b[:, :, None] * _upper_tri(ww),  # (0,0)
        sin2b[:, :, None] * w,  # (1,0)
        sinb[:, :, None] * u,  # (0,1)
        cosb[:, :, None] * wu.flatten(2),  # (1,1)
    ]
    return torch.cat(feats, dim=2)


def encode_pooled(params: SirenParams) -> Tensor:
    """Fully invariant encoding [B, F]: sum over neurons of per-neuron invariants."""
    return encode_per_neuron(params).sum(dim=1)
