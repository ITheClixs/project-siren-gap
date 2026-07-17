"""Minimal D_inf-invariant / S_n-equivariant layer (Ch3.7 seed; T5).

Per-neuron channels are the PO-4 invariant features (D_inf acts trivially on
them by construction); neuron mixing is DeepSets-style, hence permutation
equivariant: layer(g . theta) = perm_g(layer(theta)). The full Ch3.7
contribution (sign-equivariant (w, u) couplings, deep-layer interleaving)
extends this module later; T5 tests the property that already holds.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from sirengap.canon.invariants import encode_per_neuron
from sirengap.models.params import SirenParams


class InvariantDeepSetsLayer(nn.Module):
    """theta -> per-neuron features [B, n, out_dim], S_n-equivariant, D_inf-invariant."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.local = nn.Linear(in_dim, out_dim)
        self.context = nn.Linear(in_dim, out_dim)

    def forward(self, params: SirenParams) -> Tensor:
        feats = encode_per_neuron(params)  # [B, n, F]
        pooled = feats.mean(dim=1, keepdim=True)  # [B, 1, F]
        return self.local(feats) + self.context(pooled)
