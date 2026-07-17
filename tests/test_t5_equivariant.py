"""T5: layer(g . theta) = g . layer(theta) for the implemented group
(D_inf acts trivially on channels; permutations act on the neuron axis)."""

import pytest
import torch

from conftest import random_params
from sirengap.canon.invariants import encode_per_neuron
from sirengap.models.equivariant import InvariantDeepSetsLayer
from sirengap.symmetry.dinf import apply, random_element


def test_t5_layer_equivariance(device: str) -> None:
    params = random_params(3, 2, (10,), 2, seed=21, device=device)
    feat_dim = encode_per_neuron(params).shape[2]
    torch.manual_seed(0)
    layer = InvariantDeepSetsLayer(feat_dim, 8).to(device)
    out = layer(params)  # [B, n, 8]
    for trial in range(4):
        gen = torch.Generator().manual_seed(400 + trial)
        g = random_element(params, gen)
        out_moved = layer(apply(g, params))
        perm = g.perm[0].to(device)
        expected = torch.gather(out, 1, perm[:, :, None].expand_as(out))
        gap = (out_moved - expected).abs().max().item()
        assert gap < 1e-4, f"equivariance violated (trial {trial}): {gap:.2e}"


def test_t5_deep_input_refused() -> None:
    params = random_params(2, 2, (8, 8), 1, seed=22)
    layer = InvariantDeepSetsLayer(4, 4)
    with pytest.raises(ValueError, match="L=1"):
        layer(params)
