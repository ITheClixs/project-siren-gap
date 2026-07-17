"""T6: phase-invariant encoding is invariant under tau, rho, sigma and
equivariant under permutations — and the protocol's original (uncorrected)
feature demonstrably fails, locking in the G0 memo finding."""

import torch

from conftest import random_params
from sirengap.canon.invariants import encode_per_neuron, encode_pooled
from sirengap.symmetry.dinf import GroupElement, apply, random_element


def test_t6_per_neuron_equivariant_pooled_invariant(device: str) -> None:
    params = random_params(3, 2, (10,), 2, seed=31, device=device)
    feats = encode_per_neuron(params)
    pooled = encode_pooled(params)
    for trial in range(6):
        gen = torch.Generator().manual_seed(500 + trial)
        g = random_element(params, gen)
        moved = apply(g, params)
        feats_moved = encode_per_neuron(moved)
        perm = g.perm[0].to(device)
        expected = torch.gather(feats, 1, perm[:, :, None].expand_as(feats))
        assert (feats_moved - expected).abs().max().item() < 2e-4
        assert (encode_pooled(moved) - pooled).abs().max().item() < 2e-4


def test_t6_negative_control_raw_bias_not_invariant(device: str) -> None:
    params = random_params(2, 2, (8,), 1, seed=32, device=device)
    gen = torch.Generator().manual_seed(600)
    g = random_element(params, gen, identity_perm=True)
    moved = apply(g, params)
    raw_b_gap = (params.hidden[0][1] - moved.hidden[0][1]).abs().max().item()
    assert raw_b_gap > 1e-2, "group element unexpectedly fixed all biases"


def test_t6_protocol_original_feature_breaks_under_rho(device: str) -> None:
    """cos(2b) * (w (x) u) — the protocol's suggested invariant — is NOT invariant
    under rho (G0-theory-scoping §2). Executable record of the corrected algebra."""
    params = random_params(2, 2, (8,), 2, seed=33, device=device)
    n = 8
    zeros = torch.zeros(2, n, dtype=torch.long)
    eye = torch.arange(n).expand(2, n).clone()
    rho = GroupElement(d=(zeros,), j=(torch.ones(2, n, dtype=torch.long),), perm=(eye,))
    moved = apply(rho, params)

    def broken_feature(p):  # cos(2b) * (w (x) u)
        w, b = p.hidden[0]
        u = p.w_out.transpose(1, 2)
        return torch.cos(2 * b)[:, :, None] * torch.einsum("bni,bnc->bnic", w, u).flatten(2)

    gap = (broken_feature(params) - broken_feature(moved)).abs().max().item()
    assert gap > 1e-2, "protocol's cos(2b)*(w x u) unexpectedly invariant — memo finding wrong?"
