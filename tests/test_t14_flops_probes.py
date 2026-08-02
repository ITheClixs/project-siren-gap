"""T14: the S5 apparatus — FLOP accounting is consistent, and probe queries are honest.

The Pareto frontier only means something if (a) the accounting is monotone and internally
consistent, and (b) the function-query model really only touches the network through its outputs.
The second is the one worth testing hardest: a "function access" baseline that peeked at weights
would silently invalidate the whole adjudication.
"""

from __future__ import annotations

import pytest
import torch

from conftest import random_params
from sirengap.eval.flops import (
    Arch,
    function_query,
    render_access,
    siren_forward,
    weight_calign,
    weight_csort,
    weight_equivariant_reader,
    weight_invariants,
    weight_raw,
)
from sirengap.eval.probes import ProbeReader, evaluate_at
from sirengap.models.forward import forward_canonical
from sirengap.symmetry.dinf import apply, random_element

ARCH = Arch(in_dim=2, width=32, layers=2, out_dim=1)


def test_t14_arch_param_count_matches_the_real_thing() -> None:
    p = random_params(2, 2, (32, 32), 1, seed=1)
    assert ARCH.n_params == p.flat().shape[1]


@pytest.mark.parametrize("k", [1, 4, 64, 1024])
def test_t14_siren_cost_is_linear_in_probes(k: int) -> None:
    assert siren_forward(ARCH, k) == k * siren_forward(ARCH, 1)


def test_t14_render_is_function_query_at_full_grid() -> None:
    """Render access is not a separate mechanism; it is querying every grid point."""
    assert render_access(ARCH, 28) == function_query(ARCH, 28 * 28)


def test_t14_costs_are_positive_and_ordered() -> None:
    fq_small = function_query(ARCH, 16)["per_inr"]
    fq_big = function_query(ARCH, 256)["per_inr"]
    assert 0 < fq_small < fq_big
    raw = weight_raw(ARCH)["per_inr"]
    assert weight_csort(ARCH)["per_inr"] > raw, "c_sort must cost more than reading raw weights"
    assert weight_calign(ARCH, 256)["per_inr"] > weight_csort(ARCH)["per_inr"], (
        "c_align does strictly more work than c_sort"
    )
    assert weight_calign(ARCH, 256)["amortized"] > 0, "the template is a real amortized cost"
    assert weight_invariants(ARCH, 320)["preprocess"] > 0
    assert weight_equivariant_reader(ARCH, 288)["per_inr"] > 0


def test_t14_probe_evaluation_agrees_with_the_canonical_forward() -> None:
    p = random_params(4, 2, (32, 32), 1, seed=5)
    coords = torch.rand(23, 2) * 2 - 1
    assert torch.allclose(evaluate_at(p, coords), forward_canonical(p, coords), atol=1e-5)


def test_t14_probe_reader_is_invariant_to_the_whole_group() -> None:
    """The decisive property: querying sees the function, so symmetry cannot touch it."""
    p = random_params(4, 2, (32, 32), 1, seed=6)
    torch.manual_seed(0)
    reader = ProbeReader(n_probes=16).double().eval()
    p = p.to("cpu")
    dbl = type(p)(
        hidden=tuple((w.double(), b.double()) for w, b in p.hidden),
        w_out=p.w_out.double(), b_out=p.b_out.double(),
    )
    gen = torch.Generator().manual_seed(7)
    moved = apply(random_element(dbl, gen, max_windings=3), dbl)
    with torch.no_grad():
        gap = (reader(dbl) - reader(moved)).abs().max().item()
    assert gap < 1e-8, f"function access must be exactly nuisance-free, got {gap:.2e}"


def test_t14_probes_receive_gradient_but_the_inr_does_not() -> None:
    p = random_params(3, 2, (32, 32), 1, seed=8)
    reader = ProbeReader(n_probes=8)
    reader(p).sum().backward()
    assert reader.probes.grad is not None and reader.probes.grad.abs().sum() > 0
    for w, b in p.hidden:
        assert w.grad is None and b.grad is None, "the fitted INR must stay frozen"


def test_t14_frozen_probes_stay_put() -> None:
    reader = ProbeReader(n_probes=9, freeze_probes=True)
    assert not reader.probes.requires_grad
    before = reader.probes.detach().clone()
    p = random_params(2, 2, (32, 32), 1, seed=9)
    reader(p).sum().backward()
    assert torch.equal(reader.probes.detach(), before)
