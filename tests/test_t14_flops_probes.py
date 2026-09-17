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


def test_phasor_reader_flops_are_ordered_sensibly() -> None:
    """The grading is not free. W12 drops the edge MLP but pays for 18 d x d per-node maps a
    round against the graph reader's two, so it prices *above* it at equal width -- and both
    price far above reading raw weights into an MLP. The accounting must show that, since the
    S5 comparison turns on it."""
    from sirengap.eval.flops import (
        Arch,
        weight_equivariant_reader,
        weight_phasor_reader,
        weight_raw,
    )

    arch = Arch(in_dim=2, width=32, layers=2, out_dim=1)
    raw = weight_raw(arch)["per_inr"]
    phasor = weight_phasor_reader(arch, 186)["per_inr"]
    graph = weight_equivariant_reader(arch, 186)["per_inr"]
    assert raw < graph < phasor, f"raw {raw}, graph {graph}, phasor {phasor}"
    for cost in (weight_phasor_reader(arch, w) for w in (64, 128, 256)):
        assert cost["per_inr"] > 0 and cost["reader"] > cost["preprocess"]


def test_w12_analytic_cost_equals_counted_forward():
    """The analytic W12 reader cost must equal PyTorch's FLOP count of the actual forward pass.

    An external review found the formula double-counting the graded mixing maps and omitting the
    four message projections (163.3 against 145.7 MFLOP at width 186). Pin exact agreement at small
    widths so the accounting cannot drift from the module again.
    """
    from torch.utils.flop_counter import FlopCounterMode

    from sirengap.eval.flops import Arch, weight_phasor_reader
    from sirengap.fitting.batched import absorb_omega, init_from_seeds
    from sirengap.models.phasor import PhasorGradedReader, phasor_features

    for w, d in ((8, 10), (16, 24)):
        feats = phasor_features(absorb_omega(init_from_seeds([0], 2, (w, w), 1)))
        model = PhasorGradedReader.from_features(feats, width=d, n_classes=10).eval()
        with torch.no_grad(), FlopCounterMode(display=False) as counter:
            model(feats)
        assert weight_phasor_reader(Arch(width=w), d)["reader"] == counter.get_total_flops()


def test_graph_reader_analytic_costs_equal_counted_forward():
    """W11a and W11b are priced by the modules that actually run.

    An external review found W11a priced with W11b's multi-relational architecture (247.4 against
    96.5 MFLOP at width 424). Pin both graph readers to PyTorch's count of their own forward pass.
    """
    from torch.utils.flop_counter import FlopCounterMode

    from sirengap.eval.flops import Arch, weight_equivariant_reader, weight_raw_graph_reader
    from sirengap.fitting.batched import absorb_omega, init_from_seeds
    from sirengap.models.readers import (
        InvariantGraphReader,
        RawGraphReader,
        invariant_graph_features,
        raw_graph_features,
    )

    for w, d in ((8, 10), (16, 24)):
        params = absorb_omega(init_from_seeds([0], 2, (w, w), 1))
        raw = raw_graph_features(params)
        model_a = RawGraphReader(m=raw["x1"].shape[2] - 1, c=raw["x2"].shape[2] - 1, width=d).eval()
        with torch.no_grad(), FlopCounterMode(display=False) as counter:
            model_a(raw)
        assert weight_raw_graph_reader(Arch(width=w), d)["reader"] == counter.get_total_flops()

        inv = invariant_graph_features(params)
        model_b = InvariantGraphReader(
            n_node=inv["x1"].shape[2], n_edge=inv["e"].shape[3], n_global=inv["g"].shape[1], width=d
        ).eval()
        with torch.no_grad(), FlopCounterMode(display=False) as counter:
            model_b(inv)
        cost_b = weight_equivariant_reader(Arch(width=w), d, n_global=inv["g"].shape[1])
        assert cost_b["reader"] == counter.get_total_flops()
