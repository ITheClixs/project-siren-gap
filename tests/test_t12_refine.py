"""T12: the S4e alignment instrument recovers a planted group element exactly.

This is the control that makes the deep-identifiability hunt non-vacuous. If the search
cannot undo a group element it *knows* exists, then a large residual on a real pair says
nothing about identifiability — it only says the search is weak. Every claim S4e makes
rests on these tests passing at machine precision.
"""

from __future__ import annotations

import math

import pytest
import torch

from sirengap.canon.refine import (
    orbit_distance,
    param_distance,
    refine_alignment,
    relative_param_distance,
)
from sirengap.fitting.batched import absorb_omega, init_from_seeds
from sirengap.symmetry.dinf import apply, random_element


def _params(batch: int = 4, widths=(8, 8), in_dim: int = 2, out_dim: int = 1):
    return absorb_omega(init_from_seeds(list(range(batch)), in_dim, list(widths), out_dim))


def test_planted_element_is_undone_exactly() -> None:
    """theta* -> g.theta* -> refine back: residual must be machine zero."""
    target = _params()
    gen = torch.Generator().manual_seed(0)
    g = random_element(target, gen, max_windings=3)
    moved = apply(g, target)

    assert param_distance(moved, target).max() > 1.0, "planted element did not move anything"

    rel, diag = orbit_distance(moved, target)
    assert rel.max() < 1e-6, f"planted recovery failed: relative residual {rel.max():.3e}"
    assert diag.distance_final.max() < 1e-5


@pytest.mark.parametrize("windings", [0, 1, 5, 12])
def test_recovery_is_winding_independent(windings: int) -> None:
    """The bias circle is infinite; the closed-form j must reach any winding number."""
    target = _params(batch=3, widths=(6, 6))
    gen = torch.Generator().manual_seed(windings + 1)
    g = random_element(target, gen, max_windings=max(windings, 1))
    if windings == 0:
        g = random_element(target, gen, max_windings=1, identity_perm=True)
    moved = apply(g, target)
    rel, _ = orbit_distance(moved, target)
    assert rel.max() < 1e-6, f"windings={windings}: residual {rel.max():.3e}"


@pytest.mark.parametrize("widths", [(4,), (6, 6), (5, 7, 5)])
def test_recovery_across_depths(widths: tuple[int, ...]) -> None:
    """Coordinate descent over layers must reach the planted optimum at L = 1, 2 and 3."""
    target = _params(batch=2, widths=widths)
    gen = torch.Generator().manual_seed(7)
    moved = apply(random_element(target, gen, max_windings=2), target)
    rel, _ = orbit_distance(moved, target)
    assert rel.max() < 1e-6, f"widths={widths}: residual {rel.max():.3e}"


def test_multichannel_output() -> None:
    """c > 1 exercises the outgoing-column cost with vector u."""
    target = _params(batch=3, widths=(6, 6), out_dim=3)
    gen = torch.Generator().manual_seed(11)
    moved = apply(random_element(target, gen, max_windings=2), target)
    rel, _ = orbit_distance(moved, target)
    assert rel.max() < 1e-6


def test_refinement_never_increases_distance() -> None:
    """Coordinate descent is monotone; unrelated networks must not get worse."""
    a = _params(batch=4, widths=(8, 8))
    b = _params(batch=4, widths=(8, 8), in_dim=2)
    b = apply(random_element(b, torch.Generator().manual_seed(99), max_windings=2), b)
    _, diag = refine_alignment(a, b)
    assert bool((diag.distance_final <= diag.distance_start + 1e-9).all())
    for k in range(1, len(diag.per_sweep)):
        assert bool((diag.per_sweep[k] <= diag.per_sweep[k - 1] + 1e-9).all())


def test_unrelated_networks_keep_a_large_residual() -> None:
    """The null: two independent draws are not in one orbit, so the residual must stay big."""
    a = _params(batch=4, widths=(8, 8))
    b = absorb_omega(init_from_seeds([100, 101, 102, 103], 2, [8, 8], 1))
    rel, _ = orbit_distance(a, b)
    assert rel.min() > 0.05, f"unrelated pair collapsed to {rel.min():.3e}"


def test_alignment_preserves_the_function() -> None:
    """Every move is in G, so f is unchanged — asserted inside refine_alignment."""
    target = _params(batch=3, widths=(6, 6))
    gen = torch.Generator().manual_seed(3)
    moved = apply(random_element(target, gen, max_windings=4), target)
    aligned, _ = refine_alignment(moved, target, check_preservation=True)
    assert relative_param_distance(aligned, target).max() < 1e-6


def test_pure_phase_shift_of_2pi_is_free() -> None:
    """tau_1 = (d=0, j=2) is the identity on the function and must cost nothing."""
    target = _params(batch=2, widths=(5, 5))
    w, b = target.hidden[0]
    shifted = list(target.hidden)
    shifted[0] = (w, b + 2.0 * math.pi)
    from sirengap.models.params import SirenParams

    moved = SirenParams(hidden=tuple(shifted), w_out=target.w_out, b_out=target.b_out)
    rel, _ = orbit_distance(moved, target)
    assert rel.max() < 1e-6
