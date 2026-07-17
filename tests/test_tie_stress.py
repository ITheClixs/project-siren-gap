"""Tie/strata stress tests for canonicalizers (advisor review G1, Theorist 2).

Policy under test: ON the degenerate strata (dead w=0, invisible u=0, parallel
frequencies, exact phase-boundary b = pi/2), canonicalizers must still (a) run,
(b) preserve the function (they only ever apply group elements), and (c) be
deterministic. Orbit collapse is NOT guaranteed there — PO-3/PO-5 predict
exactly this — and one test documents a witnessed failure rather than hiding it.
"""

import math

import torch

from conftest import random_coords, random_params
from sirengap.canon.csort import c_sort
from sirengap.constants import TOL_FUNC
from sirengap.models.forward import max_functional_gap
from sirengap.models.params import SirenParams


def _with_layer0(params: SirenParams, w=None, b=None, w_out=None) -> SirenParams:
    w0, b0 = params.hidden[0]
    return SirenParams(
        hidden=((w if w is not None else w0, b if b is not None else b0),) + params.hidden[1:],
        w_out=w_out if w_out is not None else params.w_out,
        b_out=params.b_out,
    )


def test_dead_neuron_function_preserved_and_deterministic(device: str) -> None:
    params = random_params(2, 2, (8,), 1, seed=71, device=device)
    w0, _ = params.hidden[0]
    w_dead = w0.clone()
    w_dead[:, 3, :] = 0.0  # dead neuron
    params = _with_layer0(params, w=w_dead)
    coords = random_coords(256, 2, seed=72, device=device)
    canon1, _ = c_sort(params)
    canon2, _ = c_sort(params)
    assert max_functional_gap(params, canon1, coords) < TOL_FUNC
    assert torch.equal(canon1.flat(), canon2.flat())


def test_invisible_neuron_function_preserved(device: str) -> None:
    params = random_params(2, 2, (8,), 1, seed=73, device=device)
    w_out = params.w_out.clone()
    w_out[:, :, 5] = 0.0  # invisible neuron (u = 0)
    params = _with_layer0(params, w_out=w_out)
    coords = random_coords(256, 2, seed=74, device=device)
    canon, _ = c_sort(params)
    assert max_functional_gap(params, canon, coords) < TOL_FUNC


def test_phase_boundary_exact_half_pi(device: str) -> None:
    """b exactly pi/2 — the phase-reduction discontinuity set (PO-5 corollary)."""
    params = random_params(2, 2, (8,), 1, seed=75, device=device)
    _, b0 = params.hidden[0]
    b_edge = b0.clone()
    b_edge[:, 0] = math.pi / 2
    b_edge[:, 1] = -math.pi / 2
    params = _with_layer0(params, b=b_edge)
    coords = random_coords(256, 2, seed=76, device=device)
    canon, _ = c_sort(params)
    assert max_functional_gap(params, canon, coords) < TOL_FUNC
    b_new = canon.hidden[0][1]
    assert b_new.min() >= -math.pi / 2 - 1e-6 and b_new.max() < math.pi / 2 + 1e-6


def test_duplicate_neurons_document_collapse_limit(device: str) -> None:
    """Two identical neurons: c_sort output is well-defined and function-preserving;
    orbit collapse across a *perturbed* pair is expectedly fragile (documented)."""
    params = random_params(1, 2, (8,), 1, seed=77, device=device)
    w0, b0 = params.hidden[0]
    w_dup, b_dup = w0.clone(), b0.clone()
    w_dup[:, 4, :] = w_dup[:, 2, :]
    b_dup[:, 4] = b_dup[:, 2]  # exact duplicate pair -> sorting-key tie
    params = _with_layer0(params, w=w_dup, b=b_dup)
    coords = random_coords(256, 2, seed=78, device=device)
    canon, diag = c_sort(params)
    assert max_functional_gap(params, canon, coords) < TOL_FUNC
    # diagnostics must flag the tie: minimal sorted-key gap is ~0
    assert diag.key_min_gaps[0].min().item() < 1e-6
