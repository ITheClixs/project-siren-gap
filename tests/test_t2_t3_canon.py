"""T2: canonicalizers preserve the function. T3: orbit collapse + idempotence."""

import pytest
import torch

from conftest import random_coords, random_params
from sirengap.canon.calign import c_align
from sirengap.canon.csort import c_sort
from sirengap.constants import TOL_FUNC
from sirengap.models.forward import max_functional_gap
from sirengap.symmetry.dinf import apply, random_element

WIDTHS = [(16,), (16, 12)]


@pytest.mark.parametrize("widths", WIDTHS)
def test_t2_csort_preserves_function(device: str, widths: tuple[int, ...]) -> None:
    params = random_params(4, 2, widths, 3, seed=11, device=device)
    coords = random_coords(256, 2, seed=12, device=device)
    canon, _ = c_sort(params)
    assert max_functional_gap(params, canon, coords) < TOL_FUNC


@pytest.mark.parametrize("widths", WIDTHS)
def test_t2_calign_preserves_function(device: str, widths: tuple[int, ...]) -> None:
    params = random_params(4, 2, widths, 3, seed=13, device=device)
    template = random_params(1, 2, widths, 3, seed=14, device=device)
    probes = random_coords(512, 2, seed=15, device=device)
    canon, _ = c_align(params, template, probes)  # raises internally if broken
    assert max_functional_gap(params, canon, probes) < TOL_FUNC


@pytest.mark.parametrize("widths", WIDTHS)
def test_t3_csort_orbit_collapse_and_idempotence(device: str, widths: tuple[int, ...]) -> None:
    params = random_params(3, 2, widths, 3, seed=16, device=device)
    canon, _ = c_sort(params)
    for trial in range(4):
        gen = torch.Generator().manual_seed(200 + trial)
        moved = apply(random_element(params, gen), params)
        canon_moved, _ = c_sort(moved)
        gap = (canon.flat() - canon_moved.flat()).abs().max().item()
        assert gap < 1e-4, f"orbit collapse failed (trial {trial}): {gap:.2e}"
    canon2, _ = c_sort(canon)
    assert (canon.flat() - canon2.flat()).abs().max().item() < 1e-5


@pytest.mark.parametrize("widths", WIDTHS)
def test_t3_calign_orbit_collapse_and_idempotence(device: str, widths: tuple[int, ...]) -> None:
    params = random_params(3, 2, widths, 3, seed=17, device=device)
    template = random_params(1, 2, widths, 3, seed=18, device=device)
    probes = random_coords(512, 2, seed=19, device=device)
    canon, _ = c_align(params, template, probes)
    for trial in range(4):
        gen = torch.Generator().manual_seed(300 + trial)
        moved = apply(random_element(params, gen), params)
        canon_moved, _ = c_align(moved, template, probes)
        gap = (canon.flat() - canon_moved.flat()).abs().max().item()
        assert gap < 1e-4, f"orbit collapse failed (trial {trial}): {gap:.2e}"
    canon2, _ = c_align(canon, template, probes)
    assert (canon.flat() - canon2.flat()).abs().max().item() < 1e-5
