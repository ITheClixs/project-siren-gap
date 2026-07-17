"""T1: the implemented D_inf wreath action preserves the network function."""

import pytest
import torch

from conftest import random_coords, random_params
from sirengap.constants import TOL_FUNC
from sirengap.models.forward import max_functional_gap
from sirengap.symmetry.dinf import GroupElement, apply, random_element


@pytest.mark.parametrize("widths", [(16,), (16, 12)])
def test_random_group_elements_preserve_function(device: str, widths: tuple[int, ...]) -> None:
    params = random_params(4, 2, widths, 3, seed=1, device=device)
    coords = random_coords(256, 2, seed=2, device=device)
    for trial in range(5):
        gen = torch.Generator().manual_seed(100 + trial)
        g = random_element(params, gen)
        gap = max_functional_gap(params, apply(g, params), coords)
        assert gap < TOL_FUNC, f"trial {trial}: gap {gap:.2e}"


def test_generators_individually(device: str) -> None:
    """tau, rho, sigma each preserve f (single-generator sanity, L=1)."""
    params = random_params(2, 2, (8,), 1, seed=3, device=device)
    coords = random_coords(128, 2, seed=4, device=device)
    n = 8
    zeros = torch.zeros(2, n, dtype=torch.long)
    eye = torch.arange(n).expand(2, n).clone()
    cases = {
        "tau": GroupElement(d=(zeros,), j=(2 * torch.ones(2, n, dtype=torch.long),), perm=(eye,)),
        "rho": GroupElement(d=(zeros,), j=(torch.ones(2, n, dtype=torch.long),), perm=(eye,)),
        "sigma": GroupElement(d=(torch.ones(2, n, dtype=torch.long),), j=(zeros,), perm=(eye,)),
    }
    for name, g in cases.items():
        gap = max_functional_gap(params, apply(g, params), coords)
        assert gap < TOL_FUNC, f"{name}: gap {gap:.2e}"


def test_action_changes_parameters(device: str) -> None:
    """Negative control: the action is nontrivial on parameters."""
    params = random_params(2, 2, (8,), 1, seed=5, device=device)
    gen = torch.Generator().manual_seed(6)
    g = random_element(params, gen)
    moved = apply(g, params)
    assert (params.flat() - moved.flat()).abs().max() > 1e-2
