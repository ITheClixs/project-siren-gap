"""Shared fixtures: device parametrization (CPU always, MPS when available) and
generic random-INR builders with biases spread over several tau-windings."""

from __future__ import annotations

import pytest
import torch

from sirengap.models.params import SirenParams

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


@pytest.fixture(params=DEVICES)
def device(request: pytest.FixtureRequest) -> str:
    return request.param


def random_params(
    batch: int,
    in_dim: int,
    widths: tuple[int, ...],
    out_dim: int,
    seed: int,
    device: str = "cpu",
    bias_windings: float = 4.0,
) -> SirenParams:
    """Generic random INR; biases ~ N(0, (bias_windings*pi/2)^2) exercise tau."""
    gen = torch.Generator().manual_seed(seed)
    hidden = []
    fan = in_dim
    for n in widths:
        w = torch.randn(batch, n, fan, generator=gen)
        b = torch.randn(batch, n, generator=gen) * (bias_windings * torch.pi / 2)
        hidden.append((w.to(device), b.to(device)))
        fan = n
    w_out = torch.randn(batch, out_dim, fan, generator=gen).to(device)
    b_out = torch.randn(batch, out_dim, generator=gen).to(device)
    return SirenParams(hidden=tuple(hidden), w_out=w_out, b_out=b_out)


def random_coords(n_pts: int, in_dim: int, seed: int, device: str = "cpu") -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return (torch.rand(n_pts, in_dim, generator=gen) * 2 - 1).to(device)
