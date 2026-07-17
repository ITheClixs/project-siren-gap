"""Batched SIREN fitter (protocol A.1).

Trains B INRs simultaneously in the *internal* SIREN parameterization
h^l = sin(omega_0 * (W h + b)) with the official SIREN init, then absorbs
omega_0 into the stored weights so downstream code sees only the canonical form
h^l = sin(W' h + b') (see docs/THINKING/G0-theory-scoping.md §0).

Per-INR losses share no parameters, so Adam on their mean is exactly
independent per-INR Adam up to the constant 1/B factor in the learning-rate
scaling of gradients; this makes P-shared-det deterministic per INR (T9).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from sirengap.constants import OMEGA_0
from sirengap.models.params import SirenParams


@dataclass(frozen=True)
class FitResult:
    params: SirenParams  # stored (canonical) form, omega absorbed
    final_loss: Tensor  # [B] per-INR MSE
    loss_curve: tuple[float, ...]  # mean-over-B MSE at each logged step


def make_coord_grid(height: int, width: int, device: str = "cpu") -> Tensor:
    """Pixel-center coordinates in [-1, 1]^2, row-major: [height*width, 2]."""
    ys = torch.linspace(-1.0, 1.0, height, device=device)
    xs = torch.linspace(-1.0, 1.0, width, device=device)
    grid = torch.cartesian_prod(ys, xs)
    return grid.flip(1)  # (x, y) ordering


def init_siren_train(
    batch: int,
    in_dim: int,
    widths: tuple[int, ...],
    out_dim: int,
    generator: torch.Generator,
    shared_init: bool,
) -> list[Tensor]:
    """Official SIREN init in the internal parameterization. Returns flat tensor list
    [W_0, b_0, ..., W_L-1, b_L-1, W_out, b_out] on CPU (move to device afterwards)."""

    def uniform(shape: tuple[int, ...], bound: float) -> Tensor:
        n_draw = 1 if shared_init else batch
        t = (torch.rand((n_draw, *shape), generator=generator) * 2 - 1) * bound
        return t.expand(batch, *shape).clone() if shared_init else t

    tensors: list[Tensor] = []
    fan_in = in_dim
    for i, n in enumerate(widths):
        w_bound = (1.0 / fan_in) if i == 0 else math.sqrt(6.0 / fan_in) / OMEGA_0
        b_bound = 1.0 / math.sqrt(fan_in)
        tensors += [uniform((n, fan_in), w_bound), uniform((n,), b_bound)]
        fan_in = n
    w_bound = math.sqrt(6.0 / fan_in) / OMEGA_0
    tensors += [uniform((out_dim, fan_in), w_bound), uniform((out_dim,), 1.0 / math.sqrt(fan_in))]
    return tensors


def forward_train(tensors: list[Tensor], x: Tensor) -> Tensor:
    """Internal-parameterization forward: sin(omega_0 * (W h + b)) per sine layer."""
    n_sine = (len(tensors) - 2) // 2
    w0, b0 = tensors[0], tensors[1]
    h = torch.sin(OMEGA_0 * (torch.einsum("bji,pi->bpj", w0, x) + b0[:, None, :]))
    for layer in range(1, n_sine):
        w, b = tensors[2 * layer], tensors[2 * layer + 1]
        h = torch.sin(OMEGA_0 * (torch.einsum("bji,bpi->bpj", w, h) + b[:, None, :]))
    w_out, b_out = tensors[-2], tensors[-1]
    return torch.einsum("bci,bpi->bpc", w_out, h) + b_out[:, None, :]


def absorb_omega(tensors: list[Tensor]) -> SirenParams:
    """Convert internal parameterization to stored canonical form: (W,b) *= omega_0."""
    n_sine = (len(tensors) - 2) // 2
    hidden = tuple(
        (tensors[2 * i].detach() * OMEGA_0, tensors[2 * i + 1].detach() * OMEGA_0)
        for i in range(n_sine)
    )
    return SirenParams(hidden=hidden, w_out=tensors[-2].detach(), b_out=tensors[-1].detach())


def fit_batch(
    targets: Tensor,
    coords: Tensor,
    widths: tuple[int, ...],
    steps: int,
    lr: float = 1e-3,
    seed: int = 0,
    shared_init: bool = False,
    device: str = "cpu",
    log_every: int = 100,
) -> FitResult:
    """Fit B INRs to targets [B, P, c] on coords [P, in_dim] (full-batch, deterministic)."""
    if targets.ndim != 3 or coords.ndim != 2 or targets.shape[1] != coords.shape[0]:
        raise ValueError(f"bad shapes: targets {tuple(targets.shape)}, coords {tuple(coords.shape)}")
    gen = torch.Generator().manual_seed(seed)
    tensors = [
        t.to(device).requires_grad_(True)
        for t in init_siren_train(
            targets.shape[0], coords.shape[1], widths, targets.shape[2], gen, shared_init
        )
    ]
    targets = targets.to(device)
    coords = coords.to(device)
    opt = torch.optim.Adam(tensors, lr=lr)
    curve: list[float] = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        pred = forward_train(tensors, coords)
        per_inr = ((pred - targets) ** 2).mean(dim=(1, 2))
        # sum (not mean): Adam on the sum is exactly independent per-INR Adam,
        # because per-INR losses share no parameters (protocol A.1 / T9)
        per_inr.sum().backward()
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            curve.append(float(per_inr.detach().mean().cpu()))
    with torch.no_grad():
        final = ((forward_train(tensors, coords) - targets) ** 2).mean(dim=(1, 2)).cpu()
    return FitResult(params=absorb_omega(tensors), final_loss=final, loss_curve=tuple(curve))


def psnr(mse: Tensor, data_range: float = 2.0) -> Tensor:
    """PSNR for targets in [-1, 1] (data_range 2.0) from per-INR MSE."""
    return 10.0 * torch.log10(data_range**2 / mse.clamp_min(1e-12))
