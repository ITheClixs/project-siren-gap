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
    """Internal-parameterization forward: sin(omega_0 * (W h + b)) per sine layer.

    x: shared coords [P, in] or per-INR coords [B, P, in] (minibatch mode).
    """
    n_sine = (len(tensors) - 2) // 2
    w0, b0 = tensors[0], tensors[1]
    eq0 = "bji,pi->bpj" if x.ndim == 2 else "bji,bpi->bpj"
    h = torch.sin(OMEGA_0 * (torch.einsum(eq0, w0, x) + b0[:, None, :]))
    for layer in range(1, n_sine):
        w, b = tensors[2 * layer], tensors[2 * layer + 1]
        h = torch.sin(OMEGA_0 * (torch.einsum("bji,bpi->bpj", w, h) + b[:, None, :]))
    w_out, b_out = tensors[-2], tensors[-1]
    return torch.einsum("bci,bpi->bpc", w_out, h) + b_out[:, None, :]


def init_from_seeds(
    seeds: list[int], in_dim: int, widths: tuple[int, ...], out_dim: int
) -> list[Tensor]:
    """Per-INR inits with individually recorded seeds (P-random); identical seeds
    give identical inits (shared protocols). Returns the flat tensor list, batched."""
    per_inr: list[list[Tensor]] = []
    for s in seeds:
        gen = torch.Generator().manual_seed(int(s))
        per_inr.append(init_siren_train(1, in_dim, widths, out_dim, gen, shared_init=False))
    return [torch.cat([p[i] for p in per_inr], dim=0) for i in range(len(per_inr[0]))]


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
    init_seeds: list[int] | None = None,
    coord_batch: int | None = None,
    fit_seed: int = 0,
) -> FitResult:
    """Fit B INRs to targets [B, P, c] on coords [P, in_dim].

    Deterministic full-batch by default (P-shared-det / P-random given init_seeds).
    coord_batch=k enables per-INR coordinate minibatching driven by fit_seed
    (P-shared-stoch). init_seeds overrides (seed, shared_init) with per-INR seeds.
    """
    if targets.ndim != 3 or coords.ndim != 2 or targets.shape[1] != coords.shape[0]:
        raise ValueError(f"bad shapes: targets {tuple(targets.shape)}, coords {tuple(coords.shape)}")
    b_sz, n_pts, ch = targets.shape
    if init_seeds is not None:
        if len(init_seeds) != b_sz:
            raise ValueError("init_seeds length must equal batch size")
        raw = init_from_seeds(init_seeds, coords.shape[1], widths, ch)
    else:
        gen = torch.Generator().manual_seed(seed)
        raw = init_siren_train(b_sz, coords.shape[1], widths, ch, gen, shared_init)
    tensors = [t.to(device).requires_grad_(True) for t in raw]
    targets = targets.to(device)
    coords = coords.to(device)
    mb_gen = torch.Generator().manual_seed(fit_seed)
    opt = torch.optim.Adam(tensors, lr=lr)
    curve: list[float] = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        if coord_batch is None:
            x_step, t_step = coords, targets
        else:
            idx = torch.randint(0, n_pts, (b_sz, coord_batch), generator=mb_gen).to(device)
            x_step = coords[idx]  # [B, k, in]
            t_step = torch.gather(targets, 1, idx[:, :, None].expand(-1, -1, ch))
        pred = forward_train(tensors, x_step)
        per_inr = ((pred - t_step) ** 2).mean(dim=(1, 2))
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
