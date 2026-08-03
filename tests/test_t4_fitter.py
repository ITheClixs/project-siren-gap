"""T4: fitter sanity — loss decreases, PSNR reasonable, and the convention lock
(internal forward == canonical forward after omega absorption) holds exactly."""

import torch

from sirengap.constants import TOL_FUNC
from sirengap.fitting.batched import (
    absorb_omega,
    fit_batch,
    forward_train,
    init_siren_train,
    make_coord_grid,
    psnr,
)
from sirengap.models.forward import forward_canonical


def _smooth_targets(batch: int, side: int, seed: int) -> torch.Tensor:
    """Low-frequency random images in [-1, 1]: [B, side*side, 1]."""
    gen = torch.Generator().manual_seed(seed)
    coords = make_coord_grid(side, side)
    freqs = torch.randn(batch, 3, 2, generator=gen) * 2.0
    phases = torch.rand(batch, 3, generator=gen) * 6.28
    amps = torch.randn(batch, 3, generator=gen)
    waves = torch.sin(torch.einsum("bki,pi->bpk", freqs, coords) + phases[:, None, :])
    img = torch.einsum("bpk,bk->bp", waves, amps)
    img = img / img.abs().amax(dim=1, keepdim=True)
    return img[:, :, None]


def test_convention_lock_absorption_exact(device: str) -> None:
    """forward_canonical(absorbed params) must equal the fitter's internal forward."""
    gen = torch.Generator().manual_seed(42)
    tensors = [t.to(device) for t in init_siren_train(3, 2, (16, 12), 1, gen, shared_init=False)]
    coords = make_coord_grid(12, 12, device=device)
    internal = forward_train(tensors, coords)
    canonical = forward_canonical(absorb_omega(tensors), coords)
    gap = (internal - canonical).abs().max().item()
    assert gap < TOL_FUNC, f"convention mismatch: {gap:.2e}"


def test_fit_reduces_loss_and_reaches_reasonable_psnr(device: str) -> None:
    targets = _smooth_targets(4, 16, seed=7)
    coords = make_coord_grid(16, 16)
    result = fit_batch(targets, coords, widths=(32, 32), steps=400, lr=1e-3, seed=0, device=device)
    assert result.loss_curve[-1] < 0.25 * result.loss_curve[0], (
        f"loss did not drop: {result.loss_curve[0]:.4f} -> {result.loss_curve[-1]:.4f}"
    )
    assert psnr(result.final_loss).min().item() > 20.0

    # stored params render the same images (canonical form round-trips the fit)
    rendered = forward_canonical(result.params.to("cpu"), coords)
    mse = ((rendered - targets) ** 2).mean(dim=(1, 2))
    assert torch.allclose(mse, result.final_loss, atol=1e-5)


def test_t4_final_grad_norm_falls_with_budget() -> None:
    """S8's stationarity measure must behave like one: more steps, smaller relative gradient.

    Render fidelity does not test this -- a network can interpolate the sampled grid while
    sitting far from a stationary point -- which is exactly the confound the convergence
    sweep exists to remove.
    """
    torch.manual_seed(0)
    coords = make_coord_grid(8, 8)
    targets = torch.rand(4, 64, 1) * 2 - 1
    norms = []
    for steps in (50, 800):
        r = fit_batch(targets, coords, (16, 16), steps=steps, init_seeds=[0, 1, 2, 3])
        assert r.final_grad_norm is not None
        assert r.final_grad_norm.shape == (4,)
        assert torch.isfinite(r.final_grad_norm).all()
        norms.append(float(r.final_grad_norm.median()))
    assert norms[1] < norms[0], f"gradient norm did not fall with budget: {norms}"
