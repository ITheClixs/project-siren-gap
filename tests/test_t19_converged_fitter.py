"""T19: the converged fitter (S12) -- decayed schedule plus a per-INR stationarity stop.

Every corpus in this program so far is fitted by constant-step Adam and none reaches
stationarity: S8 measured the endpoint gradient norm *rising* between 1000 and 10000 steps,
because Adam's step size does not shrink with the gradient. These tests pin the two properties
the S12 corpora need, and one property they must not lose.
"""

from __future__ import annotations

import torch

from sirengap.fitting.batched import fit_batch


def _problem(b: int = 4, p: int = 64):
    torch.manual_seed(0)
    coords = torch.rand(p, 2) * 2 - 1
    targets = torch.sin(3 * coords[:, :1]) * torch.cos(2 * coords[:, 1:])
    return targets.expand(p, 1).unsqueeze(0).repeat(b, 1, 1).contiguous(), coords


def test_decay_beats_constant_step_past_the_end_of_descent() -> None:
    """The claim is regime-specific, and the test has to sit in that regime.

    Constant-step Adam is competitive while it is still descending -- S8 found 1000 steps to be
    its best budget -- and degrades afterwards, because its step size does not shrink with the
    gradient and the iterate diffuses in a band set by the learning rate. The decayed schedule
    should win at budgets past that point, which is where the S12 corpora will sit.
    """
    targets, coords = _problem(b=4, p=256)
    const = fit_batch(targets, coords, widths=(32, 32), steps=3000, lr=1e-3, seed=0, device="cpu")
    decayed = fit_batch(targets, coords, widths=(32, 32), steps=3000, lr=1e-3, seed=0,
                        device="cpu", schedule="cosine", lr_final=1e-6)
    assert float(decayed.final_grad_norm.median()) < float(const.final_grad_norm.median())
    # and the fit is no worse for it
    assert float(decayed.final_loss.median()) <= 2 * float(const.final_loss.median())


def test_stationarity_stop_freezes_converged_inrs() -> None:
    """An INR under the tolerance must stop moving, so later steps cannot undo its convergence."""
    targets, coords = _problem()
    r = fit_batch(targets, coords, widths=(16, 16), steps=300, lr=1e-3, seed=0, device="cpu",
                  schedule="cosine", lr_final=1e-6, stop_grad_norm=1e-1)
    # a deliberately loose tolerance: every INR should trip it and stop early
    assert r.stopped_at is not None
    assert int(r.stopped_at.max()) < 300, "no INR stopped despite a loose tolerance"


def test_stop_is_per_inr_not_global() -> None:
    targets, coords = _problem(b=4)
    targets = targets.clone()
    targets[0] *= 0.01  # one much easier problem, which should converge sooner
    r = fit_batch(targets, coords, widths=(16, 16), steps=300, lr=1e-3, seed=0, device="cpu",
                  schedule="cosine", lr_final=1e-6, stop_grad_norm=1e-2)
    assert r.stopped_at is not None
    assert len(set(int(x) for x in r.stopped_at)) > 1, "stopping is global, not per-INR"


def test_the_default_path_is_bit_identical_to_the_frozen_fitter() -> None:
    """Existing corpora must remain reproducible: no new argument may change the default path."""
    targets, coords = _problem()
    a = fit_batch(targets, coords, widths=(16, 16), steps=120, lr=1e-3, seed=3, device="cpu")
    b = fit_batch(targets, coords, widths=(16, 16), steps=120, lr=1e-3, seed=3, device="cpu",
                  schedule="constant", lr_final=None, stop_grad_norm=None)
    for x, y in zip(a.params.flat(), b.params.flat()):
        assert torch.equal(x, y)
    assert a.stopped_at is None


def test_converged_fit_is_deterministic() -> None:
    targets, coords = _problem()
    kw = dict(widths=(16, 16), steps=200, lr=1e-3, seed=1, device="cpu",
              schedule="cosine", lr_final=1e-6, stop_grad_norm=1e-3)
    a = fit_batch(targets, coords, **kw)
    b = fit_batch(targets, coords, **kw)
    assert torch.equal(a.params.flat(), b.params.flat())
    assert torch.equal(a.stopped_at, b.stopped_at)
