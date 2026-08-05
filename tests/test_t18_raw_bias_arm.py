"""T18: the raw-bias arm (W12b), the control that separates coordinates from architecture.

W12 changes two things at once against W11a: it represents the bias in phasor coordinates, and
it reads those coordinates with a graded message-passing skeleton. W12u removed the grading and
kept the coordinates. This arm does the converse -- the identical skeleton, the identical block
structure, capacity re-solved by the same rule, but the bias enters *raw* instead of lifted.

It must therefore be non-invariant, and non-invariant in a specific way: a winding b -> b + pi j
is exactly what the phasor lift quotients, so the raw arm has to move under it. A control that
accidentally stayed invariant would prove nothing, which is the failure mode audited for W12u.
"""

from __future__ import annotations

import pytest
import torch

from sirengap.fitting.batched import absorb_omega, init_from_seeds
from sirengap.models.params import SirenParams
from sirengap.models.phasor import CHARACTERS, PhasorGradedReader, phasor_features
from sirengap.symmetry.dinf import apply, random_element


def _params(batch: int = 6, widths: tuple[int, ...] = (32, 32), out_dim: int = 1) -> SirenParams:
    raw = init_from_seeds(list(range(batch)), in_dim=2, widths=widths, out_dim=out_dim)
    return absorb_omega([t * 1.0 for t in raw])


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max() / a.abs().max().clamp_min(1e-12))


def test_raw_bias_features_carry_the_bias_unlifted() -> None:
    p = _params()
    f = phasor_features(p, raw_bias=True)
    b1, b2 = p.hidden[0][1], p.hidden[1][1]
    # the bias enters once, in the sign-covariant block, as itself
    assert torch.allclose(f["l1"][(1, 0)][:, :, -1], b1)
    assert torch.allclose(f["l2"][(1, 0)][:, :, -1], b2)
    # and nowhere else: no trig channel of the bias survives anywhere
    for layer in ("l1", "l2"):
        for c in CHARACTERS:
            t = f[layer][c]
            if t.shape[2] == 0:
                continue
            assert not torch.isclose(t, torch.cos(b1)[:, :, None]).all(), (layer, c)


def test_raw_bias_keeps_the_skeleton_and_the_edge_coupling() -> None:
    p = _params()
    graded, raw = phasor_features(p), phasor_features(p, raw_bias=True)
    assert set(raw) == set(graded)
    assert set(raw["l1"]) == set(CHARACTERS) and set(raw["l2"]) == set(CHARACTERS)
    assert torch.allclose(raw["edge"], graded["edge"])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_raw_bias_arm_is_not_invariant(seed: int) -> None:
    """The point of the control: it must move under the group, or it proves nothing."""
    p = _params()
    torch.manual_seed(0)
    feats = phasor_features(p, raw_bias=True)
    model = PhasorGradedReader.from_features(feats, width=64, n_classes=10).eval()
    gen = torch.Generator().manual_seed(seed)
    g = random_element(p, gen, max_windings=3)
    with torch.no_grad():
        base = model(feats)
        moved = model(phasor_features(apply(g, p), raw_bias=True))
    assert _rel(base, moved) > 1e-3, "the raw-bias control is accidentally invariant"


def test_raw_bias_arm_moves_under_a_pure_winding() -> None:
    """Sharper: the winding alone is what the phasor quotients, so it alone must break this."""
    p = _params()
    torch.manual_seed(0)
    feats = phasor_features(p, raw_bias=True)
    model = PhasorGradedReader.from_features(feats, width=64, n_classes=10).eval()
    shifted = SirenParams(
        hidden=((p.hidden[0][0], p.hidden[0][1] + 2 * torch.pi),
                (p.hidden[1][0], p.hidden[1][1])),
        w_out=p.w_out, b_out=p.b_out,
    )
    with torch.no_grad():
        base, moved = model(feats), model(phasor_features(shifted, raw_bias=True))
    assert _rel(base, moved) > 1e-3
    # while the phasor reader of W12 does not move under the same shift
    torch.manual_seed(0)
    pf = phasor_features(p)
    graded_model = PhasorGradedReader.from_features(pf, width=64, n_classes=10).eval()
    with torch.no_grad():
        gb, gm = graded_model(pf), graded_model(phasor_features(shifted))
    assert _rel(gb, gm) < 1e-4


def test_raw_bias_reader_trains_end_to_end() -> None:
    p = _params(batch=8)
    torch.manual_seed(0)
    feats = phasor_features(p, raw_bias=True)
    model = PhasorGradedReader.from_features(feats, width=32, n_classes=10)
    logits = model(feats)
    assert logits.shape == (8, 10)
    logits.sum().backward()
    grads = [q.grad for q in model.parameters() if q.grad is not None and q.numel()]
    assert grads and any(float(g.abs().sum()) > 0 for g in grads)


def test_empty_blocks_carry_no_parameters_and_no_zero_element_tensors() -> None:
    """MPS cannot allocate zero-element tensors, so empty blocks must be skipped, not encoded.

    The semantics must match a zero-column linear map exactly: output zeros, no parameters, and
    nothing invented to fill the block.
    """
    from sirengap.models.phasor import GradedLinear

    dims = {(0, 0): 3, (1, 0): 4, (0, 1): 0, (1, 1): 0}
    lin = GradedLinear(dims, width=8, graded=True)
    assert set(lin.empty) == {(0, 1), (1, 1)}
    assert all(p.numel() > 0 for p in lin.parameters()), "no zero-element parameter may exist"

    x = {c: torch.randn(2, 5, d) for c, d in dims.items()}
    y = lin(x)
    assert all(y[c].shape == (2, 5, 8) for c in dims)
    assert float(y[(0, 1)].abs().sum()) == 0.0 and float(y[(1, 1)].abs().sum()) == 0.0


def test_raw_bias_reader_has_no_zero_element_parameters() -> None:
    p = _params()
    torch.manual_seed(0)
    model = PhasorGradedReader.from_features(
        phasor_features(p, raw_bias=True), width=32, n_classes=10)
    empty = [n for n, q in model.named_parameters() if q.numel() == 0]
    assert not empty, f"zero-element parameters break MPS: {empty}"
