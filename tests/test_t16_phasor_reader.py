"""T16: the phasor-graded reader (rung W12) is exactly G-invariant on RAW parameters.

W11b is G-invariant too, but only because it is handed W10's invariants: it inherits whatever
that fixed family discards. W12 is the construction the external review asked for -- a reader
that quotients D_infinity on the parameters themselves.

The contract:

  1. the reader's *logits* are unchanged by any element of
     (D_inf wr S_n) x (D_inf wr S_p), including large windings, to fp32 tolerance;
  2. that holds for windings far outside the range any fitted network occupies (|j| <= 40),
     because the phasor quotients the integer translation exactly rather than approximately;
  3. it is not invariant by being degenerate -- different INRs give different logits;
  4. the graded feature map assigns each block the character the algebra says it has, which is
     the property the equivariant layers rely on.
"""

from __future__ import annotations

import pytest
import torch

from sirengap.fitting.batched import absorb_omega, init_from_seeds
from sirengap.models.params import SirenParams
from sirengap.models.phasor import CHARACTERS, PhasorGradedReader, phasor_features
from sirengap.symmetry.dinf import apply, random_element

TOL = 1e-4  # relative; fp32 message passing over 32 neurons accumulates more than the encoders do


def _params(batch: int = 6, widths: tuple[int, ...] = (32, 32), out_dim: int = 1) -> SirenParams:
    raw = init_from_seeds(list(range(batch)), in_dim=2, widths=widths, out_dim=out_dim)
    return absorb_omega([t * 1.0 for t in raw])


def _reader(params: SirenParams, width: int = 64) -> PhasorGradedReader:
    torch.manual_seed(0)
    feats = phasor_features(params)
    model = PhasorGradedReader.from_features(feats, width=width, n_classes=10)
    model.eval()
    return model


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max() / a.abs().max().clamp_min(1e-12))


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_logits_are_group_invariant(seed: int) -> None:
    params = _params()
    model = _reader(params)
    gen = torch.Generator().manual_seed(seed)
    g = random_element(params, gen, max_windings=3)
    with torch.no_grad():
        base = model(phasor_features(params))
        moved = model(phasor_features(apply(g, params)))
    assert _rel(base, moved) < TOL


@pytest.mark.parametrize("windings", [10, 40])
def test_invariance_survives_large_windings(windings: int) -> None:
    """The phasor quotients the integer translation exactly, so |j| is unbounded."""
    params = _params()
    model = _reader(params)
    gen = torch.Generator().manual_seed(7)
    g = random_element(params, gen, max_windings=windings)
    with torch.no_grad():
        base = model(phasor_features(params))
        moved = model(phasor_features(apply(g, params)))
    assert _rel(base, moved) < TOL


@pytest.mark.parametrize("out_dim", [1, 3])
def test_reader_separates_distinct_inrs(out_dim: int) -> None:
    params = _params(batch=6, out_dim=out_dim)
    model = _reader(params)
    with torch.no_grad():
        logits = model(phasor_features(params))
    assert logits.shape == (6, 10)
    spread = (logits - logits.mean(dim=0, keepdim=True)).abs().max()
    assert float(spread) > 1e-4


@pytest.mark.parametrize("layer", ["l1", "l2"])
def test_feature_blocks_carry_their_declared_character(layer: str) -> None:
    """Each graded block must pick up exactly (-1)^(a*d + c*j) under a single-neuron element."""
    params = _params(batch=1)
    feats = phasor_features(params)
    gen = torch.Generator().manual_seed(11)
    for d in (0, 1):
        for j in (0, 1):
            g = random_element(params, gen, max_windings=0, identity_perm=True)
            depth = 0 if layer == "l1" else 1
            ds = [torch.zeros_like(x) for x in g.d]
            js = [torch.zeros_like(x) for x in g.j]
            ds[depth][:, 0] = d
            js[depth][:, 0] = j
            g = type(g)(d=tuple(ds), j=tuple(js), perm=g.perm)
            moved = phasor_features(apply(g, params))
            for (a, c) in CHARACTERS:
                before = feats[layer][(a, c)][0, 0]
                after = moved[layer][(a, c)][0, 0]
                sign = (-1.0) ** (a * d + c * j)
                assert torch.allclose(after, sign * before, atol=1e-5), (
                    f"{layer} block {(a, c)} wrong under (d={d}, j={j})"
                )


def test_feature_scale_is_itself_group_invariant() -> None:
    """A shift here would silently destroy every character; scaling must not, and must not
    depend on which orbit representative the corpus happens to be written in."""
    from sirengap.models.phasor import apply_scale, feature_scale

    params = _params(batch=16)
    base = phasor_features(params)
    gen = torch.Generator().manual_seed(5)
    moved = phasor_features(apply(random_element(params, gen, max_windings=6), params))

    s1, s2 = feature_scale(base), feature_scale(moved)
    for layer in ("l1", "l2"):
        for c in CHARACTERS:
            assert torch.allclose(s1[layer][c], s2[layer][c], atol=1e-5), f"{layer} {c}"
    assert torch.allclose(s1["edge"], s2["edge"], atol=1e-5)

    # and the scaled features still feed an invariant reader
    model = _reader(params)
    with torch.no_grad():
        a = model(apply_scale(base, s1))
        b = model(apply_scale(moved, s1))
    assert _rel(a, b) < TOL
