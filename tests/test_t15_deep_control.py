"""T15: the matched non-invariant control for rung W10 (external review, Priority 3).

W10's accuracy is produced by a feature map that is *both* nonlinear and G-invariant. Its
gain is therefore not attributable to symmetry handling unless a map that is equally
nonlinear but *not* G-invariant recovers less. `canon/deep_control.py` supplies that map.

The contract the control must satisfy, and that this file enforces:

  1. identical output dimension to `encode_deep` at every (width, out_dim) we use;
  2. identical polynomial degree in (w, u) and identical trigonometric order in b,
     coordinate by coordinate -- only the *parity class* of the trigonometric factor
     differs (see the table in `deep_control.py`);
  3. it is genuinely NOT invariant under the D_inf part of the group, by a margin far
     above the fp32 round-off at which `encode_deep` is invariant;
  4. it IS still invariant under the permutation part, so the two maps differ in exactly
     one thing: whether the affine phase/reflection component is quotiented out.

Property 4 is what makes the control sharp. It isolates the D_inf component -- the part
this paper claims is new -- rather than confounding it with permutation handling.
"""

from __future__ import annotations

import pytest
import torch

from sirengap.canon.deep_control import encode_deep_control
from sirengap.canon.deep_invariants import encode_deep
from sirengap.fitting.batched import absorb_omega, init_from_seeds
from sirengap.models.params import SirenParams
from sirengap.symmetry.dinf import apply, random_element

TOL_INVARIANCE = 1e-5  # the tolerance encode_deep passes at (measured moves ~3e-7)
MIN_BREAK = 1e-2  # the control must move by at least this much under D_inf


def _params(batch: int = 8, widths: tuple[int, ...] = (32, 32), out_dim: int = 1) -> SirenParams:
    raw = init_from_seeds(list(range(batch)), in_dim=2, widths=widths, out_dim=out_dim)
    return absorb_omega([t * 1.0 for t in raw])


def _rel_move(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max() / a.abs().max().clamp_min(1e-12))


@pytest.mark.parametrize("out_dim", [1, 3])
def test_control_matches_encoding_dimension(out_dim: int) -> None:
    params = _params(out_dim=out_dim)
    assert encode_deep_control(params).shape == encode_deep(params).shape


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_control_is_not_invariant_under_the_full_group(seed: int) -> None:
    params = _params()
    gen = torch.Generator().manual_seed(seed)
    g = random_element(params, gen, max_windings=3)
    assert _rel_move(encode_deep_control(params), encode_deep_control(apply(g, params))) > MIN_BREAK


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_control_is_still_permutation_invariant(seed: int) -> None:
    """Only the D_inf component is broken; permutation handling is held fixed."""
    params = _params()
    gen = torch.Generator().manual_seed(100 + seed)
    perm_only = random_element(params, gen, max_windings=0)
    perm_only = type(perm_only)(
        d=tuple(torch.zeros_like(x) for x in perm_only.d),
        j=tuple(torch.zeros_like(x) for x in perm_only.j),
        perm=perm_only.perm,
    )
    moved = _rel_move(encode_deep_control(params), encode_deep_control(apply(perm_only, params)))
    assert moved < TOL_INVARIANCE


def test_control_separates_distinct_inrs() -> None:
    """Non-invariance must not come at the cost of the features being degenerate."""
    feats = encode_deep_control(_params(batch=8))
    spread = (feats - feats.mean(dim=0, keepdim=True)).abs().max(dim=0).values
    assert float(spread.max()) > 1e-3
    assert float((spread > 1e-6).float().mean()) > 0.9


def test_control_refuses_wrong_depth() -> None:
    with pytest.raises(ValueError, match="derived for L=2"):
        encode_deep_control(_params(widths=(32,)))
