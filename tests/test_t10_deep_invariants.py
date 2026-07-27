"""T10: the L=2 deep invariant encoding (rung W10) is invariant under the full group.

The claim in canon/deep_invariants.py is that every emitted feature is unchanged by any
element of prod_l (D_inf^{n_l} semidirect S_{n_l}). This test applies random elements
(including large windings and non-trivial permutations) and requires the encoding to move
by less than the numerical tolerance, while a control confirms the encoding is not
trivially constant across different INRs.
"""

from __future__ import annotations

import pytest
import torch

from sirengap.canon.deep_invariants import encode_deep
from sirengap.fitting.batched import init_from_seeds, absorb_omega
from sirengap.models.params import SirenParams
from sirengap.symmetry.dinf import apply, random_element

TOL_INVARIANCE = 1e-5  # relative; measured moves are ~3e-7 (fp32 round-off), not near this


def _params(batch: int = 8, widths: tuple[int, ...] = (32, 32), out_dim: int = 1) -> SirenParams:
    raw = init_from_seeds(list(range(batch)), in_dim=2, widths=widths, out_dim=out_dim)
    return absorb_omega([t * 1.0 for t in raw])


def _rel_move(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max() / a.abs().max().clamp_min(1e-12))


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_encode_deep_is_group_invariant(seed: int) -> None:
    params = _params()
    gen = torch.Generator().manual_seed(seed)
    g = random_element(params, gen, max_windings=3)
    assert _rel_move(encode_deep(params), encode_deep(apply(g, params))) < TOL_INVARIANCE


def test_encode_deep_invariant_under_permutation_only() -> None:
    params = _params()
    gen = torch.Generator().manual_seed(99)
    g = random_element(params, gen, max_windings=0)
    assert _rel_move(encode_deep(params), encode_deep(apply(g, params))) < TOL_INVARIANCE


def test_encode_deep_separates_distinct_inrs() -> None:
    """Control: invariance must not come from the encoding being constant."""
    feats = encode_deep(_params(batch=8))
    spread = (feats - feats.mean(dim=0, keepdim=True)).abs().max(dim=0).values
    assert float(spread.max()) > 1e-3
    assert float((spread > 1e-6).float().mean()) > 0.9  # nearly every feature varies


def test_encode_deep_refuses_wrong_depth() -> None:
    with pytest.raises(ValueError, match="derived for L=2"):
        encode_deep(_params(widths=(32,)))
    with pytest.raises(ValueError, match="derived for L=2"):
        encode_deep(_params(widths=(16, 16, 16)))
