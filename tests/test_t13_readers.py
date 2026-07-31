"""T13: the W11 readers are exactly what they claim, and no more.

W11a must be invariant to neuron permutations and must *not* be invariant to D_infinity — the
second half matters as much as the first, because W11a exists to represent the field's
permutation-only coverage. If it accidentally became D_infinity-invariant it would stop answering
the question it was built for.

W11b must be invariant to the whole product group. Its features are the same ones W10 uses, so
this is the same invariance T10 certifies, carried through a learned equivariant network.
"""

from __future__ import annotations

import pytest
import torch

from conftest import random_params
from sirengap.models.readers import (
    InvariantGraphReader,
    RawGraphReader,
    invariant_graph_features,
    raw_graph_features,
)
from sirengap.symmetry.dinf import apply, random_element


def _perm_only(params, seed: int):
    """A group element with trivial D_infinity part: permutations alone."""
    gen = torch.Generator().manual_seed(seed)
    g = random_element(params, gen, max_windings=0)
    zeros = tuple(torch.zeros_like(d) for d in g.d)
    return type(g)(d=zeros, j=tuple(torch.zeros_like(j) for j in g.j), perm=g.perm)


def _raw_reader(params, seed: int = 0) -> RawGraphReader:
    torch.manual_seed(seed)
    m = params.hidden[0][0].shape[2]
    c = params.w_out.shape[1]
    return RawGraphReader(m=m, c=c, width=32, rounds=2).double().eval()


def _inv_reader(params, seed: int = 0) -> InvariantGraphReader:
    f = invariant_graph_features(params)
    torch.manual_seed(seed)
    return InvariantGraphReader(
        n_node=f["x1"].shape[2], n_edge=f["e"].shape[3], n_global=f["g"].shape[1],
        width=32, rounds=2,
    ).double().eval()


def _dbl(f: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.double() for k, v in f.items()}


@pytest.mark.parametrize("trial", range(4))
def test_t13a_raw_reader_is_permutation_invariant(trial: int) -> None:
    params = random_params(3, 2, (8, 8), 1, seed=500 + trial)
    reader = _raw_reader(params)
    with torch.no_grad():
        base = reader(_dbl(raw_graph_features(params)))
        moved = reader(_dbl(raw_graph_features(apply(_perm_only(params, trial), params))))
    gap = (base - moved).abs().max().item()
    assert gap < 1e-8, f"W11a is not permutation-invariant: {gap:.2e}"


def test_t13b_raw_reader_is_not_dinf_invariant() -> None:
    """The negative control: W11a must move under sign/phase, or it is not the field's model."""
    params = random_params(3, 2, (8, 8), 1, seed=77)
    reader = _raw_reader(params)
    gen = torch.Generator().manual_seed(9)
    g = random_element(params, gen, max_windings=2, identity_perm=True)
    with torch.no_grad():
        base = reader(_dbl(raw_graph_features(params)))
        moved = reader(_dbl(raw_graph_features(apply(g, params))))
    gap = (base - moved).abs().max().item()
    assert gap > 1e-3, (
        f"W11a came out D_infinity-invariant (gap {gap:.2e}); it would no longer represent "
        "permutation-only coverage"
    )


@pytest.mark.parametrize("trial", range(4))
def test_t13c_invariant_reader_is_group_invariant(trial: int) -> None:
    params = random_params(3, 2, (8, 8), 1, seed=600 + trial)
    reader = _inv_reader(params)
    gen = torch.Generator().manual_seed(700 + trial)
    g = random_element(params, gen, max_windings=3)
    with torch.no_grad():
        base = reader(_dbl(invariant_graph_features(params)))
        moved = reader(_dbl(invariant_graph_features(apply(g, params))))
    gap = (base - moved).abs().max().item()
    assert gap < 1e-6, f"W11b is not group-invariant: {gap:.2e}"


def test_t13d_invariant_reader_multichannel() -> None:
    params = random_params(2, 2, (8, 8), 3, seed=88)
    reader = _inv_reader(params)
    gen = torch.Generator().manual_seed(11)
    with torch.no_grad():
        base = reader(_dbl(invariant_graph_features(params)))
        moved = reader(_dbl(invariant_graph_features(apply(random_element(params, gen, 3), params))))
    assert (base - moved).abs().max().item() < 1e-6


def test_t13e_invariant_reader_is_not_constant() -> None:
    """Invariance is trivial to achieve by ignoring the input; check it does not."""
    a = random_params(4, 2, (8, 8), 1, seed=1)
    b = random_params(4, 2, (8, 8), 1, seed=2)
    reader = _inv_reader(a)
    with torch.no_grad():
        out_a = reader(_dbl(invariant_graph_features(a)))
        out_b = reader(_dbl(invariant_graph_features(b)))
    assert (out_a - out_b).abs().max().item() > 1e-3, "W11b ignores its input"


def test_t13f_readers_refuse_wrong_depth() -> None:
    p1 = random_params(2, 2, (8,), 1, seed=3)
    with pytest.raises(ValueError, match="L=2"):
        raw_graph_features(p1)
    with pytest.raises(ValueError, match="L=2"):
        invariant_graph_features(p1)
