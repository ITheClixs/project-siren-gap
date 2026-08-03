"""Matched non-invariant control for the L=2 invariant encoding (rung W10c).

An external review made the following objection, and it is correct: rung W10 is *both* a
nonlinear feature map and a G-invariant one, so its accuracy does not by itself attribute
anything to symmetry. A map with the same nonlinearity and no invariance might do just as
well, in which case W10's gain is ordinary feature engineering.

This module is that control. It emits, coordinate for coordinate, the same monomials in
(w, u) with the same trigonometric order in b as `deep_invariants.encode_deep`, pooled the
same way, at the same dimension -- with only the *parity class* of the trigonometric
factors changed, so that the emitted coordinates are no longer D_inf-invariant.

Parity bookkeeping under g_{d,j}: (w, b, u) -> ((-1)^d w, (-1)^d b + pi j, (-1)^{d+j} u),
writing eps_i = (-1)^{d_i + j_i} for the sign the layer-2 Gram G_il -> eps_i eps_l G_il
picks up:

    factor          transforms as          factor            transforms as
    ------------------------------------   -------------------------------------
    sin b_i         eps_i                  cos b_i           (-1)^{j_i}
    sin 2b_i        (-1)^{d_i}             cos 2b_i          1
    w_i             (-1)^{d_i}             ||w_i||^2         1

W10 coordinate                        control coordinate                    residual sign
-----------------------------------   -----------------------------------   -------------
(sin b_i sin b_l) G_il                (cos b_i cos b_l) G_il                (-1)^{d_i+d_l}
(cos b_i w_i . cos b_l w_l) G_il      (sin b_i w_i . sin b_l w_l) G_il      (-1)^{d_i+d_l}
(sin 2b_i w_i . sin 2b_l w_l)         (cos 2b_i w_i . cos 2b_l w_l)         (-1)^{d_i+d_l}
||w_i||^2                             sin(2b_i) ||w_i||^2                   (-1)^{d_i}
cos 2b_i                              sin b_i                               eps_i
G_ii                                  cos(b_i) G_ii                         (-1)^{j_i}
||w2_k||^2                            sin(2 b2_k) ||w2_k||^2                (-1)^{d_k}
cos 2b2_k                             sin b2_k                              eps_k
||u_k||^2                             cos(b2_k) ||u_k||^2                   (-1)^{j_k}
(sin b2_k) u_k                        (cos b2_k) u_k                        (-1)^{d_k}

Every control coordinate has the same total degree in (w, u) as the W10 coordinate it
replaces and a trigonometric order in {0,1,2} drawn from the same set; the three matrices
are still symmetric and still transform as M -> P M P^T, so the *permutation* half of the
group is quotiented out identically by the same eigenvalue spectra and the same
||w||^2 sort key. The two maps therefore differ in exactly one property: whether the
affine phase/reflection component D_inf has been removed.

Non-invariance is asserted numerically in T15, at a margin four orders of magnitude above
the fp32 round-off at which `encode_deep` passes the same test.
"""

from __future__ import annotations

import torch
from torch import Tensor

from sirengap.canon.deep_invariants import _spectrum
from sirengap.models.params import SirenParams, outgoing


def layer1_control(params: SirenParams) -> Tensor:
    """Parity-swapped counterpart of `deep_invariants.layer1_features`: [B, 6n]."""
    w, b = params.hidden[0]  # [B, n, m], [B, n]
    gram = torch.einsum("bki,bkl->bil", outgoing(params, 0), outgoing(params, 0))  # [B, n, n]

    sin_b, cos_b, sin_2b, cos_2b = torch.sin(b), torch.cos(b), torch.sin(2 * b), torch.cos(2 * b)
    wb = sin_b[:, :, None] * w  # W10 uses cos b here
    wc = cos_2b[:, :, None] * w  # W10 uses sin 2b here

    a_mat = cos_b[:, :, None] * cos_b[:, None, :] * gram  # W10 uses sin b sin b
    b_mat = torch.einsum("bim,blm->bil", wb, wb) * gram
    c_mat = torch.einsum("bim,blm->bil", wc, wc)

    w_norm2 = (w * w).sum(dim=2)
    order = torch.argsort(w_norm2, dim=1, descending=True)  # the same invariant sort key
    per_neuron = [sin_2b * w_norm2, sin_b, cos_b * torch.diagonal(gram, dim1=1, dim2=2)]

    return torch.cat(
        [_spectrum(a_mat), _spectrum(b_mat), _spectrum(c_mat)]
        + [torch.gather(x, 1, order) for x in per_neuron],
        dim=1,
    )


def layer2_control(params: SirenParams) -> Tensor:
    """Parity-swapped counterpart of `deep_invariants.layer2_features`: [B, (3 + c) p]."""
    w2, b2 = params.hidden[1]  # [B, p, n], [B, p]
    u = params.w_out.transpose(1, 2)  # [B, p, c]

    order = torch.argsort((w2 * w2).sum(dim=2), dim=1, descending=True)  # the same sort key

    def by_order(x: Tensor) -> Tensor:
        return torch.gather(x, 1, order if x.ndim == 2 else order[:, :, None].expand_as(x))

    feats = [
        by_order(torch.sin(2 * b2) * (w2 * w2).sum(dim=2)),
        by_order(torch.sin(b2)),
        by_order(torch.cos(b2) * (u * u).sum(dim=2)),
        by_order(torch.cos(b2)[:, :, None] * u).flatten(1),
    ]
    return torch.cat([f if f.ndim == 2 else f.flatten(1) for f in feats], dim=1)


def encode_deep_control(params: SirenParams) -> Tensor:
    """Non-invariant encoding matched to `encode_deep` in shape, degree and pooling."""
    if params.n_layers != 2:
        raise ValueError(
            f"encode_deep_control is derived for L=2 (got L={params.n_layers}); it exists "
            "only as the matched control for encode_deep"
        )
    return torch.cat([layer1_control(params), layer2_control(params)], dim=1)
