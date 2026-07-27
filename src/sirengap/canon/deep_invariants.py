"""Deep phase-invariant encoding for L=2 sine networks (rung W10; Ch3.6, OPEN_PROBLEMS #4).

`canon/invariants.py` implements PO-4's per-neuron invariants and deliberately refuses
L >= 2, because a hidden neuron's outgoing vector u is itself acted on by the *next*
layer's group, so the (0,1)/(1,1) parity classes stop being invariant. This module
supplies the missing coupling for L=2.

Notation (stored/canonical form): h1 = sin(W1 x + b1), h2 = sin(W2 h1 + b2), y = W3 h2 + b3,
with W1 [n, m], W2 [p, n], W3 [c, p]. The group (PO-1) acts per neuron as
    g_{d,j}: (w, beta, u) |-> ((-1)^d w, (-1)^d beta + pi j, (-1)^{d+j} u)
plus a permutation within each layer. For a layer-1 neuron i, u_i is the column W2[:, i];
for a layer-2 neuron k, u_k is the column W3[:, k].

Key object: the layer-2 Gram  G = W2^T W2  in R^{n x n}.

  * Layer-2 row sign flips cancel inside each product W2[k,i] W2[k,l]; layer-2 row
    permutations only reindex the sum; layer-2 phase shifts touch b2 and W3, not W2.
    So G is invariant under the entire layer-2 group.
  * Under layer-1's action, u_i picks up eps_i := (-1)^{d_i + j_i}, hence
    G_il -> eps_i eps_l G_il, and G -> P G P^T under a layer-1 permutation P.

Layer-1 parity bookkeeping (d = reflection, j = winding):
    w_i, sin 2b_i          odd in d,  even in j
    cos 2b_i, ||w_i||^2    even,      even          (fully invariant per neuron)
    sin b_i                odd in d,  odd in j   =  eps_i
    cos b_i                even in d, odd in j
so eps_i-covariant scalars can be built as `sin b_i` or as `cos b_i w_i` (contracted),
and pairing two of them against G_il cancels every sign:

    A_il = (sin b_i sin b_l) G_il
    B_il = (cos b_i w_i . cos b_l w_l) G_il
    C_il = (sin 2b_i w_i . sin 2b_l w_l)           (each factor already invariant)

Each is symmetric and transforms as M -> P M P^T, so its **sorted eigenvalue spectrum**
is invariant under the full group. Per-neuron even scalars (||w_i||^2, cos 2b_i, G_ii)
transform as v -> P v, so their **sorted order statistics** are invariant. Layer 2 is
handled by PO-4 directly, contracted over the layer-1 index (which is permuted/sign-flipped
by layer 1) so that only norms and the eps-cancelling `sin b2_k u_k` survive.

The construction is invariance-tested numerically against random group elements (T10).
"""

from __future__ import annotations

import torch
from torch import Tensor

from sirengap.models.params import SirenParams, outgoing


def _sorted_desc(x: Tensor) -> Tensor:
    return torch.sort(x, dim=-1, descending=True).values


def _spectrum(mat: Tensor) -> Tensor:
    """Sorted eigenvalues of a batch of symmetric matrices [B, n, n] -> [B, n]."""
    sym = 0.5 * (mat + mat.transpose(-1, -2))
    return _sorted_desc(torch.linalg.eigvalsh(sym.double())).to(mat.dtype)


def layer1_features(params: SirenParams) -> Tensor:
    """Layer-1 invariants coupled through the layer-2 Gram: [B, 6n]."""
    w, b = params.hidden[0]  # [B, n, m], [B, n]
    gram = torch.einsum("bki,bkl->bil", outgoing(params, 0), outgoing(params, 0))  # [B, n, n]

    sin_b, cos_b, sin_2b, cos_2b = torch.sin(b), torch.cos(b), torch.sin(2 * b), torch.cos(2 * b)
    wb = cos_b[:, :, None] * w
    wc = sin_2b[:, :, None] * w

    a_mat = sin_b[:, :, None] * sin_b[:, None, :] * gram
    b_mat = torch.einsum("bim,blm->bil", wb, wb) * gram
    c_mat = torch.einsum("bim,blm->bil", wc, wc)

    # per-neuron even scalars share one invariant sort key so they stay associated
    w_norm2 = (w * w).sum(dim=2)
    order = torch.argsort(w_norm2, dim=1, descending=True)
    per_neuron = [w_norm2, cos_2b, torch.diagonal(gram, dim1=1, dim2=2)]

    return torch.cat(
        [_spectrum(a_mat), _spectrum(b_mat), _spectrum(c_mat)]
        + [torch.gather(x, 1, order) for x in per_neuron],
        dim=1,
    )


def layer2_features(params: SirenParams) -> Tensor:
    """PO-4 invariants of layer 2, contracted over the layer-1 index: [B, (3 + c) p]."""
    w2, b2 = params.hidden[1]  # [B, p, n], [B, p]
    u = params.w_out.transpose(1, 2)  # [B, p, c] — column k of w_out is neuron k's u

    order = torch.argsort((w2 * w2).sum(dim=2), dim=1, descending=True)  # [B, p]

    def by_order(x: Tensor) -> Tensor:
        return torch.gather(x, 1, order if x.ndim == 2 else order[:, :, None].expand_as(x))

    feats = [
        by_order((w2 * w2).sum(dim=2)),  # ||w_k||^2
        by_order(torch.cos(2 * b2)),
        by_order((u * u).sum(dim=2)),  # ||u_k||^2
        by_order(torch.sin(b2)[:, :, None] * u).flatten(1),  # (0,1) class, eps-cancelling
    ]
    return torch.cat([f if f.ndim == 2 else f.flatten(1) for f in feats], dim=1)


def encode_deep(params: SirenParams) -> Tensor:
    """Full invariant encoding of an L=2 sine network: [B, F].

    A single joint sort key is used inside each layer, so features stay associated
    across the concatenation the way a canonicalizer would associate them, while
    remaining invariant because every sorted quantity is itself invariant.
    """
    if params.n_layers != 2:
        raise ValueError(
            f"encode_deep is derived for L=2 (got L={params.n_layers}); "
            "L=1 is canon.invariants.encode_pooled, L>2 is OPEN_PROBLEMS #4"
        )
    return torch.cat([layer1_features(params), layer2_features(params)], dim=1)
