"""The D-infinity wreath-product action on sine-INR weights (PO-1).

Per hidden neuron, every group element decomposes as (d, j) with d in {0,1},
j in Z, acting by
    w -> (-1)^d w,   b -> (-1)^d b + pi * j,   u -> (-1)^(d+j) u,
where u is the neuron's outgoing column. Special cases: sigma = (d=1, j=0),
rho = (d=0, j=1), tau_k = (d=0, j=2k). Function preservation for all (d, j):
    (-1)^(d+j) u * sin((-1)^d (w.x + b) + pi j)
  = (-1)^(d+j) u * (-1)^j sin((-1)^d (w.x + b))
  = (-1)^(d+j) u * (-1)^j (-1)^d sin(w.x + b) = u * sin(w.x + b).
Per layer the group is D_inf wreath S_n: neuron-wise (d, j) plus a permutation
acting jointly on (rows of W_l, b_l, columns of the outgoing matrix).

Reference for all sign conventions: docs/THINKING/G0-theory-scoping.md §1–2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from sirengap.models.params import SirenParams, outgoing, replace_layer


@dataclass(frozen=True)
class GroupElement:
    """One element of prod_l (D_inf^{n_l} semidirect S_{n_l}), batched over B INRs.

    d[l]: [B, n_l] in {0,1}; j[l]: [B, n_l] integer; perm[l]: [B, n_l] index rows.
    perm[l][b] is a permutation of range(n_l): new neuron slot i holds old neuron
    perm[l][b, i].
    """

    d: tuple[Tensor, ...]
    j: tuple[Tensor, ...]
    perm: tuple[Tensor, ...]


def random_element(
    params: SirenParams,
    generator: torch.Generator,
    max_windings: int = 3,
    identity_perm: bool = False,
) -> GroupElement:
    """Sample a random group element matching `params` shapes (CPU generator)."""
    ds, js, perms = [], [], []
    for w, _ in params.hidden:
        b_sz, n = w.shape[0], w.shape[1]
        ds.append(torch.randint(0, 2, (b_sz, n), generator=generator))
        js.append(torch.randint(-max_windings, max_windings + 1, (b_sz, n), generator=generator))
        if identity_perm:
            perms.append(torch.arange(n).expand(b_sz, n).clone())
        else:
            perms.append(torch.argsort(torch.rand(b_sz, n, generator=generator), dim=1))
    return GroupElement(d=tuple(ds), j=tuple(js), perm=tuple(perms))


def _apply_neuronwise(
    w: Tensor, b: Tensor, out_w: Tensor, d: Tensor, j: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    sign_w = 1.0 - 2.0 * d.to(w.dtype)  # (-1)^d, [B, n]
    sign_u = sign_w * torch.where(j % 2 == 0, 1.0, -1.0).to(w.dtype)  # (-1)^(d+j)
    w_new = sign_w[:, :, None] * w
    b_new = sign_w * b + math.pi * j.to(b.dtype)
    out_new = out_w * sign_u[:, None, :]
    return w_new, b_new, out_new


def _apply_perm(
    w: Tensor, b: Tensor, out_w: Tensor, perm: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    idx_rows = perm[:, :, None].expand_as(w)
    w_new = torch.gather(w, 1, idx_rows)
    b_new = torch.gather(b, 1, perm)
    idx_cols = perm[:, None, :].expand_as(out_w)
    out_new = torch.gather(out_w, 2, idx_cols)
    return w_new, b_new, out_new


def apply(g: GroupElement, params: SirenParams) -> SirenParams:
    """Apply a group element; returns new params, f is exactly preserved (T1)."""
    result = params
    for layer in range(params.n_layers):
        w, b = result.hidden[layer]
        out_w = outgoing(result, layer)
        dev = w.device
        d_l, j_l, p_l = (t.to(dev) for t in (g.d[layer], g.j[layer], g.perm[layer]))
        w, b, out_w = _apply_neuronwise(w, b, out_w, d_l, j_l)
        w, b, out_w = _apply_perm(w, b, out_w, p_l)
        result = replace_layer(result, layer, w, b, out_w)
    return result
