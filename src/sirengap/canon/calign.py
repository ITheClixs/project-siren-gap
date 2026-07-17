"""c_align: template canonicalizer via activation matching (Ch3.2, protocol A.2).

Scope qualifier (advisor G1): alignment is exact w.r.t. the *implemented* group
(PO-1); maximality of that group is PO-2 (proved L=1, conjectural deeper).
Function preservation is asserted on every call regardless.

Per hidden layer: phase-reduce, compute activations on a fixed probe set, solve
the assignment against template activations maximizing |corr|, take the
correlation sign as the sigma fix, propagate permutation+signs to the outgoing
matrix, recurse. The sigma/rho sign algebra follows the verified table in
docs/THINKING/G0-theory-scoping.md §2; function preservation is asserted (T2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

from sirengap.canon.assign import hungarian
from sirengap.canon.csort import _phase_reduce
from sirengap.constants import TOL_FUNC
from sirengap.models.forward import forward_canonical, max_functional_gap
from sirengap.models.params import SirenParams, outgoing, replace_layer


@dataclass(frozen=True)
class CAlignDiagnostics:
    corr_gaps: tuple[Tensor, ...]  # per layer [B]: spectral-ish margin = mean matched |corr|
    min_matched_corr: tuple[Tensor, ...]  # per layer [B]: weakest matched |corr|


def template_activations(template: SirenParams, probes: Tensor) -> list[Tensor]:
    """Layer activations of a single-INR template (B=1) on probes: list of [M, n_l]."""
    if template.batch != 1:
        raise ValueError("template must have batch size 1")
    acts = []
    w0, b0 = template.hidden[0]
    h = torch.sin(torch.einsum("bji,pi->bpj", w0, probes) + b0[:, None, :])[0]
    acts.append(h)
    for w, b in template.hidden[1:]:
        h = torch.sin(h @ w[0].T + b[0][None, :])
        acts.append(h)
    return acts


def _center_normalize(a: Tensor) -> Tensor:
    a = a - a.mean(dim=-2, keepdim=True)
    return a / (a.norm(dim=-2, keepdim=True) + 1e-12)


def c_align(
    params: SirenParams,
    template: SirenParams,
    probes: Tensor,
    solver: Callable[[Tensor], Tensor] = hungarian,
    check_preservation: bool = True,
) -> tuple[SirenParams, CAlignDiagnostics]:
    t_acts = template_activations(template, probes)
    result = params
    b_sz = params.batch
    corr_gaps, min_corrs = [], []

    h = probes[None, :, :].expand(b_sz, -1, -1)  # [B, M, n_0]
    for layer in range(params.n_layers):
        w, b = result.hidden[layer]
        out_w = outgoing(result, layer)
        w, b, out_w = _phase_reduce(w, b, out_w)

        acts = torch.sin(torch.einsum("bji,bpi->bpj", w, h) + b[:, None, :])  # [B, M, n]
        a_hat = _center_normalize(acts)
        t_hat = _center_normalize(t_acts[layer])  # [M, n]
        corr = torch.einsum("bpi,pj->bij", a_hat, t_hat)  # [B, n_model, n_template]

        perms, signs = [], []
        for k in range(b_sz):
            idx = solver(corr[k].abs())  # idx[t] = model neuron for template slot t
            perms.append(idx)
            signs.append(torch.sign(corr[k][idx, torch.arange(corr.shape[2])]))
        perm = torch.stack(perms).to(w.device)
        sign = torch.stack(signs).to(w.dtype).to(w.device)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)

        # permutation: template slot t takes model neuron perm[t]
        w = torch.gather(w, 1, perm[:, :, None].expand_as(w))
        b = torch.gather(b, 1, perm)
        out_w = torch.gather(out_w, 2, perm[:, None, :].expand_as(out_w))
        # sigma fix where matched correlation is negative: flip (w, b, outgoing col)
        w = sign[:, :, None] * w
        b = sign * b
        out_w = out_w * sign[:, None, :]
        # sigma may push b = -pi/2 to +pi/2; restore the fundamental domain
        w, b, out_w = _phase_reduce(w, b, out_w)

        matched_abs = torch.stack(
            [corr[k].abs()[perm[k], torch.arange(corr.shape[2])] for k in range(b_sz)]
        )
        corr_gaps.append(matched_abs.mean(dim=1))
        min_corrs.append(matched_abs.min(dim=1).values)

        result = replace_layer(result, layer, w, b, out_w)
        h = torch.sin(torch.einsum("bji,bpi->bpj", w, h) + b[:, None, :])

    if check_preservation:
        gap = max_functional_gap(params, result, probes)
        if gap > TOL_FUNC:
            raise RuntimeError(f"c_align broke the function: gap {gap:.2e} > {TOL_FUNC}")
    return result, CAlignDiagnostics(tuple(corr_gaps), tuple(min_corrs))
