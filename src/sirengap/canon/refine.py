"""Optimal alignment of one network to another *within* its symmetry orbit (S4e instrument).

`c_align` matches neurons by activation correlation against a template. That is the right
canonicalizer to *ship* (it needs no knowledge of a counterpart's parameters), but it is a
heuristic choice of orbit representative, so a large post-alignment parameter residual does
not by itself mean "no group element brings these two together".

This module answers the sharper question the deep-identifiability hunt needs:

    given theta and a target theta*, what is  min_{g in G} || g . theta - theta* || ?

The minimisation is *exact per layer*, given the other layers held fixed:

1. **Per-neuron D-infinity is closed form.** Layer l's element acts on neuron i only through
   (w_i, b_i, u_i), so for a candidate template slot t the cost decomposes:

       ||(-1)^d w_i - w*_t||^2 + ((-1)^d b_i + pi j - b*_t)^2 + ||(-1)^(d+j) u_i - u*_t||^2.

   The u-term depends on j only through its parity, and for a fixed (d, parity) the optimal
   j is the integer of that parity nearest (b*_t - (-1)^d b_i) / pi. Four (d, parity) cases
   therefore give the exact minimum over the whole infinite group D_infinity.

2. **The permutation is then an assignment problem.** With cost[i][t] = that per-neuron
   minimum, the Hungarian algorithm gives the globally optimal permutation for the layer.

Across layers the choices interact (layer l permutes the *columns* of W_{l+1}, whose rows
layer l+1 permutes), so the layers are swept by coordinate descent. Each sweep is
non-increasing in the objective and the procedure is run to a fixed point.

Every move is an element of G, so the function is preserved exactly; `refine_alignment`
asserts that, and the residual it reports is an upper bound on the true orbit distance
that is tight per layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from sirengap.canon.assign import hungarian
from sirengap.constants import TOL_FUNC
from sirengap.models.forward import max_functional_gap
from sirengap.models.params import SirenParams, outgoing, replace_layer


@dataclass(frozen=True)
class RefineDiagnostics:
    """Per-INR trace of the coordinate-descent sweeps."""

    sweeps: int
    distance_start: Tensor  # [B] ||theta - theta*|| before refinement
    distance_final: Tensor  # [B] after
    per_sweep: tuple[Tensor, ...]  # distance after each sweep, [B] each
    converged: Tensor  # [B] bool: last sweep improved by < tol


def param_distance(a: SirenParams, b: SirenParams) -> Tensor:
    """Euclidean distance between flattened parameter vectors, per INR: [B]."""
    return (a.flat() - b.flat()).norm(dim=1)


def relative_param_distance(a: SirenParams, b: SirenParams) -> Tensor:
    """||a - b|| / ||b||, per INR: [B]. Scale-free version of `param_distance`."""
    return param_distance(a, b) / (b.flat().norm(dim=1) + 1e-12)


def _best_dinf_cost(
    w: Tensor, b: Tensor, u: Tensor, w_t: Tensor, b_t: Tensor, u_t: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Exact min over (d, j) in D_infinity of the per-neuron cost, for every (i, t) pair.

    Shapes: w [B, n, m], b [B, n], u [B, c, n] (outgoing columns);
            w_t [B, n, m], b_t [B, n], u_t [B, c, n] are the target's.
    Returns (cost [B, n, n], d [B, n, n], j [B, n, n]) indexed [batch, model i, template t].
    """
    uu = u.transpose(1, 2)  # [B, n, c]
    uu_t = u_t.transpose(1, 2)  # [B, n, c]

    # squared-norm expansions, broadcast over (i, t)
    w2 = (w * w).sum(-1)[:, :, None]  # [B, n, 1]
    wt2 = (w_t * w_t).sum(-1)[:, None, :]  # [B, 1, n]
    w_dot = torch.einsum("bim,btm->bit", w, w_t)  # [B, n, n]
    u2 = (uu * uu).sum(-1)[:, :, None]
    ut2 = (uu_t * uu_t).sum(-1)[:, None, :]
    u_dot = torch.einsum("bic,btc->bit", uu, uu_t)

    best_cost = best_d = best_j = None
    for d in (0, 1):
        sw = 1.0 - 2.0 * d  # (-1)^d
        cost_w = w2 + wt2 - 2.0 * sw * w_dot  # ||sw*w_i - w*_t||^2
        delta = b_t[:, None, :] - sw * b[:, :, None]  # [B, n, n]
        for parity in (0, 1):
            # nearest integer to delta/pi with the required parity
            q = (delta / math.pi - parity) / 2.0
            j = 2.0 * torch.round(q) + parity
            cost_b = (math.pi * j - delta) ** 2
            su = sw * (1.0 - 2.0 * parity)  # (-1)^(d+j), j parity fixed
            cost_u = u2 + ut2 - 2.0 * su * u_dot
            cost = cost_w + cost_b + cost_u
            if best_cost is None:
                best_cost = cost
                best_d = torch.full_like(cost, float(d))
                best_j = j
            else:
                take = cost < best_cost
                best_cost = torch.where(take, cost, best_cost)
                best_d = torch.where(take, torch.full_like(cost, float(d)), best_d)
                best_j = torch.where(take, j, best_j)
    return best_cost, best_d, best_j


def _apply_layer_choice(
    w: Tensor, b: Tensor, out_w: Tensor, d: Tensor, j: Tensor, perm: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Apply per-neuron (d, j) then the permutation `perm` (slot t takes model neuron perm[t])."""
    sign_w = 1.0 - 2.0 * d
    sign_u = sign_w * torch.where(j % 2 == 0, 1.0, -1.0)
    w = sign_w[:, :, None] * w
    b = sign_w * b + math.pi * j
    out_w = out_w * sign_u[:, None, :]
    w = torch.gather(w, 1, perm[:, :, None].expand_as(w))
    b = torch.gather(b, 1, perm)
    out_w = torch.gather(out_w, 2, perm[:, None, :].expand_as(out_w))
    return w, b, out_w


def refine_alignment(
    params: SirenParams,
    target: SirenParams,
    max_sweeps: int = 24,
    tol: float = 1e-10,
    check_preservation: bool = True,
    probes: Tensor | None = None,
) -> tuple[SirenParams, RefineDiagnostics]:
    """Minimise ||g . params - target|| over g in G by exact per-layer coordinate descent.

    `params` and `target` must have identical shapes. Returns the aligned parameters and a
    trace. When `check_preservation` is set, the function is asserted unchanged on `probes`
    (a default probe grid is drawn if none is given) — every move is in G, so any failure is
    an implementation bug, not a modelling choice.
    """
    if params.widths() != target.widths():
        raise ValueError(f"width mismatch: {params.widths()} vs {target.widths()}")
    if params.batch != target.batch:
        raise ValueError(f"batch mismatch: {params.batch} vs {target.batch}")

    result = params
    start = param_distance(result, target)
    per_sweep: list[Tensor] = []
    prev = start
    sweeps_run = 0
    converged = torch.zeros(params.batch, dtype=torch.bool)

    for sweep in range(max_sweeps):
        sweeps_run = sweep + 1
        for layer in range(params.n_layers):
            w, b = result.hidden[layer]
            u = outgoing(result, layer)
            w_t, b_t = target.hidden[layer]
            u_t = outgoing(target, layer)

            cost, d_all, j_all = _best_dinf_cost(w, b, u, w_t, b_t, u_t)
            # `hungarian` maximises a score, so feed it the negated cost
            perm = torch.stack([hungarian(-cost[k]) for k in range(cost.shape[0])]).to(w.device)

            # gather the (d, j) chosen for each template slot's assigned model neuron
            slots = torch.arange(perm.shape[1], device=perm.device)[None, :].expand_as(perm)
            d_sel = d_all[torch.arange(perm.shape[0])[:, None], perm, slots]
            j_sel = j_all[torch.arange(perm.shape[0])[:, None], perm, slots]
            # (d, j) are indexed by model neuron; scatter back before permuting
            d_row = torch.zeros_like(d_sel).scatter_(1, perm, d_sel)
            j_row = torch.zeros_like(j_sel).scatter_(1, perm, j_sel)

            w, b, u = _apply_layer_choice(w, b, u, d_row, j_row, perm)
            result = replace_layer(result, layer, w, b, u)

        dist = param_distance(result, target)
        per_sweep.append(dist)
        improved = (prev - dist).abs()
        converged = improved < tol
        prev = dist
        if bool(converged.all()):
            break

    if check_preservation:
        if probes is None:
            g = torch.linspace(-1.0, 1.0, 12, dtype=params.hidden[0][0].dtype)
            m = params.hidden[0][0].shape[2]
            probes = torch.stack(torch.meshgrid(*([g] * m), indexing="ij"), dim=-1).reshape(-1, m)
        gap = max_functional_gap(params, result, probes.to(params.hidden[0][0].device))
        if gap > TOL_FUNC:
            raise RuntimeError(f"refine_alignment broke the function: gap {gap:.2e} > {TOL_FUNC}")

    return result, RefineDiagnostics(
        sweeps=sweeps_run,
        distance_start=start,
        distance_final=prev,
        per_sweep=tuple(per_sweep),
        converged=converged,
    )


def orbit_distance(
    params: SirenParams, target: SirenParams, **kwargs
) -> tuple[Tensor, RefineDiagnostics]:
    """Relative post-alignment residual ||g.theta - theta*|| / ||theta*||, per INR: [B]."""
    aligned, diag = refine_alignment(params, target, **kwargs)
    return relative_param_distance(aligned, target), diag
