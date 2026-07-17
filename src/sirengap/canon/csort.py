"""c_sort: template-free canonicalizer, exact w.r.t. the *implemented* group (Ch3.1).

Scope qualifier (advisor G1): "exact" means exact orbit collapse for the group of
Theorem PO-1 (D_inf wreath S_n per layer) on generic inputs. Whether that group is
ALL functional symmetries is PO-2: proved for L=1 (ch1-symmetry.tex, Thm 6.4),
conjectural for deeper networks. Behavior at strata/ties: function preservation
always holds (only group elements are applied); orbit collapse is not guaranteed
at ties (tests/test_tie_stress.py documents the policy).

Steps per hidden layer (docs/THINKING/G0-theory-scoping.md §1–2 for the algebra):
  (a) phase reduction: k = floor(b/pi + 1/2); b <- b - k*pi in [-pi/2, pi/2);
      outgoing column *= (-1)^k  (tau/rho bookkeeping);
  (b) sigma fix: flip (w, b, u) where <w_i, v_ref> < 0 for a fixed random unit
      v_ref (per layer, seeded); re-reduce phase for boundary b = -pi/2 -> pi/2;
  (c) permutation fix: lexicographic sort on rounded keys (|w|, b, w.v_ref).

Returns canonicalized params + diagnostics (sigma margins, key tie counts) —
the discontinuity sets predicted by PO-5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from sirengap.models.params import SirenParams, outgoing, replace_layer


@dataclass(frozen=True)
class CSortDiagnostics:
    sigma_margins: tuple[Tensor, ...]  # per layer: |<w_i, v_ref>| [B, n]
    key_min_gaps: tuple[Tensor, ...]  # per layer: min adjacent sorted-key gap [B]


def _phase_reduce(w: Tensor, b: Tensor, out_w: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    k = torch.floor(b / math.pi + 0.5)
    b_new = b - k * math.pi
    sign_u = torch.where(k.long() % 2 == 0, 1.0, -1.0).to(w.dtype)
    return w, b_new, out_w * sign_u[:, None, :]


def _sigma_fix(
    w: Tensor, b: Tensor, out_w: Tensor, v_ref: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    proj = torch.einsum("bni,i->bn", w, v_ref)
    flip = proj < 0
    sign = torch.where(flip, -1.0, 1.0).to(w.dtype)
    w_new = sign[:, :, None] * w
    b_new = sign * b
    out_new = out_w * sign[:, None, :]
    return w_new, b_new, out_new, proj.abs()


def _lex_keys(w: Tensor, b: Tensor, v_ref: Tensor, decimals: int) -> Tensor:
    """Rounded lexicographic key per neuron: (|w|, b, w.v_ref) -> [B, n, 3]."""
    norm = w.norm(dim=2)
    proj = torch.einsum("bni,i->bn", w, v_ref)
    keys = torch.stack([norm, b, proj], dim=2)
    return torch.round(keys * 10**decimals) / 10**decimals


def _lex_argsort(keys: Tensor) -> Tensor:
    """Stable lexicographic argsort over neurons; keys [B, n, K] -> perm [B, n].

    Computed on CPU (stable sort support + determinism), result moved back.
    """
    device = keys.device
    keys = keys.detach().cpu()
    b_sz, n, _ = keys.shape
    order = torch.arange(n).expand(b_sz, n)
    for col in reversed(range(keys.shape[2])):
        vals = torch.gather(keys[:, :, col], 1, order)
        idx = torch.argsort(vals, dim=1, stable=True)
        order = torch.gather(order, 1, idx)
    return order.to(device)


def c_sort(
    params: SirenParams, seed: int = 0, decimals: int = 6
) -> tuple[SirenParams, CSortDiagnostics]:
    result = params
    margins_all, gaps_all = [], []
    for layer in range(params.n_layers):
        w, b = result.hidden[layer]
        out_w = outgoing(result, layer)
        gen = torch.Generator().manual_seed(seed + layer)
        v_ref = torch.randn(w.shape[2], generator=gen).to(w.device, w.dtype)
        v_ref = v_ref / v_ref.norm()

        w, b, out_w = _phase_reduce(w, b, out_w)
        w, b, out_w, margins = _sigma_fix(w, b, out_w, v_ref)
        w, b, out_w = _phase_reduce(w, b, out_w)  # boundary b = -pi/2 -> +pi/2 after sigma

        keys = _lex_keys(w, b, v_ref, decimals)
        perm = _lex_argsort(keys)
        w = torch.gather(w, 1, perm[:, :, None].expand_as(w))
        b = torch.gather(b, 1, perm)
        out_w = torch.gather(out_w, 2, perm[:, None, :].expand_as(out_w))

        sorted_norm = torch.gather(keys[:, :, 0], 1, perm)
        gaps = (
            (sorted_norm[:, 1:] - sorted_norm[:, :-1]).abs().min(dim=1).values
            if keys.shape[1] > 1
            else torch.full((keys.shape[0],), float("inf"))
        )
        margins_all.append(margins)
        gaps_all.append(gaps)
        result = replace_layer(result, layer, w, b, out_w)
    return result, CSortDiagnostics(tuple(margins_all), tuple(gaps_all))
