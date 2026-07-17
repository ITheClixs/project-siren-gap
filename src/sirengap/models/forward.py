"""Single source of truth for the canonical forward pass (protocol Ch1).

h^0 = x;  h^l = sin(W^l h^{l-1} + b^l);  f(x) = W_out h^L + b_out.
No omega factor here — omega_0 is absorbed into stored weights at save time.
The fitter (sirengap.fitting) must delegate to this module for any rendering
used by tests or downstream studies (T1 depends on this being the only forward).
"""

from __future__ import annotations

import torch
from torch import Tensor

from sirengap.models.params import SirenParams


def forward_canonical(params: SirenParams, x: Tensor) -> Tensor:
    """Evaluate a batch of sine INRs on shared coordinates.

    x: [P, n_0] shared coordinates. Returns [B, P, c].
    """
    if x.ndim != 2 or x.shape[1] != params.hidden[0][0].shape[2]:
        raise ValueError(f"coords shape {tuple(x.shape)} incompatible with first layer")
    w0, b0 = params.hidden[0]
    h = torch.sin(torch.einsum("bji,pi->bpj", w0, x) + b0[:, None, :])
    for w, b in params.hidden[1:]:
        h = torch.sin(torch.einsum("bji,bpi->bpj", w, h) + b[:, None, :])
    return torch.einsum("bci,bpi->bpc", params.w_out, h) + params.b_out[:, None, :]


def max_functional_gap(a: SirenParams, b: SirenParams, x: Tensor) -> float:
    """max_x |f_a(x) - f_b(x)| over the coordinate batch (used by T1/T2)."""
    with torch.no_grad():
        return (forward_canonical(a, x) - forward_canonical(b, x)).abs().max().item()
