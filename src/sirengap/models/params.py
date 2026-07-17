"""Batched sine-INR parameter container (canonical form, omega absorbed).

Shapes (B = batch of INRs):
  hidden[l] = (W_l, b_l) with W_l: [B, n_l, n_{l-1}], b_l: [B, n_l]
  w_out: [B, c, n_L], b_out: [B, c]

Immutable: every transformation returns a new SirenParams.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SirenParams:
    hidden: tuple[tuple[Tensor, Tensor], ...]
    w_out: Tensor
    b_out: Tensor

    def __post_init__(self) -> None:
        if len(self.hidden) == 0:
            raise ValueError("SirenParams needs at least one hidden (sine) layer")
        b = self.hidden[0][0].shape[0]
        for i, (w, bias) in enumerate(self.hidden):
            if w.ndim != 3 or bias.ndim != 2:
                raise ValueError(f"layer {i}: expected W [B,n,m], b [B,n]")
            if w.shape[0] != b or bias.shape[0] != b or w.shape[1] != bias.shape[1]:
                raise ValueError(f"layer {i}: inconsistent shapes {w.shape} / {bias.shape}")
        if self.w_out.shape[0] != b or self.b_out.shape[0] != b:
            raise ValueError("output layer batch mismatch")
        if self.w_out.shape[2] != self.hidden[-1][0].shape[1]:
            raise ValueError("output layer fan-in mismatch with last hidden layer")

    @property
    def batch(self) -> int:
        return self.hidden[0][0].shape[0]

    @property
    def n_layers(self) -> int:
        return len(self.hidden)

    def widths(self) -> tuple[int, ...]:
        return tuple(w.shape[1] for w, _ in self.hidden)

    def to(self, device: torch.device | str) -> "SirenParams":
        return SirenParams(
            hidden=tuple((w.to(device), b.to(device)) for w, b in self.hidden),
            w_out=self.w_out.to(device),
            b_out=self.b_out.to(device),
        )

    def clone(self) -> "SirenParams":
        return SirenParams(
            hidden=tuple((w.clone(), b.clone()) for w, b in self.hidden),
            w_out=self.w_out.clone(),
            b_out=self.b_out.clone(),
        )

    def flat(self) -> Tensor:
        """Concatenate all parameters per INR: [B, D]."""
        parts = []
        for w, b in self.hidden:
            parts += [w.reshape(self.batch, -1), b.reshape(self.batch, -1)]
        parts += [self.w_out.reshape(self.batch, -1), self.b_out.reshape(self.batch, -1)]
        return torch.cat(parts, dim=1)


def outgoing(params: SirenParams, layer: int) -> Tensor:
    """Outgoing weight matrix of hidden layer `layer`: next hidden W or w_out.

    Returns tensor [B, n_next, n_layer]; column i couples neuron i of `layer`.
    """
    if layer < params.n_layers - 1:
        return params.hidden[layer + 1][0]
    return params.w_out


def replace_layer(
    params: SirenParams,
    layer: int,
    w: Tensor,
    b: Tensor,
    out_w: Tensor,
) -> SirenParams:
    """Return new params with hidden layer `layer` = (w, b) and its outgoing matrix = out_w."""
    hidden = list(params.hidden)
    hidden[layer] = (w, b)
    if layer < params.n_layers - 1:
        _, nb = hidden[layer + 1]
        hidden[layer + 1] = (out_w, nb)
        return SirenParams(hidden=tuple(hidden), w_out=params.w_out, b_out=params.b_out)
    return SirenParams(hidden=tuple(hidden), w_out=out_w, b_out=params.b_out)
