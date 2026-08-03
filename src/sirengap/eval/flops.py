"""Analytic FLOP accounting for the three access models (S5, RQ6).

Proposition PO-6 says a *complete* invariant of the weights carries exactly the information of the
realised function. If that is right, weight access cannot beat function access on information, and
the field's justification has to be **computational** — canonicalize once, decode many times. S5
adjudicates that on a FLOPs--accuracy frontier, so the accounting has to be explicit, analytic and
auditable rather than a wall-clock measurement on one laptop.

Conventions, fixed once so every number in the paper means the same thing:

* One multiply--accumulate is **2 FLOPs**. Matrix product $[a,b]\\times[b,c]$ costs $2abc$.
* Elementwise transcendentals (``sin``) are counted at **1 FLOP** each. They are a rounding error
  against the matmuls at these sizes and the choice is stated rather than hidden.
* Costs are **per INR**, at inference, for one forward pass of the whole pipeline: whatever the
  access model must compute to turn one fitted network into one prediction.
* **Amortized** costs (work done once per corpus, not once per INR) are reported separately, since
  that distinction is exactly what PO-6's corollary says the field's case rests on.

Every function here returns a plain dict so the numbers can be summed, printed and diffed.
"""

from __future__ import annotations

from dataclasses import dataclass

MAC = 2  # FLOPs per multiply-accumulate
SIN = 1  # FLOPs per transcendental


@dataclass(frozen=True)
class Arch:
    """The fitted INR under study."""

    in_dim: int = 2
    width: int = 32
    layers: int = 2
    out_dim: int = 1

    @property
    def n_params(self) -> int:
        w = self.width
        n = self.in_dim * w + w  # layer 1
        n += (self.layers - 1) * (w * w + w)  # hidden layers
        return n + w * self.out_dim + self.out_dim  # readout


def siren_forward(arch: Arch, n_points: int) -> int:
    """Evaluating one INR at `n_points` coordinates — the cost every query-based model pays."""
    w = arch.width
    flops = n_points * (MAC * arch.in_dim * w + w * SIN)
    for _ in range(arch.layers - 1):
        flops += n_points * (MAC * w * w + w * SIN)
    return flops + n_points * MAC * w * arch.out_dim


def mlp_forward(dims: list[int]) -> int:
    """A plain MLP forward pass, the ladder's frozen decoder shape."""
    return sum(MAC * a * b for a, b in zip(dims[:-1], dims[1:]))


def matched_decoder(in_dim: int, n_classes: int = 10) -> int:
    return mlp_forward([in_dim, 1024, 512, 256, n_classes])


# ------------------------------------------------------------------ access models


def function_query(arch: Arch, n_probes: int, n_classes: int = 10) -> dict[str, int]:
    """Function access: evaluate the INR at learned probe coordinates, classify the outputs."""
    read = siren_forward(arch, n_probes)
    head = matched_decoder(n_probes * arch.out_dim, n_classes)
    return {"probe_eval": read, "reader": head, "per_inr": read + head, "amortized": 0}


def render_access(arch: Arch, side: int, n_classes: int = 10) -> dict[str, int]:
    """Render access: evaluate on the full fit grid, then classify the image."""
    return function_query(arch, side * side, n_classes)


def weight_raw(arch: Arch, n_classes: int = 10) -> dict[str, int]:
    """Weight access, no preprocessing (rungs W1/W3)."""
    head = matched_decoder(arch.n_params, n_classes)
    return {"preprocess": 0, "reader": head, "per_inr": head, "amortized": 0}


def weight_csort(arch: Arch, n_classes: int = 10) -> dict[str, int]:
    """`c_sort`: phase reduction and a lexicographic sort per layer. No probes, no assignment."""
    w = arch.width
    # phase reduction touches every hidden neuron's (w, b, u); the sort is n log n comparisons
    reduce_cost = arch.layers * w * (arch.in_dim + 1 + arch.out_dim) * 4
    sort_cost = arch.layers * int(w * (w.bit_length())) * 4
    pre = reduce_cost + sort_cost
    head = matched_decoder(arch.n_params, n_classes)
    return {"preprocess": pre, "reader": head, "per_inr": pre + head, "amortized": 0}


def weight_calign(arch: Arch, n_probes: int, n_classes: int = 10) -> dict[str, int]:
    """`c_align`: probe activations, then a Hungarian assignment against a template, per layer.

    The template's own activations are computed **once per corpus**, so they are amortized; the
    per-INR cost is the model's activations plus the assignment. Hungarian is O(n^3).
    """
    w = arch.width
    acts = siren_forward(arch, n_probes)  # model activations on the probe set
    corr = arch.layers * MAC * n_probes * w * w  # correlation matrix per layer
    hungarian = arch.layers * (w**3)
    pre = acts + corr + hungarian
    head = matched_decoder(arch.n_params, n_classes)
    return {
        "preprocess": pre, "reader": head, "per_inr": pre + head,
        "amortized": siren_forward(arch, n_probes),  # the template, once per corpus
    }


def weight_invariants(arch: Arch, encoding_dim: int, n_classes: int = 10) -> dict[str, int]:
    """W10: the exact L=2 invariant encoding, then the matched decoder.

    Dominated by the layer-2 Gram (n^2 inner products of length n) and the eigendecompositions of
    three n x n matrices, taken at the usual ~9n^3 for a symmetric eigensolver.
    """
    w = arch.width
    gram = MAC * w * w * w
    mats = 3 * MAC * w * w * arch.in_dim
    eig = 3 * 9 * w**3
    pre = gram + mats + eig
    head = matched_decoder(encoding_dim, n_classes)
    return {"preprocess": pre, "reader": head, "per_inr": pre + head, "amortized": 0}


def weight_equivariant_reader(
    arch: Arch, width: int, rounds: int = 2, relations: int = 8,
    n_global: int = 320, invariant_features: bool = True, n_classes: int = 10,
) -> dict[str, int]:
    """W11: a graph reader over the weights. Message passing dominates."""
    w, d = arch.width, width
    pre = 0
    if invariant_features:
        # same Gram and matrices as W10, but no eigendecomposition
        pre = MAC * w * w * w + 3 * MAC * w * w * arch.in_dim
    enc = MAC * w * (3 if invariant_features else arch.in_dim + 1) * d
    per_round = (
        MAC * w * w * 3 * 64 + MAC * w * w * 64 * relations  # edge MLP over n^2 pairs
        + MAC * w * d * relations * d                        # value projection
        + MAC * w * w * relations * d                        # message aggregation
        + MAC * w * 2 * d * d                                # node update
    )
    head = mlp_forward([2 * d + (n_global if invariant_features else 0), 256, n_classes])
    total = pre + enc + rounds * per_round + head
    return {"preprocess": pre, "reader": total - pre, "per_inr": total, "amortized": 0}


def weight_phasor_reader(
    arch: Arch, width: int, rounds: int = 2, n_classes: int = 10,
) -> dict[str, int]:
    """W12: the phasor-graded reader on raw parameters.

    It has no edge MLP over the n^2 pairs -- the coupling is the weight matrix itself, so a
    message is one dense contraction against a [p, n] matrix -- but it is nonetheless the most
    expensive reader here, and by a wide margin. The grading costs: each round applies 10
    bilinear mixes and 8 graded linear maps, all d x d over every node, against the graph
    reader's two. At width 186 that is ~163 MFLOP/INR against ~54, and the accounting says so
    rather than flattering the construction.

    Costs, per INR:

      features    the phasor lift, the two Gram diagonals and E^2 -- O(w^2) and O(w * in_dim)
      embed       four graded linear maps per layer
      per round   4 message contractions (2 x [p,n] x [n,d]), 10 bilinear mixes, 8 graded
                  linear maps, on 2w nodes
      head        4d -> 2d -> d -> classes
    """
    w, d, m, c = arch.width, width, arch.in_dim, arch.out_dim
    n_blocks = len(("00", "10", "01", "11"))

    feats = MAC * w * w + MAC * w * m + SIN * 4 * w * 2  # E^2 and its two contractions, phasors
    embed = MAC * 2 * w * (m + 4 + c) * d                # four blocks per layer, small fan-in
    per_round = (
        4 * MAC * w * w * d          # the four message contractions against E and E^2
        + 10 * MAC * 2 * w * d * d   # graded bilinear: 10 unordered character pairs
        + 2 * n_blocks * MAC * 2 * w * d * d  # graded linear, both layers
    )
    head = mlp_forward([4 * d, 2 * d, d, n_classes])
    total = feats + embed + rounds * per_round + head
    return {"preprocess": feats, "reader": total - feats, "per_inr": total, "amortized": 0}


def summarize(name: str, cost: dict[str, int], acc: float | None = None) -> str:
    per = cost["per_inr"]
    amort = cost.get("amortized", 0)
    line = f"{name:34s} {per / 1e6:9.3f} MFLOP/INR"
    if amort:
        line += f"  (+{amort / 1e6:.3f} M amortized)"
    if acc is not None:
        line += f"   acc {acc:6.2f}"
    return line
