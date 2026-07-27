# S1 Addendum 01 — rung definitions made operational

**Date:** 2026-07-27 · **Status:** written *before any ladder cell has been computed* (the
implementation was smoke-tested on one 256-INR shard for shape and runtime only; no accuracy,
no decoder run). Amends `docs/prereg/S1.md` (hash 8c029cf43f01a94c) — that file stays as frozen;
this addendum is the operational reading, and is itself hashed into the ledger.

Reason for an addendum rather than an edit: the frozen prereg fixed *what* each rung means but
left four choices implicit that only surfaced at implementation. Silently resolving them inside
code would make the registration unfalsifiable, so they are resolved here, in writing, first.

## A1. W5's alignment template

`c_align` needs a template. **Chosen: the corpus's shared initialization θ₀ (init_seed 0),
as a batch-1 network**, with probes = a deterministic 256-point subsample of the fit grid.

Rationale: θ₀ is data-independent (no label or image information can leak through the template)
and it is exactly the frame W1 already lives in, so W5 asks "how close to W1's frame can alignment
bring P-random?" — the like-for-like question the ladder is built around. Alternatives rejected:
a data-derived template (e.g. the first training INR) leaks corpus statistics into the feature map
and makes the rung depend on an arbitrary image; a per-class template would leak labels outright.

## A2. W8's composition

The frozen text says "best composition of {W4 or W5} ⊕ W6-style augmentation" without an order.
Order matters, and one of the two orders is degenerate: **augment-then-canonicalize is identically
W4**, because `c_sort` is exact with respect to the implemented group (Ch3.1), so the augmentation
is undone before the decoder ever sees it — the rung would measure nothing.

**Chosen: canonicalize first, then apply the bounded augmentation in the canonical frame** at
train time (fresh element per minibatch, |j| ≤ 1, as in W6 and anchor A2). This asks the
non-degenerate question: once the symmetry has been removed exactly, does group-shaped jitter
still help as a regularizer? W8 stays in the augmentation-bearing class (15 seeds).

## A3. W9's frame

**Chosen: R group elements drawn once per seed and shared by every INR in the corpus**; the rung's
feature is the mean of the flattened weights over those R transformed copies. Sharing the elements
is what makes this a frame rather than per-sample noise — with per-INR draws the rung would just be
a noisier W3. The decoder runs at **R = 64**; R ∈ {4, 16} are reported as sensitivity cells, not as
separate registered hypotheses.

Note for the write-up: parity-odd coordinates cancel toward zero under this average, so W9 is
expected to behave as a *lossy* invariantization. That is the point of including it — it is the
cheap baseline against which W10's exact invariants are read.

## A4. W10 is now an exact L=2 invariant encoding, not an L=1-scoped one

The frozen prereg described W10 as "phase-invariant encoding (L=1-scoped: layer-1 + output
invariant features, PO-4 Φ)" — a fallback, because `canon/invariants.py` refuses L ≥ 2 (a hidden
neuron's outgoing u is acted on by the next layer's group, so PO-4's (0,1) and (1,1) classes are
not invariant for our L=2 corpora; OPEN_PROBLEMS #4).

That gap has since been closed: `canon/deep_invariants.py` couples the two layers through the
layer-2 Gram G = W₂ᵀW₂, which is invariant under the entire layer-2 group, and pairs it with
sign-cancelling layer-1 factors (`sin b_i` carries exactly ε_i = (−1)^{d_i+j_i}; `cos b_i w_i`
carries ε_i after contraction). The resulting matrices transform as M ↦ P M Pᵀ, so their sorted
eigenvalue spectra are invariant under the full group; per-neuron even scalars are emitted as
order statistics under a shared invariant sort key. D = 320 for the frozen w32 L2 corpora.
Invariance is verified numerically (T10): residual move under random group elements with windings
up to 3 and non-trivial permutations is ≈ 3 × 10⁻⁷ relative — fp32 round-off.

**W10 therefore means `encode_deep`**, which strictly dominates the scoped fallback (it is exactly
invariant where the fallback was only partially so). H-S1-6 is re-registered verbatim — same point,
same ±3 pt interval — against this definition; the original row is retained in the ledger for
calibration accounting, exactly as the P-A/B/C → v2 amendments were. No accuracy number for either
version exists at the time of writing, so the amendment cannot be selection-driven.

## A5. W7's control gets its own cell

`W7-1/8` (the K corpus subsampled to one view per image, matching W3's row count) is run and
reported as a first-class cell rather than a footnote, so the 8×-rows confound named in the frozen
prereg §1 has a number attached to it.
