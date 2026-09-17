# S19 addendum 01 — W11a with permutation-safe standardization — FROZEN

**Frozen:** 2026-09-17, before the corrected cell is computed. Ledger row `H-S19-16`.

## Why

A second external review found that the W11a training pipeline was not permutation invariant.
`feature_stats` pools every axis but the last, and the W11a edge tensor `W2` had no channel axis, so
its mean and standard deviation were indexed by layer-one neuron slot. Fixed statistics indexed by
slot do not commute with a relabelling of the neurons, so the standardized reader was not the
permutation-invariant control the paper describes. The reader itself was invariant; the input
standardization was not.

The fix (`src/sirengap/models/readers.py`) gives the edge tensor a trailing channel axis, so its
statistics are pooled over both neuron axes. `tests/test_t13_readers.py::test_t13a2` now checks
invariance of the standardized pipeline with statistics fitted on slot-heterogeneous networks.

## Arm

`scripts/33_w11_equivariant.py --variants a --out-name W11a_fixednorm`, MNIST `P-random`, width 424,
five seeds, the frozen schedule of S1-w11, everything else unchanged. Anchors W1 and W3 as in
`results/ladder/mnist/W11.json`. Runs on the Apple GPU after the S19 chain and the A9 published
readers have finished.

## Prediction

The corpus is exchangeable over neuron slots, so the per-slot statistics fitted on 55k networks are
close to the pooled ones, and the reported cell should move by about seed noise.

| id | quantity | point | 80% interval |
|---|---|---|---|
| H-S19-16 | $s$ of W11a with pooled edge statistics, MNIST `P-random` | 0.265 | [0.23, 0.30] |

## Reading rule

Whatever the outcome, the corrected cell replaces 0.265 wherever W11a is reported in the body, and
the old cell is kept in the appendix as the buggy-normalization run. If the corrected $s$ leaves the
interval upward by more than 0.10, the W11a-to-W12 attribution of the 2x2 square is recomputed with
the corrected cell and the reading that the step from W11a to W12 "says more about W11a than about
that family" is re-examined. The orbit-intervention W11a cell (recovered 0.631) used the same
standardization and is not rerun; it is marked as such.
