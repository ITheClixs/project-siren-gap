# S15 addendum 01 — the factorial on FashionMNIST and CIFAR-10 — FROZEN

**Frozen:** 2026-09-13, after the MNIST arm of S15 was scored and before either new arm is run.
Ledger rows `H-S15-8..13`, `P-S15-D..F`.

## Why

The MNIST factorial overturned the nested reading: reordering neurons carries the largest Shapley
share (44.2), sign flips next (29.9), $\pi$ shifts 4.3, $2\pi$ windings 0.4. One dataset is not a
pattern. This runs the identical design (`scripts/75_s15_orbit_factorial.py`, unchanged) on the
`P-shared-det` corpora of FashionMNIST and CIFAR-10.

## Exposure

The MNIST values above are known and anchor these predictions. The full group's damage on these
corpora is not known; the only related known numbers are their W1 − W3 gaps (70.3 and 31.6).

## Predictions (80% intervals)

| id | quantity | point | 80% interval |
|---|---|---|---|
| H-S15-8 | FashionMNIST, full-cell $\Delta$ | 68 | [60, 72] |
| H-S15-9 | FashionMNIST, $\phi_\pi$ | 38 | [25, 50] |
| H-S15-10 | FashionMNIST, $\phi_\sigma$ | 26 | [15, 38] |
| H-S15-11 | CIFAR-10, full-cell $\Delta$ | 29 | [22, 33] |
| H-S15-12 | CIFAR-10, $\phi_\pi$ | 15 | [8, 24] |
| H-S15-13 | CIFAR-10, $\phi_\sigma$ | 11 | [5, 18] |

| id | probability statement | P |
|---|---|---|
| P-S15-D | $\phi_\pi$ is the largest Shapley value on both datasets | 0.65 |
| P-S15-E | $\phi_\rho + \phi_\tau$ is below 20% of the full-cell $\Delta$ on both datasets | 0.80 |
| P-S15-F | the non-group scramble costs less than 5 points on both datasets | 0.60 |

## What the paper will say

The body reports the Shapley split on all three datasets. The ordering is stated as general only if
it holds on all three; otherwise the paper reports it per dataset.
