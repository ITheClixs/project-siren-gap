# Replication Anchors

Two literature phenomena must replicate qualitatively before the pipeline is trusted (protocol
§0.1.6). Anchor cells fixed *now* (per G0 advisor review, Empiricist 2) so success can't be gamed.

## A1 — Shared-init INRs are far more decodable than random-init (Papa et al. direction)

- **Cell:** MNIST, sine, L=2 w=32 (or the post-quality-gate config, frozen at G3), rungs W1
  (P-shared-det) vs W3 (P-random), matched decoder MLP [D→1024→512→256→10], 5 decoder seeds,
  INR-validation-split early stopping, test split touched once.
- **Success criterion (binding):** acc(W1) − acc(W3) > 10 accuracy points with 95% CI excluding 0
  (paired by seed). **Expectation (non-binding):** gap ≥ 25 points.
- **Delta documentation duty:** Papa et al. used JAX fitters, different widths/steps/datasets
  (Neural Field Arena); numeric differences expected and reported in this file, not hidden.

## A2 — Weight-space augmentation improves random-init INR classification (Shamsian et al. direction)

- **Cell:** same dataset/config, rung W6 (group augmentation during decoder training: fresh
  permutation + σ sign + τ phase ops per step, τ range matched to measured bias statistics) vs W3
  raw, same decoder/seeds/statistics.
- **Success criterion (binding):** acc(W6) − acc(W3) > 1 accuracy point with 95% CI excluding 0.
- **Caveat (resolved at G1 close-read, 2026-07-17):** Shamsian's §4.1 defines both "SIREN negation"
  (σ) and "SIREN bias" (τ/ρ with unbounded k) — and their Table 3 shows unbounded SIREN-bias
  augmentation *severely hurts* (e.g. DWS 4.69 vs ≈18 no-aug). A2 therefore replicates the
  *direction* with our bounded aug family (perm + σ + bias-range-matched τ, k ∈ {−1,0,1}); W6 cites
  them as origin. **Pre-registered risk note:** if bounded-τ augmentation also hurts, that is a
  finding consistent with their Table 3 and evidence for canonicalization-over-augmentation — not a
  pipeline failure; the A2 binding criterion then applies to the perm+σ aug variant only, with the
  τ result reported alongside.

## Status

| anchor | status | result | delta memo |
|---|---|---|---|
| A1 | **PASSED** (2026-07-18) | W1−W3 = **+80.43 pts** [95% CI 80.17, 80.69], paired t p=1.1e−11, Wilcoxon p=.0625 (n=5 floor), d≈383. W1 = 94.36±0.21 (linear probe 88.9, kNN 84.1); W3 = 13.92±0.28 (linear probe 10.1 = chance, kNN 10.5 = chance). | Direction matches Papa et al. Table 1 emphatically; magnitude far larger than their graph-net setting (+40–120% rel.) because our matched-MLP reader gets *no* permutation structure — raw random-init weights are chance-level for linear/kNN readers. Registered QG-3 interval [12,45] **missed** (obs. 80.4) — miscalibration scored in notebook. |
| A2 | **PASSED** (2026-07-18) | W6−W3 = **+4.35 pts** [3.46, 5.25], t p=1.7e−4, d=6.0 (W6 = 18.27±0.55). | Direction matches Shamsian with our *bounded* family (perm+σ+τ, |j|≤1) where their unbounded SIREN-bias aug *hurt* — consistent with the truncated-group-averaging account (PO-12). Augmentation recovers ~5% of the 80-pt gap: large headroom for exact canonicalization at S1. |

Artifacts: results/anchors/anchors_mnist.json (+ run log). Note: probes_W6 in the JSON duplicates
W3's probe numbers (probes ignore train-time augmentation by design; label is misleading — fix at
S1 refactor).
