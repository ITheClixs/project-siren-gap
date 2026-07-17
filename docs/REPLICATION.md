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
| A1 | pending (G3) | — | — |
| A2 | pending (G3/G4) | — | — |
