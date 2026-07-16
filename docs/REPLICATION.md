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
- **Caveat:** if the G1 close-read shows Shamsian's augs already include τ-phase ops, W6 cites them
  as origin and A2 becomes a direct replication; otherwise A2 is a "direction" replication with our
  aug family. Either way the delta is documented here.

## Status

| anchor | status | result | delta memo |
|---|---|---|---|
| A1 | pending (G3) | — | — |
| A2 | pending (G3/G4) | — | — |
