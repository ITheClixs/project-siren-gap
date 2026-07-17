# G3 Design Memo — INR-Bench sine generation (protocol C.1 template)

**Date:** 2026-07-17 · **Phase:** G3 (Ch4 dataset, sine track, quality gates, anchors)

## Question in my own words
Produce the sine INR corpora (MNIST first, then FMNIST/CIFAR) under the four protocols with
fit-quality certified by a *task-referenced* gate, metadata schema-valid, resumable overnight,
inside the compute budget — such that every downstream rung (S1) is a pure decoder swap.

## What the three closest papers would predict / why they might be wrong
- **Papa et al. (2312.10531):** shared init will dominate downstream; steps matter (overtraining
  RQ2). Might be wrong in magnitude for our matched-MLP reader (they used graph nets).
- **Shamsian et al. (2402.04081):** random-init corpora will overfit weight-space learners without
  augmentation. Their SIREN-bias failure predicts our bias distributions matter — we log them.
- **MWT line (2503.18123):** absolute accuracies for independently-fitted zoos will look low vs
  meta-learned regimes. Expected; scope note already in papers plan.

## Designs considered
**A. Protocol semantics (chosen):**
- P-shared-det: one global θ₀ (init_seed=0), full-batch coords, fixed steps ⇒ F deterministic per
  image (T9-verified at production config).
- P-shared-stoch: same θ₀, per-image fit_seed driving 256-coordinate minibatches ⇒ isolates pure
  optimization noise against P-shared-det (ladder W1−W2).
- P-random: per-image init_seed, **full-batch** ⇒ isolates pure init/symmetry+basin effect
  (W3 vs W1) without conflating minibatch noise. (Alternative A′: minibatched P-random — rejected:
  confounds two nuisance sources in one rung; noise is already measured by W2.)
- P-random-K: K=8 independent init_seeds per train image (train split only, scoping law L3).
**B. Steps/arch tuning (chosen: pilot-then-freeze):** start L=2 w=32 steps 1k (protocol seed
values); pilot sweep steps ∈ {300, 500, 1000} on 2k images; freeze the smallest config passing the
quality gate; never touch after freeze. (Alternative B′: PSNR-threshold-only gate — rejected:
protocol demands task-referenced gate; PSNR thresholds are proxy and image-difficulty-confounded.)
**C. Resume/queue (chosen: shard-level resume):** generation writes one safetensors shard +
metadata parquet per batch (B=256, ~17 s/batch ⇒ checkpoint « 10 min); restart scans existing
shards and continues. `caffeinate -i` wraps the run. (Alternative C′: full YAML job-queue daemon —
deferred until multi-config comparative runs (G6); shard-resume meets the protocol's
checkpoint/resume/crash-safety requirements for single-corpus jobs; logged as a scoped deferral,
not a waiver, since §0.4 mandates the capability, which this provides.)
**D. Data pipeline (chosen: stdlib IDX/pickle loaders):** MNIST/FMNIST from official S3 mirrors
(IDX format), CIFAR-10 python pickles; no torchvision dependency (keeps dep budget at 10/12).
Raw image cache in `data/` (gitignored).

## Identity & splits
image_id: dataset index for train files (0..59999), 100000+index for test files (global
uniqueness, T8). Splits: train = 0..54999, val = 55000..59999 (INR-side validation for decoder
early stopping), test = official 10k. init_seed ranges: train/val draw from [1e6, 2e6), test from
[3e6, 4e6) — disjoint by construction (T8-checked). Targets in [−1,1]; PSNR reported for [0,1]
range equivalently.

## Quality gate (task-referenced, per protocol Ch4)
Small CNN (2×conv+pool, fc) trained on real pixel train images; gate: accuracy on *rendered* test
INRs ≥ accuracy on real test pixels − 1.0 pt. Applied per protocol on the 10k test renders (pilot:
2k subset for tuning; final gate on full test split after generation).

## PREREG (numbers registered before pilot renders are evaluated)
- QG-1: frozen config will be w32 L=2 steps ≤ 1000 with gate pass on MNIST (point: steps=500,
  80% interval {300..1000}).
- QG-2: mean test-render PSNR of frozen config ∈ [28, 40] dB ([0,1] range) point 33.
- QG-3: A1 anchor (frozen configs, 5 seeds): acc(W1) − acc(W3) point +30 pts, 80% CI [12, 45].
(rows added to PREDICTION_LEDGER.csv with this memo's hash)

## Pre-mortem (3 ways this produces garbage + detection)
1. **Convention drift fitter↔symmetry** (ω absorption, coordinate order) silently corrupts every
   downstream number → detection: T4 convention-lock test + a render-vs-original PNG strip
   committed with the pilot; already-green tests rerun at production config.
2. **Quality gate passes while INRs are class-informatively *broken*** (e.g., renders fine but
   PSNR correlates with class → leakage) → detection: mandatory corr(PSNR, class),
   corr(final_loss, class) audit before any S1 run; PSNR-matched control subsets if |corr|
  material (protocol Ch4).
3. **Thermal throttling corrupts throughput planning / overnight jobs die silently** → detection:
   per-shard wall-clock logged in metadata; resume-scan on restart; burn-down table updated with
   measured-vs-projected after the first full corpus.
