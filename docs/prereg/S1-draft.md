# S1 Pre-registration — DRAFT (not yet frozen; freezes before any confirmatory decoder run)

**Status:** DRAFT at G3. Numbers marked ⟨fill⟩ are completed from anchor/pilot seed-σ (power memo)
and registered in PREDICTION_LEDGER.csv at freeze time. Freeze = committed version whose hash is
recorded in the ledger; any post-freeze deviation is logged exploratory.

## Hypotheses (frozen wording)

- H-S1-1 (information preservation): P1 (oracle render → matched MLP) ≈ P0 (pixels → matched MLP);
  TOST margin 1.0 pt.
- H-S1-2 (determinism rung): W1 (P-shared-det raw weights) ≈ P1 within ⟨fill⟩ pts; the W1−P1 gap
  is the decoder-vs-representation residual under zero nuisance.
- H-S1-3 (optimization noise): W1 − W2 gap = ⟨fill⟩ (point + 80% CI).
- H-S1-4 (init/symmetry+basin): W1 − W3 gap large (anchor A1 measured ⟨fill⟩); canonicalization
  rungs W4 (c_sort) and W5 (c_align) recover fraction ⟨fill⟩ of the W3→W1 gap.
- H-S1-5 (discriminator, RQ4 first cut): W7 (K-marginalization) − W3 > W6 (bounded group aug) − W3
  by margin ⟨fill⟩ pts (TOST-backed directional claim).
- H-S1-6 (invariant front-end): W10 ∈ [W4, W5] ± ⟨fill⟩.

## Design (frozen)

Datasets: MNIST first (this prereg), FMNIST/CIFAR follow with identical analysis (family-adjusted).
Frozen INR config: sine, L=2, w=32, steps=300, lr 1e−3 (G3 pilot; quality gate passed for
P-shared-det and P-random; per-protocol gates re-verified on the full corpora before decoding).
Rungs: P0, P1, W1, W2, W3, W4, W5, W6 (k ∈ {−1,0,1} + perm + σ; bounded per close-read of
2402.04081), W7 (K=8 train-time augmented views), W8 (best composition), W9 (frame averaging,
R ∈ {4,16,64}), W10 (phase-invariant encoding, L=1-scoped variant: layer-1+output features), X1
(cross-protocol brittleness W1↔W3).
Decoder: matched MLP [D→1024→512→256→10], GELU, dropout 0.1, AdamW 1e−3 cosine, ≤100 epochs,
early stop on val (55000–59999), patience 10; 5 seeds; test split touched once per rung by the
frozen final config. Linear probe + kNN(10, cosine) at every rung.
Feature standardization: per-dimension z-score fit on the rung's train features (part of the rung).

## Analysis plan (frozen)

Per-rung: mean ± 95% t-CI over 5 seeds. Rung comparisons: paired-by-seed t AND Wilcoxon; Cohen's d;
Holm within the pre-registered family {H-S1-1..6}. Equivalence claims via TOST, margin 1.0 pt
unless stated. Label-shuffle control at W1, W3, W5: retrain with permuted labels; must collapse to
≈10%. Waterfall figure F9 built from these cells only.

## Exclusion rules (frozen)

INRs failing the per-protocol quality gate corpus-level: none excluded row-wise; if a corpus fails
its gate, the corpus is regenerated (config change = new prereg). Decoder runs that fail to reach
> 15% val accuracy (optimization collapse) are rerun with seed+1000 and flagged; > 1 collapse per
rung invalidates the rung's cell pending investigation.

## Leakage handling (pre-committed)

corr(PSNR, label) measured per corpus (G3: ≈ +0.18 P-shared-det, ≈ +0.01 P-random). If any
headline rung's result could be driven by PSNR leakage (checked by regressing per-image decoder
correctness on PSNR within class), re-run that rung on a PSNR-matched subset (protocol Ch4) and
report both numbers.

## Power memo placeholder

From anchor seed-σ ⟨fill after A1/A2⟩: MDE at α=.05, power=.8, n=5 paired = ⟨fill⟩ pts. If MDE >
smallest scientifically relevant effect (1 pt for equivalence-adjacent rungs), raise seeds before
confirmatory run.
