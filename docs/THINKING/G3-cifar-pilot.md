# G3 CIFAR-10 Pilot Memo — config sweep, prereg, and the full-vs-fallback decision rule

**Date:** 2026-07-27 · **Phase:** G3 (final corpus; extends docs/THINKING/G3-design.md)
**Status at write time:** no CIFAR fit has been run or inspected. Predictions below are registered
in `docs/PREDICTION_LEDGER.csv` against this file's hash before the sweep is launched.

## Question in my own words

MNIST and FashionMNIST corpora are frozen at sine, L=2, w=32, steps=300 and pass the
task-referenced gate on all four protocols. CIFAR-10 is the third and hardest dataset: 32×32×3
(3072 target values vs 784) with natural-image high-frequency content. Two things must be decided
*before* burning tens of hours: (i) the smallest INR config whose renders pass the task-referenced
gate, and (ii) whether the program can afford the full 50k/10k corpus or must take the pre-planned
20k/4k fallback (docs/COMPUTE_LEDGER.md).

## What the closest evidence predicts / why it might mislead

- **Our own MNIST pilot** froze at steps=300 — the *bottom* of the registered {300..1000} interval,
  and renders **beat** real pixels on the gate CNN (negative gap). Mechanism: mild INR low-pass
  smoothing removes pixel noise the CNN was never robust to. On CIFAR this mechanism should be
  *weaker or reversed*: texture is class-informative for natural images, so low-pass damage should
  cost accuracy rather than buy it. Extrapolating "steps are cheaper than you think" from MNIST is
  exactly the QG-3 error mode (§ LAB_NOTEBOOK 2026-07-17/18 miscalibration memo) — do not repeat it.
- **SIREN literature** fits single natural images at ~40 dB with w256 L5 and 10k+ steps. Our regime
  is deliberately tiny (a *zoo*, not a showcase fit), so absolute PSNR will be far lower; what the
  gate demands is only that class-relevant structure survives.
- **Papa et al. / MWT line:** zoo-scale CIFAR INRs are known-weak downstream. Expected and already
  scoped; the gate is task-referenced against a *pixel* CNN, not against literature accuracies.

## Sweep design (chosen)

Pilot corpus: `P-shared-det`, `--split val --limit 2000` (CIFAR val = train-file ids 45000–49999),
tag `pilot<steps>w<width>L<layers>`, gate evaluated on that val subset. Configs, in cost order:

| id | width | layers | steps | D (features) | role |
|---|---|---|---|---|---|
| A | 32 | 2 | 1000 | 1249 | MNIST-shaped control; expected to fail (capacity) |
| B | 64 | 3 | 500 | 8707 | cheapest plausible |
| C | 64 | 3 | 1000 | 8707 | protocol-seed midpoint |
| D | 64 | 3 | 2000 | 8707 | protocol seed value (COMPUTE_LEDGER profiling config) |
| E | 128 | 3 | 2000 | 34179 | conditional: only if D fails |

**Freeze rule (pre-committed):** freeze the *cheapest* config (ordered by measured wall-clock per
fit) whose gate passes (`acc_real − acc_render ≤ 1.0 pt`); ties broken toward smaller D. The frozen
config is then re-verified per protocol on the full corpus test split, exactly as MNIST/FMNIST were.
If no config passes, the corpus is not generated and a waiver memo replaces this plan (CIFAR
demoted, not silently weakened).

**Gate strength:** the MNIST-family gate CNN was trained 2 epochs (real-pixel ≈ 99 / 86 %). On
CIFAR-10 a 2-epoch SmallCNN is too weak to be a discriminating reference, and a weak reference makes
the gate *easier* to pass — the wrong direction for a quality gate. CIFAR uses `--gate-epochs 10`
(cache key includes the epoch count; MNIST/FMNIST cached models are untouched, so no earlier result
moves). This is a strengthening of the gate, logged here rather than in a waiver.

## Alternatives considered

- **A′ — tune ω₀ per dataset.** Rejected: ω₀ is a convention lock (absorbed into stored weights at
  creation time); changing it per dataset breaks cross-dataset comparability of the symmetry
  analysis, which is the point of the program.
- **B′ — positional-encoding front end to buy fidelity cheaply.** Rejected: relu+PE is a *treatment*
  in S3 (comparative activations); using it here would contaminate the sine track.
- **C′ — accept a PSNR threshold instead of the task-referenced gate.** Rejected for the same reason
  as at G3-design: PSNR is image-difficulty-confounded, and on CIFAR that confound is severe
  (sky-heavy images fit trivially, textured ones do not).
- **D′ — grayscale CIFAR to reuse the 1-channel path.** Rejected: destroys color, which is
  class-informative, and would make the CIFAR arm answer a different question than MNIST/FMNIST.

## Decision rule R-CIFAR (pre-committed, binding)

Let `r` = fits/s measured on the pilot at the frozen config, derated by the measured sustained
thermal-throttle factor (×0.87, R7 from the MNIST chain). Corpus sizes:

- full path: 3 protocols × 50 000 + K=8 × 45 000 train = **510 000 fits**, `Ĥ_full = 510000/r/3600`
- fallback (20k train / 2k val / 2k test): 3 × 24 000 + 8 × 20 000 = **232 000 fits**,
  `Ĥ_fb = 232000/r/3600`

Then: **choose full iff `Ĥ_full ≤ 30 h`; else choose fallback iff `Ĥ_fb ≤ 20 h`; else escalate to a
waiver memo** (CIFAR demoted to a scoping-law-L3 role: 3 × 12k, K arm dropped). Thresholds are set
so that CIFAR generation plus the S3 comparative-activation CIFAR arm stays inside the protocol's
25–60 h dataset-generation line with ≥ 200 h left for G4–G8.

**Analysis constraint carried by the fallback:** a 20k train split changes decoder sample size, so
*absolute* cross-dataset accuracy comparisons (CIFAR vs MNIST) become confounded with train-set
size. Only within-dataset rung *gaps* (the ladder's actual quantity of interest) stay valid; any
cross-dataset statement must either use a 20k-subsampled MNIST control or be dropped. Registered
here so the constraint cannot be forgotten at write-up time.

**Compute-contention rule:** CIFAR generation and any decoder run compete for the same MPS device,
which would corrupt both throughput measurement and wall-clock ledger entries. They are serialized:
generation runs detached (`nohup caffeinate -i`), decoder work waits.

## PREREG (registered before any CIFAR fit is inspected)

- **QG-4** frozen CIFAR steps: point **1000**, 80% interval {500 … 3000}.
- **QG-4b** P(config A, w32 L2, passes the gate) = **0.15**.
- **QG-5** median val-render PSNR at the frozen config: point **27 dB**, 80% interval [22, 32].
- **QG-6** P(full 50k path chosen by R-CIFAR) = **0.35**.
- **QG-7** CIFAR gate-CNN real-pixel accuracy at 10 epochs: point **66 %**, 80% interval [60, 72].
- **QG-8** sign of the gate gap at the frozen config (`acc_real − acc_render`): predicted
  **positive** (renders *lose* to pixels, unlike MNIST/FMNIST where renders won), P = 0.80.

QG-8 is the one that tests the mechanism story rather than the engineering: if CIFAR renders also
beat pixels, the "INR low-pass removes noise the CNN cannot handle" explanation generalizes further
than my texture argument allows, and the Ch4 discussion needs rewriting.

## Pre-mortem (3 ways this produces garbage + detection)

1. **Channel/layout drift.** CIFAR targets are NHWC-flattened to [N, 1024, 3] while coords are
   row-major (y, x)-flipped; a transpose bug would produce plausible-looking but scrambled renders
   that still "fit". *Detection:* the gate itself (a scrambled render cannot classify), plus an
   explicit render-vs-original PNG strip committed with the pilot.
2. **Gate passes because the reference CNN is weak, not because renders are faithful.** *Detection:*
   QG-7 registers the expected real-pixel accuracy; if the CNN lands below the interval, the gate
   result is quarantined until the reference is retrained, regardless of the reported gap.
3. **Throughput measured on a cold machine, corpus then runs 30 % slower and blows the budget.**
   *Detection:* R-CIFAR uses the ×0.87 derate measured on the MNIST chain, and the corpus job logs
   per-shard wall-clock so the burn-down is reconciled against `Ĥ` after the first 10 % of shards.
