# S1 Addendum 02 — CIFAR-10 arm of the decomposition ladder — FROZEN

**Study:** S1 (protocol Part III; dissertation Ch5), third dataset arm.
**Frozen:** 2026-07-29, **before any CIFAR-10 ladder cell has been computed.** Amends
`docs/prereg/S1.md` (hash `8c029cf43f01a94c`) and `S1-addendum-01.md`; those files stay as frozen.
This addendum is hashed into `docs/PREDICTION_LEDGER.csv` as rows `H-C1-*`.

**Why an addendum and not a re-use of the S1 numbers.** The FashionMNIST arm was run against the
MNIST-calibrated intervals and the analysis script duly printed H-S1-4a as a "MISS" (+70.31 against
a registered [79, 82]). That was a *category error avoided in the write-up*: the interval was an
MNIST magnitude, FMNIST's own ceiling is eight points lower, and the FMNIST rows were therefore
recorded as a replication of structure and deliberately **not** added to the calibration ledger
(`docs/LAB_NOTEBOOK.md`, 2026-07-28). Doing that twice would be a habit rather than an accident.
CIFAR-10 differs from both prior corpora on three axes at once — natural images instead of
centred strokes/silhouettes, three output channels instead of one, and a *different fidelity
regime* (median render PSNR 40.1 dB with the render penalty at the gate's noise floor) — so this
arm gets its own registered numbers, made **before** the data exist, and its rows **do** enter the
calibration ledger.

**Inputs already fixed and not revisable here:** the CIFAR-10 corpora (`P-shared-det`,
`P-shared-stoch`, `P-random`, `P-random-K`; L = 2, w = 32, steps = 1000, lr 1e−3, frozen by rule
R-CIFAR on 2026-07-28 and generated 2026-07-28/29), their test-split quality gates (all three
pass: render-vs-pixel gaps +0.30 / +0.84 / +0.25 pts, median PSNR 40.1 / 34.9 / 40.3 dB), the rung
definitions of `S1.md` §1 as amended by addendum 01, the decoder apparatus, and the seed policy
(5 deterministic / 15 augmentation-bearing).

**Pre-freeze exposure — declared.** The apparatus was exercised on CIFAR-10 by
`scripts/19_ladder_shapecheck.py`, which builds each rung's feature map and prints only its shape,
row alignment, dtype and finiteness. **No decoder was run and no accuracy exists at freeze time.**
An earlier invocation of `11_ladder.py --dataset cifar10 --rungs P0 W3` was started and
**terminated before it wrote any cell** (`results/ladder/cifar10/` was empty at freeze; the run
died at exit 144 during feature construction). Both facts are stated here so that the freeze can
be audited rather than trusted.

---

## 1. What is being predicted, and why these quantities

The ladder's absolute accuracies are dataset-bound: they inherit the ceiling of the task. The
*recovery fractions*

$$ f(\mathrm{W}k) \;=\; \frac{\mathrm{W}k - \mathrm{W3}}{\mathrm{W1} - \mathrm{W3}} $$

are not — they are ratios in which the task ceiling cancels, and they are the objects the thesis's
central claim is stated in. So the registration puts its weight on the fractions and on the
*orderings*, and registers the absolute levels only loosely, as instrument checks.

## 2. Registered point predictions with 80% intervals

Scored per protocol §0.1.2 (interval contains the observation ⇒ HIT). Ledger rows `H-C1-*`.

| # | quantity | point | 80% interval | rationale |
|---|---|---|---|---|
| H-C1-1 | P0 (real-pixel MLP, test %) | 53.0 | [47, 59] | a 3-hidden-layer MLP on raw CIFAR-10 pixels; the gate CNN's 66.4 is an upper reference, not this decoder |
| H-C1-2 | P1 − P0 (pts) | −0.4 | [−2.0, +0.6] | renders at 40.1 dB are near-exact; MNIST gave −0.39 at 39.2 dB, FMNIST −0.18 |
| H-C1-3 | W1 − P1 (pts) | −12 | [−22, −5] | the raw-weight deficit grew MNIST −3.2 → FMNIST −6.5 as the task hardened; CIFAR is harder again |
| H-C1-4 | W3 (test %) | 13.0 | [10.5, 17.0] | random-init weights sat ~3 pts above chance on both prior corpora |
| H-C1-5 | **W1 − W3 (pts, the perception gap)** | **27** | **[14, 41]** | ≈ W1 40 − W3 13; the gap must shrink in absolute terms because the ceiling did |
| H-C1-6 | W1 − W2 (pts, optimization noise) | −0.7 | [−2.5, +1.0] | null on both prior corpora (−0.68, −0.70); direction registered: W2 ≥ W1 |
| H-C1-7 | **f(W4)**, template-free `c_sort` | **0.17** | **[0.08, 0.30]** | 0.177 (MNIST), 0.170 (FMNIST) — the most stable number in the program |
| H-C1-8 | **f(W5)**, `c_align` to θ₀ | **0.62** | **[0.42, 0.78]** | 0.628 (MNIST), 0.664 (FMNIST); widened downward because CIFAR neurons are fit to harder targets and may match θ₀'s less cleanly |
| H-C1-9 | f(W6), bounded group augmentation | 0.04 | [−0.02, 0.12] | 0.054, 0.032 |
| H-C1-10 | f(W7), K-marginalization | 0.04 | [−0.02, 0.12] | 0.048, 0.034 |
| H-C1-11 | (W7 − W3) − (W6 − W3) (pts) | 0.0 | [−2.0, +2.0] | the H-S1-5 discriminator is dead on both prior corpora; this is its third and last chance |
| H-C1-12 | f(W9), frame averaging R = 64 | 0.00 | [−0.03, +0.05] | 0.003, −0.008 |
| H-C1-13 | **f(W10)**, exact L = 2 invariants | **0.38** | **[0.12, 0.62]** | 0.269 (MNIST), 0.428 (FMNIST) — the least stable fraction; see §3 for the CIFAR-specific mechanism that pushes it up |
| H-C1-14 | W8 (canonicalize-then-augment, test %) | 11.0 | [10.0, 14.0] | at chance on both prior corpora (10.27, 10.20) |
| H-C1-15 | X1 forward (W1-trained reader on W3 features, %) | 11.5 | [10.0, 15.0] | 10.7 on MNIST |
| H-C1-16 | X1 reverse (W3-trained reader on W1 features, %) | 12.5 | [10.0, 17.0] | 13.2 on MNIST |
| H-C1-17 | H-S1-6 bracket: signed distance of W10 outside [W4, W5] (pts) | 0 | [−3, +3] | re-registration of H-S1-6 on this corpus |

**Equivalence margin, set before data.** H-C1-2 is additionally tested by TOST at margin
**1.5 pt**, not the 1.0 pt used on MNIST. Reason, stated now: the S1 margin was chosen against a
98-point ceiling, and this corpus's paired-difference seed SD is unmeasured. Both margins will be
reported; the 1.5-pt one carries the claim. If the observed seed SD comes in at or below the S1
deterministic value (0.210 pts) the 1.0-pt result is the one to quote, and that rule is fixed here.

## 3. Registered probability calls

Scored by Brier score, as with QG-4b/QG-6/QG-8.

- **P-C1-A = 0.65.** The separated ordering `f(W5) > f(W10) > f(W4) > max(f(W6), f(W7), f(W9))`
  holds on CIFAR-10. (It held on both prior corpora. The rungs inside the `max` are all within
  noise of each other and of zero, which is why they are bundled rather than ordered.)
- **P-C1-B = 0.60.** `f(W10)_CIFAR > f(W10)_MNIST = 0.269`. *Mechanism*: W10 reads the layer-2 Gram
  paired with sign-cancelling layer-1 factors; with c = 3 output channels each neuron's outgoing
  vector uᵢ ∈ ℝ³ carries strictly more D∞-visible structure than the c = 1 case, so the exact
  invariants should have more to see. This is a genuine prediction from the encoding's algebra,
  not an extrapolation of the two observed values.
- **P-C1-C = 0.75.** The label-shuffle controls at W1, W3, W5 all land within 2 pts of chance.

## 4. Falsification conditions (pre-committed, binding)

1. **If f(W5) < 0.30**, the "template alignment recovers roughly two thirds of the perception gap"
   claim does **not** generalize to natural images, and it must be restated as a property of
   low-complexity grayscale corpora — not softened, restated, and the paper's abstract rewritten
   accordingly. (Symmetric to S1 §8.3, which bound the opposite direction and duly fired.)
2. **If f(W4) > f(W5)**, the ordering "exact alignment beats template-free sorting" is broken and
   the canonicalizer-quality reading of the 0.18 → 0.63 span is withdrawn.
3. **If (W7 − W6) > +5 pts**, the H-S1-5 discriminator is *not* dead, the two-dataset null was a
   grayscale artifact, and defense row 8's original falsification conjunction is revived.
4. **If W1 − W3 < 5 pts**, there is no perception gap to decompose on this corpus and every
   fraction above is undefined; the arm is reported as a null result about CIFAR-10 INRs and the
   cross-dataset claim is confined to the two grayscale corpora.

## 5. Analysis, exclusions, leakage — inherited

§§5–7 of `S1.md` apply verbatim: per-rung mean ± 95% t-CI over seeds; paired-by-seed t and
Wilcoxon; Holm within {H-C1-2, H-C1-3, H-C1-5, H-C1-6, H-C1-11, H-C1-17}; label-shuffle controls
at W1/W3/W5; no row-wise INR exclusion; the PSNR-vs-correctness leakage trigger at W1/W3/W7.
Measured corr(PSNR, label) on the frozen CIFAR corpora is −0.150 (P-shared-det), −0.140
(P-random), −0.175 (P-shared-stoch) — same sign across all three, unlike MNIST, so the leakage
regression is expected to fire more readily here and the matched-subset re-run is budgeted.

## 6. Compute budget and stopping rule

Projected: ~2.5 h for the 14 rungs at the S1 seed policy (the MNIST ladder took 0.9 h at 55k train
rows and D = 1185; CIFAR has 45k train rows, D = 1251 for the raw rungs and 3072 for P0/P1, and a
heavier corpus to page in). **Stopping rule:** if the chain exceeds 8 h of wallclock, the
augmentation-bearing rungs (W6, W8) drop from 15 seeds to 5 and are reported with their reduced
power stated in the table — a deviation to be logged, not hidden. No other rung may be dropped;
a partial ladder is reported as partial.
