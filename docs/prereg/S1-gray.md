# S1 Addendum 03 — luminance-CIFAR arm: is it the images or the channels? — FROZEN

**Study:** S1 (protocol Part III; dissertation Ch5), fourth dataset arm.
**Frozen:** 2026-07-30, **before any `cifar10gray` ladder cell has been computed.** Amends
`docs/prereg/S1.md` (`8c029cf43f01a94c`), addendum 01, and `S1-cifar.md` (`f7906fc6904c7c81`);
those files stay as frozen. Ledger rows `H-G1-*`, `P-G-*`.

---

## 1. The confound this exists to break

The CIFAR-10 arm produced the paper's two headline cross-dataset findings: alignment recovery
halves ($f(\mathrm{W5})$: $0.628 / 0.664 \to 0.324$) and the two exact methods **cross over**
($f(\mathrm{W10})$: $0.269 / 0.428 \to 0.534$, overtaking alignment for the first time). But
CIFAR-10 differs from the grayscale corpora on **three** axes at once:

1. natural image statistics instead of centred strokes and silhouettes,
2. output-channel count $c = 3$ instead of $c = 1$,
3. fit budget, 1000 steps instead of 300.

Axis 3 is already dead: measured parameter travel is indistinguishable across all three corpora
($0.186 / 0.191 / 0.187$; `results/fit_travel.json`), so the extra steps bought no extra
displacement. Axes 1 and 2 remain entangled. An encoder-side ablation gave partial evidence —
truncating W10's encoder to one output channel costs only $0.077$ of $0.534$, so $c$ explains about
$29\%$ of W10's rise (`EXPLORATORY_w10_channel_ablation.json`) — but that reads a network *fitted*
with three channels, which is not the same intervention as fitting one.

**Luminance CIFAR-10** (ITU-R BT.601 luma of the identical images, same $32\times32$ geometry,
same frozen architecture w32 L2, same 1000 steps, same lr) changes **axis 2 alone**.

**The paper stakes a conjecture on this.** §9 states that $\calign$ matches neurons by correlating
their layer-1 *activations*, a statistic blind to the outgoing structure — exactly the part that
grows with $c$ — "so as more of a neuron's identity moves into its outgoing vector, activation
matching identifies it less reliably while the invariants identify it better." That conjecture
makes a sharp, falsifiable prediction here, and this arm is registered to test it rather than to
decorate it.

## 2. What is fitted, and one consequence that cannot be controlled away

Two protocols only: `P-shared-det` (carries P0, P1, W1) and `P-random` (carries W3, W4, W5, W9,
W10). `P-shared-stoch` (W2) and `P-random-K` (W7) are **not run**; the ladder is reported as
**partial**, which `S1.md` §6 permits. W6 and W8 are augmentation-bearing rungs on `P-random` and
are also not run, for budget.

**Declared, uncontrollable consequence.** At $c = 1$ the network fits $1185$ parameters to $1024$
target values; at $c = 3$ it fits $1251$ to $3072$. Dropping channels therefore takes the fit from
under- to over-parameterised and median render PSNR from $\approx40$\,dB to $\approx60$\,dB
(measured on a 256-image pilot shard: $59.1$–$60.0$\,dB). This is a *consequence* of the
intervention, not a second manipulation — there is no way to change $c$ at fixed architecture and
hold parameters-per-target fixed — and it is named here so it cannot be quietly discovered later.
It is the arm's principal limitation and §6 fixes how it must be reported.

**Pre-freeze exposure — declared.** The corpora were generated and their quality gates run before
this file was written (same pattern as `S1-cifar.md`). Pilot-shard PSNR values above were seen. **No
decoder has been run on `cifar10gray` and no ladder cell exists at freeze time.**

## 3. Registered point predictions (80% intervals)

| # | quantity | point | 80% interval | rationale |
|---|---|---|---|---|
| H-G1-1 | P0 (real-pixel MLP, test %) | 47.0 | [41, 53] | grayscale discards colour, which a pixel MLP uses; RGB-CIFAR P0 was 55.8 |
| H-G1-2 | P1 − P0 (pts) | +0.2 | [−1.0, +1.5] | renders at ~60 dB are essentially exact; RGB-CIFAR gave +0.42 at 40 dB |
| H-G1-3 | W1 (test %) | 38.0 | [28, 46] | RGB-CIFAR W1 was 44.3 against a 55.8 ceiling |
| H-G1-4 | W3 (test %) | 12.5 | [10.5, 16.0] | random-init weights sat 2–3 pts above chance on all three prior corpora |
| H-G1-5 | W1 − W3 gap (pts) | 25.5 | [14, 36] | ≈ 38 − 12.5 |
| H-G1-6 | **f(W5)**, $\calign$ | **0.45** | **[0.28, 0.66]** | the discriminator: 0.32 if images drive it, ~0.63 if channels do; centred between, wide |
| H-G1-7 | **f(W10)**, exact $L{=}2$ invariants | **0.45** | **[0.26, 0.60]** | the encoder-side ablation put the $c$-attributable part of W10's rise at ~29%, i.e. $0.534 \to 0.457$ |
| H-G1-8 | f(W4), template-free $\csort$ | 0.13 | [0.06, 0.24] | 0.177 / 0.170 / 0.108 on the three prior corpora |
| H-G1-9 | f(W9), frame averaging | 0.00 | [−0.03, +0.05] | null on all three prior corpora |
| H-G1-10 | median render PSNR (dB) | 59.5 | [55, 63] | pilot shards gave 59.1–60.0 |

## 4. The discriminator, stated as probability calls

These are the point of the arm. Scored by Brier.

- **P-G-A = 0.35.** $f(\mathrm{W5}) > 0.50$ — i.e. alignment recovery substantially returns once
  $c = 1$, which is what the paper's §9 conjecture predicts. Registered *below* even odds because
  the encoder-side ablation already suggested channels are a minority of the effect for W10, and
  because I expect image statistics to carry most of it.
- **P-G-B = 0.45.** $f(\mathrm{W10}) < f(\mathrm{W5})$ — the crossover **reverses** on luminance
  CIFAR, restoring the grayscale ordering. This is the single cleanest test: the crossover is
  either a channel-count phenomenon (it reverses) or an image-complexity phenomenon (it persists).
- **P-G-C = 0.80.** The three label-shuffle controls (W1, W3, W5) land within 2 pts of chance.

## 5. What each outcome licenses (pre-committed wording)

- **$f(\mathrm{W5}) > 0.50$ and the crossover reverses.** The paper's §9 conjecture is *supported*:
  the CIFAR collapse of alignment and the rise of the invariant encoding are substantially about
  output-channel count, not about natural images. §9 is upgraded from conjecture to
  conjecture-with-direct-evidence, and the practical recommendation is restated in terms of $c$.
- **$f(\mathrm{W5}) < 0.40$ and the crossover persists.** The conjecture is **wrong** and must be
  withdrawn from §9, not softened. The cross-dataset findings are then about signal complexity, and
  the channel mechanism is demoted to the minority contribution the encoder ablation measured.
- **Anything between.** Both factors contribute; report the split implied by the observed $f$
  against the two anchors ($0.324$ at $c{=}3$, $0.628$–$0.664$ on the grayscale corpora) and state
  explicitly that neither single-cause story survives.

**In every case** the PSNR difference of §2 is reported alongside, and no claim is made that
survives only under the assumption that fidelity is irrelevant. If $f(\mathrm{W5})$ moves in the
predicted direction, the honest alternative explanation — "the fit is simply easier, so alignment
works better" — is stated in the same paragraph, and the fidelity-matched follow-up is named as
owed.

## 6. Analysis, exclusions, compute

Inherits `S1.md` §§3–7: same frozen matched-MLP decoder, same seed policy (5 seeds for the
deterministic rungs run here), paired-by-seed $t$ and Wilcoxon, bootstrap CIs on $f$, label-shuffle
controls at W1/W3/W5, no row-wise INR exclusion. Scoring uses the per-dataset registration table in
`14_ladder_analysis.py`, extended with a `cifar10gray` entry carrying the H-G1 numbers above.

**Budget:** corpus generation measured at $12.2$ fits/s $\Rightarrow$ $120{,}000$ fits $\approx3.2$ h
derated; ladder $\approx30$ min. **Stopping rule:** if generation exceeds 6 h, `P-random` is
truncated to 20k train images and the reduced $n$ is reported in the table rather than hidden.

---

## Deviation log (appended 2026-07-30)

**D1 — the quality gates ran *after* the ladder, not before.** `scripts/30_cifar_gray_corpus.sh`
invoked `04_quality_gate.py` with `--corpus/--split/--epochs`; the script takes
`--dir/--eval-split/--gate-epochs`. Both gate invocations therefore died on an argparse error and
printed `GATE FAILED`, which was a *script* failure and not a gate rejection. The chained ladder
(`31_cifar_gray_ladder.sh`) checked only that the corpus log reported completion, so it ran on
corpora whose gates had not been evaluated.

The gates were then run by hand and **both pass**: render-vs-real-pixel gaps of $+0.04$
(`P-shared-det`) and $-0.01$ (`P-random`) points, median PSNR $59.8$ and $60.0$\,dB,
corr(PSNR, label) $-0.005$ and $-0.020$. The corpora were therefore admissible throughout and the
cells stand.

Recorded because the *ordering* was wrong even though the outcome was not, and because two things
should be said plainly: (i) a gate result is a property of the corpus and cannot be influenced by
having seen ladder numbers, so there is no selection risk here; (ii) while diagnosing the failure I
saw the tail of the ladder log, which included `f_W9` and `f_W10`, before the gates were confirmed.
The registered predictions were frozen long before either.

**Fix owed:** the chain scripts should gate on the gate, not on the corpus log. `31_*.sh` is
amended to require a passing gate JSON for every protocol before starting.

**D2 — §2's declared pre-freeze exposure overstates what existed.** §2 says "the corpora were
generated and their quality gates run before this file was written", copied from the `S1-cifar.md`
template without checking. That is **false**: this file was frozen at 12:52 on 2026-07-30, four
minutes after corpus generation *started* (12:48) and nearly four hours before it finished (16:44).
At freeze time only a handful of shards existed and no gate had been attempted.

The error runs in the conservative direction — it declares *more* pre-freeze exposure than actually
occurred, so the registration is stronger than advertised rather than weaker — but a declaration of
exposure is exactly the kind of statement that has to be true, so it is corrected here rather than
left to be discovered. What §2 should have said: the pilot-shard PSNR values ($59.1$–$60.0$\,dB from
a 256-image smoke run) had been seen, and nothing else.
