# S4e Pre-registration — deep-identifiability falsification hunt — FROZEN

**Study:** S4e (protocol Part III; dissertation Ch1 §PO-2, Ch5). The pre-committed empirical
attack on **Conjecture 6.5**: for $L\ge2$ sine networks, off a proper analytic subset,
$f_\theta = f_{\theta'}$ implies $\theta' = g\theta$ for some $g \in \prod_\ell D_\infty \wr S_{n_\ell}$.

**Frozen:** 2026-07-29, before the confirmatory run. Amends nothing; this is a new study.
Registered rows enter `docs/PREDICTION_LEDGER.csv` as `P-S4e-*`.

**Why this study exists.** PO-2 is *proved* at $L=1$ and *conjectured* at $L\ge2$, while every
empirical result in the program is $L=2$. DEFENSE row 15 names that as the program's weakest
theoretical link. The proof memo (`docs/THINKING/proof-memos/PO-2-deep-attempt.md`) reached a
reduction to a Bessel–CP decomposition with two open lemmas, documented that the Bessel–Vandermonde
system is severely ill-conditioned by $n=5$ (min $|\det|$ falling $4.4\!\times\!10^{-3} \to
4.4\!\times\!10^{-13}$ from $n=2$ to $n=5$), and concluded that the empirical hunt is the better
investment. Its "Empirical wiring" section names the design used here: *fit two-layer networks to
functions realised by known two-layer teachers, align, and measure the recovery rate.*

---

## 1. Instrument

`src/sirengap/canon/refine.py` minimises $\|g\cdot\theta - \theta^*\|$ over $g \in G$ by
coordinate descent that is **exact per layer**: the per-neuron $D_\infty$ optimum is closed form
(four $(d,\ \mathrm{parity}\ j)$ cases, the optimal $j$ being the nearest integer of that parity to
$(b^*_t - (-1)^d b_i)/\pi$), and the permutation is then a Hungarian assignment on those per-pair
minima. Layers interact only through the shared matrix $W^{\ell+1}$, so sweeps are iterated to a
fixed point. Every move lies in $G$, so the function is preserved exactly and this is asserted on
every call. Verified by **T12** (`tests/test_t12_refine.py`, 13 cases): planted elements are undone
to $<10^{-6}$ relative at $L=1,2,3$, with $c=3$ outputs, and at windings up to 12.

Two reported quantities, both relative and therefore scale-free:

$$R_f = \frac{\lVert f_{\theta_s} - f_{\theta_t}\rVert}{\lVert f_{\theta_t}\rVert}
\quad\text{(held-out grid)},
\qquad
R_\theta = \min_{g\in G} \frac{\lVert g\cdot\theta_s - \theta_t\rVert}{\lVert\theta_t\rVert}.$$

## 2. Arms

| arm | construction | role |
|---|---|---|
| `planted` | $\theta^* \to g\cdot\theta^* \to$ realign | **validity control.** If the search cannot undo an element it knows exists, every other number is meaningless. |
| `sensitivity` | perturb $\theta^*$ by relative $\varepsilon$; record $R_f$, $R_\theta$ | measures $\kappa = R_\theta/R_f$, the local condition number of the inverse map $f \mapsto \theta \bmod G$. No fitting. |
| `warmstart` | start the optimiser at $g\cdot\theta^* + \varepsilon$ noise, fit to $\theta^*$'s outputs | **fitter control**, and measures the *radius of the basin* from which optimisation returns to the true orbit. |
| `teacher` | independent students fitted to a teacher's exact outputs | the sharp test: does an independently-found near-solution sit near the teacher's orbit? |
| `null` | two unrelated networks of the same shape | the scale of "large $R_\theta$". |
| `production` | same-image pairs from `P-random-K`, $w=32$, $L=2$ | the protocol's literal arm, at production width. |

## 3. Pre-freeze exposure — declared

The instrument was built and debugged this session, and pilot runs were executed and **seen**
before this file was written. Declaring them is what keeps the freeze auditable:

- `planted` at $w\in\{2,4\}$, $n=8$: max $R_\theta = 4.8\times10^{-8}$.
- `null` at $w\in\{2,4\}$, $n=8$: median $R_\theta \approx 0.39$–$0.43$, min $0.22$.
- `teacher` at $w=2$, $n\le32$, $\le$40k steps: best $R_f = 1.7\times10^{-3}$ with
  $R_\theta = 0.43$; no student recovered.
- `sensitivity` at $w\in\{2,4,8,32\}$: $\kappa$ median $0.042 / 0.026 / 0.017 / 0.006$.
- `warmstart` ladder at $w\in\{2,8,32\}$, $n=16$: recovery fraction ($R_\theta<10^{-3}$) of
  $0.88 / 0.94 / 0.00$ at $\varepsilon=10^{-4}$.

Two consequences, both binding. (i) The predictions in §5 are **informed by** those pilots and are
therefore *not* evidence of forecasting skill; they are registered so that the confirmatory run at
larger $n$ can be scored for stability, and they are flagged `pilot-informed` in the ledger so the
calibration record is not inflated. (ii) The falsification criterion in §4 was **not** met by any
pilot, and it is fixed here before the run that could meet it.

## 4. Falsification criterion (the point of the study)

**A counterexample to Conjecture 6.5 is a pair $(\theta_s, \theta_t)$ with**

1. $R_f < 10^{-5}$ (functional near-equality on a held-out grid), **and**
2. $R_\theta > 20\,\kappa\,R_f$ where $\kappa$ is that width's measured local condition number
   — i.e. the pair is far outside what local conditioning can explain, **and**
3. both networks inside $\Theta_{\mathrm{gen}}$ by the PO-3 strata audit (no dead $w$, no zero $u$,
   no parallel first-layer pair), **and**
4. the `planted` control passing at that width in the same run.

If such a pair is found at **any** width, Conjecture 6.5 is **dead**, CLAIMS row 8 is retired, and
Ch1 must be rewritten to state $D_\infty \wr S_n$ maximality for $L=1$ only. If none is found, the
conjecture is **not** thereby supported — the correct conclusion is stated in §6.

**Void conditions.** The run is void, and reported as void, if any of: `planted` max
$R_\theta > 10^{-5}$ at any width; `null` median $R_\theta < 0.15$; `warmstart` recovery $= 0$ at
*every* width and $\varepsilon$ (which would mean the fitter never recovers anything).

## 5. Registered predictions (80% intervals; all `pilot-informed`)

| # | quantity | point | 80% interval |
|---|---|---|---|
| P-S4e-1 | `planted` max $R_\theta$, worst width | 5e−8 | [1e−9, 1e−5] |
| P-S4e-2 | $\kappa$ median at $w=32$ | 0.006 | [0.003, 0.012] |
| P-S4e-3 | `warmstart` recovery at $\varepsilon{=}10^{-4}$, $w=2$ | 0.85 | [0.55, 1.00] |
| P-S4e-4 | `warmstart` recovery at $\varepsilon{=}10^{-4}$, $w=32$ | 0.00 | [0.00, 0.15] |
| P-S4e-5 | best $R_f$ over independent students, $w=2$, $n=128$ | 3e−4 | [1e−6, 3e−3] |
| P-S4e-6 | $R_\theta$ at that best-$R_f$ student | 0.35 | [0.05, 0.65] |
| P-S4e-7 | independent-student recovery rate ($R_\theta<0.05$), $w=2$ | 0.02 | [0.00, 0.15] |
| P-S4e-8 | `production` median $R_\theta$ (MNIST, same-image pairs, $w=32$) | 0.45 | [0.25, 0.70] |
| P-S4e-9 | `production` median $R_\theta$ **minus** the different-image null | 0.00 | [−0.10, +0.10] |

**Directional claims.** (a) `warmstart` recovery is **monotone non-increasing in width** at fixed
$\varepsilon$. (b) $\kappa$ is **monotone non-increasing in width** — the forward map $\theta\mapsto f$
gets *more* expansive with width, so the inverse gets *better* conditioned locally. Both are
contrary to the naive reading of the Bessel ill-conditioning result, which is about the *global*
recovery system, not the local Jacobian, and §6 must not conflate them.

**Probability call.** **P-S4e-C = 0.15** that the §4 falsification criterion is met at some width
in the confirmatory run. Scored by Brier.

## 6. What each outcome licenses (pre-committed wording)

- **Criterion met.** Conjecture 6.5 is false. Report the pair, its strata audit, and the width.
- **Criterion not met, and independent students never reach $R_f<10^{-5}$.** The registered
  conclusion is: *S4e neither confirms nor falsifies the conjecture, because the hypothesis of the
  conjecture is not reachable by optimisation at these widths.* The publishable content is then the
  **conditioning anatomy**: local recovery is well conditioned ($\kappa\ll1$) while the basin from
  which optimisation returns to the true orbit collapses with width. This must be written as a
  statement about *optimisation and empirical reach*, **not** as evidence that identifiability
  holds — the distinction is the whole point of registering this in advance.
- **Criterion not met, and some independent student does reach $R_f<10^{-5}$ with small
  $R_\theta$.** Positive empirical support, stated as support and not as proof.

## 7. Analysis, exclusions, compute

No row-wise exclusion of students; students that diverge (non-finite loss) are reported as a count
and dropped, and more than 10% divergence at any width invalidates that width's cell. Widths
$\{2,4,8,16,32\}$, $L=2$, $c=1$, teachers are random SIREN initialisations (generically in
$\Theta_{\mathrm{gen}}$; audited). Students: Adam, cosine decay to lr/300, 40 000 steps,
$64\times64$ fitting grid, held-out probes on an $81\times81$ grid so that grid-fitting is not
mistaken for function-fitting. **Budget:** ≤ 1.5 h active. **Stopping rule:** if the confirmatory
run exceeds 3 h, the $w=16$ and $w=32$ teacher cells drop to $n=32$ and this is logged as a
deviation rather than hidden.

---

## Deviation log (appended 2026-07-30; the frozen text above is unchanged)

**Deviation D1 — compute re-plan after the stopping rule fired.** The first confirmatory run was
launched at $n=128$ for every arm and every width with 40 000 steps throughout. Its cost had been
estimated at ~1.5 h (§7); the true cost is ~12 h, because the per-step time on MPS is dominated by
*batch size* rather than width and was mis-estimated by a factor of ~20 (measured: 96 ms/step at
$n=128$, $w=32$ on a $64\times64$ grid, against an assumed ~5 ms). The run was **killed at 6 h 55 m**
with only the `planted` arm complete, and none of its numbers are used.

§7's stopping rule anticipated this in kind but not in scale ("drop the $w=16$ and $w=32$ teacher
cells to $n=32$"). The re-plan, logged here **before** the replacement run:

| arm | frozen spec | re-plan | why |
|---|---|---|---|
| `teacher` | $n=128$, 40 000 steps | $n=128$ at $w\in\{2,4\}$; $n=64$ at $w=8$; $n=32$ at $w\in\{16,32\}$; steps unchanged | P-S4e-5/6/7 are registered at $w=2$, $n=128$, which is preserved exactly. §7's allowance is extended from $\{16,32\}$ to $\{8\}$ as well. |
| `warmstart` | inherits 40 000 steps | **8 000 steps**, $n=32$ | 8 000 is the value of the declared pilot (§3) that produced the registered P-S4e-3/4 predictions, so scoring them at 8 000 steps is *more* faithful to what was registered than 40 000 would be. |
| `sensitivity`, `null` | $n=128$ | $n=32$ | both report medians over a batch and involve no fitting; $n=32$ changes nothing but wall-clock. |

**Nothing about the falsification criterion (§4), the void conditions (§4), the registered
intervals (§5) or the outcome wording (§6) is changed.** Device remains MPS: it was measured at
3–4$\times$ faster than CPU for this workload, so the cost was not a device-choice error.

**Deviation D2 — restarts added to the alignment search.** The first launch tripped the §4 void
condition (`planted` max $R_\theta = 0.22$ at $w=2$). Cause: plain per-layer coordinate descent
stalls in a joint local optimum on ~10% of width-2 pairs — visible at $n=128$, invisible at the
$n=16$ pilot scale. `refine_alignment` now takes the best of `n_restarts` descents (restart 0 is the
identity, so it can never be worse). Planted recovery is now 0/128 failures at every width, max
$4.3\times10^{-8}$. This strengthens the instrument in the direction that makes the hunt *harder* to
pass — a smaller $R_\theta$ is a weaker case for a counterexample — so it cannot inflate the result.
