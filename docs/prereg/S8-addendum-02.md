# S8 Addendum 02 — a mechanism call on the 10000-step arm, made before it is decoded

**Written:** 2026-08-04, after the 300/1000/3000 decode and **before any 10000-step cell exists**.
The 10000-step corpora are still being fitted; nothing at that budget has been decoded, scored or
seen. `S8.md` is frozen (hash `ab228c6deb526eed`) and is not edited, and no registered interval,
call or falsifier is added, removed or moved. This addendum records a belief that changed while
three of the four registered arms were being decoded, so that it can be scored rather than
asserted afterwards.

---

## What the first three budgets show

| budget | PSNR (shared) | PSNR (random) | rel. grad norm (shared) | travel (shared) | $f(\mathrm{W5})$ |
|---|---|---|---|---|---|
| 300   | 39.11 dB | 37.41 dB | $5.46\times10^{-3}$ | 0.186 | 0.502 |
| 1000  | **69.25 dB** | **64.62 dB** | $\mathbf{2.25\times10^{-4}}$ | 0.194 | 0.489 |
| 3000  | 61.91 dB | 62.43 dB | $4.53\times10^{-3}$ | 0.194 | 0.470 |

Two things in that table were not anticipated by the registration.

**Fit quality is not monotone in fit budget.** PSNR rises 30 dB from 300 to 1000 steps and then
*falls* 7.3 dB from 1000 to 3000, on both protocols, while the endpoint gradient norm falls 24×
and then rises 20×. The registration's premise --- that "steps" indexes progress toward
stationarity --- holds between 300 and 1000 and fails after it.

**Parameter travel saturates almost immediately.** The median relative distance from $\theta_0$ is
0.186, 0.194, 0.194 across a 10× budget: flat to the third decimal after 1000 steps.

## The mechanism I claim

The fitter is constant-learning-rate Adam with no schedule (`src/sirengap/fitting/batched.py`,
`lr = 1e-3`). Adam's step size is set by the ratio of first to second moments, so it does **not**
shrink as the gradient does: near a minimum the iterate stops descending and starts diffusing in a
band whose width is set by the learning rate, not by the number of steps. Under that reading, the
1000-step corpus is a near-stationary endpoint caught just as descent finishes, and the 3000-step
corpus is a *sample from the fluctuation band*, which is why its PSNR is worse, its endpoint
gradient norm is an order of magnitude larger, and its two protocols land on nearly the same PSNR
(61.9 vs 62.4) after differing by 4.6 dB at 1000 steps.

If that is right, more budget does not buy stationarity under this fitter, and the 10000-step arm
lands in the same band rather than below it.

## Calls, scoreable against the frozen S8 quantities

These are probabilities on the **already-registered** quantities. They are not new hypotheses and
they do not replace the frozen intervals; they record what I expect now, given exposure to the
three earlier arms, which are separate registered cells decoded on protocol.

- **0.72** — **P-S8-C resolves FALSE**: the median relative gradient norm does *not* fall by 10×
  between 300 and 10000 steps (it would have to reach $5.5\times10^{-4}$; the 3000-step arm sits at
  $4.5\times10^{-3}$).
- **0.70** — **H-S8-6 misses high**: the gradient norm at 10000 steps is above the registered
  interval's upper edge of $3\times10^{-3}$.
- **0.75** — **H-S8-7 misses low**: travel at 10000 steps is below the registered interval's lower
  edge of 0.20, because travel saturated by 1000 steps at 0.194 and there is no mechanism left to
  move it.
- **0.85** — **H-S8-4 hits**: $f(\mathrm{W5})$ at 10000 steps is inside $[0.20, 0.65]$.
- **0.88** — **H-S8-5 hits**: $f(\mathrm{W5})$ at 10000 minus at 300 is inside $[-0.40, +0.05]$
  (it is $-0.032$ at 3000).
- ~~**0.60** — median PSNR on the shared corpus at 10000 steps is in $[58, 66]$ dB.~~
  **Withdrawn as a call; see the declared exposure below. It is reported as an observation, not
  scored.**

## Declared exposure — the PSNR call was contaminated when I wrote it

`03_generate_inrbench.py` prints a per-shard median PSNR to `results/s8/run_master.log`, the same
file the decode writes to, in the form `shard_011008: 256 fits in 823.3s psnr_med=58.0dB`. I had
read several of those lines while checking that the resumed chain was alive, **before** writing the
PSNR call above, and they sit at 57.6–59.3 dB across the 10000-step shards. So that call was
informed by fitted values of the quantity it predicts, and betting a $[58,66]$ band was not a
forecast. It is struck out rather than deleted, and the shard PSNRs are reported as an observation.

The other five calls are clean: the generator prints PSNR and throughput only. Endpoint gradient
norm, parameter travel and every $f$ are computed by `48_s8_sweep.py` at decode time, which has not
run at this budget, and none of them appears anywhere in the log. The seed-count guard added in
`48_s8_sweep.py` still holds for the decode itself.

This is the third exposure in this study's family (`S8-addendum-01`, `S9-addendum-01`) and the first
one caught *in the same session that created it* rather than afterwards. The generalizable defect:
a long-running generator and a scorer that writes registered quantities share one log file, so
monitoring the job at all exposes one of the quantities. Separating the generator's progress log
from the results log is the mechanical fix, and it is recorded as owed rather than done here,
because changing the log path mid-chain would break the resume.

## What follows if the first three resolve as stated

`S8.md` §"If P-S8-C fails" already commits the consequence: **the sweep did not reach stationarity
and cannot answer the review's question** as posed. That commitment stands and will be executed;
this addendum does not soften it. What it adds is the reason --- the failure would be a property of
the *optimizer*, not of the budget, so the honest repair is a fitter with a decaying schedule or an
explicit stationarity stopping rule, not more steps. That repair is out of scope here and is
recorded as future work rather than run.

It also sharpens why $f(\mathrm{W5})$ is flat in budget, which is what the review actually asked.
Alignment to $\theta_0$ keeps working not because the fits are under-trained but because they never
leave $\theta_0$'s neighbourhood at all: travel is 0.194 at every budget past 300. The lazy regime
is not an artifact of a short budget under this fitter, and that is a stronger answer to the review
than the one the registration expected to give.
