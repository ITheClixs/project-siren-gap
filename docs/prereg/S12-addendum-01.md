# S12 Addendum 01 — second attempt at the same bar, staged

**Written:** 2026-08-11, before any corpus of this attempt is fitted.
`S12.md` is frozen and unedited. The first attempt's failure stands on the record: it is reported
in the paper, in CLAIMS row 63, and P-S12-B is already scored FALSE at Brier 0.49.

---

## What changed, and what did not

**Unchanged: the bar.** Median relative endpoint gradient norm below $10^{-4}$ on both protocols,
render PSNR matched between protocols within 2 dB, gate evaluated before anything is decoded. The
threshold is *not* loosened. Loosening it after seeing 1.03--1.36e-4 is precisely the move the gate
exists to prevent.

**Changed: the budget.** The step cap goes from 6000 to 12000. The first attempt's diagnosis was
that the schedule anneals correctly but the corpora run out of budget just above the bar, with five
of six between 1.03 and 1.36e-4 and one outlier at 7.34e-4. Doubling the cap is the cheapest
intervention consistent with that diagnosis, and the per-INR stop means already-converged INRs cost
nothing extra.

## Staged, to avoid spending 24 hours on a repeat failure

Replication 0 is fitted first, both protocols. Its gradient-norm condition is checked **alone**,
before replications 1 and 2 are launched:

- if replication 0 clears $10^{-4}$ on both protocols, the remaining two are fitted and the study
  proceeds to the full gate as registered in S12 §3;
- if it does not, the attempt stops there and is reported as a second failure. We do not fit two
  more replications to average our way over the bar.

This staging is declared here rather than decided later. It changes only which corpora are fitted,
not what is measured or what threshold applies.

## What is still owed if the gate passes

Everything S12 §4 registers: the seven intervals, the three probability calls, and the decode gated
on the validity check. P-S12-B is **already resolved FALSE** by the first attempt and is not
re-scored — that call asked whether the conditions would be met *on the first attempt*, and they
were not.

## Declared exposure

We have seen the first attempt's corpus statistics: gradient norms 1.03e-4 to 7.34e-4, render PSNR
68--71 dB, and the render-matching failure on replication 1 at $-2.16$ dB. That is unavoidable —
the gate reported them — and it informs the budget change above. It does not touch any decoded
quantity, because nothing was decoded.
