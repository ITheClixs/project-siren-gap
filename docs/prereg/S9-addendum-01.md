# S9 Addendum 01 — declared exposure before the confirmatory run

**Written:** 2026-08-03, after a pre-run engineering check and **before** the confirmatory run.
`S9.md` is frozen (hash `1c74280c55a1a3f0`) and is not edited.

---

## What happened

The confirmatory W12 run is queued behind about seven hours of S8 corpus generation. Rather than
discover a scaling failure after that wait, I ran the rung at production scale with **one seed and
a two-epoch cap** as a smoke test, purely to check memory, throughput and the capacity rule.

It returned $67.40\%$, i.e. $f = 0.665$, in 118 seconds at width 186 ($1{,}874{,}898$ parameters).

**I have therefore seen an unconverged, single-seed estimate of the registered quantity before the
confirmatory run.** That is exposure, and it is declared here rather than left implicit.

## What it does and does not change

- **The registration stands as written.** All of `S9.md` was frozen before this run, including the
  point estimates, the intervals, the three probability calls, and the pre-committed withdrawal.
  Nothing in it is amended.
- **H-S9-1 registered $0.55$ with an $80\%$ interval of $[0.35, 0.72]$.** The smoke value sits
  inside that interval, so the exposure does not create an incentive to have registered differently
  — but the reader should know it exists, and the interval should be read as having been fixed
  before, not after.
- **P-S9-C, registered at $0.25$, is now likely to resolve TRUE.** Two epochs and one seed will not
  beat five seeds with early stopping at 100 epochs; the confirmatory number should be higher, not
  lower. Section 4 of the frozen registration commits, in advance, to *withdrawing* this paper's
  practical claim that frame choice beats reader architecture if that happens. That commitment is
  binding and is not renegotiated in light of having seen the number early.
- **The smoke artifact is deleted** rather than reported, and the confirmatory cell is the one that
  reaches `results/ladder/mnist/W12.json`.

## The lesson, for the template

A pre-run scaling check on the registered quantity is exposure even when it is not intended as a
measurement. The template should either require such checks to run on a held-out shard with labels
shuffled, or require a declaration like this one. We take the second route here because the check
had already been run by the time the problem was noticed, and adopt the first for future studies.
