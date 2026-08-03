# S8 Addendum 01 — declared exposure, and a lesson not applied

**Written:** 2026-08-03, before the confirmatory sweep.
`S8.md` is frozen (hash `ab228c6deb526eed`) and is not edited.

---

## What happened

The S8 decode script runs unattended at the end of roughly eleven hours of corpus generation, so
a bug in it would waste the sweep. I smoke-tested it on the one budget whose corpora already
existed, at **one seed on CPU**:

```
steps=300  W1=88.00  W3=11.30  gap=76.70  f(W4)=0.107  f(W5)=0.485  f(W10)=0.234
           |grad|=7.38e-03  psnr=37.4dB  travel=0.186
```

Two of those are registered quantities: **H-S8-1** (W1 at 300 steps, registered $88.0\,[82,93]$)
and **H-S8-3** ($f(\mathrm{W5})$ at 300 steps, registered $0.60\,[0.45,0.72]$). I have therefore
seen single-seed estimates of both before the confirmatory five-seed run.

## The part that is worse than the S9 case

`S9-addendum-01.md`, written earlier the same day, ends with exactly this lesson:

> A pre-run scaling check on the registered quantity is exposure even when it is not intended as a
> measurement. The template should either require such checks to run on a held-out shard with
> labels shuffled, or require a declaration like this one.

I then ran another pre-run check on registered quantities without applying it. The declaration
route was available and I took it only after the fact, again. The lesson is not that a smoke test
is wrong --- it caught nothing this time but it is the reason the eleven-hour job is not at risk ---
it is that the template's guard has to be **mechanical**, not a note in a previous addendum.

## What changes

- **Nothing in `S8.md`.** All predictions, intervals, calls and the falsifier were frozen before
  any corpus was fitted, and are unedited.
- **The confirmatory numbers are the five-seed MPS run**, which regenerates `results/s8/sweep.json`;
  the smoke report was deleted rather than kept.
- **For the record**, the exposed values sit inside their registered intervals, so the exposure
  creates no incentive problem — but that is luck, not design, and it is the second time today.
- **Mechanical guard, adopted now.** `48_s8_sweep.py` and any future scorer must refuse to write a
  report unless invoked at the registered seed count and device, or with an explicit
  `--smoke` flag that shuffles the training labels. That is implemented alongside this addendum so
  that the guard exists in code rather than in a lesson.
