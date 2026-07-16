# PROJECT SIREN-GAP

**The weight-space perception gap: symmetry, basins, and the limits of reading semantics from
neural network parameters.**

Dissertation-scale research program: why machine-learning systems fail to read semantics from the
raw weights of independently trained implicit neural representations (INRs) — and whether that
failure is nuisance variability (parameter symmetries + basin multiplicity) rather than an
information deficit. Theory (the infinite-dihedral symmetry structure of sine networks, exact
canonicalization, identifiability), instruments (canonicalizers, a D∞≀Sₙ-equivariant architecture,
the INR-Bench dataset), six pre-registered empirical studies, and a FLOPs-matched adjudication
against function- and render-access baselines.

**Status: G0 complete** (literature deep scan, novelty gates passed, theory scoping). Next: G1
(property tests T1–T9, throughput profiling, close-read memos). See `docs/LAB_NOTEBOOK.md`.

All computation: single MacBook Air (M4), PyTorch MPS/CPU. No cloud.

## Layout

```
src/sirengap/    fitting/ symmetry/ canon/ models/ geometry/ data/ eval/ queue/
tests/           property tests T1–T9 (CPU-runnable)
configs/         one YAML per experiment, no hidden defaults
scripts/         numbered idempotent entrypoints (00_lit_scan.sh, …)
results/         committed per-seed CSVs + figures (raw shards gitignored)
paper/           paperA/ paperB/ paperC/ thesis/
docs/            LAB_NOTEBOOK, prereg/, THINKING/, ADVISOR_REVIEWS/, ledgers, RELATED_WORK, …
```

Process transparency is part of the artifact: the lab notebook, prediction ledger,
pre-registrations, and adversarial advisor reviews are committed alongside the code.

## Make targets

`make test` · `make lit-scan` · (`demo`, `figures`, `thesis` arrive at later gates)
