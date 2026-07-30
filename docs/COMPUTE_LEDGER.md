# Compute Ledger

Hardware: MacBook Air M4, unified memory, fanless. Rules: checkpoint ≥ every 10 min; resumable;
`caffeinate -i` for long runs; > 10 h jobs need written justification here. Program budget ≤ 350 h.

## Burn-down (updated at every Gate)

**Two clocks, and the difference is large.** Detached overnight chains keep running across machine
sleep, so a shard's recorded wall-clock absorbs the sleep interval. Budget accounting uses
**active compute** = median seconds-per-fit × number of fits (robust to those stalls); wall-clock is
tracked separately because it is what governs scheduling.

| gate | date | spent this gate (h, active) | cumulative (h) | remaining vs 350 | notes |
|---|---|---|---|---|---|
| G0 | 2026-07-16 | ~0 (API only) | ~0 | 350 | lit scan; no training compute |
| G1 | 2026-07-17 | ~0.3 (tests + profiling) | ~0.3 | ~349.7 | throughput measured (below) |
| G3 (interim) | 2026-07-17 | ~4 (pilots + MNIST chain) | ~4.3 | ~345.7 | **steps frozen at 300 (quality gate)**; sustained thermal throttle 55→48 fits/s (−13%, fanless, R7) |
| G3 (corpora complete) | 2026-07-23 | 11.1 total fitting, all corpora | ~12 | ~338 | 1.31 M INRs across MNIST+FMNIST × 4 protocols. **Wall-clock for the same work: 123.4 h** — the gap is machine sleep inside detached shards, not compute. Median 0.03 s/fit (MNIST cfg) = ~33 fits/s, better than the 15 fits/s G1 projection |
| G4 (in progress) | 2026-07-27 | ~0.2 (microcosm optimizer census) + ladder/pilot in flight | — | — | S1 ladder on MNIST + CIFAR-10 pilot sweep running detached |

### Measured corpus costs (from metadata, 2026-07-27)

| corpus | fits | active h | wall h |
|---|---|---|---|
| mnist P-shared-det / P-random / P-shared-stoch | 3 × 70 000 | 0.49 / 0.49 / 0.16 | ~0 (ran awake) |
| mnist P-random-K (K=8) | 440 000 | 3.10 | 25.1 |
| fashionmnist P-shared-det / P-random / P-shared-stoch | 3 × 70 000 | 0.78 / 0.79 / 0.27 | 13.3 / 11.8 / 3.2 |
| fashionmnist P-random-K (K=8) | 440 000 | 4.91 | 69.9 |
| **total** | **1 308 256** | **11.1** | **123.4** |

## Measured throughput (G1, `scripts/01_profile_fitting.py`, torch 2.13.0, M4)

| config | device | fits/s @ protocol steps | notes |
|---|---|---|---|
| MNIST-like (28², 1ch, w32 L2, 1k steps) | MPS | **~15.0** | flat across B ∈ {64, 256, 1024} — saturated at B=64 |
| MNIST-like | CPU | ~9.8 | MPS/CPU ≈ 1.5× only — CPU is a viable determinism fallback |
| CIFAR-like (32², 3ch, w64 L3, 2k steps) | MPS | **~2.1** | B=1024 peak mem 3.4 GB |
| CIFAR-like | CPU | ~1.1 | |

Footnotes: (i) targets were random noise — steps/sec is architecture-bound and image-independent,
so fits/s is valid for budgeting even though real images may need fewer steps to pass the quality
gate (pilot decides steps; these are upper-bound-ish); (ii) batching beyond B=64 buys nothing on
MPS at these sizes — use B=256 for checkpoint granularity; (iii) optimization levers if needed:
step-count tuning at the quality gate, fp16 storage (R8), torch.compile — NOT the 2410.04779
init-scaling trick (init is a treatment variable in this program).

## Corpus projections at measured throughput (MPS)

| corpus | fits | est. hours |
|---|---|---|
| sine MNIST: {shared-det, shared-stoch, random} × 70k + random-K K=8 × 60k train | 690k | ~12.5 |
| sine FashionMNIST: same | 690k | ~12.5 |
| sine CIFAR-10 full: 3 × 50k + 8 × 40k train | 470k | ~62 |
| sine CIFAR-10 **fallback 20k/4k**: 3 × 24k + 8 × 20k | 232k | ~31 |
| comparative activations (L2): 4 acts × 2 protocols × (MNIST 70k + CIFAR-fallback 24k) | 752k | ~36 (CIFAR-full: ~67) |
| S4 repeated fits: 200 × 32 × 2 datasets | 12.8k | ~1.5 |
| **fitting total (CIFAR fallback path)** | | **~94** |
| **fitting total (CIFAR full path)** | | **~156** |

With decoder/metanetwork training (30–60 h) and analysis (~10 h): fallback path ≈ 135–165 h,
full path ≈ 195–225 h — both inside the 350 h budget; full-CIFAR decision deferred to the G3 pilot
(quality-gate step counts may cut these substantially). Thermal-throttling curve still owed
(sustained-run measurement piggybacks on the first overnight G3 job rather than a synthetic burn).

## Budget targets vs revision

| class | protocol target (h) | G1 revised est. (h) |
|---|---|---|
| sine dataset generation | 25–60 | 25 (MNIST+FMNIST) + 31–62 (CIFAR) |
| comparative activations | 15–35 | 36–67 (over target if CIFAR-full; fallback keeps ≈ target) |
| repeated-fit corpus (S4) | 10–20 | ~2 (well under) |
| decoder/metanetwork training | 30–60 | unchanged (measure at S1 pilot) |
| analysis | ~10 | unchanged |

## Per-experiment-class estimates vs measured

| experiment class | est. wall-clock | measured | est. peak mem | measured | sidecar refs |
|---|---|---|---|---|---|
| fit-throughput profile | ~0.2 h | 0.15 h | < 4 GB | 3.4 GB (CIFAR B=1024) | results/profiling/fit_throughput.json |

## Justifications for > 10 h jobs

(none yet)

## S4e actuals (2026-07-29/30)

| item | wallclock | note |
|---|---|---|
| killed launch #1 | ~1 min | tripped the prereg void condition (planted control); numbers discarded |
| killed launch #2 | **6 h 55 m** | blew the 3 h stopping rule; numbers discarded. Cause: per-step cost mis-estimated ~20x |
| confirmatory run | ~40 min | 5 widths, per-width batch sizes after deviation D1 |
| budget control (5x steps, w=16/32) | ~38 min | exploratory; separates basin size from training budget |
| candidate verification | ~5 min | adjudicates the falsification candidate from parameters |

**Measured, and worth keeping:** fitting cost on MPS is dominated by **batch size, not width** —
96 ms/step at n=128/w=32 on a 64x64 grid, against 19.7 ms at n=32/w=32 and 34.5 ms at n=128/w=8.
So w=8 at n=128 is *more* expensive than w=32 at n=32. MPS measured 3-4x faster than CPU for this
workload (27 vs 78 ms/step at n=32), so the earlier over-run was a scale error, not a device error.
Lesson for future projections: benchmark one step at the intended (n, width) before projecting, and
project from batch size first.
