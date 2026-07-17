# Compute Ledger

Hardware: MacBook Air M4, unified memory, fanless. Rules: checkpoint ≥ every 10 min; resumable;
`caffeinate -i` for long runs; > 10 h jobs need written justification here. Program budget ≤ 350 h.

## Burn-down (updated at every Gate)

| gate | date | spent this gate (h) | cumulative (h) | remaining vs 350 | notes |
|---|---|---|---|---|---|
| G0 | 2026-07-16 | ~0 (API only) | ~0 | 350 | lit scan; no training compute |
| G1 | 2026-07-17 | ~0.3 (tests + profiling) | ~0.3 | ~349.7 | throughput measured (below) |
| G3 (interim) | 2026-07-17 | ~4 (pilots + full MNIST chain) | ~4.3 | ~345.7 | **steps frozen at 300 (quality gate)** — MNIST protocol now ~24 min, full 4-protocol chain ~3.5 h vs 12.5 h projected at 1k steps; sustained thermal throttle measured 55→48 fits/s (−13%, fanless, R7); anchors/decoders extra |

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
