# Compute Ledger

Hardware: MacBook Air M4, unified memory, fanless. Rules: checkpoint ≥ every 10 min; resumable;
`caffeinate -i` for long runs; > 10 h jobs need written justification here. Program budget ≤ 350 h.

## Burn-down (updated at every Gate)

| gate | date | spent this gate (h) | cumulative (h) | remaining vs 350 | notes |
|---|---|---|---|---|---|
| G0 | 2026-07-16 | ~0 (API only) | ~0 | 350 | lit scan; no training compute |

## Budget targets (protocol App. E; to be re-estimated from G1 profiling)

| class | target (h) | measured throughput | est. revised (h) |
|---|---|---|---|
| sine dataset generation (all protocols) | 25–60 | — (G1 profiling) | — |
| comparative activations (L2 scope) | 15–35 | — | — |
| repeated-fit corpus (S4, L3 scope) | 10–20 | — | — |
| decoder/metanetwork training | 30–60 | — | — |
| analysis | ~10 | — | — |
| slack / reruns / red-team controls | remainder | — | — |

## Per-experiment-class estimates vs measured

(populated from G1 profiling script: fits/sec at batch sizes {64, 256, 1024} × {MPS, CPU},
peak memory, thermal-throttling curve over 30 min sustained)

| experiment class | est. wall-clock | measured | est. peak mem | measured | sidecar refs |
|---|---|---|---|---|---|

## Justifications for > 10 h jobs

(none yet)
