# Literature Query Set (saved; re-run at every Gate)

**Tooling:** `scripts/00_lit_scan.sh` (arXiv API, Atom; 3 s politeness delay) →
`scripts/parse_atom.py` for compact reading → snapshot committed to `docs/lit_snapshots/`.
Supplementary: `paper-search` CLI (Semantic Scholar/CrossRef et al.) for citation counts and
non-arXiv venues; Hugging Face papers for model/dataset lineage when needed.

## Cadence

| when | action |
|---|---|
| every Gate G1–G8 | full re-run of s01–s16; diff new titles vs previous snapshot; log verdict deltas in LAB_NOTEBOOK |
| any design decision touching published work | targeted query, logged here with date |
| before each study prereg | re-check that study's close-read list |

## Seed queries (protocol §0.3) → script blocks

| script name | query | rationale |
|---|---|---|
| s01_wsl | all:"weight space learning" | field pulse |
| s02_inr_cls | abs:"implicit neural representation" AND abs:"classification" | direct competitors |
| s03_param_sym | abs:"parameter symmetries" OR abs:"parameter space symmetry" | theory neighbors |
| s04_canon | abs:"canonicalization" AND cat:cs.LG | canonicalization methods |
| s05_periodic_sym | abs:"periodic activation" AND abs:"symmetry" | Gate-1 tripwire (0 hits at G0) |
| s06_siren_sym | abs:"SIREN" AND abs:"symmetry" | Gate-1 tripwire |
| s07_metanets | abs:"metanetworks" OR abs:"neural functionals" | architecture line |
| s08_ws_inr | abs:"weight space" AND abs:"implicit neural" | intersection |
| s09_ti_ws | ti:"weight space" | title sweep (catches surveys/positions) |
| s10_nef_init | abs:"neural field" AND abs:"initialization" | protocol-variable literature |
| s11/s12_titles | known-title lookups | ID verification for named works |
| s13_recent_ws | abs:"weight space" + submittedDate ∈ [current-gate-window] | scoop watch (update date range each run) |
| s14_recent_inr | abs:"implicit neural representation" + same window | scoop watch |
| s15_class_signal | abs:"class signal" OR all:"HyperINR" | specific close-read hunts |
| s16_identif | abs:"identifiability" AND abs:"neural network" AND cat:cs.LG | PO-2 line (noisy; refine at G1) |

Protocol seed phrases not yet mapped to their own blocks (covered implicitly; split out if a gate
re-scan misses something): "model alignment re-basin" (s04/s09 catch it), "learned canonicalization
frame averaging" (s04), "probe-based INR" (s02/ProbeGen lineage), "sample complexity invariance"
(tracked via citation-following from 2102.10333/2102.13219/2106.07148).

## Targeted follow-ups logged

| date | query | outcome |
|---|---|---|
| 2026-07-16 | id_list batches (36 IDs) | 34 resolve as expected; 2306.12447 ≠ Expand-and-Cluster (wrong guess) |
| 2026-07-16 | all:"expand-and-cluster" | correct ID 2304.12794 |
| 2026-07-16 | flagged-abstracts batch (20 IDs) | gate evidence; see G0-novelty-gates.md |

## Maintenance rules

- Update s13/s14 date windows every run (script edit + commit).
- Every new citation enters `paper/references.bib` only after its ID appears in a verification batch.
- s16 recall is poor (title noise) — at G1 replace with `abs:"identifiable" AND abs:"activation"` +
  citation-graph walk from Sussmann/Fefferman entries via paper-search CLI (Semantic Scholar).
