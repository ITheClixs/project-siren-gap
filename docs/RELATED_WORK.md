# Related Work (living document)

**Last scan:** 2026-07-16 (G0) · **Entries:** 60 numbered (55 arXiv-verified on 2026-07-16, 4
classics without arXiv IDs, 1 OpenReview-unverified) + 12 watched in §G · **Target:** ≥ 60 curated
with delta columns by G7 (numbered entries still need delta-memo depth added as studies approach).
All arXiv IDs below were verified against the arXiv API on 2026-07-16 (`scripts/00_lit_scan.sh`;
snapshot in `docs/lit_snapshots/`). Non-arXiv items marked ✗-arXiv. Deltas for post-cutoff (2026)
papers are abstract-based until the G1 close-reads.

## Access-model taxonomy

| access model | what the perceiver touches | representative methods |
|---|---|---|
| weight-access | raw/canonicalized parameters | DWSNets, NFN, NFT, Monomial-NFN, GMN, ScaleGMN, SANE, inr2vec, **ours** |
| function-query | adaptive/learned input probes, outputs only | ProbeGen, ProbeX/MVProbe, Expand-and-Cluster (recovery) |
| render-access | dense grid evaluation → pixel model | oracle-render rung P1, CNN baselines, ARC (hybrid: arch. redesign) |
| regime-change (not perception of independent fits) | shared/meta-learned init, hypernetwork generation | functa, spatial functa, MWT line, HyperINR, IFT-2601.23181, CertMix |

## Mandatory close-reads (Appendix D.0) — half-page adversarial delta memos due before dependent studies

| work | ID (verified) | close-read due | one-line delta (G0, abstract-level) |
|---|---|---|---|
| Papa et al., How to Train NeF Representations | 2312.10531 | G1 (before G3 fitters) | interventional on fit hyperparams; no symmetry/basin decomposition; we subsume shared-init as W1 vs W3 and must replicate (anchor A1) |
| Shamsian et al., WS augmentations | 2402.04081 (+2311.08851) | G1 | states nuisance premise, treats by augmentation; no exactness/decomposition; check whether any aug is a τ-phase shift (Gate-1 tripwire) |
| Clustered or Routed? | 2605.08281 | G1, before S2 design | meta-learned MWT regime; geometry↛reader-accuracy; bias column = causal route; adversarial to S2 — prereg must name their dissociation as an outcome |
| IFT semantics (HyperINR-IFT line) | 2601.23181 (+HyperINR 2304.04188) | G2 | hypernetwork-generated weights; IFT data↔weight mapping; regime sidesteps independent-fit nuisance; complementary theory lens |
| SANE | 2406.09997 | G5 (before S5) | task-agnostic sequential weight tokens for large models; no exact symmetry handling; S5 contender-context |
| ScaleGMN | 2406.10685 | G5 | graph metanetwork with scaling equivariance (ReLU-oriented); no periodic-activation groups; S5 contender |
| Monomial-NFN | 2409.11697 | **G1 (maximality proof)** | perm×sign for sin — σ only; τ/ρ affine ⇒ outside monomial family; our Ch3.7 extends equivariance to full D∞≀Sₙ |
| ProbeGen | 2410.10811 | G5 (before S5 FLOPs matching) | function-query baseline; 30–1000× FLOPs savings claim → the S5 Pareto must beat/match this frontier honestly |

Added at G0 (not in protocol, same treatment): 2602.01083 (expressivity of perm-equivariant WSN —
**PO-6 novelty check, G1**), 2604.23720 (quasi-equivariant metanetworks, G1), surveys 2506.13018 +
2603.10090 (sine sections, G1).

**Close-read status (2026-07-17):** DONE — Papa 2312.10531, Shamsian 2402.04081 (+ SIREN-bias-aug
failure finding), 2605.08281, Monomial-NFN 2409.11697 (maximality open question), 2602.01083,
2604.23720, both surveys (memos in docs/THINKING/close-reads/). Remaining per schedule: IFT
2601.23181 (G2), SANE + ScaleGMN + ProbeGen (G5 / before S5).

## A. Closest neighbors — INR weight-space perception

| # | work | access | symmetries handled | init protocol | delta-to-us |
|---|---|---|---|---|---|
| 1 | inr2vec, 2302.05438 | weight | none (relies on alignment) | **shared init** | encoder on aligned INRs; we explain *why* shared init is load-bearing (W1 vs W3) |
| 2 | Papa et al., 2312.10531 | weight | none | factorial incl. shared | see close-read table |
| 3 | Shamsian et al., 2402.04081 | weight | perm-aware arch + augs | random | see close-read table |
| 4 | Data augs in DWS, 2311.08851 | weight | perm arch; aug taxonomy | random | predecessor of #3; aug taxonomy to mine for W6 |
| 5 | Clustered or Routed?, 2605.08281 | weight (meta) | none | meta-learned | see close-read table |
| 6 | MWT end-to-end, 2503.18123 | regime-change | none (explicitly) | meta-learned + learned LR | SOTA line (CIFAR-10 59.6% no-aug); changes data-generating process — reference point, not rung |
| 7 | IFT semantics, 2601.23181 | regime-change | n/a | hypernetwork | see close-read table |
| 8 | ARC, 2503.15156 | arch. redesign | input-space robustness | per-image fits | redesigns the INR for classifiability; we hold SIREN fixed and explain the gap |
| 9 | INR2JLS / DNG-Encoder, 2607.02166 | weight | sequential-graph structure | unspecified (G1 check) | claims +10% CIFAR-100-INR; S5 contender-context |
| 10 | Implicit-Zoo, 2406.17438 | dataset | — | per-image | large INR corpus; INR-Bench delta: protocol factorial + audits + gates |
| 11 | functa, 2201.12204 | regime-change | avoided by construction | shared base + modulations | sidesteps nuisance; we study the regime it avoids |
| 12 | spatial functa, 2302.03130 | regime-change | same | same | scaling of #11 |
| 13 | CertMix, 2607.04123 | regime-change | avoided via shared anchor | shared anchor | shared-anchor SIREN weight linearity observation → supports PO-9; different domain |

## B. Metanetworks / weight-space architectures

| # | work | notes / delta |
|---|---|---|
| 14 | DWSNets, 2301.12780 | perm-equivariant linear layers; no sign/phase |
| 15 | NFN, 2302.14040 | perm-equivariant functionals; basis for our Ch3.7 extension |
| 16 | NFT, 2305.13546 | attention variant |
| 17 | Universal Neural Functionals, 2402.05232 | general perm-equivariant construction |
| 18 | Graph Metanetworks, 2312.04501 | graphs over params; perm by construction |
| 19 | Kofinas et al., 2403.12143 | neural-graph representation; S5 baseline candidate |
| 20 | ScaleGMN, 2406.10685 | + scaling equivariance (ReLU) |
| 21 | Monomial-NFN, 2409.11697 | + sign for sin/tanh; **maximality scoped to linear actions** — our τ/ρ are affine (G1 close-read) |
| 22 | SANE, 2406.09997 | sequential tokens, scalability |
| 23 | Quasi-Equivariant Metanetworks, 2604.23720 | relaxed equivariance; expressivity trade-off; G1 close-read (Gate-1 tripwire) |
| 24 | Expressivity of perm-equiv WSN, 2602.01083 | universality theory on weight+function space; **PO-6 novelty check** |
| 25 | ProbeGen, 2410.10811 | function-query; FLOPs-frontier anchor for S5 |
| 26 | MVProbe, 2605.23410 | multi-view (Gram) probing; ProbeX lineage; S5 context |
| 27 | Adversarial attacks in WS classifiers, 2502.20314 | robustness context for Paper C limitations |

## C. Alignment, symmetry, canonicalization (theory backbone)

| # | work | notes |
|---|---|---|
| 28 | Git Re-Basin, 2209.04836 | permutation alignment for merging; our c_align generalizes the assignment step to D∞≀Sₙ |
| 29 | Entezari et al., 2110.06296 | LMC-modulo-permutation conjecture; S4b uses the post-alignment barrier construction |
| 30 | Equivariant WS alignment, 2310.13397 | learned alignment; comparison for Ch3.4 |
| 31 | Kaba et al., 2211.06489 | learned canonicalization framework (Ch3.4 basis) |
| 32 | Frame averaging, 2110.03336 | principled alternative when canonicalization discontinuous (Ch3.5, rung W9) |
| 33 | Impossibility of continuous canonicalization, 2402.16077 | PO-5's general backbone |
| 34 | Empirical impact of parameter symmetries, 2405.20231 | removes symmetries by architecture; we remove by canonicalization on fixed arch — complementary intervention |
| 35 | Parameter-symmetry survey, 2506.13018 | catalogue; G1 check of sine sections |
| 36 | WSL survey, 2603.10090 | field taxonomy (WSU/WSR/WSG); Paper C positioning |
| 37 | Complete symmetry classification, shallow ReLU, 2604.14037 | 2026; non-differentiability technique; complements our analytic-activation program |
| 38 | Symmetries of 3-layer ReLU, 2605.18319 | 2026; semi-algebraic fibers + poly-time functional-equivalence decision; methodological template for PO-2 deep |
| 39 | Neuron identifiability & LMC, 2606.04754 | 2026; effective function classes, approximate equivalence — informs PO-10 definitions |
| 40 | DeepWeightFlow, 2601.05052 | re-basin canonicalization for weight generation; perm-only |
| 41 | Hecht-Nielsen 1990 ✗-arXiv | classic perm+sign observation for odd activations |

## D. Identifiability (PO-2 toolbox)

| # | work | notes |
|---|---|---|
| 42 | Sussmann 1992 ✗-arXiv | tanh, single hidden layer, sign+perm uniqueness |
| 43 | Fefferman 1994 ✗-arXiv | sigmoid deep nets via pole structure; our Bessel-lattice analogue |
| 44 | Phuong & Lampert, ICLR 2020 ✗-arXiv-unverified (OpenReview) | ReLU functional vs parametric equivalence |
| 45 | Expand-and-Cluster, 2304.12794 | parameter recovery by overparameterized refitting; S4e tooling candidate |

## E. Invariance & sample complexity (PO-12)

| # | work | notes |
|---|---|---|
| 46 | Elesedy & Zaidi, 2102.10333 | strict generalization benefit under equivariance |
| 47 | Mei–Misiakiewicz–Montanari, 2102.13219 | invariant kernels/features effective-sample gains |
| 48 | Bietti–Venturi–Bruna, 2106.07148 | geometric stability sample complexity |
| 49 | Learning with exact invariances in poly time, 2502.19758 | computational side of PO-12 |

## F. Weights-as-data context

| # | work | notes |
|---|---|---|
| 50 | Unterthiner et al., 2002.11448 | weight statistics predict accuracy — S5 baseline |
| 51 | Hyper-representations, 2110.15288 | SSL on weights |
| 52 | Model Zoos, 2209.14764 | zoo datasets |
| 53 | HyperDiffusion, 2303.17015 | generative weights for fields |
| 54 | G.pt, 2209.12892 | generative checkpoint models |
| 55 | SIREN, 2006.09661 | the instrument itself (ω₀ convention source) |
| 56 | NeRF, 2003.08934 | field context |
| 57 | WIRE, 2301.05187 | Gaussian-wavelet activation family (S3 candidate variant) |
| 58 | FINER, 2312.02434 | variable-periodic activation — our no-clean-symmetry control (PO-7 derivation owed) |
| 59 | Fast sinusoidal NF training, 2410.04779 | weight-scaling init 10× speedup — **compute-budget lever for G3; caution: init is a treatment variable in our design** |
| 60 | FiRe, 2606.29414 | NTK preconditioning of periodic INRs — PO-9 tooling |

## G. 2026 field pulse (context; scoop-watch)

Weight-space output Jan–Jul 2026 ≈ 30 relevant papers — field is hot (R5 risk medium). Watched but
not yet integrated: WeightCLIP 2607.03551 (dataset-aligned WS latents), Render-Don't-Decode
2605.06298 (INR-state world models — render-vs-decode framing adjacent to RQ6 vocabulary, different
problem), position paper 2605.18632 (WS as generative modality), WARP 2607.01686 (training-data
recovery from weights), task-restricted RNN symmetries 2606.18457, curvature-via-symmetries
2606.00442, atlas position 2503.10633, LMC-at-scale 2606.23607, mode-connectivity-via-symmetry
2505.23681, INR signal-processing survey 2604.15047, NNiT width-agnostic aligned generation
2603.00180, Text2Weight 2508.13633.
