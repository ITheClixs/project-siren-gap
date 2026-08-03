# Provenance of every component

**Purpose.** An external review asked, correctly, that the boundary between what this program
inherits and what it adds be stated component by component rather than left to a related-work
section at the end. This file is that ledger. It is maintained alongside `RELATED_WORK.md`: that
file surveys the field, this one adjudicates ownership.

**Rule.** Nothing in this table may be described in the paper as ours if this file says otherwise,
and every "claimed new" row must name the search that failed to find a prior source.

**Last novelty scan:** 2026-08-03 (G8, eight targeted arXiv queries; snapshot in
`docs/lit_snapshots/G8-novelty-scan.txt`). Earlier scans: G0 2026-07-16, G2.

---

## 1. Theory

| # | Component | Status | Closest prior work | What is ours | Evidence in the paper |
|---|---|---|---|---|---|
| T1 | The maps $\sigma:(w,b,u)\mapsto(-w,-b,-u)$ and $\rho: b\mapsto b+\pi,\ u\mapsto -u$ preserve a sine network's function | **prior work** | Shamsian et al. 2024 (2402.04081) use both as SIREN weight-space augmentations; Tran et al. 2024 (2409.11697) fold the sign symmetry of $\sin$ into a monomial-matrix framework | nothing — cited at first mention in the abstract, in contribution 1, and again at Def. 1 | Def. 1 |
| T2 | These generate $D_\infty$ per neuron and $D_\infty\wr S_n$ per layer; normal form $g_{d,j}$ and its composition law | **ours (closure and framing)** | Tran et al. classify *linear* (monomial-matrix) symmetries and leave the general sine case open. No source we found states the closure or names the group | the identification of the generated group, the normal form, and the observation that the phase generators are **affine** and therefore outside every monomial-matrix action | Lemma 2, Thm 3, Remark after Thm 3 |
| T3 | Generic identifiability at $L=1$: functional equality forces $\theta' = g\theta$ for a unique $g \in D_\infty \wr S_n$ | **ours** | The classical identifiability line: Hecht-Nielsen 1990 (permutation/sign for odd activations), Sussmann 1992 (minimal tanh), Kůrková–Kainen 1994, Fefferman 1994 (sigmoid networks from pole structure) | the periodic-activation analogue, by Fourier atoms, with an explicit and measurable generic stratum $\Theta_{\mathrm{gen}}$ rather than an asymptotic condition; the phase component has no counterpart in any of those groups | Thm 5, proof in App. |
| T4 | No continuous exact canonicalizer exists | **specialization of prior work** | Dym, Lawrence & Siegel 2024 (2402.16077) prove the general topological obstruction and motivate weighted frames | an explicit one-neuron loop for the sine action, which shows the obstruction is *global* — it meets every loop winding the bias circle — rather than confined to sorting-key ties as in the permutation case. **Not an independent discovery**, and the paper says so at the proposition | Prop. 7 and the paragraph after it |
| T5 | A complete $G$-invariant factors through the realized function | **corollary, not a contribution** | Quotient-factorization folklore; adjacent to the weight-space universality line (2602.01083) | only the specialization to the orbit equivalence proved in T3. Demoted in the paper from a headline contribution to a consequence of Thm 5, with its $L=2$ scope flagged | Prop. 8, Cor. 9, Remark 10 (scope) |
| T6 | Exact cross-layer invariants at $L=2$ by coupling through the second-layer Gram $G_2 = W_2^\top W_2$ | **claimed new** | Per-neuron monomial invariants in NFN / Monomial-NFN; general invariant-theoretic pooling. The failure mode this repairs — a hidden neuron's outgoing vector being acted on by the *next* layer's group — is stated in Monomial-NFN but not repaired for the affine phase group | the matrices $A,B,C$, the parity bookkeeping that makes them sign-cancelling, and their spectra as invariants of the full product group | Prop. 11; T10 asserts invariance to $3\times10^{-7}$ relative; T15 supplies the matched non-invariant control |

**Search behind the "claimed new" row T6.** Eight arXiv queries on 2026-08-03: `abs:"Gram matrix" AND
abs:"weight space" AND cat:cs.LG` (0 hits), `all:"cross-layer invariant" AND cat:cs.LG` (0),
`abs:"infinite dihedral" AND cat:cs.LG` (0), `abs:"periodic activation" AND abs:"symmetry"` (0),
`abs:"canonicalization" AND abs:"implicit neural representation"` (0), `abs:"weight space" AND
abs:"invariant" AND abs:"eigenvalue"` (1 unrelated), `abs:"SIREN" AND abs:"weight space"` (2, both
already tracked), `abs:"invariant" AND abs:"sine" AND abs:"neural network" AND abs:"phase"` (1
unrelated). A null result from keyword search is weak evidence and is reported as such; it is not a
proof of novelty.

## 2. Method

| # | Component | Status | Closest prior work | What is ours |
|---|---|---|---|---|
| M1 | Aligning two networks by a permutation, for merging or mode connectivity | **prior work** | Git Re-Basin (Ainsworth et al. 2023), Entezari et al. 2022 | nothing |
| M2 | $c_{\mathrm{align}}$: alignment to a *fixed reference network* over the full $D_\infty\wr S_n$ | **ours (extension)** | M1 supplies the assignment idea; Kuhn's algorithm supplies the solver | the exact per-neuron $D_\infty$ minimizer — the cost depends on the winding only through its parity, so four cases give the exact optimum over an infinite group — inside the assignment, and the finding that any fixed reference network works, including an unrelated random one |
| M3 | Weight-space augmentation by group elements | **prior work** | Shamsian et al. 2024 | nothing; rung W6 measures it |
| M4 | Frame averaging | **prior work** | Puny et al. 2022 | nothing; rung W9 measures one instantiation of it, and the paper states that a different winding distribution or normalization could behave differently |
| M5 | Permutation-equivariant weight-space readers | **prior work** | DWSNets, NFN/NFT, GMN, ScaleGMN, UNF | nothing; W11a is a member of that family, deliberately not the strongest, and the limitations section says so |
| M6 | Function-query classification of an INR | **prior work** | ProbeGen (Horwitz et al. 2024) and the probe line | the FLOPs-matched frontier against weight access, and the amortization pricing |

## 3. Experiment

| # | Component | Status | Closest prior work | What is ours |
|---|---|---|---|---|
| E1 | The shared-vs-independent initialization discrepancy on INR corpora | **prior observation** | inr2vec (De Luigi et al. 2023) relies on shared init; Papa et al. 2024 intervene on fit hyperparameters and report downstream deltas | we replicate it seed-for-seed as a pre-registered anchor and then decompose it |
| E2 | The decomposition ladder (W1–W10, X1) | **ours** | no prior source applies one frozen decoder across exact, inexact and invariant treatments of the same corpora | the design, the pre-registration, and the recovery fraction $f_A$ |
| E3 | The orbit-only intervention (S6) | **ours**, prompted by external review | none known | holding each fitted network and its function fixed while randomizing the group, so the measured degradation has one cause |
| E4 | The matched non-invariant control (S7) | **ours**, prompted by external review | none known | a feature map matched to W10 in monomial degree, trigonometric order, pooling and dimension, differing only in whether the $D_\infty$ component is quotiented out |
| E5 | Pre-registration with interval forecasts and public scoring | **practice imported** | Munafò et al. 2017, Nosek et al. 2018; scoring rules from Gneiting & Raftery 2007 | applying it to a methods study in this field, and reporting the misses |

## 4. Corrections already forced by this ledger

- The claim that Prop. 12 certified a causal lower bound on the symmetry share was withdrawn; the
  quantity is now named an *algorithm-relative recoverable fraction* throughout.
- The CIFAR-10 headline no longer quotes the W10 encoding's $0.534$ as a reframing result; the
  strongest orbit-valued reframing there is $f(\mathrm{W5}) = 0.325$.
- Prop. 8 was demoted from a headline contribution to a consequence of Thm 5.
- Prop. 7 is presented as the sine instance of Dym et al.'s obstruction, not as an independent
  impossibility theorem.
- The 72/28 pooling-versus-incompleteness split was removed as unidentified.
