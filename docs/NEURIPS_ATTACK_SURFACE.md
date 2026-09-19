# Reviewer attack surface and bottlenecks (NeurIPS target)

Opened 2026-09-11. One row per objection a competent OpenReview reviewer is likely to raise, with
severity, the evidence it rests on, and what was done. Status is updated in place; nothing is
deleted, so the file doubles as a record of what the paper looked like before the pass.

Severity: **S** = likely to sink the paper (desk reject, or a reject-level concern shared by two
reviewers); **H** = costs a score point with most reviewers; **M** = one reviewer raises it; **L** =
polish.

## 0. What the paper claims (for orientation)

1. The function-preserving group of a sine MLP layer is $D_\infty \wr S_n$ (containment, all depths).
2. It is generically *exhaustive* at $L=1$ (Fourier atoms) and at $L=2$ (Jacobi-Anger, Bohr lattice,
   decay exponent, Bessel-tail independence). Consequences: no continuous canonicalization; a
   complete invariant carries exactly the function's information.
3. A pre-registered ladder on 1.8M fitted SIRENs: exact reframing recovers 63/66/32% of the
   shared-vs-random gap; inexact treatments recover <=13%.
4. Orbit intervention: randomizing only the group on a shared-init corpus costs 79.1 of 80.4 points;
   sign flips carry ~63, permutations ~15, windings ~1.
5. W12, a phasor-graded reader exactly invariant on raw parameters: s = 0.917 at matched capacity;
   3rd/3rd/2nd among weight-only readers on the standard INR benchmarks.
6. Function access (64 learned probes) beats every weight pipeline at ~100x less inference compute.

## 1. The theorem (what it says, how it is proved)

Depth two, $f(x) = \sum_j v_j\sin(\sum_i W_{ji}\sin(\langle w_i,x\rangle+b_i)+c_j)+\beta$.
Jacobi-Anger makes $f$ almost periodic on $\mathbb{R}^m$ with frequencies $\Omega(k)=\sum_ik_iw_i$,
$k\in\mathbb{Z}^{n_1}$, and coefficient $T(k)$ equal to one real part of a Bessel sum selected by the
parity of $\sum_ik_i$. Step 1: the frequency module is a lattice with basis $(w_i)$, so two
realizations differ by $U\in GL(n_1,\mathbb{Z})$. Step 2: along $m\,\Omega(u)$ the coefficient decays
like $e^{-|u|_1 m\log m}$, so $|u|_1$ is a property of $f$; an $\ell^1$ isometry of
$\mathbb{Z}^{n_1}$ is a signed permutation, which pins layer 1 up to $D_\infty\wr S_{n_1}$ (bias mod
$\pi$ from the phase of $T(e_i)$). Step 3: on the all-even grid and on one odd family per coordinate,
Bessel tails at distinct arguments are linearly independent, which matches layer-2 rows, recovers
magnitudes, and collapses $n_1n_2$ signs to one per row. Output bias last.

Full argument: `docs/THINKING/proof-memos/PO-2-deep-proof-v2.md`; checks:
`scripts/72_l2_proof_checks.py`.

## 2. Attack points

| # | Sev | Objection (as a reviewer would write it) | Evidence | Status |
|---|---|---|---|---|
| A1 | **S** | *Body exceeds the page limit.* | Baseline build: Conclusion + Limitations run onto p.10; limit is 9 (ICLR and NeurIPS). | open: must cut ~0.4 page before any addition |
| A2 | **S** | *"The L=2 proof has gaps."* v1 argued on one line $x=tv$ and never returned to the vectors $w_i$; the genericity condition referred to the line, not to $\theta$; neuron labels were not linked across coordinates; the chosen freeze vanishes at zeros of $J_2$, not excluded; scalar output only although CIFAR corpora have $c=3$ and the corollary is applied to them. | Line-by-line audit 2026-09-11. | **fixed in v2** (proof memo + checks); independent referee run pending; paper appendix to be replaced |
| A3 | **S** | *Internal contradictions.* Appendix "Limitations in full" says "Identifiability is proved only at L=1 ... two open lemmas"; says uncertainty "resolves one level of five" while §4 says items+seeds were resampled; "Widths are 32-64" and "one width" while §7 reports width 128; W11a paragraph blames the phase component, which §7 prices at ~1 point; W12 attribution quotes superseded +0.337/+0.315. | grep of `sections/*.tex` | **fixed** (appendix limitations rewritten) |
| A4 | **S** | *The function-access result is ProbeGen's.* Kahana et al. (ICLR'25) already show probing beats weight-space readers on the same MNIST/FMNIST-INR benchmarks at 30-1000x fewer FLOPs. The paper never cites it in the body; its bib entry has the wrong authors (lists "Horwitz, Kahana, Hoshen"; actual: Kahana, Horwitz, Shuval, Hoshen). | ProbeGen Table 3: 98.4 / 87.7 (128 probes). | open -> cite, position S5 as a controlled pricing on corpora with a known gap plus the $Kc\ll P$ condition |
| A5 | **H** | *Leaderboard is selective.* Table omits NG-GNN with probe features (94.7/74.2), NG-T (97.3/74.8) and ProbeGen (98.4/87.7). "Third of seven" reads as cherry-picked. | ScaleGMN paper §5 text; ProbeGen Table 3. | open -> add rows, marked as using function queries; this *supports* §8 |
| A6 | **H** | *Missing identifiability prior art.* Vlačić & Bölcskei (Adv. Math. 2021) treat deep networks with arbitrary affine symmetries via the null-net theorem; sine admits the 3-neuron null net $\sin z+\sin(z+2\pi/3)+\sin(z+4\pi/3)=0$, so their full-generality identifiability fails for sine and genericity is necessary. Also Rolnick & Kording 2020, Bona-Pellissier et al. 2023 (ReLU). | Paper cites Hecht-Nielsen, Sussmann, Fefferman only. | open -> cite and position |
| A7 | **H** | *Genericity is measure-theoretic, complement dense; every float network violates (A1).* | Inherent to the rational-independence route. | stated in v2 + appendix limitation; rebuttal: statements are almost-sure under any parameter density, which is what the information corollary needs |
| A8 | **H** | *Paper argues with its earlier versions.* "An earlier version of this paper claimed...", "overturns this paper's earlier claim", "is withdrawn" (5 places in the body). Confusing for reviewers and hints at a public prior version (anonymity). | intro l.55, results l.39, orbit l.33, reader l.47, conclusion l.20. | open -> rewrite as direct claims |
| A9 | **H** | *Significance.* The paper concludes weight access is dominated; why decompose a gap in a dominated paradigm, and one that strong equivariant readers (ScaleGMN 96.6 on MNIST-INR) already largely close? | §8 and bench table. | answer in framing: the contribution is the exact group + exhaustion theory and a causal measurement of what that group explains; the sign-flip dominance is actionable for architecture design; scope where weights win (expensive queries, run-level targets) |
| A10 | **H** | *Theory contribution buried.* The only genuinely new theorem (L=2) gets a 6-line paragraph; its proof ideas are not in the body. | §3. | open -> theorem + proof-idea paragraph in body (needs space from A1) |
| A11 | M | *Scale:* width 32 (one arm at 128), depth 2, 2-D images, laptop. | — | limitation stated; no cheap fix; width-128 arm is the answer |
| A12 | M | *Non-stationary fits.* | S8, S12 attempts | limitation stated; width-128 corpus interpolates (126 dB) |
| A13 | M | *W12 is not SOTA; CIFAR win over ScaleGMN unexplained.* | S14 killed the mechanism story. | report as is |
| A14 | M | *Proposition 6 / Corollary are folklore.* | — | already demoted to a consequence |
| A15 | M | *Clarity:* 15+ rung names (W1-W12, W10c, W11a/b, W12u/b/ub), calibration, addenda. | — | move nomenclature to one table; keep body to W1, W3, W5, W10, W12 |
| A16 | L | Typos: "is will be released"; "We keep it in the main text" for an appendix table. | appendix | **fixed** |

## 3. Bottlenecks, ranked

1. **Page budget (A1)** gates every other body fix. The body is over by roughly a third of a page
   and needs another half page for A4/A5/A6/A10.
2. **Proof credibility (A2, A7).** v2 closes the gaps found; it still needs a human reader. The
   memo's own warning stands.
3. **Positioning against probing (A4, A5).** Cheap to fix, expensive if a reviewer finds it first.
4. **Narrative hygiene (A3, A8, A15).** Cheap, high value.
5. **Significance framing (A9).** Needs rewriting of intro and conclusion, not new experiments.

## 3b. Review round 2 (2026-09-13): three simulated ICLR reviewers

R2 (empirical/code) and R3 (novelty/presentation) both rated **5 (marginally below), confidence 4**.
R1 (theory) pending. Merged, deduplicated, ranked by (score impact / cost). Status: **done**,
**running**, **decide** (author decision needed), **todo**.

| # | Sev | Finding | Source | Fix | Status |
|---|---|---|---|---|---|
| B1 | **S** | Benchmark INRs imported without absorbing SIREN's $\omega_0 = 30$ (scripts 62/64/65). Stored biases ~0.02, phasors degenerate (mean cos $b$ = 1.000); W12 moves 60x/41x/0.10 under the benchmark nets' true phase symmetry. Accuracies stand, "exactly invariant, runs unchanged" is false there. Verified independently. | R2 W1 | `scripts/73_absorb_omega_benchmark.py` builds `P-dws-bench-w0`; renders confirm canonical form (MNIST 0.96 nearest-mean vs 0.19); invariance re-audited on w0 corpora: logits move <=7e-6 at $|j|\le40$; W12 rerun via `74_bench_w0_chain.sh`. MNIST 93.51 (was 93.08). | **running** (FMNIST, CIFAR) |
| B2 | **S** | Anonymity: public repo `ITheClixs/project-siren-gap` repeats the title and claims; tracked `dist/arxiv/METADATA.txt` names author + ETH. ICLR allows preprints, but double-blind is effectively lost. | R3 11 | make repo private until 2026-12-16, or strip identifying files; provide anonymized supplement | **decide** |
| B3 | H | The group action itself is in ScaleGMN App. A.4.4 (depth-$L$, $b' = Qb + O\pi$) + Shamsian; "what is new is the group they close into" overclaims. | R3 1 | credit Thm po1 to them; claim only exhaustion (L=1, L=2) as the theory contribution; fix contribution 1, Remark, conclusion, provenance row T2 | todo |
| B4 | H | S6 attribution is order-dependent (nested arms); "sign 63 / winding 1 / perm 15" and "D_inf dominates 4:1" not identified; 79/80 is partly a floor effect; arm (i) was run before its prereg (disclosed in S6.md only). | R2 W3, W4 | $\sigma\times\tau\times\pi$ factorial with Shapley attribution + non-group scramble control (~2-3 h); disclose exposure in body | todo (needs prereg addendum) |
| B5 | H | Significance: the gap belongs to a symmetry-blind MLP; ScaleGMN-B reads the independent-init benchmark at 96.6; paper concludes weights are dominated. | R3 2 | reframe as symmetry attribution method + identifiability theory + design guidance; run ScaleGMN / NG-GNN under the orbit intervention (1-2 days) | todo |
| B6 | H | No figure or table in the 9-page body; ~96 decimals in prose; label zoo (W1..W12, W10c, W11a/b, W12u/b/ub; S1..S14 codes); f vs s mixed in headline. | R3 4, 5 | Fig. 1 (ladder + $\Delta_{sym}$ split) + 2 tables in body; move §4 (depth-two invariant encoding) out; descriptive names; strip study codes | todo |
| B7 | H | Paper argues with itself: "registered" x52, forecast misses narrated in body, self-calibration section. | R3 6 | pre-registration once in §5; forecasts to one appendix table; positive estimand statements | partly done (self-history removed) |
| B8 | H | Title question never answered. | R3 7 | answer: group sufficient for 79/80; exact-invariant reader closes 92%; residual 7.8 vs 5.4 function-side | todo |
| B9 | M | Leaderboard: Monomial-NFN (68.43/61.15/34.23) and quasi-equivariant Monomial-NFN missing; NFT/Inr2Array reports 98.5/79.3/63.4 on NFN-procedure corpora (verified); "±" is population SD; baselines are from ScaleGMN's Table 1, not "their papers". | R2, R3 9 | add rows with corpus-provenance note; say "quoted from ScaleGMN Table 1"; distances to ScaleGMN-B instead of ranks | todo |
| B10 | M | 5.4-pt function-side movement is one seed per protocol at K=16. | R2 W5 | 5 seeds at K=16/64/256 on P-shared-det (~15 min) | todo |
| B11 | M | Anchor rests on one $\theta_0$ draw. | R2 W7 | two more $\theta_0$ draws on the 10k subset: W1, W5, S6 B=0 (~2-3 h) | todo |
| B12 | M | W12's gain over W11a credited to quotienting; own 2x2 gives +0.291 to skeleton; W12ub is the fair comparator. | R2 W6 | reader.tex now also quotes +0.360 over W12ub | **done** |
| B13 | M | Unsupported/inaccurate statements: W12 "keeps the pattern at 14x params" (never run at 128); "at matched FLOPs"; "103x" (102.4); TOST p 2.5e-5 (is 5.9e-4 = max of one-sided p's); hier "2000 items" (10000); "both controls registered" (W12u was not); W12 table caption with superseded +0.337/+0.315; appendix "earlier draft of this paper", "we did not build" a raw-parameter reader, "L=1 theorem is the guarantee we have". | R2 W10/W16/W17, R3 A3/A8 | all corrected in both builds; generator caption in `50_score_s9.py` updated | **done** |
| B14 | M | ICLR AI-use statement inconsistent with paper: says App. S attributes the phasor idea component by component (no such row); says "editing" where CLAIMS row 62 says drafting and editing. | R3 10 | make the statement accurate (ICLR mandates it) | **decide** |
| B15 | M | Winding contribution depends on init convention: official SIREN init spans ~±3.4 bias periods, benchmark nets stay in one; "fitted SIRENs do not populate the winding directions" contradicted by own corpora. | R2 W2 | report bias spread per corpus; fix the sentence; optional small-bias-init arm (~5-6 h) | todo |
| B16 | L | Missing cites: Herrmann et al. 2024 (probing RNN weights), Albertini-Sontag-Maillot 1993, Kůrková-Kainen 1994, ProbeX, DEEP-ALIGN; ethics statement wording; "Apple M4 laptop" in body; abstract 252 words. | R3 12 | text | todo |
| B17 | L | Stats minor: bootstrap over 5 seeds anti-conservative; S6 single group draw; S10-S14 missing from PREDICTION_LEDGER; coverage denominators inconsistent (74/98 vs 55/75 vs 49/63); FLOPs tied to one decoder template. | R2 W8/W10/W11 | text + small reruns | todo |

### R1 (theory), rating 6, confidence 4: "nothing fatal; every theorem stated is proved as stated"

| # | Sev | Finding | Fix | Status |
|---|---|---|---|---|
| C1 | H | Thm deep was two-sided (both parameters generic); headline reads as standard identifiability. | One-sided theorem: $\theta$ generic, $\theta'$ arbitrary; $n_1'\ge n_1$, then $n_2'\ge n_2$, equal widths give the orbit. Body statement, proof idea and appendix proof rewritten; memo §8b. R1 re-check: "correct, no gaps"; its three wording repairs (λ'' may vanish; Remark only for $G$-invariant readers; provenance-table po5 cell; "at any width" in the appendix restatement) applied. | **done** |
| C2 | H | Vlačić–Bölcskei misread: the three-phase null network is reducible in their sense, not a counterexample; their framework already expresses affine symmetries. | related.tex and appendix Scope rewritten: framework expresses maps like ours, null-net condition for sine open. | **done** |
| C3 | H | po6/Cor undefined ("complete"), unproved, oracle is not the 64-probe reader, "computational" should include statistics. | completeness defined; po6 stated as factorization + Bayes-risk equality with proof; corollary demoted to Remark with the caveats. | **done** |
| C4 | M | "Global rather than at ties" overstated vs Dym et al.; bias fold is discontinuous only on a null set. | body and app:po5 now say the source is the $\Z$ factor; po5 stated at any width with one-line proof. | **done** |
| C5 | M | "(A1) excludes a closed set when $n_1\le m$" false. | corrected: nowhere dense, not closed. | **done** |
| C6 | L | Conjecture stronger than L=2 theorem. | "off a countable union of proper analytic subsets". | **done** |
| C7 | L | Necessary strata incomplete. | layer-two parallel rows and zero columns added. | **done** |
| C8 | L | Prop w12 imprecise (per-node sign matrix; degree-two channels; both layers; canonical form). | statement tightened, canonical-form hypothesis added (ties to B1). | **done** |
| C9 | L | Thm po1 is a one-line verification; "outside every linear action" loose. | now a Proposition; "monomial in no embedding (a bias shift is a linear transvection in homogeneous coordinates)"; maximality points to both theorems. | **done** |
| C10 | L | Routing counterexample needs bounded biases; can be made invariant. | stated with $c=\tau_{My}\circ c_0$ and $M$ above the corpus bias range. | **done** |
| C11 | M | Say what Thm deep buys the paper. | one sentence in body: certifies $\Dinf\wr S_n$ is exactly the nuisance the intervention randomizes and the reader quotients; non-quantitative. | **done** |
| C12 | L | CLAIMS row 7 said "resolving Tran Rmk 4.5"; that remark concerns ReLU/tanh. Related work said Tran "leave the general sine case open". | CLAIMS row 7 and both paper sentences softened. | **done** |
| C13 | M | Optional: lattice identifiability at every depth ($n_1$ minimal; first layers agree up to $GL(n_1,\Z)$), ~90% feasible in hours. | not started | todo |

### Further round-2 fixes (2026-09-13)

- **B6 partly done:** Figure 1 (the ladder, `fig1_ladder.pdf`) now in the body; §4 "Exact
  invariants at depth two" moved into Appendix `app:w10enc` to pay for it. Body still ends on p. 9.
  Still open: a body table for the orbit intervention, plain rung names.
- **B7 partly done:** forecast narration removed from §5, §6 and §7 of the body.
- **B8 done:** abstract now says the group is sufficient to produce nearly the whole gap without
  mediating the same share of the natural one.
- **R2 W12 done:** genericity sentence corrected on every corpus (median smallest first-layer angle
  about 2e-3 rad; below float32 resolution for at least 1 INR in 20), body and appendix.
- **B9 prepared:** `scripts/63_bench_table.py` now reads the omega-absorbed cells, adds
  Monomial-NFN (own protocol, marked), notes Inr2Array on differently built corpora, says baselines
  are from ScaleGMN's Table 1 and that ± is population s.d.; regenerate when the rerun finishes.

### Round-2 experiments (2026-09-13), all pre-registered before running

| Study | What it answers | Result | Paper change |
|---|---|---|---|
| S15 (MNIST) | order-free attribution of the 79-point intervention (B4) | Shapley: permutation 44.2, sign 29.9, $\pi$ shift 4.3, winding 0.4; 3 draws within 0.24; non-group scramble 1.1. 3/7 intervals, P-S15-B false | "sign flips carry ~63" and "$D_\infty$ dominates $S_n$ 4:1" removed everywhere; orbit §, intro, conclusion, reader § rewritten; appendix `app:s15s17` |
| S16 | function-side movement with seeds (B10) | $D_{16}=6.44$, $D_{64}=-0.25$ [-0.34,-0.14], $D_{256}=0.03$ | residual now "not fit quality either" at $K=64$; function § updated |
| S17 | dependence on the draw of $\theta_0$ (B11) | W1 88.50/84.61/88.46; $f(c_{align})$ 0.502/0.475/0.561; $\Delta$ 77.6/70.2/75.6; P-S17-A false (range 0.086) | body sentence + appendix table |
| B1 rerun | $\omega_0$-corrected leaderboard | W12 93.51 / 74.81 / 44.20: 3rd/3rd/1st among weight-only | leaderboard table, reader §, abstract ("leads on one") |
| S15 add. 01 | same factorial on FashionMNIST and CIFAR-10 | FashionMNIST 39.1/20.5/4.9/0.4 of 64.9; CIFAR-10 10.3/10.6/5.1/0.7 of 26.8; 6/6 intervals, P-S15-D and P-S15-E false | per-dataset split in orbit §, appendix table, README |
| S18 | W12 at width 128 | $s = 0.881$ (85.32%) vs alignment 0.677; shared-det arm running | width-128 paragraph; residual pending |
| L items | review polish | ProbeX, DEEP-ALIGN, Herrmann, Albertini-Sontag-Maillot, Kůrková-Kainen cited; bootstrap and FLOPs caveats; abstract shortened; calibration table now 121 intervals / 90 hits (74%), matching the ledger; S6 calibration sentence corrected for S15 | done |

## 4. Log

- 2026-09-11: audit; v2 proof and checks written; appendix limitations de-staled; baseline page
  count measured (body ends on p.10).
- 2026-09-11/12: A1 fixed (body back within 9 pages after compressing §4, §5, §7, §8, §10 and the
  conclusion). A2 fixed and strengthened after re-examination: v1's (G2) and (G7) proved
  unnecessary, so $\Thgentwo$ is five conditions with a single rational-independence condition, whose
  complement is dense only when $n_1 > m$; paper appendix replaced by v2, body theorem now states
  vector outputs, uniqueness and the proof idea. A4, A5, A6 fixed (ProbeGen cited with correct
  authors and positioned in §1 and §8; bench table gains NG-GNN/NG-T with probes and ProbeGen as a
  separate "also querying" block; Vlačić–Bölcskei cited with the null-net contrast; Jacobi–Anger
  expansion credited to Yüce et al. and Novello). A8 fixed (no self-history in the body). Five
  uncited bibliography entries removed. Checks: `scripts/72_l2_proof_checks.py` all pass, including
  the general-ray counterexample and the (A7)-free bias recovery.
- 2026-09-12: A10 addressed (theorem and three-step proof idea in §3 body); A9 partly (intro now a
  four-item contribution list; conclusion names the regime where weight access could win). Open:
  A15 (rung nomenclature), an independent human read of the proof, the NeurIPS-format port, and
  the AI-use statement wording (author's decision; see chat of 2026-09-12).
