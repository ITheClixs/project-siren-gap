# Lab Notebook — PROJECT SIREN-GAP

Chronological, append-only. One entry per work session. Claims → CLAIMS.md; predictions →
PREDICTION_LEDGER.csv; deviations → docs/waivers/.

---

## 2026-07-16 — Session 1: G0 (literature deep scan, novelty gates, theory scoping)

**Gate status:** G0 **PASSED** (both novelty gates green, provisional on 5 close-reads due G1).

**Done:**
- Repo skeleton per protocol §0.4; git initialized (first commit this session); scan tooling
  committed (`scripts/00_lit_scan.sh`, `scripts/parse_atom.py`).
- Literature deep scan: 18 arXiv API queries + 3 targeted follow-ups; 36 Appendix-D IDs verified
  (34 correct; Expand-and-Cluster corrected to 2304.12794; Phuong & Lampert has no arXiv version —
  OpenReview, marked unverified); 26 abstracts close-scanned. Snapshot: `docs/lit_snapshots/`.
- Novelty gates decided (docs/THINKING/G0-novelty-gates.md): **Gate-1 PASS** — nobody exploits the
  τ/ρ phase part of the sine symmetry group (Monomial-NFN covers σ only, and its maximality theorem
  is scoped to linear actions; our τ/ρ are affine). **Gate-2 PASS** — nuisance premise is stated in
  the literature (Shamsian) and fit-hyperparameter effects are studied (Papa), but no causal
  decomposition into optimization/symmetry/basin components exists.
- Theory scoping memo (docs/THINKING/G0-theory-scoping.md): all 12 POs scoped with strategies,
  timeboxes, downgrade paths.
- Registered predictions P-A, P-B, P-C (S3 orderings) and P-D (direction) in the prediction ledger
  — before any infrastructure exists.

**Findings worth recording:**
1. **The protocol's own PO-4 example invariants are wrong.** cos 2b·(w⊗u) fails ρ; sin 2b·(w⊗u)
   fails σ. Full parity classification worked out in the scoping memo §2 (the protocol's ⚠️ THINK
   marker anticipated this). Consequence: T6 and the W10 front-end must be built from the corrected
   table, and A.2's sign algebra gets the memo table as its reference.
2. **2026 scoop-risk is real:** ≈ 30 relevant weight-space papers in 6 months. Two 2026 papers from
   Appendix D exist and are important: 2605.08281 (bias column = causal readout route; geometry ↛
   reader accuracy in meta-learned regime — adversarial to S2, must be named in S2 prereg) and
   2601.23181 (IFT, hypernetwork regime). New-to-protocol finds: 2602.01083 (expressivity theory —
   PO-6 novelty check), 2604.14037/2605.18319 (complete ReLU symmetry classifications — technique
   templates), 2606.04754 (neuron identifiability — PO-10 definitions), CertMix 2607.04123
   (shared-anchor SIREN linearity — supports PO-9).
3. **Decision:** meta-learned-init methods (MWT line) are a different access model (they change the
   data-generating process); they enter papers as labeled reference points, never as ladder rungs.
4. **Decision:** sine phase canonicalization interacts with the *known* class-signal route (biases,
   per 2605.08281) — noted as a candidate mechanism hook for S1/S2 interpretation; do not let this
   bias analysis (it is a hypothesis, not a result).

**Deviations:** none. **Waivers:** none. **Compute:** ~0 h (API calls only).

**Next (G1):**
1. Close-read queue in order (novelty-gates memo §Consequences); write the 8+4 delta memos.
2. Property tests T1–T9 (write before science code) + `forward_canonical` single source of truth.
3. Throughput profiling script (fits/sec MPS vs CPU, batched-SIREN A.1); COMPUTE_LEDGER v1.
4. CI (GitHub Actions, CPU test suite).
5. Re-run scan with updated s13/s14 windows; diff.

---

## 2026-07-17 — Session 2: G1 (close-reads, T1–T9, profiling)

**Gate status: G1 PASSED.** Acceptance: T1–T9 green on MPS+CPU (49 tests, local run; CI covers CPU
only — gate satisfied by the local run per ADVISOR_REVIEWS/G1.md Systems 2); profiling + burn-down
v1 in COMPUTE_LEDGER; kill-risk close-reads done *before* any fitter/science code (G0 review
condition honored).

**Close-read findings (memos in docs/THINKING/close-reads/):**
1. **Monomial-NFN Remark 4.5 leaves sine functional maximality explicitly open** — PO-1/PO-2
   resolve it; their maximality is scoped to matrix subgroups of GL(n), τ/ρ are affine.
2. **Shamsian publish both sine symmetry ops as augmentations** ("SIREN negation" = σ, "SIREN
   bias" = ρ/τ with unbounded k) — **and SIREN-bias augmentation badly hurts** (their Table 3:
   DWS 4.69 vs ≈18 no-aug). Gate-1 verdict revised, not flipped: ops known as failed augmentation;
   classification + exact canonicalization + full-group equivariance + causal decomposition remain
   ours, now with published motivation for exactness-over-averaging. W6 design: bounded k only.
3. 2605.08281 is explicitly *not* an independently-fitted-zoo study (their words) — Gate-2 stands;
   S2 prereg must carry their "routed" alternative; bias route (rank 2–5, causal) noted as hook.
4. 2602.01083 = ReLU/perm *approximation* universality — PO-6 (exact, identifiability-based, D∞)
   survives; cite as counterpart. 2604.23720's Def 2.2 adopted as PO-2's statement form.
5. Surveys: sine phase symmetry absent (only a one-line Shamsian-aug citation) — tripwire closed.
6. Papa: RQ2 already hypothesizes overtraining hurts downstream — C-11 delta re-scoped to
   *mechanism* (chaos rates, post-alignment dispersion, basin counts), not the bare phenomenon.

**Engineering:**
- Core library: params/forward (canonical form, single source of truth), D∞≀Sₙ action in compact
  (d, j, perm) parameterization, c_sort, c_align (+ diagnostics), Hungarian/Sinkhorn/greedy/brute,
  phase-invariant encoding (corrected parity classes; refuses L≥2 per OPEN_PROBLEMS #4), minimal
  invariant-DeepSets layer, batched fitter (A.1; ω₀ absorbed at save; sum-loss for exact per-INR
  Adam), shard IO + schema validation.
- **T1–T9 all green, CPU + MPS (49 tests).** T6 includes an executable negative control: the
  protocol's cos(2b)·(w⊗u) demonstrably breaks under ρ. T4 verifies the convention lock
  (internal forward ≡ canonical forward after absorption).
- **T9: MPS refit determinism gap = 0.0** (torch 2.13.0, B=2/w16/150 steps) — P-shared-det viable
  on MPS; risk R1 downgraded; re-verify at production config before G3 (advisor condition).
- Throughput (M4): MNIST-config ~15 fits/s, CIFAR-config ~2.1 fits/s on MPS; flat in batch size
  beyond B=64; MPS/CPU ≈ 1.5×. Corpus projections: fallback path ≈ 94 h fitting, full-CIFAR ≈
  156 h — both inside budget; decision at G3 pilot. Lockfile committed (torch 2.13.0, numpy 2.5.1).

**Deviations:** none. **Waivers:** none. **Scan:** G0 scan is 1 day old; per-gate re-scan next
falls at G2 (s13/s14 windows to be bumped then) — logged as a judgment call, not a skip.

**Next (G2):** theory sprint per scoping memo §11 order — PO-1 write-up, PO-4 generation/separation
lemma + tie-stress tests (advisor G1 Theorist 2), PO-5, PO-3 (incl. u=0), PO-2 L=1 proof, PO-8
microcosm (profiled 2D loss surfaces, F4), PO-6 + novelty paragraph, PO-2-deep timebox (2 days,
Jacobi–Anger), PO-7 formal incl. FINER parity derivation; canonicalizer docstrings gain the
"exact w.r.t. implemented group, conditional on PO-2" qualifier; IFT close-read (2601.23181).

---

## 2026-07-17 — Session 3: G2 (theory sprint)

**Gate status: G2 PASSED** (App. E: PO-1/3/4/5 written ✓; PO-2 attempted-with-memo — exceeded:
**L=1 PROVED**, deep case downgraded with roadmap ✓; PO-8 solved ✓). Artifacts:
paper/thesis/ch1-symmetry.tex, ch2-fitmap.tex, proof-memos/PO-2-deep-attempt.md,
results/microcosm/ (F4 drafts), CLAIMS rows 3–11.

**Theory results:**
1. **PO-1 theorem** with the compact normal form g_{d,j}: (w,b,u) ↦ ((−1)^d w, (−1)^d b + πj,
   (−1)^{d+j} u), composition (d₁⊕d₂, j₂+(−1)^{d₂}j₁) ⇒ D∞ per neuron; wreath per layer;
   cross-layer commutation.
2. **PO-2 L=1 THEOREM**: on Θ_gen, functional equality ⟹ unique group relation. Final proof via
   distributional Fourier atoms (supports {±wᵢ}∪{0}, coefficients (uᵢ/2i)e^{±ibᵢ}) — resolves
   the sine case of Monomial-NFN's Remark 4.5 open question, in the maximal-symmetry-group sense
   of 2604.23720 Def 2.2.
3. **Two self-caught errors, same-session repairs (kept as thesis remarks, not sanitized):**
   (a) duplicate stratum needs only *parallel frequencies* (phasor addition merges any-bias
   parallel neurons) — Θ_gen and PO-3 corrected, and the exclusion is provably necessary;
   (b) first PO-2 draft's "hyperplane complement is connected" is false — replaced by the FT
   argument, which is simpler and stronger.
4. **PO-4 separation proposition** proved (Φ = (w⊗w, cos2b, sin2b·w, sinb·u, cosb·(w⊗u))
   separates orbits on {w≠0,u≠0}); PO-5 impossibility via an elementary τ-winding loop proof;
   PO-6 factorization + informational-equivalence corollary, positioned against 2602.01083;
   PO-7 classification incl. **FINER = ℤ₂ (odd activation; no bias-shift symmetries)** — NOT
   near-trivial as the protocol expected.
5. **Prediction amendments (pre-data, hashed):** P-A-v2, P-B-v2, P-C-v2 move FINER into the
   tanh/Gaussian class. Original rows retained for calibration accounting.
6. **PO-8 microcosm solved**: closed-form profiled loss (quadrature-certified to 6e−16); zero set
   = D∞ orbit exactly; 20 minima in the fundamental domain (1 global + 19 sidelobe); **basin
   census is non-monotone in init range with three regimes** — ±2: 100% degenerate-ridge capture
   (orbit never reached), ±10: 62% global (sweet spot ≈ ω), ±20: 51% spurious-sidelobe capture.
   Bug found by the certification itself: the naive G⁻¹ profile blows up on the w≈0 ridge;
   pseudo-inverse semantics fixed it. Caveat logged: census is Nelder-Mead-based; Adam/GD variant
   owed before F4 ships (advisor G2 Empiricist 1).
7. **PO-2-deep**: honest attempt memo — Jacobi–Anger ⇒ Bessel-CP reduction, two open lemmas
   (Bessel-Vandermonde generic rank, truncation control); numeric support shows rank holds but
   conditioning collapses by n=5 (min|det| ~ 4e−13) ⇒ S4e falsification hunt is the right next
   investment, not more proof effort.

**Chores:** tie-stress tests added (57 tests green CPU+MPS); canonicalizer scope qualifiers;
IFT 2601.23181 close-read (generated-regime complement; laziness connection noted for Ch2);
G2 re-scan clean (4 new arXiv titles, none relevant — snapshot committed).

**Deviations:** none. **Waivers:** none. **Compute:** ~0.2 h (microcosm + tests).

**Next (G3):** INR-Bench sine — pilot fits on real MNIST (quality-gate tuning: width/steps),
T9 re-verification at production config, strata audit implementation (geometry/), full corpus
generation plan + overnight queue, replication anchors A1/A2, fp32 storage, leakage audits;
plus G2 advisor queue: tectonic compile + stable labels, census optimizer variant, S4-prereg
item for the init-range prediction, scan-diff tooling.
