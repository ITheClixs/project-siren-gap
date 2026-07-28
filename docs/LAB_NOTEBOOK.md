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

---

## 2026-07-17/18 — Session 4: G3 (INR-Bench sine: MNIST complete, anchors PASSED)

**Gate status: G3 IN PROGRESS** (MNIST done + anchors done; FMNIST chain queued detached;
CIFAR pending pilot + fallback decision).

**Pilot → freeze:** MNIST sine config frozen at **L=2, w=32, steps=300, lr 1e−3**. Task-referenced
gates on official test split: P-shared-det (−0.07 pts, renders *beat* pixels), P-random (−0.07),
P-shared-stoch (−0.06) — all PASS. PSNR medians 39.2/37.5/32.2 dB. FMNIST pilot also passes at
identical config (gap 0.05) — frozen same.

**Replication anchors (binding criteria fixed pre-data):**
- **A1 PASSED: W1−W3 = +80.43 pts** [80.17, 80.69]. Raw random-init weights are *chance-level*
  for linear probe (10.1) and kNN (10.5); matched MLP reaches only 13.9. Shared-det weights:
  94.4 MLP / 88.9 linear / 84.1 kNN — near-linear decodability, consistent with PO-9 laziness.
  The perception gap in its rawest form: ~80 accuracy points destroyed by init nuisance alone
  (both protocols fully deterministic given seeds).
- **A2 PASSED: W6−W3 = +4.35 pts** [3.46, 5.25] with the *bounded* aug family — helps where
  Shamsian's unbounded SIREN-bias hurt; recovers only ~5% of the gap (headroom for W4/W5).

**Prediction scoring (protocol §0.1.2):**
- QG-1 (steps): predicted 500, 80% int {300..1000}; observed 300 — interval HIT (boundary),
  abs err 200.
- QG-2 (PSNR): predicted 33 [28,40]; observed 39.2/37.5 — HIT, abs err ~5.
- QG-3 (A1 gap): predicted 30 [12,45]; observed 80.4 — **interval MISS**, abs err 50.
  *Miscalibration memo:* I anchored on Papa's graph-net relative gains and under-weighted my own
  PO-9 reasoning (lazy shared-det ⇒ near-linear code ⇒ ceiling-level W1) and the fact that a
  plain MLP gets no permutation structure for W3 (⇒ floor-level). The thesis's own mechanism
  predicts extremes; I predicted the middle. Lesson recorded: when a registered mechanism makes a
  directional extreme prediction, do not hedge toward literature baselines from a different
  reader class. Running coverage: 2/3 intervals.
- Instrument note: Wilcoxon p=.0625 at n=5 is the two-sided floor — report alongside t per §0.5.

**Infra findings:** harness-tracked background tasks were killed twice (likely machine sleep ~
midnight); switched to detached nohup+caffeinate chains with shard-resume — survived. Sustained
throttle 55→48 fits/s (R7 measured). P-shared-stoch generation is ~2.5× faster (256-coord
minibatches). Leakage: corr(PSNR,label) = 0.203 (det) / 0.017 (random) / 0.061 (stoch) —
PSNR-matched control path pre-committed in S1 prereg draft.

**Running detached:** P-random-K MNIST (440k fits), then FMNIST 4-protocol chain (queued,
auto-starts). **Compute:** ~5 h this session (chains + decoders).

**Next:** K + FMNIST complete → gates on both; CIFAR pilot (expect steps retune + 20k/4k fallback
decision per burn-down); then G3 exit review + S1 prereg freeze (power memo from anchor σ≈0.2–0.6).

---

## 2026-07-27/28 — Session 5: G3 close-out (CIFAR) + G4 (S1 ladder) + G2 leftovers

**Gate status: G3 pilot decision made; G4 (S1) DECODED on MNIST.** The dissertation's spine
figure (F9) exists: `results/ladder/mnist/F9_waterfall.png`.

### Two integrity bugs found and fixed (both could have corrupted results silently)

1. **`src/sirengap/data/` was never committed.** The `.gitignore` rule `data/` (unanchored) matched
   the source package as well as the corpus directory, so `images.py` and `schema.py` — every
   loader and the shard schema — existed only on this machine. A fresh clone could not generate a
   single INR. Rule anchored to `/data/`; the two files are now tracked.
2. **Partial downloads were treated as complete.** `_download` skipped whenever the destination
   existed, so a transfer still in flight was handed to `tarfile`, which extracted the readable
   prefix: CIFAR-10 arrived with `data_batch_1` missing and `data_batch_2` truncated to 4.4 MB of
   31 MB. The pilot failed loudly only because the *first* batch was the missing one — had it been
   the last, generation would have fitted 2000 INRs to a silently partial dataset. Now: download to
   `.part`, verify md5, atomic rename; extract to staging, verify every member, then move into
   place. Regression test T11 (4 cases).

### Compute ledger correction: two clocks, and they differ by 10×

Detached chains keep running across machine sleep, so a shard's recorded wall-clock absorbs the
sleep interval. Measured from corpus metadata: **11.1 h active compute** (median s/fit × fits) for
all 1.31 M MNIST+FMNIST INRs, against **123.4 h wall-clock**. Budget accounting now uses active
hours (~12 h cumulative of 350); wall-clock is tracked separately for scheduling. Median 0.03 s/fit
at the MNIST config = ~33 fits/s, better than the 15 fits/s G1 projection.

### G2 advisor leftover cleared: basin census under the production optimizer

`scripts/13_microcosm_optimizers.py` repeats the PO-8 census on the *full* model (u, c trained,
not profiled out) with Adam and plain GD, 900 inits per cell.

- **The headline replicates.** Global-capture fraction at init ranges 2/5/10/20: 0.00/0.20/0.56/0.31
  (Adam), 0.00/0.18/0.58/0.33 (GD), against Nelder–Mead's 0.00/0.26/0.62/0.32. The non-monotone
  three-regime shape and the sweet spot at init range ≈ ω survive the optimizer change.
- **One sub-claim was an artifact.** "100 % degenerate-ridge capture at ±2" is Nelder–Mead on the
  profiled surface: ridge capture is 0.00 under both gradient methods, and those runs are *still
  descending* (endpoint ‖∇‖ far above tolerance) rather than sitting in a w≈0 basin. The ±20
  spurious-sidelobe fraction is optimizer-dependent too (0.48 Adam, 0.00 GD). CLAIMS row 11 is
  corrected by row 12 rather than edited. An `unconverged` class had to be added after the first
  pass assigned those endpoints to "spurious basins" — which would have been a false claim about
  the landscape.
- **Laziness, measured.** At the frozen corpus setting (Adam, lr 1e−3, 300 steps) every init ends
  unconverged, endpoint ‖∇‖ ≈ 0.5–0.7, and median |Δw| ≈ 0.24 *independently of init range*: the
  fit never leaves its initialization neighbourhood. That is the microcosm-level mechanism for the
  80-point W1−W3 gap (CLAIMS row 13).

### S1 pre-registration frozen (hash 8c029cf43f01a94c) + addendum 01

Power memo (`scripts/10_power_memo.py`) sized the seeds from measured anchor variance instead of
the protocol default. Paired-difference SD is **0.210 pts** for rungs with a fixed feature matrix
and **0.721 pts** for rungs that re-draw features each step; sizing on each SD's upper 80%
confidence limit (0.327 / 1.123) gives at n=5: MDE 0.55 pt and TOST power ≈ 1.00 for the first
class, but MDE 1.89 pts and TOST power **0.20** for the second. So W6 and W8 run at **15 seeds**,
everything else at 5.

Addendum 01 resolved four definitions before any cell was computed: W5's template is θ₀ (the shared
init — data-independent, and the frame W1 lives in); W8 is canonicalize-*then*-augment, because
augment-then-canonicalize is identically W4 once `c_sort` is exact and would measure nothing; W9's
frame is drawn once per seed and shared across INRs; and W10 became an **exact L=2 invariant
encoding** instead of the L=1-scoped fallback.

### New mathematics: deep phase-invariant encoding (W10, Ch3.6)

`canon/invariants.py` refuses L ≥ 2 because a hidden neuron's outgoing u is acted on by the next
layer's group. The missing coupling for L=2 is the layer-2 Gram **G = W₂ᵀW₂**, which is invariant
under the *entire* layer-2 group (row sign flips cancel inside each product, row permutations
reindex the sum, phase shifts touch b₂ and W₃ only) while picking up ε_i ε_l under layer 1, with
ε_i = (−1)^{d_i+j_i}. Since `sin b_i` carries exactly ε_i and `cos b_i · w_i` carries ε_i after
contraction, the matrices A = (sin b_i sin b_l)G, B = (cos b_i w_i · cos b_l w_l)G and
C = (sin 2b_i w_i · sin 2b_l w_l) are sign-cancelling and transform as M ↦ P M Pᵀ — so their sorted
eigenvalue spectra are invariant under the full group. Per-neuron even scalars are emitted as order
statistics under a shared invariant sort key; layer 2 uses PO-4 contracted over the layer-1 index.
D = 320 for the frozen corpora. T10 measures the residual move under random group elements
(windings ≤ 3, non-trivial permutations) at **≈ 3 × 10⁻⁷ relative** — fp32 round-off, not tolerance
slack — with controls for non-constancy and wrong depth.

### S1 ladder — results (MNIST, matched MLP)

| rung | P0 | P1 | W1 | W2 | W3 | W4 | W5 | W6 | W7 | W7-1/8 | W8 | W9 | W10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| acc | 97.97 | 97.59 | 94.36 | 95.04 | 13.92 | 28.19 | **64.41** | 18.12 | 17.75 | 14.59 | 10.27 | 14.13 | 35.54 |
| f = (·−W3)/(W1−W3) | — | — | 1 | — | 0 | .177 | **.628** | .054 | .048 | — | <0 | .003 | .269 |

Label-shuffle controls collapse (10.65 / 11.03 / 9.79 at W1 / W3 / W5). W1 reproduces anchor A1
seed-for-seed. Scoring against the frozen registration:

- **H-S1-1 HIT.** P1 − P0 = −0.39; TOST equivalent at margin 1.0 (p 2.5e−05). The fit destroys no
  class information — this kills "it's all decoder inadequacy" (defense row 4 now answered).
- **H-S1-2 HIT.** W1 − P1 = −3.23 (registered −3.8).
- **H-S1-3 MISS.** W1 − W2 = **−0.68**, registered +2.0 [0, 6]. Optimization noise is not a nuisance
  at all; stochastic fitting is marginally *better*. The rung is null and the entire gap is
  init/basin. (Mechanism, in hindsight: minibatch coordinate sampling is a mild regularizer, and
  under laziness both protocols stay in the same basin anyway.)
- **H-S1-4a HIT** to the decimal. W1 − W3 = +80.43 (registered 80.4).
- **H-S1-4b HIT.** f(W4) = 0.177 (registered 0.06, interval to 0.20).
- **H-S1-4c BIG MISS.** f(W5) = **0.628** (registered 0.10, interval to 0.30).
- **H-S1-5 MISS.** W7 − W6 = −0.52, Holm p = .21 — because *neither* intervention works
  (f = .048, .054). Of W7's small gain, +3.16 pts is explained by row count alone (W7 vs W7-1/8).
- **H-S1-6 HIT.** W10 = 35.54 sits inside [W4, W5].

Two unregistered results that matter: **W8 collapses to chance (10.27)** — augmenting inside the
canonical frame destroys the frame the decoder just gained — and **X1 shows total brittleness**: a
W1-trained reader scores 10.7 on W3 features, 13.2 in reverse.

### What the c_align result does to the thesis

Registered position was that canonicalization recovers little. It recovers **63 %**. The honest
restatement, which is a *better* thesis: the perception gap decomposes as ≈ 63 % symmetry-orbit
scatter (removable by an exact, function-preserving change of frame) + ≈ 37 % residual that no
canonicalization can touch — while the *template-free* canonicalizer, the one available when no
shared init exists, recovers only 18 %. The 18 → 63 span is canonicalizer quality, not information.
Defense row 8's pre-registered falsification conjunction lost its third conjunct (W7−W3 ≫ W6−W3),
so the basin claim is now carried by the residual and by S4 dispersion, and is stated at that
strength. Defense row 15's weakest link is now the template: θ₀ exists only because the corpus was
built with a known shared init. Sensitivity check (exploratory) is running.

**Calibration lesson (repeat of the QG-3 lesson, same direction).** Both big misses under-trusted a
registered mechanism relative to literature priors from a different reader class. Where I trusted
the mechanism the intervals hit, including ±80.4 to the decimal. Running interval coverage: 8/12.

**Deviations:** none. **Waivers:** none. **Compute this session:** ~1.5 h active (ladder ~0.9 h,
microcosm census ~0.2 h, CIFAR pilot in flight).
