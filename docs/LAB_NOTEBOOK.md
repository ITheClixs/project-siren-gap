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

### Session 5 addendum — CIFAR pilot scored, calibration accounting corrected

**CIFAR pilot sweep (all four registered configs).** Every config that ran passed the
strengthened 10-epoch gate: w32 L2 s1000 (gap **+0.25**, 40.1 dB), w64 L3 s500 (+0.05, 68.1 dB),
w64 L3 s1000 (−0.05, 75.9 dB). The pre-committed freeze rule takes the cheapest passing config, so
**CIFAR-10 is frozen at w32 L2, steps 1000, lr 1e−3** — the same architecture as MNIST/FMNIST,
which is worth more for cross-dataset comparability than the extra fidelity of the w64 arms.

The sign of the gate gap **tracks render fidelity**: positive (renders lose) at 40 dB, ≈0 at 68 dB,
negative (renders win, as on MNIST) at 76 dB. That is what the low-pass account predicts — smoothing
costs texture until the render is essentially exact, at which point the small denoising benefit
returns — and it is a stronger form of the QG-8 evidence than the single registered sign, though
the *mechanism* remains a conjecture (G3 review, Empiricist 2).

**Prediction scoring** (new file `docs/PREDICTION_OUTCOMES.csv`, one row per registered call):
QG-4 steps HIT (point exact, 1000); QG-5 PSNR **MISS** (registered 27 [22,32], observed 40.1 — I
under-rated the overparameterization: 8707 parameters against 3072 target values interpolates);
QG-7 gate-CNN accuracy HIT (67.45 in [60,72]); QG-4b probability call poorly resolved (P=0.15 on
config A passing; it passed, Brier 0.72); QG-8 well resolved (Brier 0.04).

**Correction to this session's earlier entry.** I wrote "running interval coverage 8/12"; the
correct count at that moment was 7/11, and with the CIFAR rows scored it is now **9/14 = 64%
coverage against nominal 80%** — i.e. the intervals are too narrow, not too wide. The five misses
are QG-3, QG-5, H-S1-3, H-S1-4c, H-S1-5. Three of them (QG-3, QG-5, H-S1-4c) are the same failure
mode: hedging a registered mechanism toward priors from a different setting. Two (H-S1-3, H-S1-5)
are a second, distinct mode: assuming a nuisance exists and then registering a contrast that could
not exist once the nuisance turned out to be null. Both modes are now named in defense row 14, and
the coverage number is reported rather than the successes alone.

**Correction to the addendum above, after config D landed.** All *four* registered configs pass
(w64 L3 s2000: gap 0.00, 67.3 dB), not the three I listed. And the fidelity claim needs weakening:
the gate gaps are +0.25 at 40.1 dB, then +0.05 / 0.00 / −0.05 at 68.1 / 67.3 / 75.9 dB. What the
data support is "the render penalty is real at 40 dB and has vanished into gate noise (±0.05) by
~68 dB", not a clean monotone trend in fidelity — the three high-fidelity configs are not ordered
by PSNR in their gaps.

Separately, **PSNR is non-monotone in steps** at w64 L3: 68.1 dB at 500 steps, 75.9 at 1000, 67.3
at 2000. Fitting longer made the fit *worse*, at fixed architecture and lr. That is a directly
relevant datum for Papa et al.'s overtraining question (RQ2) and it was obtained for free from a
config sweep, so it is exploratory, not registered — flagged for a proper steps-sweep at S2.

**R-CIFAR applied (2026-07-28).** Clean uncontended probe at the frozen config: median
**0.0821 s/fit = 12.18 fits/s**, derated ×0.87 → 10.59 fits/s. Projected full-path corpus
(540 000 fits, the corrected count) = **14.2 h** against the rule's 30 h threshold, so the
**full 50k/10k path is taken**; the fallback would have been 6.1 h. QG-6 registered P(full) = 0.35
→ scored, Brier 0.42. Corpus generation launched detached (`scripts/17_g4_chain.sh`, third stage).

*Infrastructure note worth remembering:* the decision job hung for ~15 minutes on a self-inflicted
deadlock — `pgrep -f "09_cifar_pilot"` matches **any shell whose command line contains that
string**, including the polling one-liner I had started to watch for the pilot's exit, so the job
waited on a phantom that was itself. Fixed in `16_cifar_corpus.sh` / `17_g4_chain.sh` by matching
the python process (`pgrep -fl ... | grep .venv/bin/python`) and documented in the scripts.

### W5 template sensitivity (EXPLORATORY, 2026-07-28) — the weakest link does not break

Defense row 15 named W5's template as the thesis's weakest link: θ₀ exists only because the corpus
was built with a known shared init, so a recovery fraction of 0.628 might have been an artifact of
privileged access rather than a statement about canonicalization. Five templates, same rung
otherwise (`scripts/15_w5_template_sensitivity.py`, exploratory watermark):

| template | test acc | linear probe | f = (W5−W3)/(W1−W3) |
|---|---|---|---|
| θ₀, the corpus's shared init (the registered W5) | 64.41 | 58.31 | 0.628 |
| an **unrelated** random init, seed 12345 | 65.44 | 60.09 | **0.640** |
| an unrelated random init, seed 777 | 62.24 | 56.44 | 0.601 |
| a fitted P-shared-det INR (image 0) | 58.29 | 50.83 | 0.552 |
| a fitted P-random INR (image 0) | 54.76 | 47.79 | 0.508 |

**The result holds under every template, and the shared init is not special** — an unrelated random
draw does marginally *better* (0.640 vs 0.628). What alignment buys is a consistent frame, and any
fixed reference network supplies one; knowing the corpus's initialization is not required. This
removes the caveat that would otherwise have shrunk the practical claim to c_sort's 0.177.

Second, unregistered observation: **fitted INRs make worse templates than random inits** (0.51–0.55
vs 0.60–0.64). Conjecture — a fitted network's neurons are specialized to its own image, so
activation matching against it imports that image's structure as bias, while an untrained init is
generic. Untested; it suggests a cheap improvement (align to a *random* fixed net, never to a
corpus member) and belongs in Ch3 as a design note, not as a finding.

Third: every template clears f > 0.5, so the falsification condition written into the frozen
registration (§8.3) is met unambiguously rather than at a knife edge. The rewrite of the
canonicalization claim stands on all five arms, not on one.

### S1 ladder replication on FashionMNIST (2026-07-28)

| rung | P0 | P1 | W1 | W2 | W3 | W4 | W5 | W6 | W7 | W7-1/8 | W8 | W9 | W10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FMNIST | 89.62 | 89.44 | 82.97 | 83.67 | 12.66 | 24.61 | 59.34 | 14.86 | 15.05 | 13.20 | 10.20 | 12.12 | 42.77 |

**Recovery fractions replicate the MNIST structure almost exactly** (FMNIST vs MNIST):
f(W4) 0.170 vs 0.177 · **f(W5) 0.664 vs 0.628** · f(W6) 0.032 vs 0.054 · f(W7) 0.034 vs 0.048 ·
f(W9) −0.008 vs 0.003 · f(W10) **0.428 vs 0.269**. W8 sits at chance on both (10.20 / 10.27), and
the null optimization-noise result is near-identical: W1 − W2 = −0.70 (MNIST −0.68).

So every qualitative claim from the MNIST ladder survives a dataset change: template alignment
recovers about two thirds, template-free sorting about one sixth, augmentation and marginalization
about a thirtieth, frame averaging nothing, and augmenting inside the canonical frame destroys it.
The one substantive difference is W10, where the exact invariants do markedly better on FashionMNIST
(0.428) than on MNIST (0.269) — plausibly because garment silhouettes are more separable from
coarse spectral statistics than digit strokes are, but that is a conjecture and is written as one.

**Scoring note (important, and a category error avoided).** The analysis script scores every dataset
against the same registered intervals, so it printed H-S1-4a as a MISS on FMNIST (+70.31 against
[79, 82]). That is not a real miss: the interval was registered as a *MNIST* magnitude, calibrated
on the MNIST anchor, and FMNIST's ceiling is 8 points lower to begin with (P0 89.6 vs 98.0). Only
the dataset-agnostic hypotheses transfer — the sign and nullity results (H-S1-3, H-S1-5), the
bracket claim (H-S1-6, HIT), and the recovery fractions. **The FMNIST rows are therefore recorded as
a replication of structure and are NOT added to the calibration ledger**; adding them would inflate
the denominator with predictions that were never made about this dataset.

---

## G5 — the CIFAR-10 arm (2026-07-29)

### Where the last session ended, and what had finished overnight

The G4 chain (`scripts/17_g4_chain.sh`) completed at **05:44:16** on 2026-07-29 after ~15.4 h of
detached wall-clock. All four CIFAR-10 protocols generated at the R-CIFAR-frozen config
(w32 L2, 1000 steps, lr 1e−3): 60 000 INRs each for `P-random`, `P-shared-det`, `P-shared-stoch`
and 360 000 for `P-random-K`. All three test-split gates pass — render-vs-real-pixel gaps
**+0.25 / +0.30 / +0.84** pts at median PSNR **40.3 / 40.1 / 34.9** dB. Corpus total across the
program is now **1 840 000 fitted INRs**.

### The registration came first (and this time it counts)

The FashionMNIST arm had been scored against MNIST-calibrated magnitudes and duly printed a fake
MISS; the write-up caught it, and the rows were kept out of the calibration ledger. Doing that
twice would be a habit. So `docs/prereg/S1-cifar.md` (**sha256-16 `f7906fc6904c7c81`**) was frozen
*before any CIFAR cell existed* with **17 interval rows H-C1-1…17 and 3 probability calls
P-C1-A/B/C of its own**, and those rows **do** enter the ledger.

Two discipline notes recorded in the file itself so the freeze can be audited rather than trusted:
an accuracy-producing invocation of `11_ladder.py --dataset cifar10 --rungs P0 W3` was started and
**killed before it wrote any cell** (`results/ladder/cifar10/` was empty at freeze), and the
instrument check that replaced it (`scripts/19_ladder_shapecheck.py`) prints only shape, row
alignment, dtype and finiteness — no decoder, no accuracy.

The category error is now blocked by the **instrument** rather than by prose:
`14_ladder_analysis.py` carries a per-dataset registration table, and an arm with none of its own
prints *not scored* instead of a verdict. (It also had a latent crash — it globbed its own
`S1_analysis.json` output as if it were a rung cell, so any second run died with `KeyError: 'rung'`.)

### Results

| rung | P0 | P1 | W1 | W2 | W3 | W4 | W5 | W10 | W6 | W7 | W9 | W8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CIFAR-10 | 55.81 | 56.23 | 44.29 | 45.19 | 12.64 | 16.05 | 22.92 | 29.54 | 16.57 | — | 12.68 | 10.53 |

**The headline is that the two-thirds law is not a law.** f(W5) = **0.325** against a registered
0.62 [0.42, 0.78] — a clean MISS, and only just clear of the pre-committed falsifier at 0.30. The
grayscale figure was a property of grayscale corpora.

**And the two exact methods cross over.** f(W10) = **0.534** > f(W5) = 0.325, reversing MNIST
(0.269 < 0.628) and FashionMNIST (0.428 < 0.664). This was *called in advance*: P-C1-B registered
P = 0.60 that f(W10)\_CIFAR > f(W10)\_MNIST on the grounds that with c = 3 output channels each
neuron's outgoing uᵢ ∈ ℝ³ carries strictly more D∞-visible structure than the c = 1 case (the
encoding grows D 320 → 384). That resolved correctly; the companion call P-C1-A, that the
grayscale ordering would persist (P = 0.65), resolved incorrectly, and the crossover is why. It
also breaks H-C1-17: W10 leaves the [W4, W5] bracket **upward**, by 6.6 pts.

**The obvious confound is dead, cheaply.** CIFAR was fitted for 1000 steps against 300 for the
grayscale corpora, so "the fits travelled further from θ₀ and are harder to align back" is the
first thing to check. `scripts/23_fit_travel.py` measures it with **no new fitting**: median
relative travel ‖θ_T − θ₀‖/‖θ₀‖ = **0.186 / 0.191 / 0.187** and median layer-1 direction cosine to
init = **0.998 / 0.999 / 0.999** (MNIST / FMNIST / CIFAR). Indistinguishable, and CIFAR is if
anything the least moved. The extra 700 steps bought no extra displacement. PO-9 laziness is
equally present everywhere; the f(W5) drop is not a fit-length artifact. *I had been about to
spend ~5 h generating an MNIST-at-1000-steps corpus to test this; a 3-minute measurement on
existing data answered it.*

**Everything else replicates.** P1 ≈ P0 (in fact P1 is marginally *better*, +0.42 — the mild
denoising a 40 dB render provides). W1 − W2 = −0.90: optimization noise is null on a third
dataset. X1 is at chance both ways (10.4 forward, 12.0 reverse). W8 collapses to 10.53. W9
recovers 0.001.

### A logical overreach of my own, corrected

I had written — here, in DEFENSE row 8, and in the first draft of the paper — that the residual
after c_align is "provably not symmetry, because c_align preserves the function exactly". **That is
wrong**, and it is the strong form of a claim whose weak form is right. c_align is an *exact* group
element but a *heuristic* choice of representative: two parameter vectors in the same orbit can be
sent to different representatives, so a better exact reframing can only raise f. What is proved is
the other direction:

> **f is a certified lower bound on the symmetry-attributable share**, because an exact reframing
> creates no information about the signal; the complement is an *upper* bound on the non-symmetry
> share, not a proof of it.

Our own history is the cautionary example: improving the frame once, c_sort → c_align, dropped the
"residual" from 82% to 37%. The paper now states this as a proposition (§6) with the caveat that
W10 needs a further one — it is a *nonlinear* invariant feature map, so part of its gain may be
feature engineering rather than group removal, and its 0.534 is not a certificate of the same
strength as c_align's 0.325.

### Deviations, waivers, compute

**Deviations:** none. **Waivers:** none. **Compute this session:** CIFAR ladder ~1.2 h active on
MPS; travel diagnostic ~3 min; figures/tables/paper build ~2 min. Test suite green (69 tests).

---

## S4e — the deep-identifiability hunt (2026-07-29/30)

DEFENSE row 15 has named PO-2's deep case as the weakest theoretical link since G4: the
identifiability theorem is proved at L=1 while every empirical result in the program is L=2. S4e is
the pre-committed empirical attack, registered in `docs/prereg/S4e.md`
(**sha256-16 `aa5426a4245bd22f`**) with the falsification criterion, the void conditions and the
permitted conclusions all fixed before the run.

### New instrument: an exact minimiser over G

`c_align` is a *heuristic* choice of orbit representative, so its residual cannot answer "is there
**any** g ∈ G bringing these two together?". `src/sirengap/canon/refine.py` answers that directly,
and the minimisation is **exact per layer** given the others fixed:

- the per-neuron D∞ cost ‖(−1)^d w_i − w*_t‖² + ((−1)^d b_i + πj − b*_t)² + ‖(−1)^{d+j} u_i − u*_t‖²
  depends on j **only through its parity**, so four (d, parity) cases give the exact minimum over
  the whole *infinite* group;
- the permutation is then a Hungarian assignment on those per-pair minima;
- layers interact only through the shared W_{l+1}, so they are swept by coordinate descent.

T12 (13 cases) is the control that makes the hunt non-vacuous: planted elements are undone to
< 1e−6 relative at L = 1, 2, 3, with c = 3, at windings up to 12. One bug found: `canon.assign.
hungarian` **maximises a score** and was being handed a cost.

### Two process failures, both caught by the apparatus rather than by luck

**The void condition fired.** The first confirmatory launch returned planted max R_θ = 0.22 at
w = 2 — plain coordinate descent stalls in a joint local optimum on ~10% of width-2 pairs, which is
invisible at the n = 16 pilot scale and fatal at n = 128. Prereg §4 declares that void. Diagnosed
(best of 20 restarts → residual *exactly* 0, so the group search was never the limitation), fixed by
restarts, re-verified at 0/128 failures every width. **That run's numbers are not used.**

**The stopping rule fired.** The second launch was killed at **6 h 55 m** against a 3 h budget.
Cause: I estimated ~5 ms/step, the truth is 96 ms/step at n = 128, w = 32 — and the cost is
dominated by **batch size, not width** (w = 8 at n = 128 costs more than w = 32 at n = 32), which I
had backwards. Benchmarked rather than re-guessed: MPS is 3–4× faster than CPU here, so the device
was never the problem. Deviations D1/D2 appended to the prereg; frozen text untouched; no numbers
from the killed runs used.

### Results (confirmatory, `results/s4e/s4e.json`)

| w | planted R_θ | basin (ε=1e−5) | κ | best R_f | R_θ there | unrelated R_θ |
|---|---|---|---|---|---|---|
| 2 | 4.3e−08 | 78% | 0.0422 | **5.9e−08** | **1.2e−07** | 0.468 |
| 4 | 3.7e−08 | 91% | 0.0351 | 1.6e−02 | 0.319 | 0.451 |
| 8 | 3.0e−08 | 91% | 0.0198 | 1.1e−02 | 0.475 | 0.368 |
| 16 | 3.1e−08 | 44% | 0.0146 | 7.9e−03 | 0.353 | 0.292 |
| 32 | 3.3e−08 | **0%** | 0.0055 | 1.2e−03 | 0.334 | 0.233 |

**κ falls with width** (0.042 → 0.0055): the forward map θ ↦ f is strongly *expansive*, so local
recovery is **well** conditioned. This is the opposite of the naive reading of the proof memo's
Bessel–Vandermonde collapse, and the distinction has to be kept — that determinant governs the
*global* recovery system, not the local Jacobian. Registered directional claim (b) **holds**;
claim (a), monotone basin decay, **fails** (78% at w=2 < 91% at w=4).

**The basin's volume collapses, not its depth.** Started *inside* the basin, 78–91% of runs return
at w ≤ 8, 44% at w = 16, none at w = 32 — where the optimiser walks from R_θ = 1e−5 out to 1.3e−1
while the function barely improves. A budget control at **5× the step count** changes nothing
(`scripts/28_s4e_budget_control.sh`: 0% at w=32, identical R_f floor), so this is not
under-training. Meanwhile *independent* restarts find the basin only at w = 2. Those two arms
together are what license the specific claim; neither alone would.

**One student recovered its teacher exactly.** At w = 2, 1 of 128 students reached
R_f = 5.9e−08 and aligned to R_θ = 1.2e−07 — float32 relative epsilon, max per-coordinate relative
disagreement 2.8e−06, i.e. **6–7 significant figures**, and 2.6e−07 of the unrelated-network scale.
Direct positive evidence for Conjecture 6.5.

**Production arm.** Same-image K-fit pairs at w = 32: R_θ = 0.279 after optimal alignment.
Different-image pairs: 0.280. Difference **−0.001**. Modulo the entire group, two fits of the same
image are no closer than two fits of different images — the W1-vs-W3 gap seen from parameter space
instead of through a decoder. (Their R_f is 0.73, so these pairs are nowhere near function-equal;
the arm characterises the corpus, it does not test the conjecture.)

### The registered criterion fired, and it was wrong to

Read literally, the w = 2 student meets §4: R_f < 1e−5 **and** R_θ = 1.2e−07 > 20κR_f = 5.0e−08. It
is a **false positive**, and the criterion is defective in two independent ways:

1. **Ratio-only, no absolute floor.** As R_f → machine epsilon, 20κR_f falls *below* the smallest
   residual a float32 aligner can represent (1.19e−07). Any *exact* recovery fires it.
2. **κ is the wrong null for an optimiser residual.** κ was measured on *random* perturbation
   directions; a minimiser's residual lies in the **flattest** directions of the loss, exactly where
   R_f is least sensitive to R_θ. So R_θ/R_f > κ is *expected* for any converged minimiser (2.10
   observed against 0.042), and the ratio cannot separate "different configuration" from "same
   orbit, found by descent".

A ratio against the planted control does not repair it either: for a *single* INR the planted pair
aligns to exactly 0.0 (the 4.3e−08 in §5 is a max over 128 INRs), so that ratio divides by zero. My
first verification script did exactly that and printed "GENUINE COUNTEREXAMPLE" — a bug in the
script, caught before it reached any document. Adjudication has to be **absolute**
(`scripts/29_s4e_verify_candidate.py`).

Amendment **A1** adds R_θ > 1e−3, is marked **post-hoc**, leaves §4 as frozen, and P-S4e-C is still
scored against the criterion **as written** (fired ⇒ observed 1, Brier 0.7225). Quietly editing §4
would have been the one unrecoverable move here.

### Verdict and calibration

**Conjecture 6.5 survives** — one width's direct positive evidence, no counterexample. But
identifiability at L = 2 has **no empirical content at production width**: the configuration that
would witness it is unreachable, and the optimiser leaves the true orbit even when placed on it. The
remaining route is analytic, not empirical, which retires S4e as the answer to DEFENSE row 15 and
hands the question back to the two open lemmas.

**7/9 intervals hit.** Both misses are one event: P-S4e-5 (best R_f, registered 3e−4 [1e−6, 3e−3],
observed 5.9e−08) and P-S4e-6 (R_θ there, registered 0.35 [0.05, 0.65], observed 1.2e−07), because
the n = 32 pilot that informed them never sampled the global basin while the n = 128 run did. Those
rows were flagged `pilot-informed` in the ledger *before* the run for exactly this risk. Program
coverage is now **30/40 = 75%** (grayscale 9/14, CIFAR 14/17, S4e 7/9).

**A third calibration failure mode, new this session:** a registered *criterion* — not a point
prediction — can be under-specified in a way only data reveals. Checking a criterion against its
instrument's resolution at registration time is now part of the template.

**Deviations:** D1, D2, A1, all logged in the prereg. **Waivers:** none. **Compute:** ~7 h wasted on
the killed run, ~40 min for the confirmatory run, ~40 min for the budget control, ~5 min for the
candidate verification.

---

## Luminance-CIFAR arm — a conjecture of ours, tested and withdrawn (2026-07-30)

The paper's two headline cross-dataset findings — alignment recovery halving on CIFAR, and the
crossover where the exact invariant encoding overtakes it — were confounded across three axes:
image statistics, output-channel count, and fit budget. Axis 3 was already dead (measured travel
0.186/0.191/0.187). This arm kills axis 2 by fitting **luminance CIFAR-10**: the identical images at
the identical geometry, architecture and 1000-step budget, with c = 3 → 1.

It was registered (`docs/prereg/S1-gray.md`, **b84b660829aa6d40**) specifically to test a mechanism
**this program had already put in print** — §9 of the paper conjectured that c_align matches on
layer-1 activations, a statistic blind to the outgoing structure that grows with c, so alignment
should recover at c = 1. Two probability calls carried it, both registered below even odds
(P-G-A = 0.35, P-G-B = 0.45), and §5 pre-committed the wording for every outcome including the one
that withdraws the conjecture.

### Result: the conjecture is wrong

| corpus | c | PSNR | P0 | gap W1−W3 | f(W5) | f(W10) |
|---|---|---|---|---|---|---|
| MNIST | 1 | 39.2 dB | 97.97 | 80.43 | **0.628** | 0.269 |
| FashionMNIST | 1 | 43.4 dB | 89.62 | 70.31 | **0.664** | 0.428 |
| **CIFAR-10 luminance** | **1** | **59.8 dB** | 47.50 | 27.62 | **0.324** | **0.493** |
| CIFAR-10 RGB | 3 | 40.1 dB | 55.81 | 31.64 | **0.324** | **0.534** |

f(W5) at c = 1 is **0.324** against **0.324** at c = 3 on the same images — identical to three
decimals. The crossover does not reverse (f(W10) = 0.493 > f(W5)). **Luminance CIFAR behaves like
RGB CIFAR, not like the grayscale corpora.** Both probability calls resolved against me; **9/10
intervals hit** (the miss is f(W4) = 0.058 against a lower edge of 0.06 — by 0.002).

Per §5 the conjecture is **withdrawn from the paper, not softened**. That is done.

### The arm also kills the alternative explanation it owed

§5 required that, if the result had come out the other way, I state the rival reading — "the fit is
simply easier at c = 1, so alignment works better". It came out this way instead, and the same
number now refutes that rival directly: dropping channels makes the fit **over-parameterised**
(1185 params to 1024 targets) and takes median PSNR from 40 dB to **59.8 dB**. So this corpus is
fitted *far more accurately* than MNIST's 39.2 dB and still aligns *far worse*. **Higher fidelity
does not buy alignment recovery.**

### What I will and will not say now

Eliminated: fit length, output-channel count, render fidelity. What remains is image statistics.
**I am deliberately not offering a replacement mechanism.** The one I offered was specific,
motivated by the encoding's algebra, and false. Supplying a second story on the same evidence,
immediately after the first was falsified, is how a programme talks itself into a narrative. The
paper reports the eliminations and names the experiment that would identify the cause.

### Two deviations, logged rather than hidden

**D1 — the gates ran after the ladder.** `30_cifar_gray_corpus.sh` called `04_quality_gate.py` with
`--corpus/--split/--epochs` where it takes `--dir/--eval-split/--gate-epochs`, so both gate calls
died on argparse and printed `GATE FAILED` — a *script* failure, not a gate rejection. The chained
ladder checked only that the corpus log said "complete", so it ran on ungated corpora. Gates were
then run by hand and **both pass** (gaps +0.04 and −0.01 pts, PSNR 59.8/60.0, leakage ≈ 0), so the
corpora were admissible throughout and the cells stand. Recorded anyway, because the ordering was
wrong even though the outcome was not; and noted that while diagnosing I saw the log tail including
f_W9 and f_W10 before the gates were confirmed. A gate result is a property of the corpus and cannot
be influenced by having seen ladder numbers, so there is no selection risk — but the chain now gates
on a passing gate JSON rather than on the corpus log.

**D2 — the frozen prereg overstated its own pre-freeze exposure.** §2 said the corpora and gates
existed before the file was written, copied from the `S1-cifar.md` template. False: the file was
frozen at 12:52, four minutes after generation *started* and four hours before it finished. The
error runs conservative — it declares more exposure than occurred — but a declaration of exposure
has to be true, so it is corrected in the deviation log.

**Compute:** 3.9 h generation (120k fits, throttling from 12.2 to 9.1 fits/s), ~15 min ladder,
~10 min gates.

---

## Rung W11 — reader architecture against frame choice (2026-08-02)

The ladder's design — change the feature map, freeze the reader — is what makes it interpretable
and is also its most exposed flank: the field does not read weights with a plain MLP. W11 supplies
the comparison, pre-registered at **e3bbc081a5810956** with §5 fixing the wording for every outcome
including the one that withdraws the paper's practical claim.

**Two readers** (`src/sirengap/models/readers.py`, T13 with 12 cases):
- **W11a** bipartite message passing on raw weights, S_n-equivariant, **not** D∞-invariant. That
  negative property is asserted by test, because it is exactly the coverage the DWSNets/NFN/GMN
  family has for sine networks — the phase generators are affine and outside every monomial-matrix
  action. If W11a had accidentally become D∞-invariant it would have stopped answering the question.
- **W11b** W10's *own* invariants with **learned** equivariant pooling in place of sorted
  eigenvalue spectra.

**Capacity matched by rule, not by tuning:** widths chosen as the closest parameter count to the
frozen decoder's 1,873,162 → 424 (W11a, +0.4%) and 288 (W11b, −1.5%). No row loses for being smaller.

### Results (MNIST, P-random, 5 seeds) — 5/5 intervals hit

| rung | construction | reader | acc | f |
|---|---|---|---|---|
| W4 | c_sort | matched MLP | 28.19 | 0.177 |
| **W11a** | perm-equivariant, raw weights | graph (1.88M) | **35.26** | **0.265** |
| W10 | exact invariants, eigenvalue pooling | matched MLP | 35.54 | 0.269 |
| **W11b** | same invariants, learned pooling | graph (1.85M) | **56.24** | **0.526** |
| W5 | c_align | matched MLP | 64.41 | 0.628 |

**The paper's practical claim survives its first real test.** W11a at 0.265 lands essentially on
W10 (0.269) and less than half of c_align (0.628). A capacity-matched equivariant reader on raw
weights does not substitute for choosing a good orbit representative. Bounded, and stated as
bounded: one construction, one capacity, one corpus — and the limitation naming W11a as a
*simplified* member of the family was committed **before** the number arrived, deliberately, so the
bound could not be drawn only when convenient.

**OPEN_PROBLEMS #4 is closed, with a split.** Keeping the invariants and changing only the pooling
takes f from 0.269 to 0.526. Of the 0.359 shortfall to c_align: **72% (0.257) was the eigenvalue
pooling, 28% (0.102) is the invariants' incompleteness.** The pooling was the binding constraint.

That reframes the practical recommendation: the useful object is a **G-invariant equivariant
reader** — no template, no assignment problem, within 0.10 of alignment. It also partly answers
DEFENSE row 7 (Hungarian is O(n³) at width 1024): the answer may be to stop aligning.

**Calibration.** 5/5 intervals; P-W11-A (frame choice beats reader architecture) and P-W11-B
(learned pooling beats eigenvalue spectra) both resolved as registered, P-W11-C (invariant reader
overtakes alignment) correctly registered low at 0.30 and did not happen. Program coverage is now
**44/55 = 80%**, exactly nominal for the first time.

**Deviations:** none. **Compute:** 36 min (W11a) + 49 min (W11b), inside the 3 h/variant rule.
An implementation note worth keeping: W11b's first form gated every feature channel per edge, a
[B,n,n,width] tensor that is ~800 MB at width 384; restructured to multi-relational messages,
[B,n,n,8], which is exactly as permutation-covariant and ~200× smaller.

---

## S5 — the FLOPs-matched adjudication (2026-08-02)

PO-6's corollary has been in the notebook since G2: a *complete* G-invariant of the weights carries
exactly the information of the realised function, so weight access can only win on **compute**. The
program has been asserting it on a proof. S5 measures it — registered at **80bdc96ce9497c3d**, and
deliberately **adversarial to the program's own subject matter**: P-S5-A predicted at 0.85 that
querying the function would beat every weight rung on *both* axes.

**Apparatus.** `eval/flops.py` is an analytic accounting (one MAC = 2 FLOPs, per INR, at inference)
rather than wall-clock, so the frontier reproduces off this throttling laptop and can be audited
line by line. `eval/probes.py` is function access with **learned** probe coordinates — the strong
form, which can only move the function frontier *up*, i.e. against our own thesis. T14's decisive
test: the probe reader is exactly G-invariant (gap < 1e-8) and the fitted INR receives **no
gradient**; a "function access" baseline that peeked at weights would have invalidated everything.

### Result: weight access is dominated on both axes, in both regimes

| access | accuracy | MFLOP/INR |
|---|---|---|
| function-query K=16 | 51.54 | 1.385 |
| **function-query K=64** | **95.34** | **1.594** |
| function-query K=256 | 98.23 | 2.430 |
| W5 c_align (best weight rung) | 64.41 | 5.447 |
| W11b equivariant invariant reader | 56.24 | 119.1 |
| *P0 real pixels, for reference* | *97.97* | *—* |

**K=64 beats c_align by 30.9 points at 3.4× fewer FLOPs.** K=256 (98.23) exceeds the real-pixel MLP
(97.97) — learned query points are a better input to the same decoder than the pixels are.

**Amortization, the one escape PO-6 left, closes.** Weight access costs 1.70 + 3.74·T MFLOP over T
tasks; function-query costs 1.59·T. The lines never cross, because the weight reader's *per-task*
cost on a 1185-dim input already exceeds function-query's *entire* per-task cost on a 64-dim one.
General form: reading P params into a decoder of width W costs ≈2PW, querying K points costs
≈2KcW + K·siren, so function access is cheaper whenever **Kc ≪ P** — a condition on probes needed,
not on INR size. That was registered as a prediction, not noticed afterwards.

**And the nuisance never arises.** Function-query moves 5.4 points between P-random and
P-shared-det (a fit-quality effect: 37.5 vs 39.2 dB), where weight access moves **80.4**. The entire
object this program decomposes is an artifact of choosing to read parameters.

### What this does to the thesis, stated as §5 required

On these corpora, at this scale, for targets that are functions of the represented signal,
**weight-space learning is the wrong tool** — and that applies to our own canonicalizers, invariant
encoding and equivariant reader as much as to anyone's. What survives: the theory (a correct, novel
account of the symmetry structure, independent of whether one should use the representation), the
decomposition (a measurement about that structure), and the scope conditions where the case would
have to be remade — expensive-to-query representations, or targets not identifiable from the
function. Defense row 2 is answered, and not in the program's favour.

### Calibration

**5/8 intervals.** All three misses are one shape error: the accuracy-vs-K curve is far more
sigmoidal than registered — near chance until K≈16, then saturating — so K=4 (16.4 vs [28,62]) and
K=16 (51.5 vs [62,90]) came in low while K=256 (98.2) overshot the top edge by 0.23. The
probability calls all resolved correctly, including P-S5-A, the one predicting that the program's
own subject matter would lose (Brier 0.0225), and P-S5-B, which correctly doubted that amortization
would save it (0.09).

Program coverage **49/63 = 78%**; 16 probability calls, mean Brier 0.215.

**Deviations:** none. **Compute:** ~12 min for the sweep plus both controls.

---

## 2026-08-03 — the external review: novelty positioning, provenance, and two new controls

The review that arrived after S5 raised ten priorities. Priority 1 (the invalid lower-bound claim)
and Priority 2 (an orbit-only intervention) were done in the previous session. This entry records
the rest, and one data-integrity problem found on the way.

### A file that overwrote another file

`results/s6/orbit_mnist.json` did not contain arm (i). Arm (ii) had been launched without `--tag`,
so it wrote to the untagged path and overwrote the permuted arm; only `run_perm.log` survived, with
means but no per-seed values. Arm (i) has been re-run under `--tag perm`, and the log's means are
the reproduction check. Related: the chain script in `scripts/39` first fell through its wait loop
because `pgrep` takes an *extended* regex and the guard was written with the BRE spelling `\|`,
which silently matches nothing. Both are now noted in the scripts.

### Priority 3 — the matched non-invariant control (S7)

W10 is nonlinear *and* invariant, so its number attributes nothing to symmetry on its own. `W10c`
emits the same monomials in (w,u) at the same trigonometric orders, pooled by the same spectra under
the same ‖w‖² key, at the same dimension, decoded by the same apparatus — with only the parity
classes swapped. It stays exactly permutation-invariant and is broken only in D∞, so the difference
isolates the D∞ component rather than confounding it with permutation handling. Registered in
`docs/prereg/S7.md` before any cell was decoded.

### Review question 1, answered directly

Is c_align a canonicalizer on *production* corpora, where the sort keys nearly tie? Measured:
residual of c(gθ) against c(θ) on 512 held-out INRs per dataset, four random group elements each,
median 1.2e−07 and max 1.5e−07, **0% above the 1e−4 tolerance**, on both MNIST and CIFAR-10. It is a
canonicalizer on the sampled orbits. PO-5 still guarantees a discontinuity set; we simply do not hit
it.

### Proofs

Theorem PO-2 had only a sketch. Appendix A now gives it in full: analytic continuation, the atomic
Fourier transform, support matching for width and permutation, the two sign cases for the
coefficients (each landing exactly on a g_{d,j} family), and triviality of the stabilizer for
uniqueness — plus where each genericity hypothesis is used. Appendix B specifies W10 coordinate by
coordinate with shapes, keys and contractions, proves invariance of every emitted coordinate, and
states what is *not* claimed: invariant but not shown separating; spectra 1-Lipschitz by Weyl while
the shared-key order statistics are discontinuous at ties; an encoding, not a reframing.

### Positioning

`docs/PROVENANCE.md` classifies every component prior work / specialization / consequence /
extension / ours, names the closest prior work, and states the difference. Two rows rest on a null
keyword search (eight targeted arXiv queries, snapshot committed) and say so. It has already forced
five corrections, and it is reproduced as a table in the paper rather than left to a closing
paragraph. Proposition PO-6 is demoted to a consequence of PO-2 and explicitly not offered as a
competing universality theorem; PO-5 is presented as the sine instance of Dym et al.

### Style

The review's suspicion of heavy LLM assistance was aimed at the writing, and the writing deserved
it. The abstract is 238 words. The forecast ledger and calibration analysis moved to an appendix.
Categorical and self-congratulatory constructions were replaced by what was measured, with the
instantiation named — "the tools the field reaches for do not work" became a statement about the
particular winding distribution, bias parameterization and averaging level we ran, which is both
more accurate and a stronger scientific claim.

### One correction to my own limitations text

I first wrote that no D∞-aware reader was built. That is false: W11b is G-invariant and is the best
template-free weight reader here. Its limitation is structural — it is invariant because its input
already is — and the unbuilt object is a reader that quotients D∞ on the parameters directly.
