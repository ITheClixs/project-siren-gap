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
