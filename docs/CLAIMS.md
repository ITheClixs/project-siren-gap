# Claims Ledger

Every claim destined for any paper or the thesis gets a row *before* it ships. No row, no claim.

Types: `theorem` (proved), `proposition`, `conjecture` (+ supporting experiment), `confirmatory`
(pre-registered empirical), `exploratory` (E-track, watermarked), `replication`.

| # | claim | type | evidence artifact | script | commit | status |
|---|---|---|---|---|---|---|
| 1 | The per-neuron functional symmetry group of sine INRs contains ⟨ρ,σ⟩ ≅ D∞ with τ = ρ²; the per-layer group contains D∞ ≀ Sₙ (containment direction only; maximality = PO-2) | theorem (to write, G2) | thesis Ch1 §PO-1 | — | — | scoped (G0-theory-scoping §1) |
| 2 | The protocol-suggested features cos 2b·(w⊗u), sin 2b·(w⊗u) are **not** invariant under the PO-1 group; the parity-classified family {(0,0): 1,cos 2b; (1,0): sin 2b·w; (0,1): sin b·u; (1,1): cos b·(w⊗u)} ⊗ matching-parity monomials is | proposition (membership verified by hand; generation/completeness = G2 lemma) | G0-theory-scoping §2 table | tests/test_invariants.py (T6, G1) | — | membership verified on paper; needs T6 + generation proof |

| 3 | Per-neuron sine group ⟨τ,ρ,σ⟩ ≅ D∞ with normal form g_{d,j} and composition (d₁⊕d₂, j₂+(−1)^{d₂}j₁); per-layer group ⊇ D∞ ≀ Sₙ; cross-layer actions commute | theorem | ch1-symmetry.tex Thm (PO-1) + Lemma (normal form) | tests T1 | — | written G2 |
| 4 | Strata: dead (w=0), invisible (u=0), **parallel-frequency** (w_j = ±w_i, any biases — phasor merge); closed, null, each carrying extra non-group equivalences | proposition | ch1-symmetry.tex Prop (PO-3) | strata audit at G3 | — | written G2; parallel-stratum correction logged |
| 5 | Φ = (w⊗w, cos2b, sin2b·w, sinb·u, cosb·(w⊗u)) is invariant and separates D∞-orbits on {w≠0,u≠0} | proposition | ch1-symmetry.tex Prop (PO-4) | T6 | — | written G2 |
| 6 | No continuous canonicalizer exists on the generic stratum (τ-winding obstruction; elementary loop proof) | proposition | ch1-symmetry.tex Prop (PO-5) | F7 margin diagnostics | — | written G2 |
| 7 | **L=1 identifiability:** on Θ_gen (w≠0, u≠0, no parallel pair), f_θ=f_θ′ ⟹ θ′ = gθ, g ∈ D∞≀Sₙ unique — D∞≀Sₙ is a maximal symmetry group (sense of 2604.23720 Def 2.2), resolving 2409.11697 Rmk 4.5 for sine, L=1 | theorem | ch1-symmetry.tex Thm (PO-2) | S4e falsification hunt (deep case) | — | written G2; proof to be red-teamed at G2 advisor review |
| 8 | Deep identifiability | conjecture + roadmap (Bessel-CP reduction; 2 open lemmas; Bessel-Vandermonde ill-conditioning documented) | proof-memos/PO-2-deep-attempt.md | S4e | — | downgraded per timebox, honest attempt recorded |
| 9 | Complete invariants factor through function space (conditional on identifiability); informational-equivalence corollary; exact/D∞ counterpart of 2602.01083's approximation results | proposition | ch1-symmetry.tex Prop (PO-6) | S5 Pareto adjudication | — | written G2 |
| 10 | Activation classification: tanh ℤ₂; Gaussian ℤ₂ (u untouched); ReLU+PE ℝ₊ continuous; sine D∞; **FINER ℤ₂ (odd; no bias-shift symmetries — zero-gap argument)** | proposition (per row) | ch1-symmetry.tex Prop (PO-7) | S3 | — | written G2; P-A/P-B amended pre-data |
| 11 | Microcosm: profiled loss closed form (quadrature-certified 6e−16); zero set = D∞ orbit exactly; 20 minima (1 global, 19 spurious); basin census non-monotone in init range with three regimes — 100% degenerate-ridge capture at ±2, 62% global at ±10, 51% spurious-sidelobe capture at ±20 (ω=7) | theorem (forms) + numeric certification | ch2-fitmap.tex §2; results/microcosm/ (F4) | scripts/02_microcosm_po8.py | — | done G2 |

| 12 | Microcosm basin census is **optimizer-robust in its headline and wrong in one sub-claim**: the non-monotone global-capture curve replicates under Adam and plain GD on the *full* (non-profiled) model — global fraction 0.00/0.20/0.56/0.31 (Adam) and 0.00/0.18/0.58/0.33 (GD) at init ranges 2/5/10/20 vs Nelder–Mead's 0.00/0.26/0.62/0.32, sweet spot at init range ≈ ω — but **"degenerate-ridge capture" is a Nelder–Mead/profiled-surface artifact** (0.00 under both gradient methods; those runs are still descending, not in a ridge basin), and the ±20 spurious-sidelobe fraction is optimizer-dependent (0.48 Adam-converged, 0.00 GD-converged) | numeric certification (replication of row 11 under the production optimizer class) | results/microcosm/optimizer_census.json (F4 right panel) | scripts/13_microcosm_optimizers.py | — | done G4; corrects row 11's ridge/sidelobe language |
| 13 | At the **frozen corpus setting** (Adam, lr 1e−3, 300 steps) the microcosm fit is nowhere near a critical point: 100% of inits end unconverged at every init range, endpoint ‖∇‖ ≈ 0.5–0.7, and median \|Δw\| ≈ 0.24 *independently of init range* — the fit stays in its initialization neighbourhood. Mechanistic support for PO-9 laziness and hence for the W1−W3 gap | numeric | results/microcosm/optimizer_census.json | scripts/13_microcosm_optimizers.py | — | done G4 |

(rows appended as work completes)
