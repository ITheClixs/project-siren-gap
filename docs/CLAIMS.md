# Claims Ledger

Every claim destined for any paper or the thesis gets a row *before* it ships. No row, no claim.

Types: `theorem` (proved), `proposition`, `conjecture` (+ supporting experiment), `confirmatory`
(pre-registered empirical), `exploratory` (E-track, watermarked), `replication`.

| # | claim | type | evidence artifact | script | commit | status |
|---|---|---|---|---|---|---|
| 1 | The per-neuron functional symmetry group of sine INRs contains ⟨ρ,σ⟩ ≅ D∞ with τ = ρ²; the per-layer group contains D∞ ≀ Sₙ (containment direction only; maximality = PO-2) | theorem (to write, G2) | thesis Ch1 §PO-1 | — | — | scoped (G0-theory-scoping §1) |
| 2 | The protocol-suggested features cos 2b·(w⊗u), sin 2b·(w⊗u) are **not** invariant under the PO-1 group; the parity-classified family {(0,0): 1,cos 2b; (1,0): sin 2b·w; (0,1): sin b·u; (1,1): cos b·(w⊗u)} ⊗ matching-parity monomials is | proposition (membership verified by hand; generation/completeness = G2 lemma) | G0-theory-scoping §2 table | tests/test_invariants.py (T6, G1) | — | membership verified on paper; needs T6 + generation proof |

(rows appended as work completes)
