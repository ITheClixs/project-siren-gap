# Open Problems

Sub-questions not yet resolved, each with the best partial attack. ≥ 10 required by G7.

1. **PO-2 deep-case identifiability (L ≥ 2 sine).** Best attack: Jacobi–Anger spectral-lattice
   peeling (scoping memo §5, Strategy B′). Blockers: generic non-cancellation of Bessel-coefficient
   sums across neurons; lattice-basis recovery from truncated spectra; folding rational-independence
   into "generic". Fallback wired: S4e falsification hunt on production L=2 nets.
2. **Operational "effective orbit volume" for P-B.** Proposal: |G_eff| = n!·2ⁿ·(R_b/π)ⁿ with R_b =
   95% inter-quantile bias range measured post-fit per protocol/dataset. Alternative candidate:
   description length of the canonicalization data (bits to encode g given θ). Must be fixed before
   S3 prereg; the registered P-B ordering is unfalsifiable until then (advisor review, Theorist 3).
3. **Metric validity of approximate d_G.** c_align approximates the orbit distance; approximation
   error may break the triangle inequality → S4 basin clustering could be ill-defined. Attack:
   planted-g recovery experiments quantifying the gap; exemplar-based clustering robust to quasi-
   metrics; report sensitivity of basin counts to linkage choice.
4. **Layer-≥2 phase-invariant front-end.** ~~Per-neuron invariants of layer l are not invariant to
   layer l−1's group action on w-coordinates.~~ **Solved for L=2 (G4, CLAIMS row 15):** couple the
   layers through the layer-2 Gram G = W₂ᵀW₂, which is invariant under the entire layer-2 group and
   picks up ε_iε_l under layer 1, then pair it with sign-cancelling layer-1 factors (`sin b_i`
   carries exactly ε_i; `cos b_i w_i` carries ε_i after contraction). Sorted eigenvalue spectra of
   the resulting matrices are invariant under the full group; verified to 3e−7 (T10) and decoded as
   rung W10 (35.54, f = 0.269 — it beats the template-free canonicalizer).
   **Still open for L ≥ 3:** the same trick needs a Gram per successive layer and the parity
   bookkeeping compounds; whether a finite invariant family stays *separating* at depth is unknown.
   Also open: whether the eigenvalue-spectrum pooling loses separating power relative to a full
   equivariant treatment (NFN-style), which is what the 0.269-vs-0.628 span against c_align hints
   at but does not establish.
5. **MPS determinism of P-shared-det.** Unknown whether refit-same-image weight distance ≈ 0 holds
   on MPS (T9). Attack: measure; if nondeterministic, CPU fallback for the P-shared-det corpus and
   document the MPS residual as an instrument note.
6. **Does exact phase canonicalization clean or destroy the bias signal route?** 2605.08281 finds
   the bias column is a causal readout route (meta-learned regime). τ/ρ reduction rewrites biases
   into [−π/2, π/2). Hypothesis (untested): reduction *concentrates* the route (removes nuisance
   winding); counter-hypothesis: class signal partly lives in the winding number k = round(b/π),
   which reduction moves into outgoing-sign bookkeeping. Attack: S1 rung comparison W5 vs W5-minus-
   phase-step ablation + direct probe on (k, b_reduced) split. — *Candidate for a sharp, small,
   early result.*
7. **Why does phase augmentation hurt DWS far more than GNN?** (Shamsian Table 3: DWS collapses
   under unbounded SIREN-bias aug, GNN degrades mildly.) Hypothesis: DWS features are raw-bias-
   magnitude sensitive (unbounded kπ shifts blow up the input distribution); graph nets renormalize
   per-node. Attack: reproduce the contrast inside S1's W6 with bounded vs unbounded k (E-track
   note); connects PO-12's truncated-group-averaging error term to a published failure.
