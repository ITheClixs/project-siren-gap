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
4. **Layer-≥2 phase-invariant front-end.** Per-neuron invariants of layer l are not invariant to
   layer l−1's group action on w-coordinates (scoping memo §2 caveat). Attack: interleave per-layer
   invariant maps with equivariant pooling, NFN-style; or restrict W10 to layer-1 + last-layer
   features and measure the cost.
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
