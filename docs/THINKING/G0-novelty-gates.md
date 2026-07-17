# G0 Novelty Gates — Decision Memo

**Date:** 2026-07-16 · **Inputs:** 18 arXiv API queries + 3 targeted follow-ups; 36 Appendix-D ID
verifications; 26 abstracts close-scanned. Raw evidence: `docs/lit_snapshots/G0-scan-summary.txt`
(regenerate via `make lit-scan`). Queries: `docs/LIT_QUERIES.md`.

Gate semantics: each gate **passes** if the answer to its question is *no*; a *yes* triggers the
Part IV pivot logic before infrastructure is built.

---

## Gate-1 — "Does anything exploit sine translation symmetries (τ/ρ phase shifts) for INR classification?"

### Verdict: **PASS** (no exploitation found), with three documented near-misses and two confirmatory close-reads due at G1.

Evidence:

1. **Monomial-NFN (arXiv:2409.11697, NeurIPS 2024)** — the closest work. Extends NFN equivariance
   from permutations to monomial matrix groups: scaling for ReLU, **sign flips for sin/tanh** (our σ).
   Their maximality theorem ("all groups leaving the network invariant while acting on weight space
   are subgroups of the monomial matrix group") is necessarily scoped to **linear** actions: our τ/ρ
   act **affinely** on biases (b ↦ b+π with a sign twist) and are therefore outside their classified
   family while still being exact functional symmetries. Consequence: the τ/ρ part of D∞ is
   unexploited in the strongest existing symmetry-aware architecture, and our PO-1/PO-2 extend the
   known classification for sine. **Close-read of their maximality proof due at G1** to state this
   delta precisely and fairly.
2. **DeepWeightFlow (2601.05052, Jan 2026)** — canonicalization (Git Re-Basin / TransFusion) used for
   generative weight modeling: **permutations only**, no sign, no phase, different task.
3. **CertMix (2607.04123, Jul 2026)** — aligns SIREN weight spaces by **overfitting from a shared
   anchor** (equivalent to our P-shared protocol), not by symmetry canonicalization; domain is
   metamaterial design. Confirms the community handles nuisance by shared init, not by group theory.
4. Query `abs:"periodic activation" AND abs:"symmetry"` returns **zero** hits; `abs:"SIREN" AND
   abs:"symmetry"` returns nothing on point.
5. Surveys — parameter-symmetry survey (2506.13018) and WSL survey (2603.10090): symmetry
   *cataloguing*; G1 close-read to confirm sine bias-periodicity appears at most as a remark
   (Hecht-Nielsen-style folklore) and is nowhere canonicalized or exploited for perception.
6. **Quasi-Equivariant Metanetworks (2604.23720, Apr 2026)** — relaxes strict equivariance; abstract
   indicates perm/scaling scope. **Close-read due at G1** (tripwire: if their "functional identity"
   formalism includes periodic-activation phase orbits, Gate-1 needs re-adjudication).
7. Shamsian et al. (2402.04081) + predecessor (2311.08851): weight-space augmentations. Whether any
   augmentation is a *function-preserving bias phase shift* (vs. input-space transforms and
   permutation/sign ops) is not determinable from abstracts — **close-read at G1**; if they augment
   with τ-shifts, rung W6 cites them as origin and our delta shifts to exactness + decomposition.

Tripwires that would flip the verdict: any of items 5–7 revealing phase-shift exploitation; any new
arXiv hit on the re-scan queries at G1+.

### G1 update (2026-07-17, after full-text close-reads) — verdict CONFIRMED with revised precision

The close-reads (docs/THINKING/close-reads/) resolve items 6–7 and sharpen item 1:

- **Shamsian et al. DO publish the τ/ρ ops** as augmentations ("SIREN negation" = σ, "SIREN bias" =
  b+kπ with (−1)ᵏ on the next layer, k ∈ ℤ unbounded) — *and their Table 3 shows SIREN bias
  augmentation severely hurts weight-space learners* (DWS: 4.69 vs ≈18 no-aug on ModelNet40-INR;
  58.2 vs ≈68 FMNIST; 24.3 vs ≈39 CIFAR-family). So the strict reading of Gate-1 is "the ops are
  known, were tried as augmentation, and failed." Nobody canonicalizes, classifies, or builds
  equivariance for the phase component.
- **Monomial-NFN's Remark 4.5 states functional maximality for sine as an open question.** Their
  maximality is for matrix subgroups of GL(n); τ/ρ are affine. PO-1/PO-2 answer their open question.
- **Quasi-equivariant metanetworks:** sine appears only inside their learned-action
  parameterization; no phase symmetries. Their Def 2.2 (maximal symmetry group off a variety) is
  adopted as PO-2's statement form.
- **Revised Gate-1 novelty statement (binding for all papers):** the sine phase symmetries are
  *known as isolated identities and were used as (failed) augmentation*; the program's novelty is
  (a) the group-theoretic classification (D∞ ≀ Sₙ, wreath structure) with identifiability (resolving
  2409.11697's open question), (b) exact canonicalization algorithms for the full group, (c) full-
  group equivariant layers (perm/+sign/+phase ablation), (d) the causal decomposition built on
  exactness, (e) the published failure of phase *augmentation* as direct motivation for exactness
  (and a PO-12 data point).
- Gate-2 confirmed: 2605.08281's own scoping sentence — "the object of study is not an
  independently fitted SIREN zoo." 2602.01083 is ReLU/perm approximation theory; PO-6 survives with
  sharpened framing (see close-read).

## Gate-2 — "Does a published causal decomposition of the perception gap exist?"

### Verdict: **PASS** (no decomposition found; the premise is stated but never decomposed).

Evidence:

1. **Papa et al. (2312.10531)** — the closest empirical study: shared init, overtraining,
   architecture effects on downstream NeF accuracy + Neural Field Arena benchmark. Interventional on
   *fitting hyperparameters*, but no separation of symmetry vs. optimization-noise vs. basin
   components, no canonicalization rungs, no marginalization discriminator. Our S1 ladder subsumes
   its init finding as rungs W1/W3 (and must replicate it — anchor A1).
2. **Shamsian et al. (2402.04081)** — states the nuisance premise explicitly ("a given object can be
   represented by many different weight configurations; typical INR training sets fail to capture
   [that] variability") and treats it with augmentation. No decomposition, no exactness, no theory of
   which variability is removable. Anchor A2.
3. **2605.08281 (May 2026, "Clustered or Routed?")** — the most important recent neighbor. In a
   *meta-learned* (MWT) shared-anchor regime: finds weight-space geometry does **not** reliably track
   trained-reader accuracy; identifies the SIREN **bias column** as a causal readout route; concludes
   class signal is *routed through the reader*, not raw clustering. This is a mechanism study of the
   *reader* in a meta-learned regime — not a causal decomposition of the *fit-process* gap across
   init/symmetry/basin components in independently-fitted INRs. It does not gate our program, but it
   (a) is directly adversarial to S2's geometry-law premise → S2 prereg must include their
   dissociation as a named alternative outcome; (b) highlights biases as signal carriers — exactly
   the coordinate our τ/ρ act on, so phase canonicalization interacts with the *known* signal route
   (potentially our sharpest empirical hook). **Full close-read at G1, before any S2 design work.**
4. **2601.23181 (Jan 2026, IFT)** — theory of semantics in *hypernetwork-generated* weights via the
   implicit function theorem; the generated-weights regime sidesteps independent-fit nuisance
   entirely (functa-family). Complementary, not a decomposition.
5. **End-to-End INR classification (2503.18123, CVPR 2025)** + 2605.08281 define the meta-learned
   SOTA line (CIFAR-10 SIREN 59.6% no-aug). Scope note for all our papers: we study *independently
   fitted* INRs because that is the regime where the perception gap exists and is scientifically
   interesting; meta-learned inits are a different access model (they change the data-generating
   process, not the reader). Include as a labeled reference point, not a rung.

Tripwires: G1 close-reads of 2605.08281 and the WSL survey's "understanding" section; re-scan at
every gate (queries s01, s02, s08, s13).

## Consequences for design (actions taken at G0)

- Close-read queue (G1, ordered): 2605.08281 → 2409.11697 (maximality proof) → 2402.04081 +
  2311.08851 (aug list) → 2602.01083 (PO-6 novelty) → 2604.23720 → surveys (sine sections) →
  2312.10531 (overtraining details, anchor numbers) → 2410.10811 (ProbeGen budget details for S5).
- S2 prereg must pre-register the "geometry necessary-but-not-sufficient / routed" alternative
  explicitly, with the analysis that discriminates it (trained-reader vs. geometry dissociation
  cells), crediting 2605.08281.
- S5 gains one contender-context citation (INR2JLS, 2607.02166: dynamic-graph encoder, claims +10%
  on CIFAR-100-INR) and one baseline lineage note (ProbeX → MVProbe, 2605.23410).
- The 2026 explosion of weight-space work (≈ 30 relevant hits in 6 months) raises scoop risk from R5
  "low" to "medium"; re-scan cadence confirmed at *every* gate, and Paper A's theory core (D∞
  classification + exact canonicalization + identifiability) should be drafted early (G2-adjacent
  workshop-able preprint decision at G4).
