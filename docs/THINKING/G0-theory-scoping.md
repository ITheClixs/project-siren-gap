# G0 Theory Scoping Memo — PO-1 … PO-12

**Date:** 2026-07-16 · **Gate:** G0 · **Author:** research agent · **Status:** binding plan for G2

Purpose: scope every proof obligation before infrastructure exists — expected strength
(theorem / proposition / conjecture), attack strategy, risk, downgrade path, and empirical wiring.
Worked algebra included where it de-risks later code (per protocol ⚠️ THINK markers).

---

## 0. Convention lock (project-wide; T1 depends on it)

h⁰ = x ∈ [−1,1]²; hˡ = sin(Wˡ hˡ⁻¹ + bˡ), l = 1..L; f_θ(x) = W^{L+1} h^L + b^{L+1}.
SIREN ω₀ = 30 (first layer) is **absorbed into stored weights at dataset-creation time**: the shard
contains W¹_stored = ω₀·W¹_train if the fitter parameterizes sin(ω₀·(W¹x+b¹)); symmetry code sees
only the canonical form above. Verification: T1 must pass on *stored* weights re-rendered by an
independent forward implementation that knows nothing about ω₀. A convention mismatch here
silently invalidates every downstream number — the fitter and `src/sirengap/symmetry/` must share a
single source of truth (`sirengap.models.forward_canonical`).

## 1. PO-1 — Group structure. Expected: THEOREM (low risk)

Per hidden neuron i in layer l: incoming row wᵢ, bias bᵢ, outgoing column uᵢ of W^{l+1}. Generators:

- τ_k: bᵢ ↦ bᵢ + 2πk — sin(z + 2πk) = sin z. Preserves f. ✓
- ρ: (bᵢ, uᵢ) ↦ (bᵢ + π, −uᵢ) — sin(z+π) = −sin z, compensated in uᵢ. ✓
- σ: (wᵢ, bᵢ, uᵢ) ↦ (−wᵢ, −bᵢ, −uᵢ) — sin odd. ✓

Relations (verified by direct composition on (w,b,u)):
ρ² = τ₁; σ² = id; σρσ: (w,b,u) →σ (−w,−b,−u) →ρ (−w,−b+π,u) →σ (w,b−π,−u) = ρ⁻¹. ✓
⟨ρ, σ | σ²=1, σρσ=ρ⁻¹, ord(ρ)=∞⟩ ≅ **D∞** (τ = ρ² is redundant as a generator — state this).
Per layer: (D∞)^{n_l} ⋊ S_{n_l} = **D∞ ≀ S_{n_l}** acting jointly on (rows of Wˡ, bˡ, columns of W^{l+1});
full group = ∏_l D∞ ≀ S_{n_l} (hidden layers only; output affine layer not acted on).
Risk: none mathematically; the value is the clean wreath-product statement and the exactness of the
joint action bookkeeping (A.2's most likely silent bug). Write at G2 day 1.

## 2. PO-4 — Invariant theory of the per-neuron action. Expected: PROPOSITION (worked core below)

**Finding at G0 (important):** the protocol's example invariants are *not* invariant. Worked check:

Action recap: τ: b+2π · ρ: (b+π, u→−u) · σ: (−w, −b, −u).
Ansatz: m = g(b)·P(w)·Q(u), P a monomial of parity α in w-entries, Q parity β in u-entries.
Invariance ⇔
- τ: g is 2π-periodic;
- ρ: g(b+π) = (−1)^β g(b) — β even ⇒ g π-periodic; β odd ⇒ g π-antiperiodic;
- σ: g(−b) = (−1)^{α+β} g(b) — α+β even ⇒ g even; odd ⇒ g odd.

Fourier solutions g ∈ {cos kb, sin kb}: ρ forces k ≡ β (mod 2); σ picks cos/sin. Minimal-k generators:

| (α, β) | g(b) | generator examples |
|---|---|---|
| (0,0) | 1, cos 2b | ‖w‖², ‖u‖², w⊗w, u⊗u, cos 2b, cos 2b·(w⊗w), cos 2b·(u⊗u) |
| (1,0) | sin 2b | sin 2b · w |
| (0,1) | sin b | sin b · u |
| (1,1) | cos b | cos b · (w⊗u) |

The protocol's cos 2b·(w⊗u) has (α,β)=(1,1) but k even ⇒ **fails ρ** (u flips, cos 2b doesn't).
Its sin 2b·(w⊗u) has g odd with α+β even ⇒ **fails σ**. The protocol's own parenthetical flags this;
the corrected parity classification above is the fix and is already a small original result.
Each entry in the table has been checked against all three generators by hand.

**Separation sketch (to prove as a lemma at G2, generic stratum w≠0, u≠0):**
w⊗w → w up to sign; sin 2b·w combined with the chosen w-sign → sin 2b; with cos 2b → b mod π
tied to the sign convention; sin b·u and cos b·(w⊗u) then jointly resolve (u, b mod 2π): the residual
ambiguity (b, u) vs (b+π, −u) maps sin b·u → sin(b+π)(−u) = sin b·u — i.e. the features are constant
exactly on the orbit, and distinct orbits separate. Degenerate strata excluded (PO-3).
**Wiring:** this feature map (plus DeepSets pooling over neurons for the S_n part) = the
phase-invariant encoding front-end (Ch3.6, rung W10, test T6).
**Caveat for L ≥ 2:** for layer l ≥ 2 the "w" slot of neuron i is a row of Wˡ whose *coordinates* are
permuted/sign-flipped by layer l−1's group — per-neuron invariants alone are not invariant to the
previous layer's action. The front-end must be built layer-wise interleaved (same structural problem
NFN/DWS solve). Design point for Ch3.6/3.7; flagged in OPEN_PROBLEMS.

## 3. PO-5 — No continuous canonicalization. Expected: PROPOSITION

After quotienting by ⟨τ, ρ⟩ the bias coordinate lives on a circle (ℝ / πℤ with a sign-twist on u —
formally the orbit space of (b,u) under ⟨τ,ρ⟩ is a twisted line bundle over S¹). A continuous
canonicalizer is a continuous section of the quotient map on the generic stratum; the S¹ factor makes
the quotient a nontrivial principal-like bundle ⇒ no global continuous section (connectedness /
covering-space argument). Cite and adapt Dym–Lawrence–Siegel (arXiv:2402.16077, verified) for the
general framing; our case is concrete enough for a self-contained two-page proof.
Corollary to quantify: any exact canonicalizer is discontinuous on the tie set
{b ≡ π/2 mod π} ∪ {⟨w, v_ref⟩ = 0} ∪ {invariant-key ties} — exactly the margins c_sort/c_align log.
Wiring: F7 diagnostics; R3 risk.

## 4. PO-3 — Degenerate strata. Expected: PROPOSITION (easy, but add u = 0)

Strata: (i) dead neuron wᵢ = 0 (constant output sin(bᵢ) trades against output bias — a continuum);
(ii) **invisible neuron uᵢ = 0** (incoming (wᵢ,bᵢ) completely unidentifiable — protocol omits this
stratum; add it); (iii) duplicate/merge continua: (wᵢ,bᵢ) = ±(wⱼ,bⱼ) mod (π-structure) allowing
uᵢ+uⱼ trades; (iv) collinear-frequency coincidences on lines (measure-zero in weight space).
All are zero sets of nontrivial analytic equations ⇒ closed, measure zero.
Wiring: dataset audit metrics min‖wᵢ‖, min‖uᵢ‖, min pairwise orbit-distance between neurons —
report distributions per dataset (checks "generic" empirically).

## 5. PO-2 — Generic identifiability. Expected: THEOREM (L=1) + CONJECTURE (deep) with a real attack

**L = 1 (single hidden layer):** f_θ = Σᵢ uᵢ sin(⟨wᵢ,x⟩+bᵢ) + c. If f_θ ≡ f_θ′ on an open U ⊂ ℝ²,
real-analyticity extends equality to ℝ². Restrict to generic lines x(t) = x₀ + tv: v avoiding finitely
many hyperplanes gives distinct line-frequencies aᵢ = ⟨wᵢ, v⟩ within and across the two networks
(after σ-normalizing aᵢ > 0). Linear independence of {1, sin(at+c) : a > 0} via the almost-periodic
inner product (distinct frequencies orthogonal under lim_T T⁻¹∫₋ᵀᵀ) forces matched frequency sets,
amplitudes, phases mod 2π — residual ambiguities are exactly ⟨σ, ρ, τ⟩ per neuron. Vary v over an
open set to recover wᵢ from the linear functionals aᵢ(v); continuity of the matching gives a
consistent permutation. Exclusions: uᵢ ≠ 0, wᵢ ≠ 0, no duplicate neurons mod group — precisely PO-3's
strata. Moderate bookkeeping risk only.
**Deep case (L ≥ 2), Strategy B′ (Jacobi–Anger spectral lattice):** along a generic line, layer-1
outputs are trig sums; sin(Σᵢ Aᵢ sin(aᵢt+cᵢ) + φ) expands by Jacobi–Anger into an absolutely
convergent series supported on the ℤ-module {Σᵢ kᵢaᵢ}, coefficients = products of Bessel J_{kᵢ}(Aᵢ).
For generically rationally-independent {aᵢ}, the minimal generating set of the spectral support
recovers layer-1 line-frequencies; J₁-order coefficients then constrain layer-2 rows; peel and induct.
Obstructions to resolve: cross-neuron coefficient cancellation (needs a generic non-cancellation
lemma); lattice-basis recovery from a truncated spectrum; formalizing "generic" to include rational
independence. Literature: Sussmann 1992 (tanh, L=1), Fefferman 1994 (sigmoid, deep — uses pole
structure, our analogue is the Bessel lattice), Phuong & Lampert 2020 (ReLU), Expand-and-Cluster
(arXiv:2304.12794, verified) for reconstruction-flavored technique; new 2026 ReLU classifications
(2604.14037, 2605.18319) for the semi-algebraic framing of "generic fibers".
**Timebox:** 2 working days at G2 for B′; then downgrade to conjecture + falsification protocol
(S4e: exhaustive-alignment residual hunt on repeated fits of the same image, L=2 production nets).
Any confirmed functionally-equal, non-group-related, non-degenerate pair = headline finding.

## 6. PO-6 — Completeness–tautology. Expected: PROPOSITION conditional on PO-2 + **novelty check required**

Given PO-2 on the generic stratum: orbits = functional equivalence classes there, so any orbit-
separating (complete) invariant factors through θ ↦ f_θ; a complete invariant is an injective encoding
of the function. Corollary: complete-invariant weight perception is informationally function-access;
only computational advantages are possible. Escape hatches to state honestly: incomplete-but-useful
invariants; distributional/population methods; amortization.
**⚠ Novelty check (from G0 scan):** arXiv:2602.01083 ("Expressive Power of Permutation-Equivariant
Weight-Space Networks", Feb 2026) proves universality results for weight-/function-space maps —
close-read at G1 to determine whether PO-6's factorization argument (or a version of it) is already
in their framework. If yes: cite, sharpen to the D∞ case (their scope is permutation-equivariant),
and keep the FLOPs-Pareto corollary as ours.

## 7. PO-7 — Comparative classification. Expected: THEOREM-table (per-family, mostly easy)

| activation | per-neuron group | type | exact canon.? | canonicalizer |
|---|---|---|---|---|
| tanh (odd) | ⟨σ⟩ ≅ ℤ₂ | finite | yes | sign fix |
| Gaussian e^{−z²} (even) | ⟨σ′⟩ ≅ ℤ₂, σ′: (w,b)→(−w,−b), u untouched | finite | yes | sign fix (different embedding) |
| sine | D∞ = ⟨ρ,σ⟩ | infinite discrete | yes, necessarily discontinuous | phase-reduce + sign fix |
| ReLU + PE | ℝ₊ scaling (λw, λb, u/λ) | continuous non-compact | a.e. (‖(w,b)‖=1) | row-normalize |
| FINER-style | expect ≈ trivial (S_n only) | — | nothing beyond perm-sort | — |

FINER derivation owed at G2: variable-periodic sin((|z|+1)z)-style activations break τ/ρ/σ (check σ:
activation neither odd nor even once |z| enters; verify carefully — if the activation is odd, σ
survives). S_n (permutations) present for every family. Fit-quality, parameter-count, and spectral
confounds are S3's problem; the memo for S3 must carry them.

## 8. PO-8 — Microcosm. Expected: closed-form THEOREM for structure + numeric enumeration

1 neuron, 1D: model u sin(wt+b) + c, target A sin(ωt+φ) + c₀ on [−1,1], population L2 loss.
Key reduction: profile out the linear parameters — for fixed (w,b), optimal (u,c) solve a 2×2 linear
system whose entries are elementary integrals (∫sin², ∫sin, ∫sin·target over [−1,1] — closed forms in
sin/cos of (w±ω)). The profiled loss ℓ(w,b) is an explicit 2D surface: enumerate critical points
exactly/numerically, classify global minima (the D∞ orbit of (ω,φ) pulled back) vs spurious minima
(finite-domain spectral-leakage sidelobes at w ≈ ω ± kπ-ish). Predict basin counts as a function of
init distribution analytically; verify by brute-force descent from a grid of inits. Extend numerically
to 2 neurons / two-tone. Deliverable: F4 + the first precise separation of "symmetry copies" from
"genuine basins". This is where PO-10's definitions get calibrated.

## 9. PO-9 / PO-10 / PO-11 — fit-map anatomy (empirical probes with theory hooks)

- **PO-9 laziness:** rel. NTK movement ‖K_T−K₀‖/‖K₀‖ and weight travel vs width {16,32,64,128}.
  Consequence to state: ε-lazy + P-shared-det ⇒ θ_T ≈ θ₀ + Φ(y), Φ linear ⇒ W1 linear probe ≈ pixel
  linear probe (exact if Φ full-rank on pixel span; degradation bounded by conditioning). Registered
  direction P-D: the W1-vs-pixel linear-probe gap shrinks with width. External support found at G0:
  CertMix (2607.04123) observes approximate linearity of a physical functional in shared-anchor SIREN
  weight space; FiRe (2606.29414) gives NTK conditioning analysis of periodic INRs. Cite both.
- **PO-10 basins:** d_G(θ,θ′) = min over group of ‖θ−g·θ′‖ (approximated by c_align; approximation gap
  measured on synthetic pairs with planted g — recovery-rate diagnostic). Basin equivalence = small
  d_G OR post-alignment linear-mode-connected (barrier < ε_barrier, pre-registered at S4).
  Note from G0 scan: 2606.04754 ("neuron identifiability", Jun 2026) formalizes approximate
  equivalence classes and merging-without-alignment — close-read before finalizing definitions.
- **PO-11 chaos:** finite-difference sensitivity of F under P-shared-det; divergence curves from
  δ₀-separated inits. C-11 registered at S4 prereg (non-monotone decodability vs fit length).
  Papa et al. studied overtraining effects on downstream accuracy — close-read to record exactly what
  they found so C-11's delta is honest (mechanism via chaos/basin counts, not the bare phenomenon).

## 10. PO-12 — sample complexity under nuisance groups. Expected: PROPOSITION-lite + registered ordering

For finite groups: strict-benefit results (Elesedy–Zaidi 2102.10333) and invariant-kernel gains
(Mei–Misiakiewicz–Montanari 2102.13219; Bietti et al. 2106.07148) give effective-sample ×|G| flavored
improvements. For D∞'s non-compact τ: replace |G| by an *effective orbit count* using the measured
95% bias range R_b after fitting: |G_eff| ≈ n!·2ⁿ·(R_b/π)ⁿ per layer. Derivation obligation: state
the bounded-range adaptation explicitly (truncated group averaging error term). Predicted S5 ordering
(register numerically at S5 prereg): equivariant ≥ exact canonicalization ≥ augmentation ≥ raw, gaps
growing as training-set size shrinks. Also connects to 2502.19758 (exact invariances in poly time) —
cite as the computational side.

## 11. Attack order & timeboxes (G2)

PO-1 (0.5d) → PO-4 incl. separation lemma (1d) → PO-5 (1d) → PO-3 (0.5d) → PO-2 L=1 (1.5d) →
PO-8 (1.5d) → PO-6 + novelty memo vs 2602.01083 (1d) → PO-2 deep B′ (timebox 2d, then downgrade
decision memo) → PO-7 formal incl. FINER derivation (1d). Total ≈ 10 working days of theory at G2.

## 12. Registered predictions (hashed into PREDICTION_LEDGER.csv)

- **P-A (S3, canonicalization consistency ordering):** tanh ≈ Gaussian > sine > ReLU+PE > FINER.
- **P-B (S3, raw-weight decodability ordering; operational orbit-volume definition owed before S3
  prereg, see OPEN_PROBLEMS):** FINER ≥ tanh ≈ Gaussian > sine > ReLU+PE.
  Rationale: every family keeps S_n; per-neuron factors add ℤ₂ (tanh, Gaussian), effectively-finite
  D∞ copies within the realized bias range (sine, larger), a continuum (ReLU+PE, largest); FINER adds
  ≈ nothing. Raw decodability predicted inversely ordered by effective orbit volume.
- **P-C (S3, post-canonicalization collapse toward the shared-init ceiling):** tanh ≈ Gaussian reach
  it first; sine next (tie-sensitivity residual); ReLU+PE next (margin fragility near ‖(w,b)‖≈0);
  FINER moves least (little to remove; residual = basin dispersion).
- **P-D (S1/PO-9, direction):** W1-vs-pixel linear-probe gap is monotonically decreasing in width;
  numeric intervals to be registered at S1 prereg after the pilot fixes seed-σ.

## 13. Theory risk register

| risk | likelihood | mitigation |
|---|---|---|
| PO-2 deep intractable | high | timebox + conjecture + S4e falsification hunt (protocol R2) |
| PO-6 partially scooped by 2602.01083 | medium | G1 close-read; sharpen to D∞; keep Pareto corollary |
| PO-4 separation fails on a bigger-than-expected set | low | enlarge generator set (higher k); report |
| FINER keeps σ (odd activation) | medium | derive carefully; adjust P-A/P-B *before* S3 prereg if needed, with a logged amendment |
| A.2 sign algebra bug | medium | six-line algebra check written *before* c_align code (T2/T3 enforced); the PO-4 table above is the reference |
