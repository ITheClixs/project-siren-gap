# Proof-attempt memo (C.3): PO-2 deep case (L ≥ 2 sine identifiability)

**Date:** 2026-07-17 · **Timebox:** 2 working days equivalent; spent this session on the
reduction; verdict below. **Statement attempted:** off a proper analytic subset, $f_\theta =
f_{\theta'}$ for $L\ge2$ sine networks implies $\theta' = g\theta$, $g \in \prod_l D_\infty \wr S_{n_l}$.

## Strategy B′: Jacobi–Anger spectral lattice → CP decomposition

Along a generic line $x = tv$, layer-1 outputs are $s_i(t) = \sin(a_i t + b_i)$, $a_i = \langle
w_i, v\rangle$ (σ-normalized $a_i > 0$, distinct — the L=1 machinery). A second-layer neuron
contributes $v_j \sin(\sum_i W_{ji} s_i(t) + c_j)$. Jacobi–Anger,
$e^{iz\sin\phi} = \sum_{k\in\mathbb Z} J_k(z) e^{ik\phi}$, gives
$$e^{i\,\mathrm{inner}_j(t)} = e^{ic_j} \prod_i \sum_{k_i} J_{k_i}(W_{ji}) e^{ik_i(a_i t + b_i)},$$
absolutely convergent (Bessel super-exponential decay in order). Hence $f|_{\mathrm{line}}$ is
almost-periodic with spectrum in the module $\Lambda = \{\sum_i k_i a_i : k \in \mathbb Z^{n_1}\}$
and Fourier–Bohr coefficient at $k$:
$$T(k) \;=\; \sum_{j} v_j\, e^{ic_j} \Big(\prod_i J_{k_i}(W_{ji})\Big)\, e^{i\sum_i k_i b_i}
\quad(\text{plus the conjugate branch}).$$
For generic $v$ the $a_i$ are rationally independent, so the representation of each spectral
point by $k$ is unique and $T$ is well-defined on $\mathbb Z^{n_1}$.

**Key observation.** After pulling out the known phases $e^{i k\cdot b}$ (recoverable only mod
the layer-1 ambiguities — see below), $T$ is a **rank-$n_2$ CP-like tensor in the Bessel basis**:
$T'(k) = \sum_j \lambda_j \prod_i J_{k_i}(W_{ji})$, $\lambda_j = v_j e^{ic_j}$. If the "Bessel
Vandermonde" matrices $[J_k(x)]_{k=0..K,\,x\in\{W_{1i},..,W_{n_2 i}\}}$ have full column rank for
generic arguments, Kruskal-type uniqueness of CP decompositions applies and $(\lambda_j, |W_{ji}|)$
are identifiable up to permutation of $j$ — i.e., layer-2 rows up to exactly the layer-2
permutation, with layer-2's $D_\infty$ ambiguities entering through $J_{-k} = (-1)^k J_k$
(σ of layer 1 flips $k_i \mapsto -k_i$) and $c_j \mapsto c_j + \pi$ with $v_j \mapsto -v_j$
(ρ of layer 2, visible as $\lambda_j \mapsto -\lambda_j$... which trades against $J$ signs —
bookkeeping to pin down). Then peel: subtract nothing, instead recurse — layer-1 parameters
$(a_i, b_i)$ are read from the module generators and the first-order coefficients
$T(e_i) = \sum_j \lambda_j J_1(W_{ji}) \prod_{i'\neq i} J_0(W_{ji'}) e^{ib_i}$.

## What was actually attempted, and where it stands

1. **Reduction correctness** (this memo, above): believed sound; the two-sided (conjugate) branch
   and the $\pm k$ folding need one careful lemma ("spectral folding") — drafted, not verified.
2. **Bessel-Vandermonde rank lemma** (open): $\det[J_{k}(x_j)]_{k,j}$ for distinct $x_j > 0$ is a
   real-analytic function of $(x_j)$; at small arguments $J_k(x) = (x/2)^k/k!\,(1 + O(x^2))$, so
   the leading term of the determinant is a Vandermonde-like product $\prod_{j} (x_j/2)^{k_j}$
   alternant — nonvanishing for distinct small $x_j$, hence the determinant is not identically
   zero, hence generically nonzero. *Gap:* "generic" here must be intersected with the network's
   actual parameter measure; the small-argument expansion argument gives a proper analytic
   exceptional set — this part looks completable with care. **Numeric support with a warning:**
   the microcosm script checks 200 random draws per size; min |det| = 4.4e-3 (n=2), 5.2e-8 (n=3),
   6.9e-9 (n=4), 4.4e-13 (n=5) — nonzero throughout (consistent with generic rank) but *severely
   ill-conditioned* already at n=5, which means the truncation-control lemma (item 3) is not a
   formality: noise-robust recovery through a near-singular Bessel system will degrade
   exponentially in width. This tempers expectations for any constructive/algorithmic use of the
   deep identifiability argument and strengthens the case that S4e (empirical falsification hunt)
   is the right investment rather than more proof effort now.
3. **Truncation control** (open): identifying CP factors from finitely many observed coefficients
   with an infinite tail; Bessel decay gives explicit tail bounds, but a clean statement of "the
   truncated tensor's decomposition converges to the true factors" was not reached in the timebox.
4. **Cross-neuron cancellation:** subsumed by the CP-rank condition (Kruskal) — no separate lemma
   needed if 2 holds with Kruskal-rank margins; not verified in the timebox.

## Downgrade decision

**Conjecture** (Ch1, Conj. 6.5) with a proved-modulo-two-lemmas roadmap: (a) Bessel-Vandermonde
generic rank (analytic-determinant argument, numerically supported), (b) truncation control.
Honest status: this is a genuine attack with a plausible completion path, not a stuck attempt;
estimated 1–2 weeks of focused work to close, which the program's budget does not owe until the
empirical falsification hunt (S4e) reports. If S4e finds a counterexample pair, the conjecture
dies and the CP machinery becomes the tool for *characterizing* the failure.

## Literature consulted
Sussmann 1992 (tanh L=1); Fefferman 1994 (deep sigmoid via pole geometry — our Bessel lattice is
the periodic analogue of his pole configuration); Phuong & Lampert 2020 (ReLU);
Expand-and-Cluster 2304.12794 (overparameterized recovery — relevant to S4e tooling);
2604.14037 / 2605.18319 (semi-algebraic fiber machinery for ReLU, 2026); Kruskal 1977 CP
uniqueness (via standard references).

## Empirical wiring
S4e falsification hunt (production L=2 nets); Bessel rank numeric check (microcosm script);
if needed later: synthetic L=2 identifiability probe — fit two-layer nets to functions realized
by known two-layer teachers, align, and measure recovery rate (candidate E-track study, logged in
OPEN_PROBLEMS #1).
