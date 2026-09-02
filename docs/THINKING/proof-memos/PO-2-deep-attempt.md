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

---

# Addendum, 2026-09-02: one lemma closed, one dissolved, and a new obstacle named

## (B) Truncation control is not needed for the theorem

The memo lists truncation control as an open lemma: "identifying CP factors from finitely many
observed coefficients with an infinite tail." That framing is algorithmic, and identifiability is
not an estimation problem.

If $f_\theta = f_{\theta'}$ then *every* Fourier-Bohr coefficient agrees, exactly. We hold the whole
infinite tensor, not a truncation of it, and we are therefore free to *choose* a finite sub-block on
which to run a uniqueness argument. The values on that block carry no error and there is no tail to
control. Restricting $k_i$ to $\{0,\dots,K\}$ leaves both networks' rank-$n_2$ CP decompositions
intact, so a uniqueness theorem on the block transfers to the factors.

The lemma is not open. It is unnecessary. What it was really guarding against, noise-robust recovery
through a near-singular Bessel system, matters only for a constructive algorithm.

## (A) The Bessel-Vandermonde rank lemma is closed

The argument sketched in the memo is complete once stated carefully. $\det[J_k(x_j)]_{k=0..n-1}$ is
real-analytic on $(0,\infty)^n$. Substituting $x_j = \varepsilon t_j$ and using
$J_k(x) = (x/2)^k/k!\,(1+O(x^2))$,
$$\det[J_k(\varepsilon t_j)] \;=\; \varepsilon^{n(n-1)/2}\Big(\prod_{k<n}\tfrac{1}{k!\,2^k}\Big)
\prod_{a<b}(t_b - t_a)\;\big(1 + O(\varepsilon^2)\big),$$
which is nonzero for distinct $t_j$. So the determinant is not identically zero; its zero set is a
proper analytic subset, of Lebesgue measure zero, and full column rank is generic. That is exactly
the "off a proper analytic subset" the theorem statement asks for, so the memo's worry about
intersecting with the network's parameter measure dissolves: any measure absolutely continuous with
respect to Lebesgue gives the exceptional set zero mass.

Numerically confirmed: the ratio of the determinant to the predicted leading term converges to
$1.0000$ at $\varepsilon = 10^{-2}$ and $10^{-3}$ for $n = 2,3,4,5$.

## Uniqueness scope

With every factor matrix at full column rank, $k$-rank $= n_2$ and Kruskal's condition
$n_1 n_2 \ge 2n_2 + (n_1 - 1)$ holds **for $n_1 \ge 3$ and $n_2 \ge 2$**. Production networks at
$n_1 = n_2 = 32$ are far inside it. It fails only at $n_1 \le 2$ or $n_2 = 1$, which must be excluded
or handled separately. Kruskal is conservative here: with full column rank in several modes, milder
uniqueness results apply.

## The ill-conditioning is not an obstacle to the theorem, and it does bound the numerics

Smallest singular-value ratios of the factor matrices, 500 draws each: $10^{-4}$ at $n_2 = 2$,
$10^{-7}$ at $4$, $10^{-13}$ at $8$, $10^{-27}$ at $16$, $10^{-57}$ at $32$. A nonzero determinant is
a nonzero determinant in exact arithmetic, so this does not touch the proof. It does mean **float64
cannot verify the rank claim beyond $n_2 \approx 8$**, and the numerics above should not be read as
support at production width.

## The remaining obstacle, and it is subtler than this memo recorded

What is left is the memo's item 1, layer-1 peeling and spectral folding, listed there as "believed
sound, drafted, not verified." It is harder than that suggests, for a reason not previously noted.

For generic $v$ the $a_i$ are rationally independent, so the frequency module $\Lambda$ is free of
rank $n_1$. But **a free module does not determine its basis**: any unimodular transformation gives
another generating set. Recovering the $a_i$ therefore needs more than $\Lambda$ itself. The
plausible route is the coefficient decay: $J_k(x) \sim (x/2)^{|k|}/|k|!$ falls super-exponentially in
order, so the $a_i$ should be distinguished as the frequencies carrying the largest coefficients at
$|k|_1 = 1$. Making that an argument requires a quantitative separation between order-one and
higher-order coefficients that holds uniformly on the generic stratum, and large $W_{ji}$ works
against it.

That is now the critical path, and the honest estimate is that it is the substance of the conjecture
rather than bookkeeping around it.

## Status

Two open lemmas at the last writing; one closed, one shown unnecessary, one new obstacle named and
sharpened. The conjecture is not proved. It is closer, and what remains is now a single, precisely
stated question about recovering a distinguished basis of the frequency module from coefficient
decay.

---

# Addendum 2, 2026-09-02: the peeling obstacle has a route, and the exceptional stratum is explicit

The obstacle named in addendum 1 was that a free module does not determine its basis, so the
frequency module alone cannot recover the $a_i$. The route below removes the need to identify them
by magnitude at all.

## Decay recovers the norm, not the basis

For a primitive $u \in \mathbb{Z}^{n_1}$ define
$$\rho(u) \;=\; \lim_{m\to\infty} \frac{-\log|T(mu)|}{m \log m}.$$
Since $T(mu) = \sum_j \lambda_j \prod_i J_{m u_i}(W_{ji}) e^{ic_j}$ and
$J_k(x) \sim (x/2)^k/k!$, the $j$ with the largest $\prod_i |W_{ji}|^{u_i}$ dominates and Stirling
gives $-\log|T(mu)| = |u|_1\, m\log m + O(m)$. Hence
$$\boxed{\rho(u) = |u|_1.}$$

The point is that $\rho$ is defined without reference to any basis. It is a function on the module
itself, and it reconstructs the $\ell^1$ norm that the true basis induces.

**Consequence.** Let $\theta, \theta'$ realize the same $f$. Their frequency modules coincide, so
$a' = Ua$ for some $U \in GL(n_1,\mathbb{Z})$. Both expansions are product-Bessel, so both induce the
same $\rho$, so $U$ preserves the $\ell^1$ norm on $\mathbb{Z}^{n_1}$. The linear $\ell^1$
isometries of $\mathbb{Z}^n$ are exactly the signed permutation matrices. Therefore $U$ is a signed
permutation, which is precisely the layer-1 symmetry the theorem asserts.

## Numerical status

Fitting $-\log|T(mu)| = \rho\, m\log m + \beta m + \gamma$ over $m = 10..60$ at 200-digit precision:
$\rho$ matches $|u|_1$ to a worst deviation of $0.052$ over ten directions with $|u|_1$ from 1 to 6,
and to $0.057$ over random parameter draws at $n_2 \in \{2,3,4\}$. The residual shrinks with $|u|_1$
in the way a remaining $\log\log$ correction predicts.

## The exceptional stratum, found by trying to break it

Take $W_2 = -W_1$ and $\lambda_2 = -\lambda_1$. Because $J_k(-x) = (-1)^k J_k(x)$,
$$T(mu) \;=\; \lambda_1 \prod_i J_{m u_i}(W_{1i})\,\big(1 - (-1)^{m|u|_1}\big),$$
identically zero whenever $|u|_1$ is even. Measured: $\rho = 0$ on direction $(1,1,0)$, while the
odd directions are unaffected. So the non-cancellation hypothesis is not cosmetic, and the stratum
it must exclude is explicit and algebraic: sign-tied neuron pairs with opposite readout. It is
codimension at least one, hence measure zero, hence compatible with an "off a proper analytic
subset" statement.

## Where the conjecture now stands

Every major obstacle has a route:

| step | status |
|---|---|
| Jacobi-Anger reduction to a Bessel CP tensor | verified numerically to $6\times10^{-13}$ relative |
| truncation control | not needed; identifiability has exact data |
| Bessel-Vandermonde generic rank | closed by the small-argument expansion |
| CP uniqueness | Kruskal, for $n_1 \ge 3$, $n_2 \ge 2$ |
| layer-1 peeling | route above: decay recovers $\ell^1$, isometries are signed permutations |

**This is a proof sketch, not a proof.** What is missing is rigour rather than ideas: the
asymptotic for $\rho$ needs a statement with uniform control and a proof that the limit exists on
the generic stratum, the dominance argument needs the tie cases handled rather than observed, and
recovery of the biases $b_i$ once the $a_i$ are pinned has not been written. Those are a real piece
of work, but they are the kind of work that finishes, which is not what could be said a day ago.
