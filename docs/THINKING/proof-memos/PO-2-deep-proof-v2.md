# PO-2 at depth two: a restructured proof (v2)

Status: complete argument, restructured 2026-09-11 after a line-by-line audit of v1
(`PO-2-deep-proof.md`, kept as the record of how the route was found). It runs on vector
frequencies in $\mathbb{R}^m$ rather than on one line, identifies layer two by independence of
Bessel tails rather than by constructive peeling, covers vector outputs (the CIFAR-10 corpora have
$c = 3$), and needs five genericity conditions instead of seven. §9 lists every change and why.

**Not yet read by an independent expert in almost-periodic functions or tensor identifiability.
That reading is owed before submission.**

Notation: $m$ is the input dimension and $c$ the output dimension; the layer-two biases are
$\eta_j$ and the asymptotic index is $t$, so neither letter is reused.

## 1. Setting and group

$$
f_\theta(x) \;=\; \sum_{j=1}^{n_2} v_j \,\sin\!\Big(\sum_{i=1}^{n_1} W_{ji}\,\sin(\langle w_i,x\rangle + b_i) + \eta_j\Big) + \beta,
$$
$w_i \in \mathbb{R}^m$, $b_i, \eta_j \in \mathbb{R}$, $W \in \mathbb{R}^{n_2\times n_1}$,
$v_j, \beta \in \mathbb{R}^c$. $G_1 = D_\infty \wr S_{n_1}$ acts on the triples
$(w_i, b_i, W_{\cdot i})$, $G_2 = D_\infty \wr S_{n_2}$ on $(W_{j\cdot}, \eta_j, v_j)$, each neuron
by $g_{d,q}(w,b,u) = ((-1)^d w, (-1)^d b + \pi q, (-1)^{d+q}u)$. The actions touch $W$ on columns
and rows respectively, commute, and preserve $f_\theta$.

## 2. Genericity

$\Theta_2$ is the set of $\theta$ with

- **(A1)** $\sum_i k_i w_i \neq 0$ for every $k \in \mathbb{Z}^{n_1}\setminus\{0\}$;
- **(A2)** $W_{ji} \ne 0$ and $J_0(W_{ji}) \ne 0$ for all $i,j$;
- **(A3)** for each $i$, the $|W_{ji}|$ are pairwise distinct over $j$;
- **(A4)** $\eta_j \notin \frac{\pi}{2}\mathbb{Z}$;
- **(A5)** $v_j \neq 0 \in \mathbb{R}^c$.

(A1) implies $w_i \ne 0$ and $w_i \ne \pm w_l$, so $\Theta_2$ refines the $L=1$ stratum.

**Size.** Each condition fails on a countable union of zero sets of real-analytic functions that
are not identically zero, so $\Theta_2$ has full measure and is comeagre. (A2)–(A5) exclude closed
nowhere-dense sets. So does (A1) when $n_1 \le m$, since $\mathbb{Z}$-dependence then implies
$\mathbb{R}$-dependence. When $n_1 > m$, as in every corpus ($m = 2$, $n_1 = 32$), the complement of
(A1) is dense: $n_1 > m$ vectors in $\mathbb{Q}^m$ are always $\mathbb{Z}$-dependent, so no
rational-weight network satisfies (A1), floating-point networks included. A statement on
$\Theta_2$ holds almost surely under any parameter density, which is what the information
corollary needs, but says nothing about one finite-precision network.

**Invariance.** $\Theta_2$ is $G$-invariant: $G$ permutes and negates rows and columns of $W$,
negates and permutes the $w_i$, and shifts biases by multiples of $\pi$.

## 3. Expansion and Bohr coefficients

With $\varphi_i(x) = \langle w_i,x\rangle + b_i$, Jacobi–Anger gives
$$
f_\theta(x) = \beta + \operatorname{Im}\sum_{k\in\mathbb{Z}^{n_1}} B(k)\, e^{\mathrm{i}k\cdot\varphi(x)},
\qquad B(k) = \sum_j v_j e^{\mathrm{i}\eta_j}\prod_i J_{k_i}(W_{ji}) \in \mathbb{C}^c,
$$
absolutely and uniformly convergent. Frequencies are $\Omega(k) = \sum_i k_i w_i$. Using
$B(-k) = (-1)^{|k|}B(k)$, $|k| = \sum_i k_i$, the coefficient of $e^{\mathrm{i}\Omega(k)\cdot x}$ is
$$
T(k) = \begin{cases}
\beta + \operatorname{Im}B(0), & k=0,\\
e^{\mathrm{i}k\cdot b}\operatorname{Im}B(k), & k\ne0,\ |k| \text{ even},\\
-\mathrm{i}\,e^{\mathrm{i}k\cdot b}\operatorname{Re}B(k), & |k| \text{ odd}.
\end{cases}
$$
Define $\hat f(\omega) = \lim_{R\to\infty}(2R)^{-m}\int_{[-R,R]^m} f(x)e^{-\mathrm{i}\omega\cdot x}dx$.
The cube mean of $e^{\mathrm{i}\nu\cdot x}$ is $\prod_r \sin(\nu_rR)/(\nu_rR) \to \mathbf 1_{\nu=0}$,
absolute summability passes the limit through the sum, and (A1) makes $\Omega$ injective, so
$\hat f(\Omega(k)) = T(k)$ and $\hat f = 0$ off $\Omega(\mathbb{Z}^{n_1})$. No uniqueness theorem for
almost-periodic functions is needed: $\hat f$ is defined from $f$ directly.

## 4. Bessel lemmas

**Lemma 1.** $J_k(x) = \frac{(x/2)^k}{k!}(1+E_k(x))$ with $|E_k(x)| \le e^{x^2/(4(k+1))}-1$
($k \ge 0$). *Proof:* ascending series and $k!/(s+k)! \le (k+1)^{-s}$. $\square$

**Lemma 2 (independent tails).** Let $N_1,\dots,N_n \subseteq \mathbb{Z}_{>0}$ be infinite and
$y^{(1)},\dots,y^{(r)} \in (0,\infty)^n$ pairwise distinct. The functions
$\phi_s(k) = \prod_i J_{k_i}(y^{(s)}_i)$ on $N_1\times\dots\times N_n$ are linearly independent.

*Proof.* $n = 1$: if $\sum_s\alpha_sJ_k(y_s) = 0$ on $N_1$ with some $\alpha_s \ne 0$, take the largest
$y_{s^\star}$ with $\alpha_{s^\star} \ne 0$, multiply by $k!/(y_{s^\star}/2)^k$ and let $k \to \infty$
in $N_1$: Lemma 1 gives the limit $\alpha_{s^\star}$, a contradiction. $n > 1$: group the $s$ by
$y^{(s)}_1$; for fixed $(k_2..k_n)$ the $n=1$ case kills each group's coefficient; inside a group the
tails $(y^{(s)}_2..y^{(s)}_n)$ are distinct; induct. Vector coefficients reduce to coordinates. $\square$

## 5. Decay along rays

**Proposition 3.** (a) For every $\theta$ and $u \in \mathbb{Z}^{n_1}\setminus\{0\}$,
$\liminf_t \frac{-\log\|T(tu)\|}{t\log t} \ge |u|_1$ (with $-\log 0 = \infty$).
(b) For $\theta \in \Theta_2$, $T(te_i) \ne 0$ for all large $t$ and
$\lim_t \frac{-\log\|T(te_i)\|}{t\log t} = 1$.

*Proof.* (a) $\|T(tu)\| \le \sum_j\|v_j\|\prod_i|J_{tu_i}(W_{ji})|$; for $u_i \ne 0$ Lemma 1 bounds the
factor by $2(|W_{ji}|/2)^{t|u_i|}/(t|u_i|)!$ for large $t$, $|J_0| \le 1$, and Stirling gives
$|u|_1t\log t + O(t)$. (b) Up to a unimodular factor
$T(te_i) = \sum_j v_j s^{(t)}_j J_t(W_{ji})\prod_{l\ne i}J_0(W_{jl})$ with
$s^{(t)}_j \in \{\sin\eta_j, \cos\eta_j\}$ by the parity of $t$. By (A3) one $j^\star$ maximises
$|W_{ji}|$; the rest are smaller by $O((|W_{ji}|/|W_{j^\star i}|)^t)$. The leading term is nonzero by
(A5), (A4), (A2), and its logarithm is $-t\log t + O(t)$. $\square$

**Remark.** The limit along a general ray need not be $|u|_1$ on $\Theta_2$. Rows $(a,b)$, $(b,a)$ of
$W$ with $\eta_1 = \eta_2$ and $v_2 = -v_1$ give $T(t(1,1)) = 0$ for all $t$ (checked to 80 digits).
Equality on every ray holds if the $\log|W_{ji}| - \log|W_{j'i}|$ are rationally independent, which
v1 assumed as (G2); the proof does not need it.

## 6. Layer one

Let $\theta,\theta' \in \Theta_2$, $f_\theta = f_{\theta'}$ on an open set, hence everywhere.

*Lattice.* By Prop. 3(b) $tw_i, (t+1)w_i \in \operatorname{Spec}f$ for large $t$, so
$\operatorname{Spec}f$ generates $\Lambda = \Omega(\mathbb{Z}^{n_1})$, free of rank $n_1$ with basis
$(w_i)$ by (A1). Same for $\theta'$; hence $n_1 = n_1'$ and $w'_i = \sum_l U_{il}w_l$,
$U \in GL(n_1,\mathbb{Z})$.

*Signed permutation.* Fix $i$, let $k = U_{i\cdot}$, so $tw'_i = \Omega(tk)$. $\hat f(tw'_i)$ equals
$T'(te_i)$ and $T(tk)$. Prop. 3(b) for $\theta'$ and 3(a) for $\theta$ give $1 \ge |k|_1$; $k \ne 0$,
so $k = \epsilon_i e_{\pi(i)}$. $U$ invertible makes $\pi$ a permutation: $w'_i = \epsilon_iw_{\pi(i)}$.

*Biases.* $T'(te_i) = T(t\epsilon_ie_{\pi(i)})$ for all $t$; they are
$\zeta_te^{\mathrm{i}tb'_i}X'_t$ and $\zeta_te^{\mathrm{i}t\epsilon_ib_{\pi(i)}}X_t$ with the same
$\zeta_t \in \{1,-\mathrm{i}\}$ (parity of $t$ equals parity of $t\epsilon_i$) and real vectors
$X_t, X'_t$, nonzero for large $t$. So $e^{\mathrm{i}t\delta_i} \in \mathbb{R}$ for large $t$, where
$\delta_i = b'_i - \epsilon_ib_{\pi(i)}$; the quotient of consecutive $t$ gives
$e^{\mathrm{i}\delta_i} \in \mathbb{R}$, $\delta_i \in \pi\mathbb{Z}$.

So $(w'_i, b'_i) = g_{d_i,q_i}(w_{\pi(i)}, b_{\pi(i)})$ in its first two slots. With $g_1 \in G_1$ the
corresponding element, $\theta'' = g_1^{-1}\theta' \in \Theta_2$ realises $f$ and has the first
layer of $\theta$ exactly.

## 7. Layer two

$\theta, \theta''$ share $(w,b)$, so $\operatorname{Im}B = \operatorname{Im}B''$ on $k \ne 0$ with
$|k|$ even and $\operatorname{Re}B = \operatorname{Re}B''$ on $|k|$ odd. Put
$y_j = (|W_{j1}|..|W_{jn_1}|)$, $\epsilon_{ji} = \operatorname{sign}W_{ji}$,
$\lambda_j = v_j\sin\eta_j$, $\mu_j = v_j\cos\eta_j$ (nonzero by (A4), (A5)).

*Rows.* On $(2\mathbb{Z}_{>0})^{n_1}$:
$\sum_j\lambda_j\prod_iJ_{k_i}((y_j)_i) = \sum_j\lambda''_j\prod_iJ_{k_i}((y''_j)_i)$. The $y_j$ are
distinct (A3), as are the $y''_j$. Lemma 2 on the merged list: each $y_j$ equals exactly one
$y''_{\tau(j)}$ with the same coefficient; $\tau$ is a bijection, $n_2 = n_2'$. Relabel by $\tau$.

*Signs.* On $k_i$ odd positive, other coordinates even positive:
$\sum_j\mu_j\epsilon_{ji}\prod_lJ_{k_l}((y_j)_l) = \sum_j\mu''_j\epsilon''_{ji}\prod_lJ_{k_l}((y_j)_l)$,
so $\mu_j\epsilon_{ji} = \mu''_j\epsilon''_{ji}$; $\mu_j \ne 0$ makes $s_j = \epsilon''_{ji}\epsilon_{ji}$
independent of $i$: $W''_{j\cdot} = s_jW_{j\cdot}$, $\mu''_j = s_j\mu_j$, $\lambda''_j = \lambda_j$.

*Output layer.* $s_j = 1$: $v''_je^{\mathrm{i}\eta''_j} = v_je^{\mathrm{i}\eta_j}$ in $\mathbb{C}^c$; a
coordinate with $v_{jr} \ne 0$ gives $\eta''_j = \eta_j + \pi q$, then $v''_j = (-1)^qv_j$: $g_{0,q}$.
$s_j = -1$: $v''_je^{\mathrm{i}\eta''_j} = -v_je^{-\mathrm{i}\eta_j}$, the same for $\sigma$ applied
first: $g_{1,q}$. Then all hidden parameters and $V$ agree, and $\beta'' = \beta$ since the functions
differ by a constant.

## 8. Theorem, uniqueness, scope

**Theorem.** $\theta,\theta' \in \Theta_2$ of any widths and output dimension, $f_\theta = f_{\theta'}$
on an open set $\Rightarrow$ $n_1 = n_1'$, $n_2 = n_2'$, $\theta' = g\theta$ for a unique
$g \in G_1 \times G_2$.

*Uniqueness.* A stabiliser element is trivial on layer one by (A1) and the bias equations, then on
layer two by (A3) (permutation), $W_{j\cdot} \ne 0$ (sign) and the biases (winding). $\square$

**Scope.** Both parameters are assumed generic. One-sidedly, $n_1$ is minimal: $\operatorname{Spec}f$
generates a lattice of rank $n_1$ and any realisation places it in a lattice of rank at most its own
first width. It is not shown that a generic $\theta$ has no equivalent of smaller second width
outside $\Theta_2$. The corollary in the paper compares parameters inside $\Theta_2$ and needs
nothing more; the paper must not claim minimality beyond this.

**Necessity.** (A5) is necessary ($v_j = 0$ frees $W_{j\cdot}$ and $\eta_j$), as are $w_i \ne 0$ and
$w_i \ne \pm w_l$ (phasor addition) and nonzero columns of $W$. Sine also has the null network
$\sin z + \sin(z+2\pi/3) + \sin(z+4\pi/3) = 0$, so no non-generic statement in the sense of
Vlačić–Bölcskei can hold. Whether the rational part of (A1) can be dropped is open.

**Where each condition is used.** (A1): injectivity of $\Omega$, lattice basis, uniqueness. (A2):
nonvanishing $J_0$ factors in Prop. 3(b); positive magnitudes in §7. (A3): unique dominant term in
Prop. 3(b); distinct rows in §7; uniqueness. (A4), (A5): nonvanishing leading term; $\lambda_j,
\mu_j \ne 0$.

## 8b. One-sided form (2026-09-13)

The argument needs genericity of $\theta$ only. Let $\theta \in \Theta_2$ and $\theta'$ be *any*
depth-two network with the same function.

- *Layer one.* $\operatorname{Spec}f$ generates $\Lambda$ (rank $n_1$, by Prop. 3(b) for $\theta$)
  and lies in $\Lambda' = \Omega'(\mathbb{Z}^{n_1'})$ for any $\theta'$, so $n_1 \le \operatorname{rank}
  \Lambda' \le n_1'$. If $n_1' = n_1$, $n_1$ generators of a free group of rank $n_1$ are a basis, so
  $\theta'$ satisfies (A1) automatically. Write $w_i = \sum_l M_{il}w'_l$; Prop. 3(b) for $\theta$ and
  3(a) for $\theta'$ (condition-free) give $|M_{i\cdot}|_1 \le 1$, so $M$ is a signed permutation.
  Biases as before, using nonvanishing on the $\theta$ side only.
- *Layer two.* On the even grid, rows of $\theta''$ with a zero entry vanish ($J_k(0) = 0$, $k \ge 1$);
  Lemma 2 forces each $y_j$ (nonzero coefficient) to be the magnitude vector of at least one
  $\theta''$ row, distinct $j$ distinct rows, so $n_2' \ge n_2$. If $n_2' = n_2$ the matching is a
  bijection; signs and output layer as before, using $\mu_j \ne 0$ on the $\theta$ side only.

**Theorem (one-sided).** For $\theta \in \Theta_2$ and any depth-two $\theta'$ with $f_{\theta'} =
f_\theta$: $n_1' \ge n_1$; if $n_1' = n_1$ then $n_2' \ge n_2$; if both agree, $\theta' = g\theta$
for a unique $g$. So $\theta$ is identifiable among all networks no wider in either layer. Not
shown: a bound on $n_2'$ when $n_1' > n_1$.

**Correction to §8 "Necessity".** The three-phase null network $\sin z + \sin(z+2\pi/3) +
\sin(z+4\pi/3)$ is *reducible* in Vlačić–Bölcskei's sense (identical incoming weights), so it does
not show that their null-net condition fails for sine; whether it holds is open. Necessary strata
also include layer-two parallel rows $W_{j\cdot} = \pm W_{j'\cdot}$ and zero columns of $W$.

## 9. Changes from v1

1. **One line versus $\mathbb{R}^m$.** v1 restricted to $x = tv$, pinned only the projections
   $\langle w_i, v\rangle$, and put $v$ inside a genericity condition. v2 works with vector
   frequencies and the intrinsic (A1).
2. **Cross-coordinate labels.** v1 obtained each column's multiset $\{|W_{ji}|\}_j$ without linking
   columns. Lemma 2 on the product grid matches whole rows.
3. **Hidden neurons.** v1 froze one coordinate at order 2; at a zero of $J_2$ that annihilates a
   neuron, and no condition excluded it. v2 uses every positive order in every coordinate.
4. **Vector outputs** throughout; a common bias phase forces a common sign across coordinates.
5. **Two conditions removed.** v1's (G2), rational independence of log-magnitude differences, is not
   needed: the signed-permutation step only uses rays along basis vectors, exact on one side by (A3)
   and a lower bound on the other with no condition. v1's (G7), $\operatorname{Re}B(e_i) \ne 0$, is
   not needed: $tw_i$ and $(t+1)w_i$ lie in the spectrum for large $t$, which generates the lattice,
   and the bias follows from the phases at consecutive $t$.
6. **Stated precisely:** uniqueness of $g$; $G$-invariance of $\Theta_2$; when the exceptional set is
   dense; that only two-sided genericity is proved.

v1's folding formula, Lemma 1 and the decay asymptotic (now Prop. 3 and its Remark) are kept, and
v1's numerical record still applies to them. v1's constructive peeling is correct and gives an
algorithm; v2 does not need it.

## References

- V. Vlačić and H. Bölcskei. Affine symmetries and neural network identifiability. *Advances in
  Mathematics* 376:107485, 2021.
- G. N. Watson. *A Treatise on the Theory of Bessel Functions*, 2nd ed., 1944.
- G. Yüce, G. Ortiz-Jiménez, B. Besbinar, P. Frossard. A structured dictionary perspective on
  implicit neural representations. CVPR 2022. (Jacobi–Anger expansion of sine networks.)
- T. Novello. Understanding sinusoidal neural networks. arXiv:2212.01833, 2022.
