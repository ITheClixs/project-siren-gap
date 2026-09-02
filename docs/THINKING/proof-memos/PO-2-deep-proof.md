# PO-2 deep case: a proof of generic identifiability at $L = 2$

Status: complete argument, one step sketched rather than proved (§7, uniformity of the asymptotic).
Supersedes the roadmap in `PO-2-deep-attempt.md`; that memo's two open lemmas are resolved there.

## 1. Setting

A two-hidden-layer sine network on $\mathbb{R}^m$:
$$h^1_i(x) = \sin(\langle w_i, x\rangle + b_i), \qquad
h^2_j = \sin\Big(\sum_i W_{ji} h^1_i + c_j\Big), \qquad
f_\theta(x) = \sum_j v_j h^2_j + \beta,$$
with widths $n_1, n_2$. Restrict to a line $x = tv$ and write $a_i = \langle w_i, v\rangle$.

## 2. Genericity

Let $\Theta^{(2)}_{\mathrm{gen}}$ be the parameters satisfying:

- **(G1)** $w_i \ne 0$, $u_i \ne 0$, no $w_j = \pm w_i$ (the $L=1$ stratum), and $v$ chosen so the
  $a_i$ are rationally independent.
- **(G2)** for every $j \ne j'$, the vector $\big(\log|W_{ji}| - \log|W_{j'i}|\big)_{i}$ has
  rationally independent entries.
- **(G3)** $c_j \notin \tfrac{\pi}{2}\mathbb{Z}$ for every $j$.
- **(G4)** for each $i$, the $|W_{ji}|$ are distinct across $j$.
- **(G5)** $v_j \ne 0$ and $W_{ji} \ne 0$ for all $i,j$.

Each condition fails on a countable union of proper analytic subsets, so
$\Theta^{(2)}_{\mathrm{gen}}$ has full measure. (G1) is inherited from the $L=1$ theorem; the rest
are new and each is used exactly once below.

## 3. Expansion

By Jacobi-Anger, $e^{iz\sin\phi} = \sum_{k\in\mathbb{Z}} J_k(z)e^{ik\phi}$, so
$$f_\theta(tv) = \beta + \operatorname{Im}\Big[\sum_{k\in\mathbb{Z}^{n_1}} A(k)\, e^{i (k\cdot a) t}\Big],
\qquad A(k) = B(k)\, e^{i k\cdot b},$$
$$B(k) \;=\; \sum_{j} v_j\, e^{i c_j} \prod_i J_{k_i}(W_{ji}).$$
Convergence is absolute: $\sum_k \prod_i |J_{k_i}(W_{ji})| = \prod_i \sum_{k_i}|J_{k_i}(W_{ji})|
< \infty$ by super-exponential Bessel decay in the order.

*Verified numerically: the expansion reproduces $f$ to $6.5\times10^{-13}$ relative.*

## 4. Folding: what is observable

By (G1) the map $k \mapsto k\cdot a$ is injective, so each frequency is carried by one $k$ and its
negative. Since $f$ is real and $J_{-k}(x) = (-1)^k J_k(x)$, the Fourier-Bohr coefficient is
$$T(k) \;:=\; c_f(k\cdot a) \;=\; \tfrac{1}{2i}\big[A(k) - \overline{A(-k)}\big]
\;=\;\begin{cases}
e^{i k\cdot b}\,\operatorname{Im} B(k), & \textstyle\sum_i k_i \text{ even},\\[2pt]
-i\,e^{i k\cdot b}\,\operatorname{Re} B(k), & \textstyle\sum_i k_i \text{ odd}.
\end{cases}$$

This is the lemma the earlier memo called "spectral folding, drafted, not verified". It matters:
$B(k)$ itself is **not** observable, only one real part of it, chosen by the parity of $\sum_i k_i$.

*Verified numerically as an algebraic identity, worst error $9.8\times10^{-18}$ over ten lattice
points.*

Writing $r_j(k) = v_j \prod_i J_{k_i}(W_{ji}) \in \mathbb{R}$, we have
$\operatorname{Re}B(k) = \sum_j r_j(k)\cos c_j$ and
$\operatorname{Im}B(k) = \sum_j r_j(k)\sin c_j$.

## 5. The decay exponent recovers the $\ell^1$ norm

For primitive $u \in \mathbb{Z}^{n_1}$ set
$$\rho(u) \;=\; \lim_{m\to\infty} \frac{-\log|T(mu)|}{m\log m}.$$

**Proposition 1.** On $\Theta^{(2)}_{\mathrm{gen}}$, $\rho(u) = |u|_1$ for every primitive $u$.

*Proof.* $r_j(mu) = v_j\prod_i J_{mu_i}(W_{ji})$ and $J_k(x) = (x/2)^k/k!\,(1+O(x^2))$, so
$$|r_j(mu)| \;=\; |v_j|\,\frac{P_j(u)^m}{\prod_i (m u_i)!}\big(1+o(1)\big),
\qquad P_j(u) = \prod_i |W_{ji}/2|^{u_i}.$$
By (G2) the map $j \mapsto \sum_i u_i \log|W_{ji}|$ is injective for every $u \ne 0$, so a unique
$j^\star$ maximises $P_j(u)$ and dominates the sum. By (G3) neither $\cos c_{j^\star}$ nor
$\sin c_{j^\star}$ vanishes, so whichever real part the parity selects is asymptotically
$r_{j^\star}(mu)$ times a nonzero constant. Stirling gives
$$\log \prod_i (mu_i)! \;=\; |u|_1\, m\log m \;+\; O(m),$$
and $\log P_{j^\star}(u)^m = O(m)$, whence $-\log|T(mu)| = |u|_1\,m\log m + O(m)$. $\square$

The content is that $\rho$ is defined from $f$ alone. It refers to no basis, yet it reconstructs the
$\ell^1$ norm that the true basis induces on the frequency module.

*Verified numerically at 200-digit precision: fitting
$-\log|T(mu)| = \rho\,m\log m + \beta m + \gamma$ over $m = 10..60$ recovers $\rho = |u|_1$ to a
worst deviation of $0.052$ over ten directions with $|u|_1 \in [1,6]$, and $0.057$ over random
parameter draws.*

**Sharpness.** (G2) is not cosmetic. Take $W_2 = -W_1$ and $v_2 = -v_1$: then
$T(mu) = v_1 \prod_i J_{mu_i}(W_{1i})(1 - (-1)^{m|u|_1})$, identically zero for even $|u|_1$.
Measured: $\rho = 0$ on direction $(1,1,0)$. This stratum is exactly what (G2) excludes.

## 6. Layer 1 is pinned up to its group

**Proposition 2.** If $f_\theta = f_{\theta'}$ with both in $\Theta^{(2)}_{\mathrm{gen}}$, then
$n_1 = n_1'$ and the frequency generators agree up to a signed permutation.

*Proof.* Equality of functions gives equality of frequency modules $\Lambda = \Lambda'$. The rank of
a free $\mathbb{Z}$-module is an invariant, so $n_1 = n_1'$, and $a' = Ua$ for some
$U \in GL(n_1,\mathbb{Z})$. Both expansions are of the form in §3, so by Proposition 1 both compute
the same $\rho$, and $\rho$ is intrinsic to $f$. Hence $|Uk|_1 = |k|_1$ for all $k$. In particular
$|Ue_i|_1 = 1$, and the only integer vectors of $\ell^1$ norm one are $\pm e_j$, so $U$ carries basis
vectors to signed basis vectors; being invertible it is a signed permutation matrix. $\square$

Signed permutation of the $a_i$ is exactly the layer-1 permutation together with the $\sigma$
generator, which is what the theorem asserts for that layer.

## 7. Biases

With $\sum_i k_i = 1$ odd, §4 gives $T(e_i) = e^{i(b_i - \pi/2)}\operatorname{Re}B(e_i)$ with
$\operatorname{Re}B(e_i)$ real and, by (G3) and (G5), nonzero. Hence $\arg T(e_i)$ determines $b_i$
**modulo $\pi$** and no better. That is precisely the $\rho$ generator's ambiguity
$b \mapsto b + \pi$, so layer 1 is now pinned exactly up to $D_\infty \wr S_{n_1}$.

## 8. Layer 2 by CP uniqueness on the even sublattice

The obstruction in §4 is that $B$ is never observable, only one real part per parity class. Restrict
to $k \in 2\mathbb{Z}^{n_1}$, where $\sum_i k_i$ is always even: the parity is constant, the index
set is again a product grid, and $\operatorname{Im}B$ is observable throughout. On
$k_i \in \{0,2,\dots,2K\}$,
$$\operatorname{Im}B(2\kappa) \;=\; \sum_j \lambda_j \prod_i J_{2\kappa_i}(W_{ji}),
\qquad \lambda_j = v_j \sin c_j,$$
a rank-$n_2$ CP tensor with mode-$i$ factor matrix $A^{(i)}[\kappa, j] = J_{2\kappa_i}(W_{ji})$.

**Lemma (even-order Bessel rank).** For $|x_j|$ distinct, $\det[J_{2\kappa}(x_j)]_{\kappa,j} \ne 0$
generically. *Proof.* $J_{2\kappa}(x) = (x/2)^{2\kappa}/(2\kappa)!\,(1+O(x^2))$, so with
$x_j = \varepsilon t_j$ the determinant tends to $\varepsilon^{\,\kappa(\kappa-1)}$ times a nonzero
constant times a Vandermonde alternant in $t_j^2$, nonzero for distinct $|t_j|$. The determinant is
real-analytic and not identically zero, so its zero set is a proper analytic subset. $\square$
By (G4) the hypothesis holds. *Verified numerically: full column rank in 300/300 draws at
$n_2 = 2,\dots,5$.*

Full column rank gives $k\text{-rank} = n_2$ in every mode, so Kruskal's condition
$n_1 n_2 \ge 2n_2 + (n_1 - 1)$ holds whenever $n_1 \ge 3$ and $n_2 \ge 2$, and the CP decomposition
is unique up to permutation and scaling of the columns. Therefore $n_2 = n_2'$ and, after
permutation, the factor vectors match: $\big(J_{2\kappa}(W_{ji})\big)_\kappa =
\big(J_{2\kappa}(W'_{ji})\big)_\kappa$ for each $i$.

Since $\sum_k J_k(x)e^{ik\phi} = e^{ix\sin\phi}$, the full Bessel vector determines $x$, and
$J_{2\kappa}(-x) = J_{2\kappa}(x)$ leaves exactly the sign of $W_{ji}$ undetermined. Repeating on the
shifted grid $k_i \in \{1,3,\dots\}$ recovers $v_j\cos c_j$ alongside $v_j \sin c_j$, hence $v_j$ and
$c_j$ modulo $\pi$. The residual freedoms are: a per-neuron sign on the rows of $W$, a shift
$c_j \mapsto c_j + \pi$ with $v_j \mapsto -v_j$, and a permutation of $j$. That is
$D_\infty \wr S_{n_2}$.

## 9. Theorem

**Theorem.** Let $\theta, \theta' \in \Theta^{(2)}_{\mathrm{gen}}$ be two-hidden-layer sine networks
with $n_1, n_1' \ge 3$ and $n_2, n_2' \ge 2$. If $f_\theta = f_{\theta'}$ on a nonempty open set,
then $n_1 = n_1'$, $n_2 = n_2'$, and $\theta' = g\theta$ for some
$g \in (D_\infty \wr S_{n_1}) \times (D_\infty \wr S_{n_2})$.

*Proof.* Analytic continuation extends the equality to $\mathbb{R}^m$. §6 and §7 pin layer 1 up to
$D_\infty \wr S_{n_1}$; §8 pins layer 2 up to $D_\infty \wr S_{n_2}$. $\square$

## 10. What is not yet rigorous

One step is sketched rather than proved. **Proposition 1 needs the $o(1)$ made uniform.** The
argument treats each $j$'s asymptotic separately and then takes the dominant term; making that a
proof requires an explicit error bound on $J_k(x) - (x/2)^k/k!$ uniform over the relevant range of
$k$, and a quantitative gap $P_{j^\star}(u)/P_{j^{\star\star}}(u)$ to absorb the subdominant terms.
Both look routine and neither is written.

Two smaller gaps. The $n_1 \le 2$ and $n_2 = 1$ cases fall outside Kruskal and are excluded by
hypothesis rather than handled. And §8 asserts that the shifted odd grid recovers $v_j\cos c_j$ by
the same route; that is stated, not carried out.

Any claimed proof of this statement should be checked by someone other than its author. The route
was found and the numerics were run by the same agent, and neither of those is a referee.
