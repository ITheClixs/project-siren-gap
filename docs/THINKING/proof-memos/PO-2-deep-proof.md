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
- **(G6)** $J_0(W_{ji}) \ne 0$ for all $i,j$, i.e. no $W_{ji}$ is a zero of $J_0$.

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

**Lemma 1 (explicit Bessel control).** Write $J_k(x) = \frac{(x/2)^{k}}{k!}\big(1+E_k(x)\big)$ for
$k \ge 0$. Then
$$|E_k(x)| \;\le\; \exp\Big(\frac{x^2}{4(k+1)}\Big) - 1 \;=\; O\!\big(x^2/k\big).$$

*Proof.* From the series $E_k(x) = \sum_{s\ge1}\frac{(-1)^s k!}{s!(s+k)!}(x/2)^{2s}$ and
$\frac{k!}{(s+k)!} = \prod_{r=1}^{s}(k+r)^{-1} \le (k+1)^{-s}$,
$$|E_k(x)| \;\le\; \sum_{s\ge1}\frac{1}{s!}\Big(\frac{(x/2)^2}{k+1}\Big)^{s} \;=\; e^{x^2/(4(k+1))}-1. \qquad\square$$

Because $J_{-k} = (-1)^kJ_k$ and $J_k(-x) = (-1)^kJ_k(x)$, the bound governs $|J_k(x)|$ for either
sign of $k$ and of $x$ with $k$ replaced by $|k|$. For $|k| \ge x^2$ it gives $|E_k| \le e^{1/4}-1
< 0.29$, hence a two-sided bound. This is the explicit, uniform-on-compacts, $O(1/k)$ control that
the previous draft assumed without proof.

*Verified: the bound holds in all 24 tested cases over $x \in [0.5,6]$, $k \in [1,60]$, with worst
ratio $|E_k|/\text{bound} = 0.999$, so it is valid and not vacuous; and it agrees with $x^2/(4k)$ to
three digits at $k = 1000$.*

Fix a primitive $u$ and set, for each $j$,
$$P_j(u) = \prod_{i:u_i\ne0}\big|W_{ji}/2\big|^{|u_i|}, \qquad
Q_j(u) = \prod_{i:u_i=0}\big|J_0(W_{ji})\big|,$$
$s_j = \sin c_j$ or $\cos c_j$ according to the parity of $\sum_i k_i$, and let $j^\star$ maximise
$P_j(u)$ with $\gamma(u) = P_{j^\star}(u)/P_{j^{\star\star}}(u)$ the ratio to the runner-up.

**Proposition 1.** For each fixed primitive $u$, on $\Theta^{(2)}_{\mathrm{gen}}$,
$$|T(mu)| \;=\; \frac{|v_{j^\star}|\,Q_{j^\star}(u)\,|s_{j^\star}|\;P_{j^\star}(u)^{m}}
{\prod_{i:u_i\ne0}\big(m|u_i|\big)!}\;\Big(1 + O(m^{-1}) + O\big(\gamma(u)^{-m}\big)\Big),$$
hence $-\log|T(mu)| = |u|_1\,m\log m + O(m)$ and $\rho(u) = |u|_1$.

*Proof.* Coordinates with $u_i = 0$ contribute the constant $J_0(W_{ji})$, nonzero by (G6); without
(G6) a single vanishing factor would kill a term for every $m$ at once. For the remaining
coordinates $|mu_i| \to \infty$, so once $m \ge \max_{j,i}W_{ji}^2 / \min_{i:u_i\ne0}|u_i|$ Lemma 1
applies to each, and since there are finitely many pairs $(i,j)$ the product of the $1+E$ factors is
$1 + O(1/m)$ with a constant depending only on $\max|W_{ji}|$ and $n_1$. Therefore
$$r_j(mu) \;=\; \pm\,\frac{|v_j|\,Q_j(u)\,P_j(u)^m}{\prod_{i:u_i\ne0}(m|u_i|)!}\,\big(1+O(1/m)\big).$$
The denominator does not depend on $j$. This is the point: the entire $j$-dependence sits in
$|v_j|Q_j(u)P_j(u)^m$, so comparing terms needs no control of the factorials. By (G2), $P_j(u) =
P_{j'}(u)$ would force $u \perp (\log|W_{ji}|-\log|W_{j'i}|)_i$, impossible for $u \ne 0$; hence
$j^\star$ is unique and $\gamma(u) > 1$ strictly. The other $n_2-1$ terms are therefore smaller by
$O(\gamma(u)^{-m})$ relative to the leading one. By (G3) both $|\sin c_{j^\star}|$ and
$|\cos c_{j^\star}|$ are nonzero, so whichever the parity selects is bounded below by a positive
constant independent of $m$, and no cancellation can occur in the leading term. This is the display.
Taking logarithms and applying Stirling coordinatewise,
$$\log\!\!\prod_{i:u_i\ne0}\!(m|u_i|)! \;=\; \sum_{i}\Big[m|u_i|\log(m|u_i|) - m|u_i| +
\tfrac12\log(2\pi m|u_i|)\Big] + O(1/m) \;=\; |u|_1\,m\log m + O(m),$$
while $m\log P_{j^\star}(u)$ and the constants are $O(m)$ and $O(1)$. Dividing by $m\log m$ gives
$\rho(u) = |u|_1$ with error $O(1/\log m)$. $\square$

*Verified numerically at 220-digit precision, testing the constant and not merely the exponent. The
ratio of $|T(mu)|$ to the predicted right-hand side tends to $1$ along every direction tested, and
the two error terms separate cleanly: for $u = (1,1,1)$, where $\gamma = 1.0035$, the deviation falls
$0.326 \to 0.0018$ across $m = 64 \dots 2400$ in step with $\gamma^{-m}$, while for $u = (1,1,0)$,
where $\gamma = 1.043$, the $\gamma^{-m}$ term is spent by $m \approx 150$ and the residual then
reads $0.0052, 0.0026, 0.0013$ at $m = 600, 1200, 2400$, halving as $m$ doubles exactly as the
$O(1/m)$ of Lemma 1 requires.*

**The limit is pointwise in $u$ and is not uniform.** $\gamma(u)$ may come arbitrarily close to $1$
as $u$ ranges over the lattice, and then the approach to the limit is arbitrarily slow: the measured
$\gamma = 1.0035$ direction needs $m \gtrsim 1/\log\gamma \approx 288$ before the subdominant term
decays at all, and at $m = 64$ it is still at $80\%$ of full strength. Section 6 quantifies over each
frequency separately and so needs only the pointwise statement, but any argument wanting a rate
uniform over $u$ does not have one here. Convergence of $\rho$ itself is $O(1/\log m)$, which is
brutally slow: the raw quotient for $u = (1,1,1)$ reads $2.37, 2.43, 2.49, 2.53$ at
$m = 300 \dots 2400$ against a target of $3$. Estimating $\rho$ numerically requires fitting the
$O(m)$ term rather than waiting it out.

**Sharpness.** (G2) is not cosmetic. Take $W_2 = -W_1$ and $v_2 = -v_1$: then
$T(mu) = v_1 \prod_i J_{mu_i}(W_{1i})(1 - (-1)^{m|u|_1})$, identically zero for even $|u|_1$.
Measured: $\rho = 0$ on direction $(1,1,0)$. This is exactly the stratum where $\gamma(u) = 1$ and
$j^\star$ fails to be unique.

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

(G3) earns its keep a second time here. If $\sin c_j$ vanished for some $j$ that neuron would carry
coefficient zero on this grid and be invisible to the decomposition, so the tensor would have rank
below $n_2$ and one neuron would never be recovered.

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
$J_{2\kappa}(-x) = J_{2\kappa}(x)$ leaves exactly the sign of $W_{ji}$ undetermined. So the even grid
delivers $n_2$ and the matrix $W$ up to a per-row sign and a permutation of $j$.

### The odd grid

The even grid never sees $\operatorname{Re}B$, so it cannot separate $v_j$ from $c_j$: it returns
only the single combination $\lambda_j = v_j\sin c_j$. To get the second combination we need a grid
on which $\sum_i k_i$ is *odd*, and which is still a product so that the Bessel structure survives.
Shifting one coordinate suffices. Put
$$k_1 \in \{1,3,\dots,2K+1\}, \qquad k_i \in \{0,2,\dots,2K\}\ (i \ge 2),$$
so $\sum_i k_i$ is odd throughout. There
$$\operatorname{Re}B(k) = \sum_j \mu_j \prod_i J_{k_i}(W_{ji}), \qquad \mu_j = v_j\cos c_j,$$
with mode-$1$ factor matrix $[J_{2\kappa+1}(W_{j1})]$ and the same even-order matrices as before in
modes $i \ge 2$. By (G3) again, now in the form $\cos c_j \ne 0$, every neuron is visible.

**Lemma (odd-order Bessel rank).** For $|x_j|$ distinct and nonzero,
$\det[J_{2\kappa+1}(x_j)]_{\kappa,j} \ne 0$ generically. *Proof.* With $x_j = \varepsilon t_j$ and
$J_{2\kappa+1}(x) = (x/2)^{2\kappa+1}/(2\kappa+1)!\,(1+O(x^2))$, the determinant is
$$\varepsilon^{\,n^2}\prod_\kappa \frac{1}{(2\kappa+1)!\,2^{2\kappa+1}} \cdot
\det\big[t_j^{2\kappa+1}\big]\big(1+O(\varepsilon^2)\big),
\qquad \det\big[t_j^{2\kappa+1}\big] = \Big(\prod_j t_j\Big)\prod_{a<b}\big(t_b^2 - t_a^2\big),$$
nonzero for $t_j \ne 0$ with $|t_j|$ distinct. Real-analytic and not identically zero, so its zero
set is a proper analytic subset. $\square$ The hypotheses are (G4) and (G5). *Verified numerically:
full column rank in 300/300 draws at $n_2 = 2,3,4$ and 298/300 at $n_2 = 5$, where the two misses sit
at min singular ratio $5.5\times10^{-13}$, against a $10^{-12}$ threshold, so they are conditioning
rather than rank.*

**No second CP, and hence no matching problem.** Running an independent decomposition on the odd grid
would return $\{\mu_j\}$ under its own permutation and its own column scalings, and those would then
have to be matched against $\{\lambda_j\}$ before the two could be combined. That step is
unnecessary. The even grid has already produced $W$, so every number $\prod_i J_{k_i}(W_{ji})$ is
known, and each of $\lambda$ and $\mu$ is the solution of a square linear system
$$\Big[\textstyle\prod_i J_{k_i}(W_{ji})\Big]_{k,j}\;\lambda \;=\; \big[T(k)\big]_k$$
over any $n_2$ lattice points of the appropriate parity making the matrix invertible, which the two
rank lemmas supply. Both systems are solved against the same $W$ and therefore against the same
labelling of $j$, so $\lambda_j$ and $\mu_j$ refer to the same neuron by construction.

Then $(\mu_j, \lambda_j) = v_j(\cos c_j, \sin c_j)$, so
$$|v_j| = \sqrt{\lambda_j^2 + \mu_j^2}, \qquad c_j = \operatorname{atan2}(\lambda_j, \mu_j) \bmod \pi,$$
both well defined since $v_j \ne 0$ by (G5). What is not determined is the joint sign: $(v_j, c_j)$
and $(-v_j, c_j + \pi)$ produce the same pair $(\mu_j, \lambda_j)$ and hence the same function. That
is exactly the $\rho$ generator on layer two, not a deficiency of the argument.

*Verified end to end: recovering $|v_j|$ and $c_j \bmod \pi$ from the two grids by the linear solves
above, over eight random networks at $n_1 = n_2 = 3$ with $c_j$ drawn across the full range so the
sign of $v_j$ genuinely varies, gives worst error $2.1\times10^{-13}$ in $|v_j|$ and
$2.8\times10^{-12}$ in $c_j \bmod \pi$, measured with a circular distance.*

The residual freedoms are therefore: a per-neuron sign on the rows of $W$, the shift
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

Proposition 1 is now proved rather than sketched: Lemma 1 supplies the explicit Bessel error bound,
the $j$-independence of the factorials removes the need to control them when comparing terms, and
(G2) supplies the strict dominance gap. Closing it turned up one condition the earlier draft had
silently assumed, **(G6)**, without which a coordinate outside the support of $u$ annihilates a term
for every $m$ simultaneously.

What the repair does not give is uniformity in $u$, and that appears to be a fact about the problem
rather than a defect of the argument, since $\gamma(u)$ genuinely approaches $1$ along some
directions. Section 6 does not need uniformity. A quantitative or effective version of this theorem
would.

One smaller gap remains: the $n_1 \le 2$ and $n_2 = 1$ cases fall outside Kruskal and are excluded by
hypothesis rather than handled.

The odd grid of §8 is now carried out rather than asserted. Doing so needed an odd-order analogue of
the Bessel rank lemma, and it showed that the natural route, a second CP decomposition, is the wrong
one: it would return the coefficients under an unrelated permutation and scaling, creating a matching
problem that does not have to exist. Since the even grid already yields $W$, both coefficient
vectors follow from linear solves on a shared labelling.

Any claimed proof of this statement should be checked by someone other than its author. The route
was found and the numerics were run by the same agent, and neither of those is a referee.
