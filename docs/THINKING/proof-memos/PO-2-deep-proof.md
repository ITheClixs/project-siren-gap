# PO-2 deep case: a proof of generic identifiability at $L = 2$

Status: complete argument. The asymptotic of §5 is pointwise in $u$ and not uniform, which §6 does
not need; see §10. Four corrections to an earlier draft are recorded in §10.
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
- **(G7)** $\operatorname{Re}B(e_i) \ne 0$ for every $i$, with $B$ the coefficient defined in §3.

Each condition fails on a countable union of proper analytic subsets, so
$\Theta^{(2)}_{\mathrm{gen}}$ has full measure. For (G7) this holds because
$$\operatorname{Re}B(e_i) \;=\; \sum_j v_j \cos c_j\, J_1(W_{ji}) \prod_{i' \ne i} J_0(W_{ji'})$$
is real-analytic in the parameters and not identically zero: at $n_2 = 1$, $v_1 = 1$, $c_1 = 0$ it
reduces to $J_1(W_{1i})\prod_{i'\ne i}J_0(W_{1i'})$, which is nonzero for suitable $W$. (G1) is
inherited from the $L=1$ theorem; the rest are new. (G3) is used twice, in §5 and again in §8; the
others are used once each.

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
$$T(k) \;:=\; c_f(k\cdot a) \;=\; \beta\,\mathbf{1}_{\{k=0\}} \;+\;
\tfrac{1}{2i}\big[A(k) - \overline{A(-k)}\big]
\;=\;\begin{cases}
\beta + \operatorname{Im}B(0), & k = 0,\\[2pt]
e^{i k\cdot b}\,\operatorname{Im} B(k), & k \ne 0,\ \textstyle\sum_i k_i \text{ even},\\[2pt]
-i\,e^{i k\cdot b}\,\operatorname{Re} B(k), & \textstyle\sum_i k_i \text{ odd}.
\end{cases}$$

This is the lemma the earlier memo called "spectral folding, drafted, not verified". It matters
twice over. First, $B(k)$ itself is **not** observable, only one real part of it, chosen by the
parity of $\sum_i k_i$. Second, **the origin is contaminated by the output bias.** The constant
$\beta$ sits at frequency zero and nowhere else, so $\operatorname{Im}B(0)$ is not observable until
$\beta$ is known, and an earlier draft of this memo formed the even tensor of §8 with its
$(0,\dots,0)$ entry set to $\operatorname{Im}B(0)$, which is not a quantity the data supplies. Every
recovery step below is therefore confined to $k \ne 0$, and $\beta$ is recovered last, from this
identity, once the hidden parameters are known.

*Verified numerically: the empirical mean of $f$ along a line exceeds $\operatorname{Im}B(0)$ by
$0.731275$ against $\beta = 0.731416$, so the extra term is $\beta$ and not a bookkeeping slip.*

*The $k \ne 0$ cases were verified numerically as an algebraic identity, worst error
$9.8\times10^{-18}$ over ten lattice points.*

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
$s^{(m)}_j = \sin c_j$ or $\cos c_j$ according to the parity of $\sum_i(mu)_i = m\sum_i u_i$, and
let $j^\star$ maximise $P_j(u)$ with $\gamma(u) = P_{j^\star}(u)/P_{j^{\star\star}}(u)$ the ratio to
the runner-up. The superscript is not decoration: when $\sum_i u_i$ is odd the parity alternates
with $m$, so $s^{(m)}_j$ swaps between $\sin c_j$ and $\cos c_j$ along the ray. The index $j^\star$
does not move, since $P_j(u)$ is parity-free.

**Proposition 1.** For each fixed primitive $u$, on $\Theta^{(2)}_{\mathrm{gen}}$,
$$|T(mu)| \;=\; \frac{|v_{j^\star}|\,Q_{j^\star}(u)\,\big|s^{(m)}_{j^\star}\big|\;P_{j^\star}(u)^{m}}
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
$\operatorname{Re}B(e_i)$ real and nonzero by **(G7)**. Hence $\arg T(e_i)$ determines $b_i$
**modulo $\pi$** and no better.

(G7) cannot be dropped in favour of (G3) and (G5). Those two make every *summand* of
$\operatorname{Re}B(e_i) = \sum_j v_j\cos c_j\,J_1(W_{ji})\prod_{i'\ne i}J_0(W_{ji'})$ nonzero, which
is not the same as making the sum nonzero: the expression is a linear functional of $v$, and (G5)
constrains $v$ only to have nonvanishing entries, which does not keep it out of that functional's
kernel. For $n_2 \ge 2$ the kernel meets the region cut out by (G1)-(G6), and a parameter there has
$T(e_i) = 0$, leaving $\arg T(e_i)$ undefined and $b_i$ unrecovered. Hence the separate condition. That is precisely the $\rho$ generator's ambiguity
$b \mapsto b + \pi$, so layer 1 is now pinned exactly up to $D_\infty \wr S_{n_1}$.

## 8. Layer 2 by peeling on nonzero-frequency tails

The obstruction of §4 is that $B$ is never observable, only one real part per parity class, and that
the origin carries $\beta$. Restricting to $k \in 2\mathbb{Z}^{n_1}$ fixes the parity: there
$\sum_i k_i$ is always even, the index set is a product grid, and $\operatorname{Im}B$ is observable
at every point except $k = 0$. Writing $\lambda_j = v_j \sin c_j$,
$$\operatorname{Im}B(2\kappa) \;=\; \sum_j \lambda_j \prod_i J_{2\kappa_i}(W_{ji}),
\qquad \kappa \in \mathbb{Z}_{\ge0}^{n_1} \setminus \{0\}.$$

(G3) earns its keep a second time here. If $\sin c_j$ vanished for some $j$ that neuron would carry
coefficient zero on this grid and be invisible, and one neuron would never be recovered.

**Why not CP uniqueness on a finite grid.** The natural move is to read this as a rank-$n_2$ CP
tensor on $\kappa_i \in \{0,\dots,K\}$ and invoke Kruskal, or its $N$-way extension
[Sidiropoulos and Bro 2000]. An earlier draft did exactly that. It does not work, for two
independent reasons, and both are worth stating because the route looks sound.

The first is the origin: the tensor's $(0,\dots,0)$ entry is not observable, so the object the
argument decomposes is not an object the data provides.

The second is that the finite factor matrices need not have full column rank, and **(G4) does not
give it**. The rank lemma one wants says $\det[J_{2\kappa}(x_j)]_{\kappa,j} \ne 0$ for $|x_j|$
distinct, and the small-argument Vandermonde argument proves only that this determinant is
*generically* nonzero, that is off a proper analytic subset. That subset is not among (G1)-(G7), and
it is met. Let $x_1 \ne x_2$ be positive zeros of $J_2$. Then the whole second row of
$[J_{2\kappa}(x_j)]$ vanishes, so at $K = 1$ the matrix has rank one although $|x_1| \ne |x_2|$ and
both $J_0(x_j) \ne 0$. It is worse than one bad row. At any zero of $J_2$ the recurrence
$J_{n-1}(x) + J_{n+1}(x) = (2n/x)J_n(x)$ gives $J_3 = -J_1$ from $n = 2$, then $J_0 = 2J_1/x$ from
$n = 1$ and $J_4 = -6J_1/x$ from $n = 3$, so
$$\frac{J_4(x)}{J_0(x)} \;=\; -3 \qquad\text{at every zero of } J_2,$$
independently of which zero. Rows $J_0$, $J_2$, $J_4$ are therefore proportional across any two such
arguments and the matrix stays rank one for every $K \le 2$; the first nonvanishing minor is
$(J_0, J_6)$. So no universal cutoff of the form $K \ge n_2 - 1$ exists. *Verified: with $x_1,x_2$
the first two positive zeros of $J_2$ at 140 digits, the minors $(J_0,J_2)$, $(J_0,J_4)$ and
$(J_2,J_4)$ are all $O(10^{-40})$, i.e. zero to the precision of the quoted roots, while
$(J_0,J_6) = -4.87\times10^{-2}$.*

Both problems disappear together if the argument never truncates and never touches the origin. The
tails are enough, and the tool is already in this memo.

**Lemma 2 (scale-free extraction).** Let $s_n = \sum_j \alpha_j J_n(x_j)$ with all $\alpha_j \ne 0$
and the $|x_j|$ distinct, and let $n \to \infty$ through either parity class. Then
$$\max_j |x_j| \;=\; 2\lim_{n\to\infty}\big(|s_n|\,n!\big)^{1/n},
\qquad \alpha_{j^\star} \;=\; \lim_{n\to\infty}\frac{s_n}{J_n(x_{j^\star})}.$$

*Proof.* By Lemma 1, $s_n\,n! = \sum_j \alpha_j (x_j/2)^{n}(1+E_n(x_j))$ with each $E_n \to 0$, and
this expansion holds for every order $n \ge 0$, so nothing in it distinguishes the parities. The term
of largest $|x_j|$ dominates strictly, so the sum is $\alpha_{j^\star}(x_{j^\star}/2)^{n}(1+o(1))$.
Taking $n$-th roots and using $|\alpha_{j^\star}|^{1/n} \to 1$ gives the first claim; dividing by
$J_n(x_{j^\star})$ gives the second. $\square$

The first limit is insensitive to $\alpha_{j^\star}$, so a Bessel argument can be read off before its
coefficient is known. Together the two give a peeling step: extract $|x_{j^\star}|$ and
$\alpha_{j^\star}$, subtract that term, recurse. The remaining $|x_j|$ stay distinct, so the step
repeats and the sequence is identically zero after $n_2$ rounds. *Verified at both parities on a
three-term sum with coefficients spanning $1$ to $-37.5$: the estimate of $\max_j|x_j| = 2.3$ reads
$2.2933$ (odd) and $2.2939$ (even) at $n \approx 21$, and $2.299995$ at both parities by $n \approx
801$, the two parities agreeing to six digits throughout.*

**Lemma 3 (tails are independent).** For $|x_j|$ distinct and nonzero, the sequences
$\big(J_n(x_j)\big)_{n \ge n_0}$ are linearly independent over $\mathbb{R}$, for any $n_0$ and along
either parity. *Proof.* If $\sum_j \alpha_j J_n(x_j) = 0$ for all such $n$ with some $\alpha_j \ne
0$, discard the zero coefficients and apply Lemma 2's first limit to what remains: the left side is
identically zero, so its $n$-th roots tend to $0$, while the right side tends to
$\max_j |x_j|/2 > 0$. $\square$

Lemma 3 is what replaces the finite rank lemmas. It says a finite cutoff with full column rank
*exists*, since finitely many linearly independent infinite columns have some finite set of rows of
full rank, but it does not name one, and the $J_2$ example above shows no formula in $n_2$ alone can.
Nothing below needs a named cutoff.

### Magnitudes, coefficients and labels

Fix a coordinate $i$ and any values $\kappa_{i'} \ge 0$ for $i' \ne i$, not all zero if $i$ is the
only remaining coordinate, so that the multi-index is never the origin. Then
$$\sigma^{(i)}_{\kappa}(\kappa_{-i}) \;:=\; \operatorname{Im}B(2\kappa\,e_i + 2\kappa_{-i})
\;=\; \sum_j \Big[\lambda_j \prod_{i'\ne i} J_{2\kappa_{i'}}(W_{ji'})\Big] J_{2\kappa}(W_{ji}),$$
which is Lemma 2's form in $\kappa$ with coefficients $\alpha_j = \lambda_j \prod_{i'\ne i}
J_{2\kappa_{i'}}(W_{ji'})$. Peeling in $\kappa$ returns every $|W_{ji}|$ together with its
$\alpha_j$, and the magnitudes returned do not depend on the frozen $\kappa_{-i}$, so they label the
neurons: the term carrying a given $|W_{ji}|$ is the same $j$ for every choice of $\kappa_{-i}$. This
is the observation that removed the matching problem from the odd families, used once more. By (G4)
the $|W_{ji}|$ are distinct across $j$ for each fixed $i$, which is exactly Lemma 2's hypothesis, and
by (G5) they are nonzero.

Running this for each $i$ in turn, with $\kappa_{-i} = 0$ except for a single coordinate held at
$\kappa_{i_0} = 1$ to keep the index off the origin, gives every $|W_{ji}|$ under one consistent
labelling of $j$. Dividing the recovered $\alpha_j$ by the now-known Bessel factors gives
$\lambda_j$. No cutoff is chosen, no tensor is decomposed, and $k = 0$ is never evaluated.

### Signs from the odd families

The even lattice is blind to signs, since $J_{2\kappa}(-x) = J_{2\kappa}(x)$, and it returns only
$\lambda_j = v_j\sin c_j$. Write $W_{ji} = \epsilon_{ji}|W_{ji}|$. A *row* flip
$\epsilon_{j\cdot} \mapsto -\epsilon_{j\cdot}$ with $c_j \mapsto -c_j$ and $v_j \mapsto -v_j$ is the
$\sigma$ generator on layer-two neuron $j$ and preserves $f$; a *single-entry* flip is not a group
element and does change $f$. So the observables must separate single-entry flips, and $n_1 n_2$ signs
must collapse to $n_2$ row signs. One odd family per coordinate does it, and fewer does not: on a
grid with $k_1$ odd and every other $k_i$ even, flipping $\epsilon_{ji}$ for any $i \ge 2$ leaves
that grid and the even lattice pointwise unchanged while moving $f$.

For each $i$ let $G_i$ be the indices with $k_i$ odd and $k_{i'}$ even for $i' \ne i$. There
$\sum_l k_l$ is odd, so $\operatorname{Re}B$ is observable, and no point of $G_i$ is the origin. By
$J_k(\epsilon x) = \epsilon^k J_k(x)$ every even-order factor is sign-free and only $\epsilon_{ji}$
survives:
$$\operatorname{Re}B(k) \;=\; \sum_j \big(\mu_j\,\epsilon_{ji}\big)\,J_{k_i}\big(|W_{ji}|\big)
\prod_{i' \ne i} J_{k_{i'}}\big(|W_{ji'}|\big), \qquad \mu_j = v_j\cos c_j .$$
Freezing $k_{-i}$ and peeling in $k_i$ through odd orders, which Lemma 2 covers, returns
$$m^{(i)}_j \;=\; \mu_j\,\epsilon_{ji}$$
against the same labelling, since the magnitudes $|W_{ji}|$ that identify the terms are already
known. By (G3) as $\cos c_j \ne 0$ and (G5), every $\mu_j$ is nonzero, so every neuron is visible on
every $G_i$.

Since $\mu_j \ne 0$, the vector $\big(m^{(1)}_j,\dots,m^{(n_1)}_j\big) =
\mu_j\big(\epsilon_{j1},\dots,\epsilon_{jn_1}\big)$ has all entries of the common magnitude
$|\mu_j|$, so its sign pattern determines every product
$\epsilon_{ji}\epsilon_{ji'} = \operatorname{sign}\big(m^{(i)}_j m^{(i')}_j\big)$ and leaves exactly
one sign per row undetermined, that of $\mu_j$. Fixing the representative $\epsilon_{j1} = +1$ gives
$\mu_j = m^{(1)}_j$ and pins every $W_{ji}$; the opposite choice is the row flip. The residual sign
freedom is $2^{n_2}$, one per layer-two neuron, not the $2^{n_1n_2}$ the even lattice alone leaves.

### The output bias, last

With $(\mu_j, \lambda_j) = v_j(\cos c_j, \sin c_j)$,
$$|v_j| = \sqrt{\lambda_j^2 + \mu_j^2}, \qquad c_j = \operatorname{atan2}(\lambda_j, \mu_j) \bmod \pi,$$
both well defined since $v_j \ne 0$ by (G5). Two freedoms remain and both are group elements. The
joint sign: $(v_j, c_j)$ and $(-v_j, c_j+\pi)$ give the same pair and hence the same function, which
is the $\rho$ generator on layer two. The row flip: it fixes $\lambda_j$ and negates $\mu_j$, sending
$c_j \mapsto -c_j \bmod \pi$ and $v_j \mapsto -v_j$ alongside $W_{j\cdot} \mapsto -W_{j\cdot}$, which
is $\sigma$.

Every hidden parameter is now known, so $\operatorname{Im}B(0) = \sum_j \lambda_j \prod_i
J_0(W_{ji})$ is computable, and §4's origin identity gives
$$\beta \;=\; T(0) - \operatorname{Im}B(0),$$
which is the one place $k = 0$ is used and the last step rather than the first. Equivalently $\beta$
is $f(x_0)$ minus the reconstructed hidden network at any single point.

The residual freedoms are therefore: a per-neuron sign on the rows of $W$, the shift
$c_j \mapsto c_j+\pi$ with $v_j \mapsto -v_j$, and a permutation of $j$. That is
$D_\infty \wr S_{n_2}$.

**Every width, and no case analysis.** Nothing above assumed $n_1 \ge 3$ or $n_2 \ge 2$. Peeling is
coordinatewise, so $n_1 = 1$ and $n_1 = 2$ are not special; at $n_2 = 1$ the peeling terminates after
one round and (G2) is vacuous, with Proposition 1 holding trivially because no competing $j$ exists,
so $\gamma = \infty$. The earlier draft's separate treatment of small widths existed only because
Kruskal fails there; with Kruskal gone the argument is uniform.

*Verified in two parts, because two distinct claims are involved.*

*That peeling returns the signed coefficients: at $n_1 = n_2 = 2$ with mixed signs
$\epsilon = \binom{+\,-}{-\,+}$, freezing $k_2 = 2$ and peeling in $k_1$ through odd orders at 400
digits recovers $m^{(1)}_{j^\star} = \mu_{j^\star}\epsilon_{j^\star 1} = +0.83$ to $3.8\times10^{-40}$,
sign included, with the intermediate coefficient limit accurate to $5.6\times10^{-41}$. Worth
recording that the two limits of Lemma 2 converge at very different rates: the coefficient limit
divides by $J_n(x_{j^\star})$ and so cancels the dominant behaviour exactly, while the magnitude
limit is the slow one, reading $2.2734$ against $2.3$ at $n = 181$, consistent with the $O(1/n)$ of
Lemma 1. Magnitudes are the expensive quantity here, not signs.*

*That the sign collapse is right: coefficient matrices assembled from the magnitudes $|W_{ji}|$
alone, never from the signed $W$, across fifteen configurations spanning
$(n_1,n_2) \in \{(2,2),(3,2),(3,3),(4,3),(3,4)\}$ at three seeds each, return
$m^{(i)}_j = \mu_j\epsilon_{ji}$ to worst error $6.0\times10^{-15}$, and the reconstruction obtained
by normalising $\epsilon_{j1} = +1$ reproduces $f$ on 200 random points to worst error
$2.7\times10^{-15}$. That check obtains the $m^{(i)}_j$ by finite least squares rather than by
peeling, which is legitimate for what it tests, since the collapse argument takes the $m^{(i)}_j$ as
given and is indifferent to how they were produced. Reconstructing the function rather than
comparing parameters is the right check when the residual freedom is a group action.*

*The corresponding check on the earlier two-grid version is what exposed the sign gap: flipping the
single entry $W_{ji}$ for any $i \ge 2$ moves $f$ by $3.4\times10^{-1}$ while changing the even grid
and the $k_1$-shifted odd grid by exactly $0$, and a $k_2$-shifted grid, which that version never
formed, by $1.5\times10^{-1}$. The zeros are exact, not small.*

**Remark (Kruskal, if one wants it).** When the finite minors do happen to be nonsingular, the CP
route of the earlier draft is shorter, and its arithmetic is right: $k$-rank $n_2$ in every mode
gives $\sum_i k\text{-rank}_i = n_1n_2 \ge 2n_2 + (n_1-1)$ exactly when $n_2(n_1-2) \ge n_1-1$, which
holds for $n_1 \ge 3, n_2 \ge 2$ and is tight at $(3,2)$. What the earlier draft got wrong was not
the inequality but the claim that (G4) supplies its hypotheses, and the belief that the CP route
avoids limits: reading a magnitude off a factor column determined only up to scaling uses Lemma 2
either way. Since the peeling argument needs no cutoff, no rank hypothesis and no case analysis, it
is used here instead.
## 9. Theorem

**Theorem.** Let $\theta, \theta' \in \Theta^{(2)}_{\mathrm{gen}}$ be two-hidden-layer sine networks
of any widths $n_1, n_2, n_1', n_2' \ge 1$. If $f_\theta = f_{\theta'}$ on a nonempty open set,
then $n_1 = n_1'$, $n_2 = n_2'$, and $\theta' = g\theta$ for some
$g \in (D_\infty \wr S_{n_1}) \times (D_\infty \wr S_{n_2})$.

*Proof.* Analytic continuation extends the equality to $\mathbb{R}^m$. §6 and §7 pin layer 1 up to
$D_\infty \wr S_{n_1}$ for every $n_1 \ge 1$; §8 pins layer 2 up to $D_\infty \wr S_{n_2}$ at every
width by peeling on nonzero-frequency tails, and then recovers the output bias $\beta$, on which the
group acts trivially, from the origin identity of §4. $\square$

Note that $\beta$ has to be recovered for the statement to be about $\theta$ rather than about the
hidden part of $\theta$: the conclusion $\theta' = g\theta$ ranges over all of $\theta$, and $g$
fixes $\beta$, so $\beta' = \beta$ is part of what must be shown rather than a convention.

## 10. What is not yet rigorous

Proposition 1 is now proved rather than sketched: Lemma 1 supplies the explicit Bessel error bound,
the $j$-independence of the factorials removes the need to control them when comparing terms, and
(G2) supplies the strict dominance gap. Closing it turned up one condition the earlier draft had
silently assumed, **(G6)**, without which a coordinate outside the support of $u$ annihilates a term
for every $m$ simultaneously. A second such condition, **(G7)**, turned up later in §7 and is
recorded with the other corrections below.

What the repair does not give is uniformity in $u$, and that appears to be a fact about the problem
rather than a defect of the argument, since $\gamma(u)$ genuinely approaches $1$ along some
directions. Section 6 does not need uniformity. A quantitative or effective version of this theorem
would.

Kruskal is now gone rather than optional. The rank hypotheses it needs are the one step of §8 whose
hypotheses were hardest to check, and checking them showed they do not hold on the stated stratum:
(G4) gives distinct magnitudes, the Vandermonde argument gives generic nonvanishing, and those are
not the same statement. Zeros of $J_2$ meet the difference. Peeling on tails needs no cutoff, no rank
hypothesis and no case analysis, so it now carries every width uniformly and the small-width
subsection has been absorbed rather than kept.

The odd families of §8 are now carried out rather than asserted, and it turned out that the natural
route, a second CP decomposition, is the wrong one: it would return the coefficients under an
unrelated permutation and scaling, creating a matching problem that does not have to exist. Peeling
on the even lattice already fixes the labelling by magnitude, so each odd family is read against it.

**Corrections made to earlier drafts of this memo.** Six are worth recording, because five of them
were assertions a reader could have taken on trust. The first four were found by re-reading; the
last two came from an external review on 2026-09-09 and were the two that changed the architecture
of §8 rather than a step inside it.

1. That draft used a *single* odd grid and concluded that the even grid delivers $W$ "up to a per-row
   sign". It does not. The even grid is blind to every sign, and a $k_1$-shifted odd grid recovers
   only $\epsilon_{j1}$, leaving $2^{n_2(n_1-1)}$ of freedom that is not group freedom. One odd grid
   per coordinate closes it, and the sign patterns then collapse to one sign per row.
2. §7 derived the nonvanishing of $\operatorname{Re}B(e_i)$ from (G3) and (G5). Those make each
   summand nonzero, not the sum; the expression is a linear functional of $v$ and its kernel meets
   the region. Now **(G7)**.
3. §8 invoked the all-orders identity $\sum_k J_k(x)e^{ik\phi} = e^{ix\sin\phi}$ on a grid supplying
   even orders only, and did so before the CP column scaling had been fixed. Lemma 2's scale-free
   limit is the correct instrument and was already in the memo.
4. Proposition 1's $s_{j^\star}$ was written as a constant when it alternates with $m$ whenever
   $\sum_i u_i$ is odd. The proof body was already correct; only the display was not.
5. §4's folding formula omitted the output bias, which sits at frequency zero and nowhere else, so
   $c_f(0) = \beta + \operatorname{Im}B(0)$. Every draft up to this one then built the even tensor
   with an unobservable entry at its origin, and the $n_2 = 1$ case read $\lambda_1$ straight off it.
   Recovery is now confined to $k \ne 0$ and $\beta$ is recovered last.
6. The finite rank lemmas proved *generic* nonvanishing and were then applied as though (G4) implied
   it. It does not, and the gap is reachable: at two zeros of $J_2$ the even factor matrix has rank
   one for every $K \le 2$, because $J_4/J_0 = -3$ at every zero of $J_2$. No cutoff of the form
   $K \ge n_2 - 1$ works. The repair was to stop truncating: Lemma 2 extended to either parity, plus
   Lemma 3, carries the whole of §8 on tails, and Kruskal is now a remark rather than a step.

Corrections 5 and 6 are the reason §8 is shorter than it was. Both the origin problem and the rank
problem are artifacts of working on a finite grid that includes $k = 0$; neither survives on tails.

Any claimed proof of this statement should be checked by someone other than its author. The route was
found, the numerics were run, and the four corrections above were found by the same agent, and none
of those is a referee. The first correction in particular was a one-line assertion that survived
until someone tested it, which is the specific failure mode this note exists to flag.

## References

- J. B. Kruskal. Three-way arrays: rank and uniqueness of trilinear decompositions.
  *Linear Algebra and its Applications*, 18(2):95-138, 1977.
- N. D. Sidiropoulos and R. Bro. On the uniqueness of multilinear decomposition of $N$-way arrays.
  *Journal of Chemometrics*, 14(3):229-239, 2000.
- G. N. Watson. *A Treatise on the Theory of Bessel Functions*. Cambridge University Press, 2nd
  edition, 1944. (Jacobi-Anger, the even-order expansion of $\cos(x\sin\phi)$, and the ascending
  series used in Lemma 1.)
