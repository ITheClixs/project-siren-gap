"""Numerical checks for the restructured depth-two identifiability proof (PO-2 v2).

These check identities the proof uses, not the theorem itself: preservation directions are
testable numerically, exhaustion is not. Each check prints its worst error and fails loudly.

    .venv/bin/python scripts/72_l2_proof_checks.py
"""

from __future__ import annotations

import itertools
import sys

import numpy as np
from scipy.special import jv

RNG = np.random.default_rng(20260911)
TOL = 1e-10


def net(theta, x):
    """Depth-two sine network R^m -> R^c. x has shape (N, m)."""
    w, b, W, c, V, beta = theta
    h1 = np.sin(x @ w.T + b)            # (N, n1)
    h2 = np.sin(h1 @ W.T + c)           # (N, n2)
    return h2 @ V.T + beta              # (N, c_out)


def random_theta(m=2, n1=3, n2=2, c_out=3, wscale=1.5):
    w = RNG.normal(size=(n1, m)) * 3.0
    b = RNG.uniform(-4, 4, size=n1)
    W = RNG.normal(size=(n2, n1)) * wscale
    c = RNG.uniform(-4, 4, size=n2)
    V = RNG.normal(size=(c_out, n2))
    beta = RNG.normal(size=c_out)
    return w, b, W, c, V, beta


def B(theta, k):
    """B(k) = sum_j v_j e^{i c_j} prod_i J_{k_i}(W_ji), a vector in C^c."""
    _, _, W, c, V, _ = theta
    r = np.prod(jv(np.asarray(k)[None, :], W), axis=1)       # (n2,)
    return V @ (np.exp(1j * c) * r)


def T(theta, k):
    """Folded Bohr coefficient at frequency Omega(k) = sum_i k_i w_i (section 3)."""
    _, b, _, _, _, beta = theta
    k = np.asarray(k)
    Bk = B(theta, k)
    phase = np.exp(1j * (k @ b))
    if not k.any():
        return beta + Bk.imag
    if k.sum() % 2 == 0:
        return phase * Bk.imag
    return -1j * phase * Bk.real


def check_expansion(theta, K=14, npts=40):
    """f(x) = sum_k T(k) exp(i Omega(k).x), vector outputs, frequencies in R^m."""
    w = theta[0]
    x = RNG.uniform(-3, 3, size=(npts, w.shape[1]))
    recon = np.zeros((npts, theta[4].shape[0]), dtype=complex)
    for k in itertools.product(range(-K, K + 1), repeat=w.shape[0]):
        k = np.array(k)
        recon += np.exp(1j * (x @ (k @ w)))[:, None] * T(theta, k)[None, :]
    direct = net(theta, x)
    return np.max(np.abs(recon - direct)) / np.max(np.abs(direct))


def act_layer1(theta, perm, d, q):
    """Apply g in D_inf wr S_n1 on layer 1: neuron i of the result is g_{d_i,q_i}(neuron perm[i])."""
    w, b, W, c, V, beta = theta
    s = (-1.0) ** d
    w2 = s[:, None] * w[perm]
    b2 = s * b[perm] + np.pi * q
    W2 = W[:, perm] * ((-1.0) ** (d + q))[None, :]
    return w2, b2, W2, c, V, beta


def act_layer2(theta, perm, d, q):
    w, b, W, c, V, beta = theta
    s = (-1.0) ** d
    W2 = s[:, None] * W[perm]
    c2 = s * c[perm] + np.pi * q
    V2 = V[:, perm] * ((-1.0) ** (d + q))[None, :]
    return w, b, W2, c2, V2, beta


def check_group(theta):
    n1, n2 = theta[0].shape[0], theta[2].shape[0]
    g = act_layer1(theta, RNG.permutation(n1), RNG.integers(0, 2, n1), RNG.integers(-5, 6, n1))
    g = act_layer2(g, RNG.permutation(n2), RNG.integers(0, 2, n2), RNG.integers(-5, 6, n2))
    x = RNG.uniform(-3, 3, size=(200, theta[0].shape[1]))
    return np.max(np.abs(net(g, x) - net(theta, x))), g


def check_bias_frequency(theta, g1_args):
    """Prop 5, biases: T'(e_i) equals T(eps_i e_{pi(i)}) for theta' = g1 theta."""
    perm, d, q = g1_args
    tp = act_layer1(theta, perm, d, q)
    n1 = theta[0].shape[0]
    worst = 0.0
    for i in range(n1):
        e = np.zeros(n1, dtype=int)
        e[i] = 1
        ep = np.zeros(n1, dtype=int)
        ep[perm[i]] = 1 if d[i] == 0 else -1
        worst = max(worst, np.max(np.abs(T(tp, e) - T(theta, ep))))
    return worst


def blocks(theta, Kmax=6):
    """Im B on K_ev and Re B on each odd family K_i, as arrays (section 7)."""
    n1 = theta[0].shape[0]
    ev = [2 * t for t in range(1, Kmax + 1)]
    od = [2 * t - 1 for t in range(1, Kmax + 1)]
    out = {"ev": np.array([B(theta, k).imag for k in itertools.product(ev, repeat=n1)])}
    for i in range(n1):
        grids = [od if l == i else ev for l in range(n1)]
        out[i] = np.array([B(theta, k).real for k in itertools.product(*grids)])
    return out


def check_layer2_blocks(theta):
    """Group elements on layer 2 leave every block fixed; a single-entry sign flip does not."""
    n2 = theta[2].shape[0]
    g = act_layer2(theta, RNG.permutation(n2), RNG.integers(0, 2, n2), RNG.integers(-5, 6, n2))
    b0, b1 = blocks(theta), blocks(g)
    same = max(np.max(np.abs(b0[key] - b1[key])) for key in b0)
    w, b, W, c, V, beta = theta
    W_flip = W.copy()
    W_flip[0, 1] *= -1.0
    b2 = blocks((w, b, W_flip, c, V, beta))
    moved = {key: np.max(np.abs(b0[key] - b2[key])) for key in b0}
    return same, moved


def check_tail_rank(n1=3, rows=4, Kmax=5, trials=20):
    """Lemma 3 sanity: the product-grid matrices have full column rank for distinct rows."""
    worst = np.inf
    ev = [2 * t for t in range(1, Kmax + 1)]
    for _ in range(trials):
        Y = np.abs(RNG.normal(size=(rows, n1))) * 2 + 0.1
        M = np.array([[np.prod(jv(np.array(k), Y[s])) for s in range(rows)]
                      for k in itertools.product(ev, repeat=n1)])
        sv = np.linalg.svd(M / np.abs(M).max(axis=0), compute_uv=False)
        worst = min(worst, sv[-1] / sv[0])
    return worst


def check_v1_freeze_defect():
    """v1 held a coordinate at kappa=1, i.e. order 2. At a zero of J_2 that hides a neuron."""
    from scipy.optimize import brentq

    z2 = brentq(lambda t: jv(2, t), 5.0, 5.3)            # first positive zero of J_2 (5.1356)
    return z2, jv(2, z2)


def check_general_ray_counterexample(t_max=30):
    """Remark after Prop. 3: rows (a,b),(b,a), equal biases, v2 = -v1 give T(t(1,1)) = 0 exactly,
    although the network satisfies (A1)-(A5). So the proof may only use rays along basis vectors."""
    a, b = 1.3, 0.7
    w = np.array([[1.0, 0.3], [0.2, 1.7]])
    theta = (w, np.array([0.4, -1.1]), np.array([[a, b], [b, a]]), np.array([0.9, 0.9]),
             np.array([[1.0, -1.0]]), np.array([0.0]))
    ray = max(np.max(np.abs(T(theta, np.array([t, t])))) for t in range(1, t_max))
    axis = min(np.max(np.abs(T(theta, np.array([t, 0])))) for t in range(1, 12))
    return ray, axis


def check_no_A7_needed():
    """(A7) of v1 is not needed: with Re B(e_1) = 0, T(e_1) vanishes but T(t e_1) does not for
    larger t, so the spectrum still generates the lattice and the bias still follows."""
    from scipy.optimize import brentq

    W = np.array([[1.1, 0.6], [1.9, 0.8]])
    c = np.array([0.5, -0.8])
    base = (np.array([[1.0, 0.2], [0.1, 1.3]]), np.array([0.3, 0.9]), W, c, None, np.array([0.0]))

    def re_b_e1(v2):
        theta = base[:4] + (np.array([[1.0, v2]]),) + base[5:]
        return B(theta, np.array([1, 0])).real[0]

    v2 = brentq(re_b_e1, -50, 50)
    theta = base[:4] + (np.array([[1.0, v2]]),) + base[5:]
    t_vals = [np.max(np.abs(T(theta, np.array([t, 0])))) for t in (1, 2, 3)]
    return v2, t_vals


def main() -> int:
    failures = []

    theta = random_theta()
    err = check_expansion(theta)
    print(f"[expansion, m=2, c=3] relative error {err:.2e}")
    failures += [] if err < 1e-9 else ["expansion"]

    gerr, _ = check_group(theta)
    print(f"[group preserves f, windings to |q|=5] max error {gerr:.2e}")
    failures += [] if gerr < TOL else ["group"]

    n1 = theta[0].shape[0]
    args = (RNG.permutation(n1), RNG.integers(0, 2, n1), RNG.integers(-5, 6, n1))
    berr = check_bias_frequency(theta, args)
    print(f"[bias step: T'(e_i) = T(eps_i e_pi(i))] max error {berr:.2e}")
    failures += [] if berr < TOL else ["bias"]

    same, moved = check_layer2_blocks(theta)
    print(f"[layer-2 group fixes all blocks] max change {same:.2e}")
    print("[single-entry flip W_01] block changes:", {k: f"{v:.2e}" for k, v in moved.items()})
    failures += [] if same < TOL else ["layer2-group"]
    # Flipping W_{0,1} is invisible on the even grid and on K_0, visible on K_1, as the proof says.
    ok = moved["ev"] < TOL and moved[0] < TOL and moved[1] > 1e-6
    failures += [] if ok else ["layer2-flip-pattern"]

    r = check_tail_rank()
    print(f"[Lemma 3 sanity, 3 coords, 4 rows, orders 2..10] worst normalised sigma_min {r:.2e}")
    failures += [] if r > 1e-12 else ["tail-rank"]

    ray, axis = check_general_ray_counterexample()
    print(f"[general ray] max |T(t(1,1))| = {ray:.1e} on a network in Theta_2; "
          f"min |T(t e_1)| over t<12 = {axis:.1e}")
    failures += [] if (ray < 1e-15 and axis > 1e-12) else ["general-ray"]

    v2, tv = check_no_A7_needed()
    print(f"[no (A7)] at v2 = {v2:.4f}, |T(e_1)|, |T(2e_1)|, |T(3e_1)| = "
          + ", ".join(f"{x:.1e}" for x in tv))
    failures += [] if (tv[0] < 1e-12 and tv[1] > 1e-6 and tv[2] > 1e-6) else ["no-A7"]

    z2, val = check_v1_freeze_defect()
    print(f"[v1 defect 3] J_2 vanishes at {z2:.6f} (J_2 = {val:.1e}); a v1 freeze at order 2 "
          "annihilates any neuron with |W_ji0| there, which (G1)-(G7) did not exclude")

    if failures:
        print("FAILED:", failures)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
