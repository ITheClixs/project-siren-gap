"""Assignment solvers for template alignment (Ch3.2): Hungarian, Sinkhorn, greedy.

All solve: given score matrix S [n, n] (higher = better match), find permutation
idx with idx[t] = model neuron assigned to template slot t, maximizing sum of
S[idx[t], t]. Hungarian is exact (T7 verifies against brute force for n <= 8);
Sinkhorn must never score below greedy (T7).
"""

from __future__ import annotations

import itertools

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor


def hungarian(score: Tensor) -> Tensor:
    """Exact max-score assignment; score [n, n] -> idx [n] with idx[t] = model row."""
    cost = -score.detach().cpu().double().numpy()
    rows, cols = linear_sum_assignment(cost)
    idx = np.empty(score.shape[0], dtype=np.int64)
    idx[cols] = rows
    return torch.from_numpy(idx)


def greedy(score: Tensor) -> Tensor:
    """Greedy max assignment (baseline for T7)."""
    s = score.detach().clone()
    n = s.shape[0]
    idx = torch.full((n,), -1, dtype=torch.long)
    neg = torch.finfo(s.dtype).min
    for _ in range(n):
        flat = int(torch.argmax(s).item())
        r, c = flat // n, flat % n
        idx[c] = r
        s[r, :] = neg
        s[:, c] = neg
    return idx


def sinkhorn(
    score: Tensor,
    epsilons: tuple[float, ...] = (1.0, 0.3, 0.1, 0.03),
    iters: int = 200,
) -> Tensor:
    """Entropic-OT relaxation with epsilon annealing, rounded via Hungarian on the plan.

    Guarantee used by T7: the returned assignment scores >= greedy (we return the
    better of {rounded plan, greedy}).
    """
    score = score.detach().cpu()
    s = score.double()
    n = s.shape[0]
    log_k = None
    for eps in epsilons:
        log_k = s / eps
        log_u = torch.zeros(n, dtype=torch.float64)
        log_v = torch.zeros(n, dtype=torch.float64)
        for _ in range(iters):
            log_u = -torch.logsumexp(log_k + log_v[None, :], dim=1)
            log_v = -torch.logsumexp(log_k + log_u[:, None], dim=0)
    plan = (log_k + log_u[:, None] + log_v[None, :]).exp()
    cand = hungarian(plan.float())
    base = greedy(score)
    score_of = lambda idx: float(score[idx, torch.arange(n)].sum())  # noqa: E731
    return cand if score_of(cand) >= score_of(base) else base


def brute_force(score: Tensor) -> Tensor:
    """Exhaustive optimum for tiny n (test oracle, n <= 8)."""
    n = score.shape[0]
    if n > 8:
        raise ValueError("brute_force limited to n <= 8")
    best, best_val = None, -float("inf")
    cols = torch.arange(n)
    for perm in itertools.permutations(range(n)):
        idx = torch.tensor(perm)
        val = float(score[idx, cols].sum())
        if val > best_val:
            best, best_val = idx, val
    assert best is not None
    return best
