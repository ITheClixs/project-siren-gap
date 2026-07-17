"""Statistics standard (protocol §0.5): paired tests, TOST, Holm, bootstrap CI."""

from __future__ import annotations

import numpy as np
from scipy import stats as sps


def paired_summary(a: np.ndarray, b: np.ndarray) -> dict:
    """a, b: per-seed metrics (same seeds). Reports mean diff, 95% t-CI, paired t,
    Wilcoxon, Cohen's d (paired)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    tcrit = sps.t.ppf(0.975, n - 1) if n > 1 else float("nan")
    t_stat, t_p = sps.ttest_rel(a, b) if n > 1 else (float("nan"), float("nan"))
    try:
        w_stat, w_p = sps.wilcoxon(a, b) if n > 1 else (float("nan"), float("nan"))
    except ValueError:  # all-zero differences
        w_stat, w_p = float("nan"), 1.0
    cohen = mean / float(d.std(ddof=1)) if n > 1 and d.std(ddof=1) > 0 else float("nan")
    return {
        "n": n,
        "mean_diff": mean,
        "ci95": [mean - tcrit * se, mean + tcrit * se] if n > 1 else [float("nan")] * 2,
        "t_p": float(t_p),
        "wilcoxon_p": float(w_p),
        "cohen_d": cohen,
    }


def tost_equivalence(a: np.ndarray, b: np.ndarray, margin: float) -> dict:
    """Two one-sided tests: H0 |mean diff| >= margin. Both p's < .05 => equivalent."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    t_lower = (d.mean() + margin) / se  # H0: diff <= -margin
    t_upper = (d.mean() - margin) / se  # H0: diff >= +margin
    p_lower = 1 - sps.t.cdf(t_lower, n - 1)
    p_upper = sps.t.cdf(t_upper, n - 1)
    return {
        "margin": margin,
        "p_lower": float(p_lower),
        "p_upper": float(p_upper),
        "equivalent_at_05": bool(max(p_lower, p_upper) < 0.05),
    }


def holm(pvals: list[float]) -> list[float]:
    """Holm–Bonferroni adjusted p-values (family order preserved)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


def bootstrap_ci_mean(x: np.ndarray, n_boot: int = 10000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]
