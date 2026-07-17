"""T7: Hungarian optimal (vs brute force, n <= 8); Sinkhorn never below greedy."""

import torch

from sirengap.canon.assign import brute_force, greedy, hungarian, sinkhorn


def _score_of(score: torch.Tensor, idx: torch.Tensor) -> float:
    return float(score[idx, torch.arange(score.shape[0])].sum())


def test_hungarian_matches_brute_force() -> None:
    for seed in range(20):
        gen = torch.Generator().manual_seed(seed)
        score = torch.rand(6, 6, generator=gen)
        assert abs(_score_of(score, hungarian(score)) - _score_of(score, brute_force(score))) < 1e-9


def test_sinkhorn_at_least_greedy_and_at_most_hungarian() -> None:
    for seed in range(20):
        gen = torch.Generator().manual_seed(100 + seed)
        score = torch.rand(10, 10, generator=gen)
        s_greedy = _score_of(score, greedy(score))
        s_sink = _score_of(score, sinkhorn(score))
        s_hung = _score_of(score, hungarian(score))
        assert s_sink >= s_greedy - 1e-9
        assert s_sink <= s_hung + 1e-9


def test_assignments_are_permutations() -> None:
    gen = torch.Generator().manual_seed(7)
    score = torch.rand(12, 12, generator=gen)
    for solver in (hungarian, greedy, sinkhorn):
        idx = solver(score)
        assert sorted(idx.tolist()) == list(range(12)), solver.__name__
