"""T9: P-shared-det determinism — refitting the same images yields identical
weights on CPU; MPS residual nondeterminism is measured and recorded (risk R1)."""

import json
from pathlib import Path

import torch

from sirengap.fitting.batched import fit_batch, make_coord_grid

REPORT = Path(__file__).resolve().parent.parent / "results" / "t9_mps_report.json"


def _fit_twice(device: str) -> float:
    gen = torch.Generator().manual_seed(9)
    targets = (torch.rand(2, 256, 1, generator=gen) * 2 - 1)
    coords = make_coord_grid(16, 16)
    kwargs = dict(widths=(16, 16), steps=150, lr=1e-3, seed=123, shared_init=True, device=device)
    a = fit_batch(targets, coords, **kwargs)
    b = fit_batch(targets, coords, **kwargs)
    return (a.params.flat() - b.params.flat()).abs().max().item()


def test_t9_cpu_exact_determinism() -> None:
    assert _fit_twice("cpu") == 0.0


def test_t9_mps_determinism_measured() -> None:
    if not torch.backends.mps.is_available():
        return
    gap = _fit_twice("mps")
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(
        json.dumps({"max_abs_refit_gap": gap, "torch": torch.__version__}, indent=2)
    )
    # protocol tolerance; if this ever fails, P-shared-det moves to CPU (risk R1)
    assert gap < 1e-4, f"MPS refit nondeterminism {gap:.2e} exceeds tolerance"
