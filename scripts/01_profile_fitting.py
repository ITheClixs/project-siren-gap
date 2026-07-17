#!/usr/bin/env python3
"""Fit-throughput profiling (G1): fits/sec at several batch sizes, MPS vs CPU.

Writes results/profiling/fit_throughput.json and prints a table. Numbers feed
docs/COMPUTE_LEDGER.md (burn-down v1).

Usage: .venv/bin/python scripts/01_profile_fitting.py [--quick]
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sirengap.fitting.batched import fit_batch, make_coord_grid  # noqa: E402

CONFIGS = {
    # name: (side, channels, widths, protocol_steps)
    "mnist-like": (28, 1, (32, 32), 1000),
    "cifar-like": (32, 3, (64, 64, 64), 2000),
}


def profile_one(device: str, side: int, ch: int, widths: tuple[int, ...], batch: int, steps: int) -> dict:
    gen = torch.Generator().manual_seed(0)
    targets = torch.rand(batch, side * side, ch, generator=gen) * 2 - 1
    coords = make_coord_grid(side, side)
    fit_batch(targets[:8], coords, widths=widths, steps=10, device=device)  # warmup/compile
    if device == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    fit_batch(targets, coords, widths=widths, steps=steps, device=device)
    if device == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - t0
    peak_mb = torch.mps.driver_allocated_memory() / 1e6 if device == "mps" else float("nan")
    return {
        "device": device,
        "batch": batch,
        "timed_steps": steps,
        "seconds": round(elapsed, 3),
        "steps_per_sec": round(steps / elapsed, 2),
        "peak_mem_mb": round(peak_mb, 1) if peak_mb == peak_mb else None,
    }


def main() -> None:
    quick = "--quick" in sys.argv
    timed_steps = 60 if quick else 200
    devices = (["mps"] if torch.backends.mps.is_available() else []) + ["cpu"]
    out: dict = {
        "platform": platform.platform(),
        "torch": torch.__version__,
        "timed_steps": timed_steps,
        "runs": [],
        "derived_fits_per_sec_at_protocol_steps": {},
    }
    for name, (side, ch, widths, proto_steps) in CONFIGS.items():
        for device in devices:
            batches = [64, 256, 1024] if device == "mps" else [64, 256]
            if quick:
                batches = batches[:2]
            for batch in batches:
                run = profile_one(device, side, ch, widths, batch, timed_steps)
                run["config"] = name
                out["runs"].append(run)
                # a "fit" = one INR trained for proto_steps
                fits_per_sec = run["steps_per_sec"] * batch / proto_steps
                key = f"{name}/{device}/B{batch}"
                out["derived_fits_per_sec_at_protocol_steps"][key] = round(fits_per_sec, 2)
                print(
                    f"{name:12s} {device:4s} B={batch:5d}  {run['steps_per_sec']:8.1f} steps/s  "
                    f"-> {fits_per_sec:8.2f} fits/s @ {proto_steps} steps  "
                    f"mem={run['peak_mem_mb']} MB"
                )
    dest = Path(__file__).resolve().parent.parent / "results" / "profiling"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "fit_throughput.json").write_text(json.dumps(out, indent=2))
    print(f"\nwritten: {dest / 'fit_throughput.json'}")


if __name__ == "__main__":
    main()
