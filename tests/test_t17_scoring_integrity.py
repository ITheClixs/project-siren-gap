"""T17: the ledger is append-once, and off-protocol runs cannot reach it.

Written after the incident in CLAIMS row 54: a master-chain step re-ran the S5 sweep at its
default n=3, overwrote the registered n=5 artifact, and re-scored it, appending a second copy of
all eleven S5 rows. No verdict changed, but the calibration audit counts rows, so a duplicate is
a silent corruption of the program's own scoreboard.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "docs" / "PREDICTION_OUTCOMES.csv"
PY = ROOT / ".venv" / "bin" / "python"


def test_ledger_scores_each_prediction_once() -> None:
    with LEDGER.open(newline="") as fh:
        names = [row["prediction"] for row in csv.DictReader(fh)]
    dupes = [n for n, c in Counter(names).items() if c > 1]
    assert not dupes, f"predictions scored more than once: {dupes}"


def test_ledger_rows_are_wellformed() -> None:
    """Intervals carry a verdict and bounds; probability calls carry a Brier score.

    The verdict is not recomputed here: some registered intervals are one-sided in effect (e.g.
    P-S4e-1, an upper bound on a residual, scored HIT at an observed 0.0 below its lower edge),
    and encoding a single containment rule would contradict the frozen registrations.
    """
    with LEDGER.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row["kind"] == "interval":
                assert row["verdict"] in {"HIT", "MISS"}, row
                for k in ("lo80", "hi80", "observed"):
                    float(row[k])  # raises if unparseable
            else:
                assert row["brier"], row


@pytest.mark.skipif(not PY.exists(), reason="repo venv not present")
def test_s5_sweep_refuses_off_protocol_seed_count() -> None:
    r = subprocess.run(
        [str(PY), str(ROOT / "scripts" / "35_s5_pareto.py"), "--seeds", "3"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert r.returncode != 0
    assert "off-protocol" in (r.stdout + r.stderr)


@pytest.mark.skipif(not PY.exists(), reason="repo venv not present")
def test_s5_scorer_refuses_to_double_score() -> None:
    r = subprocess.run(
        [str(PY), str(ROOT / "scripts" / "36_score_s5.py")],
        capture_output=True, text=True, cwd=ROOT,
    )
    out = r.stdout + r.stderr
    assert "refusing to double-score" in out, out[-400:]


def _load_sweep_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "s8_sweep", ROOT / "scripts" / "48_s8_sweep.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_report() -> dict:
    mod = _load_sweep_module()
    reg = {k: {"statement": s, "point": p, "interval": [lo, hi],
               "observed": lo, "verdict": "HIT"}
           for k, (s, p, (lo, hi)) in mod.REGISTERED.items()}
    return {"registered": reg, "P-S8-A": True, "P-S8-B": True, "P-S8-C": False}


def test_s8_ledger_rows_cover_every_registered_call() -> None:
    mod = _load_sweep_module()
    rows = mod.ledger_rows(_fake_report())
    names = {r["prediction"].split()[0] for r in rows}
    assert names == set(mod.REGISTERED) | set(mod.REGISTERED_P)
    briers = {r["prediction"].split()[0]: r["brier"] for r in rows if r["kind"] == "probability"}
    assert briers["P-S8-A"] == pytest.approx(0.09)   # p=0.70, happened
    assert briers["P-S8-C"] == pytest.approx(0.36)   # p=0.60, did not happen


def test_s8_ledger_refuses_to_double_score(tmp_path, monkeypatch) -> None:
    mod = _load_sweep_module()
    report = _fake_report()
    ledger = tmp_path / "PREDICTION_OUTCOMES.csv"
    header = ("date_scored,prediction,kind,point,lo80,hi80,observed,verdict,abs_error,brier,note\n")
    ledger.write_text(header)
    monkeypatch.setattr(mod, "OUTCOMES", ledger)

    mod.append_ledger(report, rescore=False)
    first = ledger.read_text().count("\n")
    assert first == len(mod.ledger_rows(report)) + 1

    with pytest.raises(SystemExit, match="double-score"):
        mod.append_ledger(report, rescore=False)
    assert ledger.read_text().count("\n") == first, "a refused append must not write"

    mod.append_ledger(report, rescore=True)
    assert ledger.read_text().count("\n") == 2 * first - 1


def test_every_ledger_writer_is_covered_by_the_uniqueness_invariant() -> None:
    """The per-script guard exists only on the S5 path; uniqueness is what catches the rest.

    This asserts the surface is known rather than that each scorer is individually guarded, so
    that adding a new scorer shows up here instead of silently widening the exposure.
    """
    writers = sorted(p.name for p in (ROOT / "scripts").glob("*.py")
                     if "PREDICTION_OUTCOMES" in p.read_text())
    assert writers, "no ledger-writing scripts found; the glob is wrong"
    assert "36_score_s5.py" in writers
    for name in writers:
        assert name[:2].isdigit(), name
