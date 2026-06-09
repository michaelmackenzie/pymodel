#!/usr/bin/env python3
"""Cross-backend binned consistency test.

Builds the simple_shapes_binned example for each of the three backends
(zmodel, hfmodel, roomodel), runs an Asimov fit, and checks that the
reported POI best-fit values agree to within a tight tolerance.

All three backends share:
  - The same per-bin expected counts (analytically computed)
  - The same observed histogram (OBS_COUNTS, total=90)
  - The same rate uncertainties (lumi 5%, bkg_norm 10%)
  - Signal yield 12, background yield 80

Usage:
    cd /path/to/pymodel
    python tests/regression_binned_consistency.py
"""
import json
import subprocess
import sys
from pathlib import Path

import dill

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(command, cwd, label=""):
    print("+", " ".join(str(c) for c in command))
    proc = subprocess.run(
        [str(c) for c in command],
        cwd=str(cwd),
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:])
        raise RuntimeError(
            f"Command failed (code {proc.returncode})"
            + (f" [{label}]" if label else "")
            + f": {' '.join(str(c) for c in command)}"
        )
    return proc.stdout


def _load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_pickle(path):
    with open(path, "rb") as fh:
        return dill.load(fh)


def _poi_fit_from_summary(summaries, backend):
    """Extract the POI best-fit from the first summary entry."""
    if not summaries:
        raise AssertionError(f"{backend}: no summaries returned")
    s = summaries[0]
    val = s.get("poi_fit")
    if val is None:
        raise AssertionError(f"{backend}: poi_fit missing from summary: {s}")
    return float(val)


# ---------------------------------------------------------------------------
# Per-backend fit runners
# ---------------------------------------------------------------------------

def _run_hfmodel(repo, cli):
    ex = repo / "examples" / "hfmodel"
    _run([sys.executable, "simple_shapes_binned.py"], cwd=ex, label="hfmodel generate")

    model  = "examples/hfmodel/simple_shapes_binned_regtest.json"
    output = "examples/hfmodel/analysis_binned_regtest.json"
    _run(cli + ["hfmodel", "build",
                "examples/hfmodel/simple_shapes_binned_card.txt", model],
         cwd=repo, label="hfmodel build")

    _run(cli + ["hfmodel", "analyze",
                "--model-file", model,
                "--output", output],
         cwd=repo, label="hfmodel analyze")

    snap = _load_json(repo / output)
    return _poi_fit_from_summary(snap.get("summaries", []), "hfmodel")


def _run_zmodel(repo, cli):
    ex = repo / "examples" / "zmodel"
    _run([sys.executable, "simple_shapes_binned.py"], cwd=ex, label="zmodel generate")

    model  = "examples/zmodel/simple_shapes_binned_regtest.pkl"
    output = "examples/zmodel/analysis_binned_regtest.pkl"
    _run(cli + ["zmodel", "build",
                "examples/zmodel/simple_shapes_binned_card.txt", model],
         cwd=repo, label="zmodel build")

    _run(cli + ["zmodel", "analyze",
                "--model-file", model,
                "--fit-mode", "binned",
                "--output", output],
         cwd=repo, label="zmodel analyze")

    snap = _load_pickle(repo / output)
    return _poi_fit_from_summary(snap.get("summaries", []), "zmodel")


def _run_roomodel(repo, cli):
    ex = repo / "examples" / "roomodel"
    _run([sys.executable, "simple_shapes_binned.py"], cwd=ex, label="roomodel generate")

    model  = "examples/roomodel/simple_shapes_binned_regtest.root"
    output = "examples/roomodel/analysis_binned_regtest.json"
    _run(cli + ["roomodel", "build",
                "examples/roomodel/simple_shapes_binned_card.txt", model],
         cwd=repo, label="roomodel build")

    _run(cli + ["roomodel", "analyze",
                "--model-file", model,
                "--output", output],
         cwd=repo, label="roomodel analyze")

    snap = _load_json(repo / output)
    return _poi_fit_from_summary(snap.get("summaries", []), "roomodel")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    repo = REPO_ROOT
    cli  = [sys.executable, "python/pymodel"]

    print("=" * 60)
    print("Binned cross-backend consistency test")
    print("=" * 60)

    hf_poi  = _run_hfmodel(repo, cli)
    z_poi   = _run_zmodel(repo, cli)
    roo_poi = _run_roomodel(repo, cli)

    print()
    print(f"  hfmodel  POI best-fit: {hf_poi:.6f}")
    print(f"  zmodel   POI best-fit: {z_poi:.6f}")
    print(f"  roomodel POI best-fit: {roo_poi:.6f}")
    print()

    # All three backends are fitting the identical model on the Asimov dataset
    # (expected data = signal + background at mu=1).  The Asimov fit should
    # converge to mu≈1 for all backends; we require agreement to within 5%.
    tol = 0.05
    pairs = [
        ("hfmodel",  hf_poi,  "zmodel",   z_poi),
        ("hfmodel",  hf_poi,  "roomodel", roo_poi),
        ("zmodel",   z_poi,   "roomodel", roo_poi),
    ]
    for name_a, val_a, name_b, val_b in pairs:
        diff = abs(val_a - val_b)
        denom = max(abs(val_a), abs(val_b), 1e-9)
        rel = diff / denom
        status = "OK" if rel <= tol else "FAIL"
        print(f"  {name_a} vs {name_b}: |{val_a:.6f} - {val_b:.6f}| / max = {rel:.4f}  [{status}]")
        if rel > tol:
            raise AssertionError(
                f"{name_a} POI={val_a:.6f} and {name_b} POI={val_b:.6f} "
                f"differ by {rel:.2%} > {tol:.0%} tolerance"
            )

    print()
    print("All three backends agree within tolerance.  Binned consistency test PASSED.")


if __name__ == "__main__":
    main()
