#!/usr/bin/env python3
"""Compare pymodel with Combine on any datacard.

    python3 scripts/compare_combine.py CARD --workdir SCRATCH [--backends roomodel,zmodel,hfmodel]
                                       [--rmax 20] [--no-combine]

The card and every shape file it references are COPIED into SCRATCH, keeping their paths
relative to the current directory (Combine's shape-file lookup), so nothing is ever written
next to the original card.  Combine (text2workspace.py, AsymptoticLimits, MultiDimFit
singles) runs in SCRATCH; then each backend computes the same quantities through the pymodel
API.  A table of the numbers and their relative differences is printed and written to
SCRATCH/compare.json.  Needs the Combine environment (combine on PATH, and
PYMODEL_ROOT_LIBS=libHiggsAnalysisCombinedLimit.so for Combine classes).
"""

import argparse
import json
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, os.path.join(REPO, "tests"))


def copy_inputs(card, workdir):
    from modelspec.datacard import parse_datacard

    dc = parse_datacard(card)
    files = {os.path.relpath(card)}
    for ch_map in dc.shapeMap.values():
        for entry in ch_map.values():
            if entry and entry[0] != "FAKE":
                path = entry[0]
                if not os.path.exists(path):  # Combine: relative to the card directory as fallback
                    path = os.path.join(os.path.dirname(card), path)
                files.add(os.path.relpath(path))
    for rel in files:
        if rel.startswith(".."):
            raise SystemExit(f"{rel} is outside the current directory; run from a directory above all inputs")
        dest = os.path.join(workdir, rel)
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        shutil.copy2(rel, dest)
    return os.path.relpath(card)


def run_combine(card, workdir, rmax):
    from combine_fixtures import Runner, limit_tree, one

    run = Runner(workdir, os.path.join(workdir, "combine.log"))
    run.run("text2workspace", ["text2workspace.py", card, "-m", "120", "-o", "cmp_ws.root"])
    opts = ["--rMax", str(rmax)]
    run.run("asymptotic", ["combine", "-M", "AsymptoticLimits", "cmp_ws.root", "-n", ".cmp"] + opts)
    out = {"expected": {}}
    for row in limit_tree(one("higgsCombine.cmp.AsymptoticLimits.mH120*.root", workdir), ["limit", "quantileExpected"]):
        q = round(row["quantileExpected"], 3)
        if q == -1:
            out["observed"] = row["limit"]
        else:
            out["expected"][q] = row["limit"]
    run.run("singles", ["combine", "-M", "MultiDimFit", "cmp_ws.root", "-n", ".cmp", "--algo", "singles"] + opts)
    rows = limit_tree(one("higgsCombine.cmp.MultiDimFit.mH120*.root", workdir), ["r"])
    out["r_hat"] = rows[0]["r"]
    out["r_68"] = [rows[1]["r"], rows[2]["r"]] if len(rows) > 2 else None
    out["commands"] = run.commands
    return out


def run_pymodel(card, backend_name, rmax):
    import numpy as np

    from inference.asymptotic import AsymptoticLimits
    from inference.fitting import Fitter
    from inference.model import observed_dataset
    from inference.scan import grid_points, profile_scan
    from modelspec.datacard import build_ir
    from stat_backends import get_backend

    model = build_ir(card, poi_range=(0.0, rmax))
    lik = get_backend(backend_name).build_likelihood(model, argparse.Namespace())
    fitter = Fitter(lik)
    res = AsymptoticLimits(lik, fitter).run()
    data = observed_dataset(model)
    free = fitter.fit(data)
    hi = max(v for v in [res.observed or 0] + [v for v in res.expected.values() if v]) or rmax
    sc = profile_scan(lik, fitter, data, lik.poi, grid_points(0.0, min(rmax, 1.5 * hi), 60), free=free)
    iv = sc["intervals"]["68"]
    return {"observed": res.observed, "expected": {round(q, 3): v for q, v in res.expected.items()},
            "r_hat": free.value(lik, lik.poi), "r_68": [iv["lower"] or 0.0, iv["upper"]],
            "flags": res.flags + sc["flags"], "notes": list(model.notes) + list(getattr(lik, "notes", []))}


def rel(a, b):
    if a is None or b is None:
        return "n/a"
    return f"{(a - b) / b:+.2%}" if b else f"{a - b:+.3g}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("card")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--backends", default="roomodel,zmodel,hfmodel")
    ap.add_argument("--rmax", type=float, default=20.0)
    ap.add_argument("--no-combine", action="store_true", help="only pymodel (e.g. without a Combine build)")
    args = ap.parse_args()
    workdir = os.path.abspath(args.workdir)
    if os.path.realpath(workdir).startswith(os.path.realpath(REPO)) or "mumep_ana" in os.path.realpath(workdir):
        raise SystemExit("--workdir must be a scratch directory outside the repository and mumep_ana")
    os.makedirs(workdir, exist_ok=True)
    card = copy_inputs(args.card, workdir)
    os.chdir(workdir)
    results = {}
    if not args.no_combine:
        results["combine"] = run_combine(card, workdir, args.rmax)
    for name in args.backends.split(","):
        try:
            results[name] = run_pymodel(card, name, args.rmax)
        except Exception as exc:  # report and continue with the other backends
            results[name] = {"error": f"{type(exc).__name__}: {exc}"}
    ref = results.get("combine", {})
    rows = [("observed", lambda r: r.get("observed"))] + \
           [(f"expected {q:g}", lambda r, q=q: r.get("expected", {}).get(q)) for q in (0.025, 0.16, 0.5, 0.84, 0.975)] + \
           [("r_hat", lambda r: r.get("r_hat")), ("r 68% up", lambda r: (r.get("r_68") or [None, None])[1])]
    names = list(results)
    print(f"{'quantity':<16}" + "".join(f"{n:>22}" for n in names))
    for label, get in rows:
        cells = []
        for n in names:
            v = get(results[n]) if "error" not in results[n] else None
            cell = "n/a" if v is None else f"{v:.5g}"
            if n != "combine" and ref:
                cell += f" ({rel(v, get(ref))})"
            cells.append(f"{cell:>22}")
        print(f"{label:<16}" + "".join(cells))
    for n in names:
        if "error" in results[n]:
            print(f"{n}: ERROR {results[n]['error']}")
        for f in results[n].get("flags", []):
            print(f"{n}: flag: {f}")
    with open(os.path.join(workdir, "compare.json"), "w") as handle:
        json.dump({k: {kk: (dict((str(a), b) for a, b in vv.items()) if isinstance(vv, dict) else vv)
                       for kk, vv in v.items()} for k, v in results.items()}, handle, indent=1)
    print(f"\nWrote {os.path.join(workdir, 'compare.json')}")


if __name__ == "__main__":
    main()
