"""Produce the Combine reference numbers used by tests/run_all.py.

Usage (in an environment where ``combine`` and ``text2workspace.py`` are on PATH; see
make_combine_fixtures.sh):

    python3 tests/combine_fixtures.py WORKDIR [--only NAME,...] [--jobs N]

For every fixture below, the inputs are COPIED into WORKDIR/<name>/ (never run next to the
cards in the repository or in mumep_ana), the example's make_inputs.py is run there, and
Combine is run with fixed seeds:

* text2workspace.py
* AsymptoticLimits (observed + expected)
* MultiDimFit --algo grid (50 points on r) and --algo singles
* FitDiagnostics (best fit, s+b and b-only fit results)
* Significance (asymptotic)
* for fixtures with ``nll_points``: Combine's own NLL (cacheutils::CachingSimNLL of model_s on
  data_obs, constraint terms included) at that many seeded parameter points of the
  text2workspace model, so the tests can compare NLL differences point by point
* for fixtures with ``hybrid``: HybridNew --LHCmode LHC-limits and LHC-feldman-cousins at
  single r points (fixed number of toys, --clsAcc 0), then Combine's own grid readout
  (--readHybridResults --grid) of the limit / interval.

Fixtures with ``measure`` run only that measurement: ``impacts`` (combineTool.py -M Impacts
--robustFit 1, with a tight minimiser tolerance) or ``scan2d`` (MultiDimFit --algo grid with two
-P parameters).

Numbers are read from the higgsCombine*.root limit trees (and the HypoTestResults / the
fitDiagnostics RooFitResults) with PyROOT and written to tests/fixtures/<name>.json with the
exact command lines and the Combine commit.
"""

import argparse
import datetime
import glob
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FIXTURE_DIR = os.path.join(HERE, "fixtures")
MUMEP_CARDS = "/exp/mu2e/app/users/mmackenz/mumep/mumep_ana/analysis/combine"
SEED = 20260927
TOYS = 1000
ROOT_LOCK = threading.RLock()  # PyROOT file access is not thread safe

# Discrete profiling: Combine's recommended options, plus an exhaustive scan of the category
# states (pymodel's Fitter profiles discrete parameters exhaustively; Combine's default
# iterative discrete minimisation gave a 1.7% different 97.5% expected limit on mumep_40_env).
ENVELOPE_OPTS = ["--cminDefaultMinimizerStrategy", "0", "--X-rtd", "MINIMIZER_freezeDisassociatedParams",
                 "--cminRunAllDiscreteCombinations"]

# name -> configuration.  "example": directory under examples/ (copied, make_inputs.py run);
# "mumep": (card, [workspaces]) copied from mumep_ana into datacards/ and workspaces/.
FIXTURES = {
    "counting": dict(example="counting", card="card.txt", grid=(0.0, 4.0),
                     hybrid=dict(cls=[0.5, 1.0, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0],
                                 fc=[0.1, 0.3, 0.6, 0.9, 1.2, 1.4, 1.6, 1.8, 2.1, 2.5])),
    "counting_multibin": dict(example="counting", card="card_multibin.txt", grid=(0.0, 4.0)),
    "templates": dict(example="templates", card="card.txt", grid=(0.0, 4.0)),
    "parametric_binned": dict(example="parametric_binned", card="card.txt", grid=(0.0, 3.0)),
    "parametric_unbinned": dict(example="parametric_unbinned", card="card.txt", grid=(0.0, 3.0)),
    "low_background_n0": dict(example="low_background", card="card_n0.txt", grid=(0.0, 6.0),
                              hybrid=dict(cls=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
                                          fc=[0.1, 0.5, 1.0, 1.5, 2.0, 2.3, 2.6, 3.0, 3.5])),
    "low_background_n1": dict(example="low_background", card="card_n1.txt", grid=(0.0, 8.0),
                              hybrid=dict(cls=[2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.5],
                                          fc=[0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0])),
    "envelope": dict(example="envelope", card="card.txt", grid=(0.0, 3.0), extra=ENVELOPE_OPTS),
    "mumem_75_hists": dict(mumep=("combine_mumem_75_evt_r0104_hists.txt",
                                  ["workspace_mumem_75_evt_r0104_hists.root"]), grid=None),
    "mumem_75_funcs": dict(mumep=("combine_mumem_75_evt_r0104_funcs.txt",
                                  ["workspace_mumem_75_evt_r0104_funcs.root"]), grid=None),
    "mumep_40_env": dict(mumep=("combine_mumep_40_evt_r0104_env.txt",
                                ["workspace_mumep_40_evt_r0104_env.root"]), grid=None, extra=ENVELOPE_OPTS),
    # multi-dimensional (p, t0) channels: 2D RooHistPdfs / RooProdPdf, 2D RooDataHist / RooDataSet
    "two_dim": dict(example="two_dim", card="card.txt", grid=(0.0, 5.0), nll_points=12),
    "two_dim_param": dict(example="two_dim", card="card_param.txt", grid=(0.0, 5.0), nll_points=12),
    "two_dim_unbinned": dict(example="two_dim", card="card_unbinned.txt", grid=(0.0, 5.0), nll_points=12),
    # autoMCStats (Barlow-Beeston-lite): TH1 templates with low-statistics weighted MC
    # bin_integrator: see patch_bin_integrator (Combine's autoMCStats normalisation is otherwise
    # integrated numerically in this ROOT version and the fits fail)
    "mcstats": dict(example="mcstats", card="card.txt", grid=(0.0, 4.0), bin_integrator=True),
    # shape systematics on RooAbsPdfs (VerticalInterpPdf / FastVerticalInterpHistPdf2)
    "pdf_shape_syst": dict(example="pdf_shape_syst", card="card.txt", grid=(0.0, 3.0)),
    "pdf_shape_syst_histpdf": dict(example="pdf_shape_syst", card="card_histpdf.txt", grid=(0.0, 3.0)),
    "pdf_shape_syst_floating": dict(example="pdf_shape_syst", card="card_floating.txt", grid=(0.0, 3.0)),
    "pdf_shape_syst_unbinned": dict(example="pdf_shape_syst", card="card_unbinned.txt", grid=(0.0, 3.0)),
    # impacts and 2D scans (measure=...): only that measurement is run (see produce_measure)
    "counting_impacts": dict(example="counting", card="card.txt", measure="impacts", r_range=(-5.0, 5.0)),
    "templates_impacts": dict(example="templates", card="card.txt", measure="impacts", r_range=(0.0, 20.0)),
    "templates_scan2d": dict(example="templates", card="card.txt", measure="scan2d", params=("r", "bkg_norm"),
                             ranges=((0.0, 3.0), (-2.5, 2.5)), points=400),
}


# ----------------------------------------------------------------------------------------

def root():
    import ROOT

    ROOT.gROOT.SetBatch(True)
    if not hasattr(ROOT, "RooMultiPdf"):
        ROOT.gSystem.Load("libHiggsAnalysisCombinedLimit.so")
    return ROOT


def combine_version() -> dict:
    exe = shutil.which("combine")
    if exe is None:
        raise RuntimeError("combine is not on PATH; source the Combine environment first")
    src = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(exe))))  # build/bin/combine
    info = {"executable": exe, "source": src}
    for key, cmd in (("commit", ["git", "rev-parse", "HEAD"]), ("describe", ["git", "describe", "--always", "--dirty"]),
                     ("branch", ["git", "rev-parse", "--abbrev-ref", "HEAD"])):
        out = subprocess.run(cmd, cwd=src, capture_output=True, text=True)
        info[key] = out.stdout.strip() if out.returncode == 0 else f"unknown ({out.stderr.strip()})"
    info["root"] = root().gROOT.GetVersion()
    return info


class Runner:
    def __init__(self, cwd, log):
        self.cwd, self.log = cwd, log
        self.commands = {}

    def run(self, key, cmd, env=None):
        """Run a command (list), record it under ``key`` and fail loudly on a non-zero exit."""
        line = " ".join(shlex.quote(c) for c in cmd)
        self.commands[key] = line
        t0 = time.time()
        with open(self.log, "a") as handle:
            handle.write(f"\n### {key}: {line}\n")
            handle.flush()
            proc = subprocess.run(cmd, cwd=self.cwd, stdout=handle, stderr=subprocess.STDOUT,
                                  env=None if env is None else dict(os.environ, **env))
        if proc.returncode != 0:
            raise RuntimeError(f"'{line}' failed with exit code {proc.returncode} (see {self.log})")
        return time.time() - t0


def limit_tree(path, branches):
    with ROOT_LOCK:
        return _limit_tree(path, branches)


def _limit_tree(path, branches):
    R = root()
    f = R.TFile.Open(path)
    if not f or f.IsZombie():
        raise FileNotFoundError(path)
    t = f.Get("limit")
    rows = []
    for entry in t:
        rows.append({b: float(getattr(entry, b)) for b in branches})
    f.Close()
    return rows


def one(pattern, cwd):
    hits = sorted(glob.glob(os.path.join(cwd, pattern)))
    if len(hits) != 1:
        raise RuntimeError(f"expected one file matching {pattern} in {cwd}, found {hits}")
    return hits[0]


def fit_result(rfr):
    if not rfr:
        return None
    out = {"status": int(rfr.status()), "covQual": int(rfr.covQual()), "minNll": float(rfr.minNll()),
           "params": {}}
    for p in rfr.floatParsFinal():
        out["params"][p.GetName()] = {"value": p.getVal(), "error": p.getError(),
                                      "error_lo": p.getErrorLo(), "error_hi": p.getErrorHi()}
    return out


def hybrid_results(path):
    """HypoTestResults saved with --saveHybridResult in a (hadded) HybridNew output."""
    with ROOT_LOCK:
        return _hybrid_results(path)


def _hybrid_results(path):
    R = root()
    f = R.TFile.Open(path)
    d = f.Get("toys")
    out = []
    for key in d.GetListOfKeys():
        obj = key.ReadObj()
        if not obj.InheritsFrom("RooStats::HypoTestResult"):
            continue
        name = key.GetName()  # HypoTestResult_mh120_r1.5_<seed>
        r = float(name.split("_r")[1].split("_")[0])
        nsb = obj.GetNullDistribution().GetSize() if obj.GetNullDistribution() else 0
        nb = obj.GetAltDistribution().GetSize() if obj.GetAltDistribution() else 0
        out.append({"r": r, "CLs": obj.CLs(), "CLs_err": obj.CLsError(), "CLb": obj.CLb(), "CLb_err": obj.CLbError(),
                    "CLsplusb": obj.CLsplusb(), "CLsplusb_err": obj.CLsplusbError(),
                    "test_stat_obs": obj.GetTestStatisticData(), "n_null_dist": nsb, "n_alt_dist": nb})
    f.Close()
    return sorted(out, key=lambda x: x["r"])


# ----------------------------------------------------------------------------------------

def prepare(name, cfg, workdir):
    """Copy the inputs to WORKDIR/<name>; return (cwd, card path relative to cwd, source)."""
    dest = os.path.join(workdir, name)
    if os.path.exists(dest):
        shutil.rmtree(dest)
    if "example" in cfg:
        src = os.path.join(REPO, "examples", cfg["example"])
        shutil.copytree(src, dest, ignore=shutil.ignore_patterns("*.root", "higgsCombine*", "*.log"))
        if os.path.exists(os.path.join(dest, "make_inputs.py")):
            subprocess.run([sys.executable, "make_inputs.py"], cwd=dest, check=True, capture_output=True)
        return dest, cfg["card"], f"examples/{cfg['example']}/{cfg['card']}"
    card, workspaces = cfg["mumep"]
    os.makedirs(os.path.join(dest, "datacards"))
    os.makedirs(os.path.join(dest, "workspaces"))
    shutil.copy(os.path.join(MUMEP_CARDS, "datacards", card), os.path.join(dest, "datacards", card))
    for ws in workspaces:
        shutil.copy(os.path.join(MUMEP_CARDS, "workspaces", ws), os.path.join(dest, "workspaces", ws))
    return dest, os.path.join("datacards", card), os.path.join(MUMEP_CARDS, "datacards", card)


def patch_bin_integrator(cwd, ws):
    """Make RooFit integrate every CMSHistErrorPropagator/CMSHistSum with RooBinIntegrator over
    the bins of its observable (the exact bin sum, which is what the class's own analytical
    integral returns).

    With ROOT 6.32, RooRealIntegral does not use CMSHistErrorPropagator::analyticalIntegral
    (the observable also reaches it through the CMSHistFunc servers) and integrates the step
    function numerically: the extended term is off by ~2e-5 relative in a parameter-dependent,
    non-smooth way, CachingAddNLL reports "integrals don't match" and falls back to it, and
    MIGRAD fails (MultiDimFit "failed", nonsensical expected limits).  The patch is stored in the
    workspace (RooAbsReal::specialIntegratorConfig) and changes nothing else."""
    with ROOT_LOCK:
        R = root()
        path = os.path.join(cwd, ws)
        f = R.TFile.Open(path)
        w = f.Get("w")
        names = []
        for func in w.allFunctions():
            if func.ClassName() not in ("CMSHistErrorPropagator", "CMSHistSum"):
                continue
            obs = w.var("CMS_th1x")
            cfg = func.specialIntegratorConfig(True)
            cfg.method1D().setLabel("RooBinIntegrator")  # CMS_th1x is a closed range
            cfg.getConfigSection("RooBinIntegrator").setRealValue("numBins", obs.getBins())
            exact = 0.0
            for b in range(obs.getBins()):
                obs.setVal(b + 0.5)
                exact += func.getVal()
            integral = func.createIntegral(R.RooArgSet(obs)).getVal()
            if abs(integral - exact) > 1e-9 * max(1.0, abs(exact)):
                raise RuntimeError(f"bin-integrator patch of {func.GetName()}: integral {integral} != bin sum {exact}")
            names.append(func.GetName())
        tmp = path + ".patched"
        w.writeToFile(tmp)
        f.Close()
        os.replace(tmp, path)
    return f"python: RooBinIntegrator (CMS_th1x bins) as specialIntegratorConfig of {', '.join(names)}"


def _declare_caching_nll(R):
    """Make cacheutils::CachingSimNLL usable from PyROOT (its header needs boost, whose include
    directory is found next to the boost library on LD_LIBRARY_PATH)."""
    if hasattr(R, "cacheutils") and hasattr(R.cacheutils, "CachingSimNLL"):
        return
    for d in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        inc = os.path.join(os.path.dirname(d.rstrip("/")), "include")
        if "boost" in d and os.path.isdir(os.path.join(inc, "boost", "ptr_container")):
            R.gInterpreter.AddIncludePath(inc)
            break
    if not R.gInterpreter.Declare('#include "HiggsAnalysis/CombinedLimit/interface/CachingNLL.h"'):
        raise RuntimeError("cannot declare Combine's CachingNLL.h (boost include directory not found)")


def nll_points(cwd, ws, n, r_range):
    """Combine's NLL (CachingSimNLL of model_s on data_obs, with the constraint terms) at the
    nominal parameters and at n - 1 seeded random points: r uniform in ``r_range``, constrained
    nuisances (those with a <name>_Pdf) N(nominal, 0.7), other floating parameters
    nominal * U(0.85, 1.15).  Only differences between points are meaningful."""
    import numpy as np

    with ROOT_LOCK:
        R = root()
        _declare_caching_nll(R)
        f = R.TFile.Open(os.path.join(cwd, ws))
        w = f.Get("w")
        mc = w.obj("ModelConfig")
        skip = {v.GetName() for v in mc.GetObservables()} | {v.GetName() for v in mc.GetGlobalObservables()}
        pars = sorted(v.GetName() for v in w.allVars() if not v.isConstant() and v.GetName() not in skip)
        nominal = {p: w.var(p).getVal() for p in pars}
        nll = R.cacheutils.CachingSimNLL(mc.GetPdf(), w.data("data_obs"), mc.GetNuisanceParameters())
        rng = np.random.default_rng(SEED)
        out = []
        for k in range(n):
            vals = dict(nominal)
            if k:
                for p in pars:
                    if p == "r":
                        vals[p] = float(rng.uniform(*r_range))
                    elif w.pdf(f"{p}_Pdf"):
                        vals[p] = float(nominal[p] + rng.normal(0.0, 0.7))
                    else:
                        vals[p] = float(nominal[p] * rng.uniform(0.85, 1.15))
                    var = w.var(p)
                    vals[p] = min(max(vals[p], var.getMin()), var.getMax())
            for p, v in vals.items():
                w.var(p).setVal(v)
            out.append({"params": vals, "nll": float(nll.getVal())})
        f.Close()
    return out


def produce(name, cfg, workdir, version, jobs):
    t_start = time.time()
    if "measure" in cfg:
        return produce_measure(name, cfg, workdir, version, jobs, t_start)
    cwd, card, source = prepare(name, cfg, workdir)
    log = os.path.join(cwd, "fixture.log")
    run = Runner(cwd, log)
    extra = list(cfg.get("extra", []))
    ws = "ws.root"
    out = {"fixture": name, "card": source, "combine": version, "generated": datetime.datetime.now().isoformat(),
           "seed": SEED, "extra_options": extra}
    run.run("text2workspace", ["text2workspace.py", card, "-m", "120", "-o", ws])
    if cfg.get("bin_integrator"):
        run.commands["workspace_patch"] = patch_bin_integrator(cwd, ws)
    if cfg.get("nll_points"):
        out["nll_points"] = nll_points(cwd, ws, cfg["nll_points"], cfg.get("grid") or (0.0, 3.0))
        run.commands["nll_points"] = f"python: cacheutils::CachingSimNLL(model_s, data_obs) at {cfg['nll_points']} points"

    run.run("asymptotic", ["combine", "-M", "AsymptoticLimits", ws, "-m", "120", "-n", f".{name}",
                           "--seed", str(SEED)] + extra)
    rows = limit_tree(one(f"higgsCombine.{name}.AsymptoticLimits.mH120*.root", cwd), ["limit", "quantileExpected"])
    out["asymptotic"] = {"observed": None, "expected": {}}
    for row in rows:
        q = round(row["quantileExpected"], 3)
        if q == -1:
            out["asymptotic"]["observed"] = row["limit"]
        else:
            out["asymptotic"]["expected"][f"{q:g}"] = row["limit"]

    grid = cfg.get("grid")
    if grid is None:  # real cards: around the asymptotic result
        hi = 1.5 * out["asymptotic"]["expected"]["0.975"]
        grid = (0.0, float(f"{hi:.3g}"))
    rng_opt = ["--setParameterRanges", f"r={grid[0]:g},{grid[1]:g}"]
    out["r_range"] = list(grid)

    run.run("multidimfit_grid", ["combine", "-M", "MultiDimFit", ws, "-m", "120", "-n", f".{name}.grid",
                                 "--algo", "grid", "--points", "50", "-P", "r", "--floatOtherPOIs", "1",
                                 "--seed", str(SEED)] + rng_opt + extra)
    rows = limit_tree(one(f"higgsCombine.{name}.grid.MultiDimFit.mH120*.root", cwd), ["r", "deltaNLL", "quantileExpected"])
    out["multidimfit_grid"] = {"n_points": 50, "best_fit": rows[0]["r"],
                               "points": [[row["r"], 2.0 * row["deltaNLL"]] for row in rows[1:]]}

    run.run("multidimfit_singles", ["combine", "-M", "MultiDimFit", ws, "-m", "120", "-n", f".{name}.singles",
                                    "--algo", "singles", "--seed", str(SEED)] + rng_opt + extra)
    rows = limit_tree(one(f"higgsCombine.{name}.singles.MultiDimFit.mH120*.root", cwd), ["r", "quantileExpected"])
    out["multidimfit_singles"] = {"best_fit": rows[0]["r"], "lower": rows[1]["r"] if len(rows) > 1 else None,
                                  "upper": rows[2]["r"] if len(rows) > 2 else None,
                                  "rows": [[row["r"], row["quantileExpected"]] for row in rows]}

    run.run("fitdiagnostics", ["combine", "-M", "FitDiagnostics", ws, "-m", "120", "-n", f".{name}",
                               "--seed", str(SEED), "--forceRecreateNLL"] + rng_opt + extra)
    with ROOT_LOCK:
        R = root()
        f = R.TFile.Open(one(f"fitDiagnostics.{name}.root", cwd))
        out["fitdiagnostics"] = {"fit_s": fit_result(f.Get("fit_s")), "fit_b": fit_result(f.Get("fit_b"))}
        f.Close()
    if out["fitdiagnostics"]["fit_s"] is None:
        raise RuntimeError(f"FitDiagnostics wrote no fit_s for {name} (see {log})")

    run.run("significance", ["combine", "-M", "Significance", ws, "-m", "120", "-n", f".{name}",
                             "--seed", str(SEED)] + extra)
    rows = limit_tree(one(f"higgsCombine.{name}.Significance.mH120*.root", cwd), ["limit"])
    out["significance"] = {"Z": rows[0]["limit"]}

    hyb = cfg.get("hybrid")
    if hyb:
        out["hybrid_cls"] = hybrid_mode(run, name, ws, cwd, "LHC-limits", hyb["cls"], 0.95, jobs)
        out["hybrid_fc"] = hybrid_mode(run, name, ws, cwd, "LHC-feldman-cousins", hyb["fc"], 0.90, jobs)
    out["commands"] = run.commands
    out["cpu_seconds_wall"] = round(time.time() - t_start, 1)
    path = os.path.join(FIXTURE_DIR, f"{name}.json")
    with open(path, "w") as handle:
        json.dump(out, handle, indent=1)
    return name, path, out["cpu_seconds_wall"]


def produce_measure(name, cfg, workdir, version, jobs, t_start):
    """Impacts (combineTool.py -M Impacts, --robustFit 1) or a 2D MultiDimFit grid."""
    cwd, card, source = prepare(name, cfg, workdir)
    run = Runner(cwd, os.path.join(cwd, "fixture.log"))
    ws = "ws.root"
    out = {"fixture": name, "card": source, "combine": version, "generated": datetime.datetime.now().isoformat(),
           "seed": SEED, "measure": cfg["measure"]}
    run.run("text2workspace", ["text2workspace.py", card, "-m", "120", "-o", ws])
    if cfg["measure"] == "impacts":
        lo, hi = cfg["r_range"]
        out["r_range"] = [lo, hi]
        # combineTool needs CMSSW_BASE/SCRAM_ARCH only for batch job templates
        env = {"CMSSW_BASE": os.environ.get("CMSSW_BASE", cwd), "SCRAM_ARCH": os.environ.get("SCRAM_ARCH", "none")}
        # a tight minimiser tolerance: with Combine's default (0.1) the fixed-nuisance fits are only
        # accurate to ~0.005 in r (e.g. templates bkg_norm: r = 1.0210 at tolerance 0.1, 1.0246 at 0.001)
        base = ["combineTool.py", "-M", "Impacts", "-d", ws, "-m", "120", "--robustFit", "1",
                "--cminDefaultMinimizerTolerance", "0.001",
                "--rMin", f"{lo:g}", "--rMax", f"{hi:g}", "--parallel", str(max(1, jobs))]
        run.run("impacts_initial", base + ["--doInitialFit"], env=env)
        run.run("impacts_fits", base + ["--doFits"], env=env)
        run.run("impacts_collect", base + ["-o", "impacts.json"], env=env)
        with open(os.path.join(cwd, "impacts.json")) as handle:
            out["impacts"] = json.load(handle)
    else:
        (px, py), ((xlo, xhi), (ylo, yhi)) = cfg["params"], cfg["ranges"]
        run.run("multidimfit_grid2d", ["combine", "-M", "MultiDimFit", ws, "-m", "120", "-n", f".{name}",
                                       "--algo", "grid", "--points", str(cfg["points"]), "-P", px, "-P", py,
                                       "--floatOtherPOIs", "1", "--seed", str(SEED),
                                       "--setParameterRanges", f"{px}={xlo:g},{xhi:g}:{py}={ylo:g},{yhi:g}"])
        rows = limit_tree(one(f"higgsCombine.{name}.MultiDimFit.mH120*.root", cwd), [px, py, "deltaNLL"])
        out["scan2d"] = {"params": [px, py], "ranges": [[xlo, xhi], [ylo, yhi]], "points": cfg["points"],
                         "best_fit": [rows[0][px], rows[0][py]],
                         "grid": [[row[px], row[py], 2.0 * row["deltaNLL"]] for row in rows[1:]]}
    out["commands"] = run.commands
    out["cpu_seconds_wall"] = round(time.time() - t_start, 1)
    path = os.path.join(FIXTURE_DIR, f"{name}.json")
    with open(path, "w") as handle:
        json.dump(out, handle, indent=1)
    return name, path, out["cpu_seconds_wall"]


def hybrid_mode(run, name, ws, cwd, mode, points, cl, jobs):
    tag = "cls" if mode == "LHC-limits" else "fc"
    base = ["combine", "-M", "HybridNew", ws, "-m", "120", "--LHCmode", mode, "-T", str(TOYS), "--clsAcc", "0",
            "--saveHybridResult"]
    if mode == "LHC-limits":
        base.append("--fullBToys")  # as many b-only as s+b toys (Combine default: 1/4)
    cmds = []
    for i, r in enumerate(points):
        cmds.append((f"hybrid_{tag}_r{r:g}", base + ["--singlePoint", f"{r:g}", "--seed", str(SEED + i),
                                                    "-n", f".{name}.{tag}.r{r:g}"]))
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(lambda kc: run.run(*kc), cmds))
    files = sorted(glob.glob(os.path.join(cwd, f"higgsCombine.{name}.{tag}.r*.HybridNew.mH120.*.root")))
    merged = f"merged_{tag}.root"
    run.run(f"hybrid_{tag}_hadd", ["hadd", "-f", merged] + [os.path.basename(p) for p in files])
    out = {"mode": mode, "toys": TOYS, "cl": cl, "points": hybrid_results(os.path.join(cwd, merged))}
    readout = ["combine", "-M", "HybridNew", ws, "-m", "120", "--LHCmode", mode, "--readHybridResults",
               "--grid", merged, "--cl", f"{cl:g}"]
    if mode == "LHC-limits":
        run.run("hybrid_cls_grid_observed", readout + ["-n", f".{name}.clsgrid"])
        rows = limit_tree(one(f"higgsCombine.{name}.clsgrid.HybridNew.mH120*.root", cwd), ["limit", "limitErr"])
        out["grid_limit"] = {"observed": rows[0]["limit"], "observed_err": rows[0]["limitErr"], "expected": {}}
        for q in (0.16, 0.5, 0.84):
            run.run(f"hybrid_cls_grid_expected_{q:g}", readout + ["--expectedFromGrid", f"{q:g}",
                                                                 "-n", f".{name}.clsgrid.exp{q:g}"])
            rows = limit_tree(one(f"higgsCombine.{name}.clsgrid.exp{q:g}.HybridNew.mH120*quant{q:.3f}*.root", cwd),
                              ["limit", "limitErr"])
            out["grid_limit"]["expected"][f"{q:g}"] = rows[0]["limit"]
    else:
        run.run("hybrid_fc_grid_upper", readout + ["-n", f".{name}.fcgrid.up"])
        up = limit_tree(one(f"higgsCombine.{name}.fcgrid.up.HybridNew.mH120*.root", cwd), ["limit", "limitErr"])
        run.run("hybrid_fc_grid_lower", readout + ["--lowerLimit", "-n", f".{name}.fcgrid.lo"])
        lo = limit_tree(one(f"higgsCombine.{name}.fcgrid.lo.HybridNew.mH120*.root", cwd), ["limit", "limitErr"])
        out["note"] = ("grid_interval is Combine's --readHybridResults readout; its --lowerLimit readout "
                       "is not meaningful when the interval starts at r = 0 (it returns the upper crossing). "
                       "The per-point p-values (CLsplusb) are the reference.")
        out["grid_interval"] = {"lower": lo[0]["limit"], "lower_err": lo[0]["limitErr"],
                                "upper": up[0]["limit"], "upper_err": up[0]["limitErr"]}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("workdir")
    ap.add_argument("--only", default="", help="comma-separated fixture names")
    ap.add_argument("--jobs", type=int, default=16, help="parallel HybridNew points / fixtures")
    args = ap.parse_args(argv)
    workdir = os.path.abspath(args.workdir)
    for forbidden in (os.path.abspath(MUMEP_CARDS), REPO):
        if workdir == forbidden or workdir.startswith(forbidden + os.sep):
            raise SystemExit(f"refusing to run Combine inside {forbidden}; choose a scratch directory")
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    names = [n for n in args.only.split(",") if n] or list(FIXTURES)
    unknown = [n for n in names if n not in FIXTURES]
    if unknown:
        raise SystemExit(f"unknown fixtures {unknown}; choose from {list(FIXTURES)}")
    version = combine_version()
    failures = []
    with ThreadPoolExecutor(max_workers=min(len(names), max(1, args.jobs // 4))) as pool:
        futures = {n: pool.submit(produce, n, FIXTURES[n], workdir, version, args.jobs) for n in names}
        for n, fut in futures.items():
            try:
                name, path, secs = fut.result()
                print(f"{name:22s} -> {path} ({secs:.0f} s)")
            except Exception as exc:  # report every failed fixture, then fail
                failures.append(n)
                print(f"{n:22s} FAILED: {exc}")
    if failures:
        raise SystemExit(f"fixtures failed: {failures}")


if __name__ == "__main__":
    main()
