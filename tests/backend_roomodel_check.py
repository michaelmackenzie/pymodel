#!/usr/bin/env python3
"""Validation of the roomodel backend (plain script, no pytest).

    python tests/backend_roomodel_check.py --workdir DIR [--real-dir DIR] [--core-dir DIR] [--skip-cli]

1. Oracle: nll_main and expected_by_process agree with inference.semantic_likelihood at 20
   random points (counting with lnN sym/asym, lnU, gmN, rateParam + param; two-channel TH1
   templates with shape systematics incl. scale != 1; RooHistPdf "parametric-histogram").
2. Combine (when text2workspace.py and the Combine library are available): total-NLL
   differences between parameter points and per-process yields against the text2workspace
   workspace of the same card (templates, shapeN, rateParam formula, the unbinned card, and
   the user's real cards copied to --real-dir/datacards + workspaces, and the autoMCStats
   example examples/mcstats).
3. Export: the exported RooWorkspace, evaluated with RooFit, reproduces NLL differences.
4. CLI on an unbinned card (limit, fit, fit -t 20, scan, fc) and on the counting cards
   in --core-dir (asymptotic and toy limits vs the reference numbers).
Exits non-zero if a check fails.
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))

from inference.model import observed_dataset  # noqa: E402
from inference.semantic_likelihood import SemanticLikelihood  # noqa: E402
from modelspec import rootinput as R  # noqa: E402
from modelspec.datacard import build_ir  # noqa: E402
from stat_backends import get_backend  # noqa: E402

BACKEND = get_backend("roomodel")
FAILURES = []


def check(ok, what):
    print(("  ok    " if ok else "  FAIL  ") + what)
    if not ok:
        FAILURES.append(what)


# ----------------------------------------------------------------------------------------
# test inputs
# ----------------------------------------------------------------------------------------

def write(path, text):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


def make_counting(wd):
    return write(os.path.join(wd, "count_mix.txt"), """imax 2
jmax 2
kmax *
---
bin a b
observation 25 9
---
bin      a    a    a    b    b    b
process  sig  bkg  oth  sig  bkg  oth
process  0    1    2    0    1    2
rate     4.0  15   3.0  2.0  6.0  1.5
---
lumi   lnN  1.05  -  1.05  1.05  -  1.05
asy    lnN  0.9/1.15  -  -  1.2/0.95  -  -
flat   lnU  -  -  1.5  -  -  -
ctl    gmN 30  -  0.5  -  -  0.2  -
bscale rateParam a bkg 1.0 [0,5]
bscale rateParam b bkg 1.0 [0,5]
bscale param 1.0 -0.1/+0.2
""")


def make_formula_card(wd):
    return write(os.path.join(wd, "count_formula.txt"), """imax 1
jmax 1
kmax *
---
bin a
observation 30
---
bin a a
process sig bkg
process 0 1
rate 5 20
---
bkgN lnN - 1.1
k1 rateParam a bkg 1.0 [0,4]
k2 rateParam a bkg 1.0 [0,4]
k2 param 1.0 0.2
kprod rateParam a sig (@0*@0+@1) k1,k2
""")


def make_templates(wd, algo="shape"):
    ROOT = R.root()
    rng = np.random.default_rng(7)
    path = os.path.join(wd, f"templates_{algo}.root")
    f = ROOT.TFile(path, "RECREATE")
    specs = {"A": (5, 0.0, 5.0), "B": (4, 0.0, 8.0)}
    for ch, (nb, lo, hi) in specs.items():
        sig = np.array([1, 3, 6, 3, 1][:nb], dtype=float) * 2.0
        bkg = np.linspace(20, 8, nb)
        for proc, nom in (("sig", sig), ("bkg", bkg)):
            for var, arr in (("", nom),
                             ("_alphaUp", nom * np.linspace(1.15, 0.9, nb)),
                             ("_alphaDown", nom * np.linspace(0.9, 1.1, nb)),
                             ("_betaUp", nom * (1.0 + 0.2 * rng.random(nb))),
                             ("_betaDown", nom * (1.0 - 0.15 * rng.random(nb)))):
                h = ROOT.TH1D(f"{ch}_{proc}{var}", "", nb, lo, hi)
                for i, v in enumerate(arr):
                    h.SetBinContent(i + 1, v)
                h.Write()
        data = ROOT.TH1D(f"{ch}_data_obs", "", nb, lo, hi)
        for i, v in enumerate(np.round(sig + bkg + rng.normal(0, 2, nb))):
            data.SetBinContent(i + 1, max(v, 0))
        data.Write()
    f.Close()
    return write(os.path.join(wd, f"templates_{algo}.txt"), f"""imax 2
jmax 1
kmax *
---
shapes * * templates_{algo}.root $CHANNEL_$PROCESS $CHANNEL_$PROCESS_$SYSTEMATIC
---
bin A B
observation -1 -1
---
bin     A    A    B    B
process sig  bkg  sig  bkg
process 0    1    0    1
rate    -1   -1   -1   -1
---
lumi   lnN   1.05  1.05 1.05 1.05
alpha  {algo} 1     1    -    0.5
beta   {algo} 0.5   -    1    1
""")


def make_mcstats(wd):
    """examples/mcstats (autoMCStats), copied into wd/mcstats with its inputs generated there."""
    src = os.path.join(HERE, "..", "examples", "mcstats")
    dest = os.path.join(wd, "mcstats")
    os.makedirs(dest, exist_ok=True)
    for f in ("card.txt", "make_inputs.py"):
        shutil.copy(os.path.join(src, f), dest)
    subprocess.run([sys.executable, "make_inputs.py"], cwd=dest, check=True, capture_output=True)
    return os.path.join(dest, "card.txt")


def make_histpdf(wd):
    ROOT = R.root()
    path = os.path.join(wd, "histpdf.root")
    x = ROOT.RooRealVar("mx", "mx", 100, 110)
    x.setBins(20)
    ws = ROOT.RooWorkspace("ws", "ws")
    for name, shape in (("sig", lambda c: math.exp(-0.5 * ((c - 105) / 1.0) ** 2)),
                        ("bkg", lambda c: 1.0 + 0.05 * (c - 100))):
        h = ROOT.TH1D(f"h_{name}", "", 20, 100, 110)
        for i in range(20):
            h.SetBinContent(i + 1, shape(h.GetBinCenter(i + 1)))
        dh = ROOT.RooDataHist(f"dh_{name}", "", ROOT.RooArgList(x), h)
        getattr(ws, "import")(ROOT.RooHistPdf(f"pdf_{name}", "", ROOT.RooArgSet(x), dh))
    hd = ROOT.TH1D("hd", "", 20, 100, 110)
    rng = np.random.default_rng(11)
    for i in range(20):
        hd.SetBinContent(i + 1, float(rng.poisson(3)))
    getattr(ws, "import")(ROOT.RooDataHist("data_obs", "", ROOT.RooArgList(x), hd))
    ws.writeToFile(path, True)
    return write(os.path.join(wd, "histpdf.txt"), """imax 1
jmax 1
kmax *
---
shapes sig * histpdf.root ws:pdf_sig
shapes bkg * histpdf.root ws:pdf_bkg
shapes data_obs * histpdf.root ws:data_obs
---
bin c
observation -1
---
bin c c
process sig bkg
process 0 1
rate 6 60
---
bN lnN - 1.1/0.93
""")


def make_unbinned(wd):
    """Unbinned RooDataSet; Gaussian signal whose mean is m0 + 0.5*dm with dm a param nuisance;
    exponential background with a floating slope; floating background norm."""
    ROOT = R.root()
    path = os.path.join(wd, "unbinned.root")
    ws = ROOT.RooWorkspace("ws", "ws")
    x = ROOT.RooRealVar("mass", "mass", 100.0, 120.0)
    x.setBins(20)
    dm = ROOT.RooRealVar("dm", "dm", 0.0, -4.0, 4.0)
    mean = ROOT.RooFormulaVar("sig_mean", "", "110.0+0.5*@0", ROOT.RooArgList(dm))
    sigma = ROOT.RooRealVar("sig_sigma", "", 1.2)
    sigma.setConstant(True)
    sig = ROOT.RooGaussian("sig", "", x, mean, sigma)
    slope = ROOT.RooRealVar("slope", "", -0.08, -1.0, 0.0)
    bkg = ROOT.RooExponential("bkg", "", x, slope)
    norm = ROOT.RooRealVar("bkg_norm", "", 1.0, 0.0, 5.0)
    getattr(ws, "import")(sig)
    getattr(ws, "import")(bkg)
    getattr(ws, "import")(norm)
    ROOT.RooRandom.randomGenerator().SetSeed(5)
    data = bkg.generate(ROOT.RooArgSet(x), 200)
    data.append(sig.generate(ROOT.RooArgSet(x), 6))
    data.SetName("data_obs")
    getattr(ws, "import")(data)
    ws.writeToFile(path, True)
    return write(os.path.join(wd, "unbinned.txt"), """imax 1
jmax 1
kmax *
---
shapes * * unbinned.root ws:$PROCESS
shapes data_obs * unbinned.root ws:data_obs
---
bin u
observation -1
---
bin u u
process sig bkg
process 0 1
rate 6 200
---
lumi lnN 1.05 -
dm param 0 1
""")


# ----------------------------------------------------------------------------------------
# checks
# ----------------------------------------------------------------------------------------

def random_points(lik, n, seed):
    rng = np.random.default_rng(seed)
    x0 = lik.nominal_values()
    pts = [x0]
    for k in range(n - 1):
        x = x0.copy()
        for i, p in enumerate(lik.parameters):
            if p.role == "poi":
                x[i] = rng.uniform(0.0, 3.0)
            elif p.role == "discrete":
                x[i] = k % p.n_states
            elif p.floating:
                if p.origin == "gmN":
                    x[i] = max(1.0, p.value + rng.normal(0, 3))
                elif p.origin == "autoMCStats" and p.constraint.kind == "poisson":
                    x[i] = p.value * rng.uniform(0.3, 2.0)
                elif p.constraint is not None and p.constraint.kind == "flat":
                    x[i] = rng.uniform(p.lo, p.hi)
                elif p.origin in ("workspace",):
                    x[i] = p.value * (1 + 0.02 * rng.normal()) if p.value else rng.normal(0, 0.01)
                elif p.origin in ("rateParam", "param") and p.constraint is None or p.origin == "param" and p.value:
                    x[i] = p.value * (1 + 0.2 * rng.normal())
                else:
                    x[i] = rng.normal(0, 1.2)
                x[i] = min(max(x[i], p.lo), p.hi)
        pts.append(x)
    return pts


def oracle_check(card, label):
    m = build_ir(card)
    lik = BACKEND.build_likelihood(m, None)
    ora = SemanticLikelihood(m)
    d1, d2 = observed_dataset(m), observed_dataset(m)
    worst_nll = worst_exp = 0.0
    for x in random_points(lik, 20, 1):
        a, b = lik.nll_main(x, lik.native(d1)), ora.nll_main(x, ora.native(d2))
        worst_nll = max(worst_nll, abs(a - b) / max(1.0, abs(b)))
        ea, eb = lik.expected_by_process(x), ora.expected_by_process(x)
        for ch in eb:
            for p in eb[ch]:
                worst_exp = max(worst_exp, float(np.max(np.abs(ea[ch][p] - eb[ch][p]) / np.maximum(np.abs(eb[ch][p]), 1e-12))))
    print(f"[oracle] {label}: max rel NLL diff {worst_nll:.2e}, max rel yield diff {worst_exp:.2e}")
    check(worst_nll < 1e-8 and worst_exp < 1e-8, f"{label} agrees with the semantic oracle")


def have_combine():
    ROOT = R.root()
    return hasattr(ROOT, "RooMultiPdf") and shutil.which("text2workspace.py") is not None


def combine_check(card, label, cwd, tol=1e-6, npts=7, round_kappas=False):
    """Compare total-NLL differences and yields with Combine's workspace of the same card.

    ``round_kappas``: Combine's ShapeTools stores the template normalisation kappas as
    RooConstVar("%f") (6 decimals); pymodel uses the exact integral ratios.  To test the rest
    of the model exactly, the roomodel likelihood is then built with kappas rounded the same
    way (the unrounded difference is printed first).  Combine also stores templates as TH1F,
    which limits the agreement to ~1e-6."""
    if round_kappas:
        combine_check(card, label + " (exact kappas)", cwd, tol=1e-3, npts=npts)
        from modelspec import semantics as S

        orig = S.template_norm_kappas

        def rounded(*args, **kw):
            out = orig(*args, **kw)
            return None if out is None else tuple(float("%f" % k) for k in out)

        S.template_norm_kappas = rounded
        try:
            return combine_check(card, label + " (kappas rounded as in Combine)", cwd, tol=tol, npts=npts)
        finally:
            S.template_norm_kappas = orig
    ROOT = R.root()
    out = os.path.join(cwd, "t2w_" + os.path.basename(card).replace(".txt", ".root"))
    log = subprocess.run(["text2workspace.py", card, "-o", out], cwd=cwd, capture_output=True, text=True)
    if log.returncode != 0:
        check(False, f"{label}: text2workspace failed: {log.stderr[-500:]}")
        return
    old = os.getcwd()
    os.chdir(cwd)
    try:
        m = build_ir(card)
    finally:
        os.chdir(old)
    lik = BACKEND.build_likelihood(m, None)
    data = observed_dataset(m)
    f = ROOT.TFile.Open(out)
    w = f.Get("w")
    mc = w.obj("ModelConfig")
    nll = mc.GetPdf().createNLL(w.data("data_obs"), ROOT.RooFit.Constrain(mc.GetNuisanceParameters()))
    alias = {}
    for ch in m.channels:
        for pr in ch.processes:
            if pr.shape.ref is not None:
                alias[pr.shape.ref.name + "_norm"] = f"shape{'Sig' if pr.is_signal else 'Bkg'}_{pr.name}_{ch.name}__norm"

    def setw(x):
        for n, v in zip(lik.names, x):
            if w.cat(n):
                w.cat(n).setIndex(int(v))
                continue
            var = w.var(n) or w.var(alias.get(n, ""))
            if var:
                var.setVal(float(v))
            elif lik.model.parameters[n].floating:
                raise KeyError(f"parameter {n} not in the Combine workspace")

    # autoMCStats: RooRealIntegral (ROOT 6.32) integrates CMSHistErrorPropagator numerically,
    # so RooFit's extended term differs from the exact bin sum by a parameter-dependent ~1e-5
    # relative amount.  Replace it by the bin sum (what the model defines) and report its size.
    xobs = w.var("CMS_th1x")
    props = [(w.function(f"prop_bin{ch.name}"), w.pdf(f"pdf_bin{ch.name}_nuis") or w.pdf(f"pdf_bin{ch.name}"))
             for ch in m.channels if ch.mcstats is not None]

    def integral_artefact():
        out = 0.0
        for prop, sumpdf in props:
            exact = 0.0
            for b in range(xobs.getBins()):
                xobs.setVal(b + 0.5)
                exact += prop.getVal()
            out += sumpdf.expectedEvents(ROOT.RooArgSet(xobs)) - exact
        return out

    integ = {(ch.name, pr.name): float(np.sum(pr.shape.contents))
             for ch in m.channels if ch.mcstats is not None for pr in ch.processes}
    mine, comb = [], []
    arte = []
    worst_y = 0.0
    for x in random_points(lik, npts, 2):
        setw(x)
        a = integral_artefact() if props else 0.0
        arte.append(a)
        comb.append(nll.getVal() - a)
        mine.append(lik.nll(x, data))
        lik.set_values(x)
        for cb in lik.mw.channels:
            ch = lik._cpp[cb.name]
            for ip, pb in enumerate(cb.procs):
                y = ch.proc_yield(ip)
                if (cb.name, pb.name) in integ:  # CMSHistFunc: coefficient = yield / template integral
                    y = y / integ[(cb.name, pb.name)] if integ[(cb.name, pb.name)] > 0 else 0.0
                fn = w.function(f"n_exp_final_bin{cb.name}_proc_{pb.name}") or \
                    w.function(f"n_exp_bin{cb.name}_proc_{pb.name}")
                worst_y = max(worst_y, abs(y - fn.getVal()) / max(abs(fn.getVal()), 1e-12))
    mine, comb = np.array(mine), np.array(comb)
    raw = float(np.max(np.abs((mine - mine[0]) - (comb - comb[0]))))
    print(f"[combine] {label}: max |dNLL - dNLL_combine| = {raw:.2e}; max rel yield diff {worst_y:.2e}"
          + (f"; Combine's numeric-integral artefact removed: up to {np.ptp(arte):.2e}" if props else ""))
    check(raw < tol and worst_y < tol, f"{label} agrees with Combine")
    return raw


def export_check(card, label, wd):
    ROOT = R.root()
    m = build_ir(card)
    lik = BACKEND.build_likelihood(m, None)
    path = os.path.join(wd, "export_" + os.path.basename(card).replace(".txt", ".root"))
    BACKEND.export(m, path, None)
    f = ROOT.TFile.Open(path)
    w = f.Get("w")
    mc = w.obj("ModelConfig")
    nll = mc.GetPdf().createNLL(w.data("data_obs"), ROOT.RooFit.Constrain(mc.GetNuisanceParameters()),
                                ROOT.RooFit.GlobalObservables(mc.GetGlobalObservables()))
    data = observed_dataset(m)
    mine, rf = [], []
    for x in random_points(lik, 8, 3):
        for n, v in zip(lik.names, x):
            (w.cat(n).setIndex(int(v)) if w.cat(n) else w.var(n).setVal(float(v)))
        rf.append(nll.getVal())
        mine.append(lik.nll(x, data))
    mine, rf = np.array(mine), np.array(rf)
    diff = float(np.max(np.abs((mine - mine[0]) - (rf - rf[0]))))
    print(f"[export] {label}: RooFit NLL of the exported workspace vs roomodel, max |ddNLL| = {diff:.2e}")
    check(diff < 1e-6, f"{label} export reproduces the likelihood")


def cli(args, wd):
    import pymodel_core as C

    out = os.path.join(wd, "cli_result.json")
    argv = ["roomodel"] + args + ["--output", out]
    print("[cli] pymodel " + " ".join(args))
    old = os.getcwd()
    os.chdir(wd)
    try:
        C.run(argv)
    finally:
        os.chdir(old)
    with open(out, "r", encoding="utf-8") as handle:
        return json.load(handle)


def unbinned_checks(card, wd):
    ROOT = R.root()
    m = build_ir(card)
    lik = BACKEND.build_likelihood(m, None)
    d = observed_dataset(m)
    x = lik.nominal_values()
    # direct check of the unbinned convention with RooFit pdf evaluations
    exp = lik.expected_by_process(x)["u"]
    ytot = sum(lik._cpp["u"].proc_yield(i) for i in range(2))
    cb = lik.mw.channels[0]
    dens = np.zeros(len(d.main["u"].values))
    for ip, pb in enumerate(cb.procs):
        y = lik._cpp["u"].proc_yield(ip)
        for j, v in enumerate(d.main["u"].values):
            cb.obs.setVal(float(v))
            dens[j] += y * pb.states[0].pdf.getVal(ROOT.RooArgSet(cb.obs))
    ref = ytot - float(np.sum(np.log(dens)))
    val = lik.nll_main(x, lik.native(d))
    print(f"[unbinned] nll_main {val:.10f} vs direct {ref:.10f}; expected {[float(v.sum()) for v in exp.values()]}")
    check(abs(val - ref) < 1e-9 * abs(ref), "unbinned nll_main convention")
    vals = lik.sample_unbinned(x, "u", 1000, np.random.default_rng(1))
    vals2 = lik.sample_unbinned(x, "u", 1000, np.random.default_rng(1))
    check(len(vals) == 1000 and np.array_equal(vals, vals2) and vals.min() >= 100 and vals.max() <= 120,
          "sample_unbinned is reproducible and inside the range")
    res = cli(["limit", card, "--method", "asymptotic"], wd)["result"]
    print(f"  asymptotic: observed {res['observed']}, expected {res['expected']}")
    check(res["observed"] is not None and math.isfinite(res["observed"]), "unbinned asymptotic limit")
    res = cli(["fit", card], wd)["result"]
    fit = res["fits"][0]
    print(f"  fit: r = {fit['parameters']['r']['value']:.4g}, slope = {fit['parameters']['slope']['value']:.4g}")
    res = cli(["fit", card, "-t", "20", "--expect-signal", "1"], wd)["result"]
    print(f"  toys: {res['toy_summary']}")
    check(res["toy_summary"]["n_valid"] >= 18, "unbinned toy fits")
    res = cli(["scan", card, "--points", "15", "--range", "0:4"], wd)["result"]
    print(f"  scan: best fit {res['best_fit']:.4g}, intervals {res['intervals']}")
    res = cli(["fc", card, "--grid", "0:4:5", "--toys-per-point", "50", "--refine", "0"], wd)["result"]
    print(f"  fc: [{res['lower']}, {res['upper']}]")
    check(res["upper"] is not None, "unbinned Feldman-Cousins")


def counting_cli(core_dir, wd):
    refs = {"count_0.txt": 0.247, "count_80.txt": 1.5695, "count_150.txt": 7.588}
    for card, ref in refs.items():
        path = os.path.join(core_dir, card)
        if not os.path.exists(path):
            print(f"[cli] {path} missing, skipped")
            continue
        res = cli(["limit", path, "--method", "asymptotic"], wd)["result"]
        print(f"  {card}: asymptotic observed {res['observed']:.4f} (reference {ref})")
        check(abs(res["observed"] - ref) < 0.01 * ref + 0.002, f"{card} asymptotic limit")
    path = os.path.join(core_dir, "count_80_lnN.txt")
    if os.path.exists(path):
        asy = cli(["limit", path, "--method", "asymptotic"], wd)["result"]
        res = cli(["limit", path, "--method", "toys", "--toys-per-point", "200", "--refine", "2"], wd)["result"]
        print(f"  count_80_lnN.txt: toy observed {res['observed']} +- {res['observed_err']} "
              f"(asymptotic {asy['observed']:.4f})")
        check(res["observed"] is not None, "count_80_lnN toy limit")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--real-dir", default=None, help="scratch copy with datacards/ and workspaces/ of the user's cards")
    ap.add_argument("--core-dir", default=None, help="directory with count_{0,80,150}.txt, count_80_lnN.txt")
    ap.add_argument("--skip-cli", action="store_true")
    a = ap.parse_args()
    wd = os.path.abspath(a.workdir)
    real_dir = os.path.abspath(a.real_dir) if a.real_dir else None
    core_dir = os.path.abspath(a.core_dir) if a.core_dir else None
    os.makedirs(wd, exist_ok=True)
    old = os.getcwd()
    os.chdir(wd)
    cards = {"counting": make_counting(wd), "formula": make_formula_card(wd), "templates": make_templates(wd),
             "templates-shapeN": make_templates(wd, "shapeN"), "histpdf": make_histpdf(wd),
             "unbinned": make_unbinned(wd), "mcstats": make_mcstats(wd)}
    os.chdir(old)
    os.chdir(wd)  # datacards use paths relative to the card directory / working directory
    for key in ("counting", "templates", "histpdf", "mcstats"):
        oracle_check(cards[key], key)
    for key in ("counting", "templates", "histpdf", "unbinned"):
        export_check(cards[key], key, wd)
    if have_combine():
        for key in ("counting", "formula", "histpdf", "unbinned"):
            combine_check(cards[key], key, wd)
        for key in ("templates", "templates-shapeN"):
            combine_check(cards[key], key, wd, round_kappas=True)
        mdir = os.path.dirname(cards["mcstats"])
        os.chdir(mdir)
        # tolerance 1e-5: Combine stores the templates and the rescaled up/down templates as TH1F
        combine_check("card.txt", "mcstats (autoMCStats)", mdir, tol=1e-5, round_kappas=True, npts=12)
        os.chdir(wd)
        if real_dir:
            rd = real_dir
            for card in ("combine_mumem_75_evt_r0104_hists.txt", "combine_mumem_75_evt_r0104_funcs.txt",
                         "combine_mumep_40_evt_r0104_env.txt", "combine_total_mumem_75_evt_r0101.txt"):
                path = os.path.join(rd, "datacards", card)
                if os.path.exists(path):
                    os.chdir(rd)
                    combine_check(os.path.join("datacards", card), card, rd, tol=1e-8)
                    os.chdir(wd)
    else:
        print("[combine] Combine not available (source combine_env.sh): comparisons skipped")
    if not a.skip_cli:
        unbinned_checks(cards["unbinned"], wd)
        if core_dir:
            counting_cli(core_dir, wd)
    os.chdir(old)
    print("\nFAILED: " + "; ".join(FAILURES) if FAILURES else "\nall roomodel checks passed")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
