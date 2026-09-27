#!/usr/bin/env python3
"""Validation of the zmodel (zfit) backend.  Plain script (no pytest in rootana).

  source setup_env.sh; export PYTHONPATH=$PYTHONPATH:<repo>/python
  python3 tests/backend_zmodel_check.py [--workdir DIR] [--real-card CARD]

Checks
  1. formula translation vs RooFormulaVar for a list of TFormula expressions;
  2. nll_main and expected yields vs the numpy oracle (SemanticLikelihood) at 20 random
     points: counting (lnN sym/asym, gmN, rateParam), two-channel TH1 templates with shape
     systematics (scale != 1, |theta| > 1) and a parametric-histogram card; rateParam
     formula yields vs direct evaluation;
  3. every RooFit class in the class map: zfit vs RooFit normalised densities (50 points) and
     bin fractions at random parameter points;
  4. parametric card (Gaussian with RooFormulaVar mean of a param nuisance, exponential with
     floating slope, double-sided RooCrystalBall, RooFormulaVar <pdf>_norm): nll_main
     differences vs RooFit createNLL(Extended) differences on the same data, unbinned,
     weighted unbinned, and binned (center and integral);
  5. timing of nll_main; reproducibility of sample_unbinned;
  6. optional: the translation of a real card's pdfs (--real-card, needs combine_env.sh for
     Combine classes).
Exit status 1 if any check fails.
"""

import argparse
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))

from inference.model import observed_dataset  # noqa: E402
from inference.semantic_likelihood import SemanticLikelihood  # noqa: E402
from modelspec import rootinput  # noqa: E402
from modelspec.datacard import build_ir  # noqa: E402
from stat_backends.zmodel import BACKEND  # noqa: E402
from stat_backends.zmodel import formula as F  # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAILURES.append(name)


def write(path, text):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


# ----------------------------------------------------------------------------------------
# 1. formulas
# ----------------------------------------------------------------------------------------

FORMULAS = ["1/2*@0", "-@0^2", "@0^@1^2", "2^-1", "@0**2", "e", "pi", "x[0]*x[1]", "a*b", "(@0>1)*5",
            "(@0>1)&&(@1<2)", "max(@0,@1)", "TMath::Max(@0,1.)", "exp(-@0)", "TMath::Exp(@0)", "1e-1*@0", "!(@0>1)",
            "-2^2", "abs(-3)", "TMath::Gaus(@0,1,2)", "TMath::Gaus(@0,1,2,true)", "sqrt(@0)/@1", "3/2", "@0/2*3",
            "log(@0)", "TMath::Power(@0,2)", "pow(2,3)", "TMath::Erf(0.5)", "erfc(@0/3)", "atan2(@0,@1)",
            "TMath::Pi()", "@0*-1", "2*-@0^2", "- - @0", "sqrt2", "2*3/4", "(1<2)/2", "@0>1 ? 1 : 2", "1./3", "-3/2",
            "7/-2", "2^3/3", "5e-1", "1.e2", ".5", "TMath::E()", "TMath::Log10(100)", "log10(@0)", "fabs(-2)",
            "@0 == 2", "@0 != 2", "TMath::Sq(3)", "TMath::Sign(2,-1)", "std::exp(1)", "exp(1)^2", "a^b", "cosh(1)",
            "@0/@1/2", "2-@0-1", "(pow(max(0., @0 - @1), 0.5) * pow(120.0 - @0, 1.5) * exp(-0.1 * @0))",
            "TMath::Min(@0, @1) + tanh(@0) - sinh(0.3) + TMath::ATan(1)"]


def check_formulas(R):
    a = R.RooRealVar("fa", "a", 2.0)
    b = R.RooRealVar("fb", "b", 3.0)
    a.SetName("a")
    b.SetName("b")
    ops, tfops = F.NumpyOps(), F.TFOps()
    worst = 0.0
    for text in FORMULAS:
        ref = R.RooFormulaVar("ftest", "ftest", text, R.RooArgList(a, b)).getVal()
        node = F.parse(text, ["a", "b"])
        for o in (ops, tfops):
            val = float(np.asarray(F.compile_formula(node, o)([2.0, 3.0])))
            worst = max(worst, abs(val - ref) / max(1.0, abs(ref)))
    check("formula translation vs RooFormulaVar", worst < 1e-14, f"({len(FORMULAS)} formulas, max dev {worst:.1e})")
    for bad in ("@0%3", "TMath::Landau(@0)", "foo(@0)", "@5", "unknown_name"):
        try:
            F.parse(bad, ["a", "b"])
            check(f"formula '{bad}' rejected", False)
        except F.FormulaError:
            pass


# ----------------------------------------------------------------------------------------
# 2. oracle comparisons
# ----------------------------------------------------------------------------------------

COUNT_CARD = """imax 2 jmax 2 kmax *
bin a b
observation 10 20
bin      a    a    a    b    b    b
process  sig  bkg1 bkg2 sig  bkg1 bkg2
process  0    1    2    0    1    2
rate     3    8    2.5  4    15   0.1
lumi lnN 1.1  1.1  -    1.1  1.1  -
asy  lnN 0.9/1.2 - -    0.85/1.1 - -
bk   lnN -    1.3  -    -    1.25 -
cr   gmN 25   -    -    0.1  -    -    0.004
norm rateParam a bkg1 1.0 [0,5]
"""

FORMULA_CARD = COUNT_CARD + "fml rateParam b bkg1 (@0*@0+0.5*@1) norm,lumi\n"


def make_templates(R, path):
    f = R.TFile(path, "RECREATE")
    rng = np.random.default_rng(3)
    edges = np.array([0.0, 1.0, 2.0, 3.5, 5.0, 8.0])
    for ch in ("ch1", "ch2"):
        d = f.mkdir(ch)
        d.cd()
        base = {"sig": np.array([1.0, 4.0, 6.0, 3.0, 1.0]), "bkg": np.array([20.0, 15.0, 10.0, 8.0, 6.0])}
        if ch == "ch2":
            base = {k: v[::-1] * 1.3 for k, v in base.items()}
        hists = {}
        for proc, arr in base.items():
            hists[proc] = arr
            for syst in ("sh1", "sh2"):
                up = arr * (1 + 0.3 * rng.uniform(-1, 1, arr.size))
                down = arr * (1 + 0.2 * rng.uniform(-1, 1, arr.size))
                hists[f"{proc}_{syst}Up"] = up
                hists[f"{proc}_{syst}Down"] = down
        hists["data_obs"] = np.round(base["sig"] + base["bkg"] + rng.normal(0, 2, 5)).clip(0)
        for name, arr in hists.items():
            h = R.TH1D(name, name, len(edges) - 1, edges)
            for i, val in enumerate(arr):
                h.SetBinContent(i + 1, val)
            h.Write()
    f.Close()


TEMPLATE_CARD = """imax 2 jmax 1 kmax *
shapes * * templ.root $CHANNEL/$PROCESS $CHANNEL/$PROCESS_$SYSTEMATIC
bin ch1 ch2
observation -1 -1
bin      ch1 ch1 ch2 ch2
process  sig bkg sig bkg
process  0   1   0   1
rate     -1  -1  -1  -1
lumi lnN 1.05 1.05 1.05 1.05
sh1 shape 1   0.5 2.0 -
sh2 shape -   2.0 1   0.7
bn  lnN  -   0.8/1.3 - 1.1
"""


def make_param_hist_ws(R, path):
    w = R.RooWorkspace("w")
    x = R.RooRealVar("x", "x", 0.0, 10.0)
    x.setBins(20)
    getattr(w, "import")(x)
    w.factory("Gaussian::sig(x, mu[4.0], sg[0.7])")
    w.factory("Exponential::bkg(x, c[-0.3])")
    w.factory("Chebychev::bkg2(x, {k1[0.2], k2[-0.1]})")
    pdf = w.factory("SUM::tot(0.1*sig, 0.6*bkg, bkg2)")
    data = pdf.generateBinned(R.RooArgSet(w.var("x")), 300)
    data.SetName("data_obs")
    getattr(w, "import")(data)
    w.writeToFile(path, True)


PARAM_HIST_CARD = """imax 1 jmax 2 kmax *
shapes * * phist.root w:$PROCESS
bin c
observation -1
bin     c   c   c
process sig bkg bkg2
process 0   1   2
rate    12  150 100
lumi lnN 1.1 1.1 -
bnrm lnN -   0.9/1.15 -
b2   rateParam c bkg2 1 [0,4]
"""


def random_point(lik, rng):
    v = lik.nominal_values()
    for i, p in enumerate(lik.parameters):
        if p.name == lik.poi:
            v[i] = rng.uniform(0.0, 3.0)
        elif p.origin == "gmN":
            v[i] = rng.uniform(10.0, 40.0)
        elif p.origin in ("rateParam",):
            v[i] = rng.uniform(0.5, 2.0)
        elif p.role == "nuisance":
            v[i] = rng.uniform(-2.5, 2.5)
    return v


def compare_oracle(label, card, npts=20):
    model = build_ir(card)
    z = BACKEND.build_likelihood(model, None)
    o = SemanticLikelihood(model)
    data = observed_dataset(model)
    rng = np.random.default_rng(11)
    worst_nll = worst_exp = 0.0
    for _ in range(npts):
        v = random_point(z, rng)
        a, b = z.nll_main(v, z.prepare(data)), o.nll_main(v, o.prepare(data))
        worst_nll = max(worst_nll, abs(a - b) / max(1.0, abs(b)))
        ez, eo = z.expected_by_process(v), o.expected_by_process(v)
        for ch in eo:
            for p in eo[ch]:
                worst_exp = max(worst_exp, float(np.max(np.abs(ez[ch][p] - eo[ch][p]) / np.maximum(1e-12, np.abs(eo[ch][p])))))
    check(f"{label}: nll_main vs oracle", worst_nll < 1e-8, f"(max rel dev {worst_nll:.1e})")
    check(f"{label}: expected yields vs oracle", worst_exp < 1e-8, f"(max rel dev {worst_exp:.1e})")
    return z


def check_formula_card(card):
    model = build_ir(card)
    z = BACKEND.build_likelihood(model, None)
    rng = np.random.default_rng(5)
    worst = 0.0
    for _ in range(20):
        v = random_point(z, rng)
        norm, lumi = v[z.index["norm"]], v[z.index["lumi"]]
        want = 15.0 * 1.1 ** lumi * 1.25 ** v[z.index["bk"]] * (norm * norm + 0.5 * lumi)
        got = float(z.expected_by_process(v)["b"]["bkg1"][0])
        worst = max(worst, abs(got - want) / abs(want))
    check("rateParam formula yield", worst < 1e-12, f"(max rel dev {worst:.1e})")


# ----------------------------------------------------------------------------------------
# 3. class map vs RooFit
# ----------------------------------------------------------------------------------------

def class_map_cases(R):
    """(name, pdf, obs, floating vars, integral-check) built in a workspace."""
    w = R.RooWorkspace("wc")
    w.factory("x[100, 110]")
    w.var("x").setBins(40)
    f = w.factory
    f("expr::gmean('@0 + @1*@2', m0[104.5], dm[0.3], es[0, -5, 5])")
    f("Gaussian::gau(x, gmean, sg[0.8, 0.2, 3])")
    f("Exponential::ex(x, c[-0.3, -2, 2])")
    ex_neg = R.RooExponential("exneg", "exneg", w.var("x"), f("cn[0.2, -2, 2]"), True)
    getattr(w, "import")(ex_neg)
    f("CBShape::cbs(x, cm[105, 102, 108], cs[0.7, 0.2, 2], ca[1.2, 0.3, 4], cnn[3, 1.1, 10])")
    f("CBShape::cbsr(x, cm, cs, car[-1.0, -4, -0.3], cnn)")
    x = w.var("x")
    cb1 = R.RooCrystalBall("cb1", "", x, f("bx0[105, 102, 108]"), f("bsg[0.6, 0.2, 2]"), f("bal[1.5, -4, 4]"),
                           f("bn[2.5, 1.1, 10]"))
    cb1r = R.RooCrystalBall("cb1r", "", x, w.var("bx0"), w.var("bsg"), f("balr[-1.1, -4, -0.2]"), w.var("bn"))
    cb2s = R.RooCrystalBall("cb2s", "", x, w.var("bx0"), w.var("bsg"), w.var("bal"), w.var("bn"), True)
    cb2 = R.RooCrystalBall("cb2", "", x, w.var("bx0"), f("bsl[0.5, 0.2, 2]"), f("bsr[0.9, 0.2, 2]"),
                           f("bal2[1.3, 0.3, 4]"), f("bnl[3, 1.1, 10]"), f("bar[2.0, 0.3, 4]"), f("bnr[5, 1.1, 20]"))
    for p in (cb1, cb1r, cb2s, cb2):
        getattr(w, "import")(p)
    f("Chebychev::cheb(x, {h1[0.2, -1, 1], h2[-0.1, -1, 1], h3[0.05, -1, 1]})")
    f("Bernstein::bern(x, {q0[1, 0, 5], q1[0.5, 0, 5], q2[2, 0, 5]})")
    f("Polynomial::poly(x, {p1[-0.005, -0.1, 0.1], p2[0.00001, -0.001, 0.001]})")
    poly0 = R.RooPolynomial("poly0", "", x, R.RooArgList(f("u0[3, 2, 4]"), f("p0b[-0.005, -0.02, -0.001]")), 0)
    getattr(w, "import")(poly0)
    f("Uniform::uni(x)")
    f("SUM::addf(fr1[0.3, 0, 1]*gau, fr2[0.2, 0, 1]*cbs, ex)")
    f("SUM::adde(ns[20, 0, 100]*gau, nb[80, 0, 500]*ex)")
    add_rec = R.RooAddPdf("addr", "", R.RooArgList(w.pdf("gau"), w.pdf("cbs"), w.pdf("ex")),
                          R.RooArgList(w.var("fr1"), w.var("fr2")), True)
    getattr(w, "import")(add_rec)
    hx = R.RooRealVar("x", "x", 100, 110)
    hx.setBins(10)
    dh = w.pdf("gau").generateBinned(R.RooArgSet(x), 2000)
    hist = R.RooHistPdf("histp", "", R.RooArgSet(x), dh, 0)
    getattr(w, "import")(hist)
    f("SUM::addh(fh[0.4, 0, 1]*histp, ex)")
    f("EXPR::gen('pow(max(0., @0 - @1), @2) * pow(120.0 - @0, 1.5) * exp(-@3*@0)', x, g0[99, 95, 100], "
      "g1[0.8, 0.3, 3], g2[0.05, 0.001, 0.3])")
    f("expr::sx('@0 + @1*@2', x, dm, es)")
    f("EXPR::gens('exp(@1*(@0-105))*(1+TMath::Erf((@0-103)/2))', sx, c)")
    cases = ["gau", "ex", "exneg", "cbs", "cbsr", "cb1", "cb1r", "cb2s", "cb2", "cheb", "bern", "poly", "poly0", "uni",
             "addf", "adde", "addr", "histp", "addh", "gen", "gens"]
    return w, cases


def translate_and_compare(R, w, name, obs_name, rng, npoints=5, integral=True, tol=1e-6, extra_note=""):
    import zfit

    from stat_backends.zmodel.roofit import Translator, bin_integrals, roofit_reference

    pdf = w.pdf(name)
    obs = w.var(obs_name)
    floating = [v for v in rootinput.floating_params(pdf, obs)]
    zp = {}

    def make_param(n):
        if n not in zp:
            zp[n] = zfit.Parameter(f"chk_{name}_{n}", w.var(n).getVal())
        return zp[n]

    tr = Translator(R, obs_name, obs.getMin(), obs.getMax(), zp, [v.GetName() for v in floating], make_param)
    zpdf = tr.pdf(pdf)
    edges = rootinput.var_edges(obs)
    xs = np.linspace(obs.getMin(), obs.getMax(), 52)[1:-1]
    worst_d = worst_b = worst_i = worst_def = 0.0
    nominal = {v.GetName(): v.getVal() for v in floating}
    try:
        for k in range(npoints):
            for v in floating:
                if k > 0:
                    lo, hi = v.getMin(), v.getMax()
                    c = nominal[v.GetName()]
                    v.setVal(min(hi, max(lo, c + 0.15 * (hi - lo) * rng.uniform(-1, 1))))
                make_param(v.GetName()).set_value(v.getVal())
            d_r, b_r, d_def = roofit_reference(R, pdf, obs_name, xs, edges, "center")
            worst_def = max(worst_def, float(np.max(np.abs(d_def - d_r)) / np.max(d_r)))
            d_z = np.asarray(zpdf.pdf(xs[:, None])).ravel()
            worst_d = max(worst_d, float(np.max(np.abs(d_z - d_r)) / np.max(d_r)))
            xc = 0.5 * (np.array(edges[:-1]) + np.array(edges[1:]))
            b_z = np.asarray(zpdf.pdf(xc[:, None])).ravel() * np.diff(edges)
            worst_b = max(worst_b, float(np.max(np.abs(b_z - b_r)) / np.max(b_r)))
            if integral:
                _, i_r, _ = roofit_reference(R, pdf, obs_name, xs[:2], edges, "integral")
                i_z = np.asarray(bin_integrals(zpdf, tr.space.obs[0], edges)).ravel()
                worst_i = max(worst_i, float(np.max(np.abs(i_z - i_r)) / np.max(i_r)))
    finally:
        for v in floating:
            v.setVal(nominal[v.GetName()])
    ok = worst_d < tol and worst_b < tol and worst_i < tol
    check(f"class map {pdf.ClassName()} '{name}'", ok,
          f"(densities {worst_d:.1e}, centre fractions {worst_b:.1e}, "
          f"bin integrals {('%.1e' % worst_i) if integral else 'not checked'}; "
          f"classes {sorted(tr.classes)}{'; RooFit default normalisation off by %.1e' % worst_def if worst_def > 1e-9 else ''})"
          f"{extra_note}")
    return tr.classes


def check_class_map(R):
    w, cases = class_map_cases(R)
    rng = np.random.default_rng(2)
    for name in cases:
        translate_and_compare(R, w, name, "x", rng)
    for bad in ("RooLandau", "RooVoigtian"):
        pdf = {"RooLandau": w.factory("Landau::lan(x, lm[104], ls[1])"),
               "RooVoigtian": w.factory("Voigtian::voi(x, vm[104], vw[1], vs[1])")}[bad]
        try:
            from stat_backends.zmodel.roofit import Translator
            Translator(R, "x", 100, 110, {}, [], lambda n: None).pdf(pdf)
            check(f"{bad} rejected", False)
        except Exception as exc:  # noqa: BLE001 - we want to see the exception type
            check(f"{bad} raises UnsupportedByBackend naming the class",
                  type(exc).__name__ == "UnsupportedByBackend" and bad in str(exc), f"({exc})")


# ----------------------------------------------------------------------------------------
# 4. parametric card vs RooFit NLL
# ----------------------------------------------------------------------------------------

def make_param_ws(R, path, kind):
    """kind: 'unbinned', 'weighted', 'binned'."""
    w = R.RooWorkspace("w")
    f = w.factory
    f("x[100, 110]")
    w.var("x").setBins(50)
    f("expr::smean('@0 + @1*@2', m0[105], dm[0.3], es[0, -5, 5])")
    f("Gaussian::sig(x, smean, ssg[0.6])")
    f("Exponential::bkg(x, slope[-0.25, -2, 1])")
    x = w.var("x")
    tail = R.RooCrystalBall("tail", "", x, f("tx0[103, 101, 105]"), f("tsl[0.7]"), f("tsr[1.1]"), f("tal[1.2]"),
                           f("tnl[3]"), f("tar[1.8]"), f("tnr[6]"))
    getattr(w, "import")(tail)
    f("expr::tail_norm('1 + 0.1*@0', es)")
    # data from a known model
    f("expr::ys('20*@0', r_true[1.0])")
    tot = f("SUM::gen_tot(ys*sig, 100*bkg, 30*tail)")
    R.RooRandom.randomGenerator().SetSeed(4321)
    data = tot.generate(R.RooArgSet(x), 150)
    if kind == "weighted":
        wv = R.RooRealVar("wgt", "wgt", 1.0)
        ds = R.RooDataSet("data_obs", "", R.RooArgSet(x, wv), R.RooFit.WeightVar(wv))
        rng = np.random.default_rng(9)
        for i in range(data.numEntries()):
            row = data.get(i)
            x.setVal(row.getRealValue("x"))
            ds.add(R.RooArgSet(x), float(rng.uniform(0.5, 1.5)))
        data = ds
    elif kind == "binned":
        data = R.RooDataHist("data_obs", "", R.RooArgSet(x), data)
    data.SetName("data_obs")
    getattr(w, "import")(data)
    w.writeToFile(path, True)


PARAM_CARD = """imax 1 jmax 2 kmax *
shapes * * {ws} w:$PROCESS
bin c1
observation -1
bin     c1  c1  c1
process sig bkg tail
process 0   1   2
rate    20  100 30
es param 0 1
bnorm rateParam c1 bkg 1 [0,3]
"""


def roofit_param_nll(R, path):
    """RooFit extended NLL of the same model, as a function of the parameter dict."""
    f = R.TFile.Open(path)
    w = f.Get("w")
    r = R.RooRealVar("r", "r", 1.0, 0, 20)
    bnorm = R.RooRealVar("bnorm", "bnorm", 1.0, 0, 3)
    ns = R.RooFormulaVar("ns", "20*@0", R.RooArgList(r))
    nb = R.RooFormulaVar("nb", "100*@0", R.RooArgList(bnorm))
    nt = R.RooFormulaVar("nt", "30*@0", R.RooArgList(w.function("tail_norm")))
    model = R.RooAddPdf("model", "", R.RooArgList(w.pdf("sig"), w.pdf("bkg"), w.pdf("tail")), R.RooArgList(ns, nb, nt))
    data = w.data("data_obs")
    nll = model.createNLL(data, R.RooFit.Extended(True))
    variables = {"r": r, "bnorm": bnorm, "es": w.var("es"), "slope": w.var("slope"), "tx0": w.var("tx0")}
    keep = (f, w, r, bnorm, ns, nb, nt, model, data, nll)

    def evaluate(point):
        for k, var in variables.items():
            var.setVal(point[k])
        return nll.getVal()

    def pdf_parts(point, edges, method):
        """Per-process (yield, bin fractions) from RooFit at ``point``."""
        for k, var in variables.items():
            var.setVal(point[k])
        x = w.var("x")
        return [(ns.getVal(), rootinput.binned_pdf_contents(w.pdf("sig"), x, edges, method)),
                (nb.getVal(), rootinput.binned_pdf_contents(w.pdf("bkg"), x, edges, method)),
                (nt.getVal(), rootinput.binned_pdf_contents(w.pdf("tail"), x, edges, method))]
    return evaluate, pdf_parts, keep


def check_param_card(R, workdir, kind, integration="center"):
    ws = f"param_{kind}.root"
    path = os.path.join(workdir, ws)
    make_param_ws(R, path, kind)
    card = write(os.path.join(workdir, f"param_{kind}.txt"), PARAM_CARD.format(ws=ws))
    model = build_ir(card, bin_integration=integration)
    z = BACKEND.build_likelihood(model, None)
    data = observed_dataset(model)
    evaluate, pdf_parts, keep = roofit_param_nll(R, path)
    rng = np.random.default_rng(17)
    names = ["r", "bnorm", "es", "slope", "tx0"]
    pts = [{"r": 1.0, "bnorm": 1.0, "es": 0.0, "slope": -0.25, "tx0": 103.0}]
    for _ in range(19):
        pts.append({"r": rng.uniform(0.2, 2.5), "bnorm": rng.uniform(0.7, 1.4), "es": rng.uniform(-2.5, 2.5),
                    "slope": rng.uniform(-0.5, 0.0), "tx0": rng.uniform(102, 104)})
    edges = list(model.channels[0].observable.edges)
    widths = np.diff(edges)
    zs, rs = [], []
    for p in pts:
        v = z.nominal_values()
        for n in names:
            v[z.index[n]] = p[n]
        zs.append(z.nll_main(v, z.native(data)))
        if kind == "binned":
            counts = data.main["c1"].counts
            parts = pdf_parts(p, edges, integration)
            nu = sum(y * np.asarray(fr) for y, fr in parts)
            if integration == "center":  # RooFit's extended binned NLL (the pymodel/Combine convention)
                rs.append(evaluate(p))
            else:  # direct formula with RooFit bin integrals
                rs.append(float(np.sum(nu - counts * np.log(nu))))
        else:
            rs.append(evaluate(p))
    dz = np.array(zs) - zs[0]
    dr = np.array(rs) - rs[0]
    worst = float(np.max(np.abs(dz - dr)))
    label = f"parametric card ({kind}{', ' + integration if kind == 'binned' else ''})"
    ref_name = "RooFit bin integrals" if integration == "integral" and kind == "binned" else "RooFit createNLL"
    check(f"{label}: nll_main differences vs {ref_name}", worst < 1e-6,
          f"(20 points, max |dNLL_z - dNLL_ref| {worst:.1e}, dNLL range {np.ptp(dr):.3g})")
    for rec in z.translation_checks:
        print(f"      create() check {rec['process']} {rec['class']}: densities {rec['density_max_rel_dev']:.1e}, "
              f"bin fractions {rec['bin_fraction_max_rel_dev']:.1e}")
    del keep
    return z, card


# ----------------------------------------------------------------------------------------
# 5. timing and sampling
# ----------------------------------------------------------------------------------------

def timing(label, z, n=300):
    data = observed_dataset(z.model)
    native = z.native(data)
    v = z.nominal_values()
    z.nll_main(v, native)
    t0 = time.perf_counter()
    for i in range(n):
        v2 = v.copy()
        v2[z.poi_index] += 1e-4 * i
        z.nll_main(v2, native)
    dt = (time.perf_counter() - t0) / n
    print(f"[INFO] {label}: nll_main {dt * 1e3:.3f} ms per call (trace+first call {z.trace_seconds:.2f} s)")
    return dt


def check_sampling_exact(z, channel):
    """chi2 of a large sample vs exact bin integrals of the total intensity."""
    v = z.nominal_values()
    edges = np.asarray(z.model.channel(channel).observable.edges)
    fine = np.linspace(edges[0], edges[-1], 200001)
    dens = z.density(v, channel, fine)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(fine))])
    integ = np.diff(np.interp(edges, fine, cdf))
    big = z.sample_unbinned(v, channel, 400000, np.random.default_rng(8))
    h, _ = np.histogram(big, bins=edges)
    exp = integ / integ.sum() * len(big)
    chi2 = float(np.sum((h - exp) ** 2 / exp))
    ndf = len(h) - 1
    check("sample_unbinned distribution (400k events, chi2 vs bin integrals)", chi2 < ndf + 5 * math.sqrt(2 * ndf),
          f"(chi2 {chi2:.1f} / {ndf})")


# ----------------------------------------------------------------------------------------
# 6. real card
# ----------------------------------------------------------------------------------------

def check_real_card(R, card):
    from stat_backends.zmodel.roofit import Translator

    model = build_ir(card)
    ch = model.channels[0]
    print(f"[INFO] real card {card}: features {sorted(model.features())}")
    rng = np.random.default_rng(1)
    classes = set()
    for proc in ch.processes:
        ref = proc.shape.ref
        w = rootinput.get_workspace(ref.file, ref.workspace)
        pdf = w.pdf(ref.name)
        try:
            classes |= translate_and_compare(R, w, ref.name, ch.observable.name, rng, npoints=1,
                                             integral=False, tol=1e-6)
        except Exception as exc:  # noqa: BLE001 - report which class fails
            print(f"[INFO]   {proc.name}: {pdf.ClassName()} -> {type(exc).__name__}: {exc}")
            classes.add(pdf.ClassName())
    print(f"[INFO] real card needs classes: {sorted(classes)}")
    z = BACKEND.build_likelihood(model, None)
    o = SemanticLikelihood(model)
    data = observed_dataset(model)
    v = z.nominal_values()
    dev = abs(z.nll_main(v, z.prepare(data)) - o.nll_main(v, o.prepare(data)))
    check("real card (fixed pdf histograms) nll_main vs oracle", dev < 1e-8, f"(|diff| {dev:.1e})")
    # the same card with every pdf translated to zfit (as if its parameters floated)
    import copy

    m2 = copy.deepcopy(model)
    for proc in m2.channels[0].processes:
        proc.shape.contents = []
    zt = BACKEND.build_likelihood(m2, None)
    for note in zt.notes:
        print(f"[INFO]   {note}")
    for proc in ch.processes:
        exp_t = float(np.sum(zt.expected_by_process(v)[ch.name][proc.name]))
        exp_h = float(np.sum(z.expected_by_process(v)[ch.name][proc.name]))
        print(f"[INFO]   {proc.name}: sum of bin-centre yields, zfit-translated {exp_t:.6g} vs IR histogram "
              f"(RooFit default normalisation) {exp_h:.6g}  (rel. {exp_t / exp_h - 1:+.1e})")
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=os.environ.get("ZMODEL_CHECK_DIR", "zmodel_check"))
    ap.add_argument("--real-card", default=None)
    args = ap.parse_args()
    os.makedirs(args.workdir, exist_ok=True)
    workdir = os.path.abspath(args.workdir)
    R = rootinput.root()
    check_formulas(R)

    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        count = compare_oracle("counting (lnN, asym lnN, gmN, rateParam)", write("count.txt", COUNT_CARD))
        check_formula_card(write("count_formula.txt", FORMULA_CARD))
        make_templates(R, "templ.root")
        templ = compare_oracle("TH1 templates, 2 channels, shape systs", write("templ.txt", TEMPLATE_CARD))
        make_param_hist_ws(R, "phist.root")
        compare_oracle("parametric-histogram", write("phist.txt", PARAM_HIST_CARD))
    finally:
        os.chdir(cwd)

    check_class_map(R)
    zu, _ = check_param_card(R, workdir, "unbinned")
    check_param_card(R, workdir, "weighted")
    zb, _ = check_param_card(R, workdir, "binned", "center")
    zi, _ = check_param_card(R, workdir, "binned", "integral")

    timing("counting", count)
    timing("templates (2 channels x 5 bins)", templ)
    timing("parametric unbinned (150 events, 3 pdfs)", zu)
    timing("parametric binned center (50 bins)", zb)
    timing("parametric binned integral (50 bins)", zi)
    check_sampling_exact(zu, "c1")
    b = zu.sample_unbinned(zu.nominal_values(), "c1", 1000, np.random.default_rng(5))
    c = zu.sample_unbinned(zu.nominal_values(), "c1", 1000, np.random.default_rng(5))
    check("sample_unbinned reproducible from the numpy rng", np.array_equal(b, c))

    if args.real_card:
        check_real_card(R, args.real_card)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        return 1
    print("all zmodel checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
