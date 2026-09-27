"""Write workspace.root with RooWorkspace "w" for a discrete-profiling (envelope) model.

Needs Combine's library (RooMultiPdf): run it in the Combine environment, or with
PYMODEL_ROOT_LIBS=libHiggsAnalysisCombinedLimit.so pointing to it.

* observable x in [1, 11] with 50 bins; data_obs is a RooDataHist of 1000 events drawn from
  a power law x^-1.2 plus 25 signal events (fixed seed);
* signal: RooGaussian(x, 6, 0.3), rate 25 in the datacard;
* background "bkg": a RooMultiPdf over three functions, each with free shape parameters:
    0  exponential  exp(p_exp * x)
    1  power law    x^p_pow
    2  Bernstein    2nd order, coefficients (1, b1, b2)
  with the category ``pdfindex`` (``pdfindex discrete`` in the datacard) and a floating
  normalisation bkg_norm.
"""

import os

import numpy as np
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
HERE = os.path.dirname(os.path.abspath(__file__))


def load_combine():
    if not hasattr(ROOT, "RooMultiPdf"):
        lib = os.environ.get("PYMODEL_ROOT_LIBS", "libHiggsAnalysisCombinedLimit.so").split(":")[0]
        if ROOT.gSystem.Load(lib) < 0:
            raise RuntimeError(f"cannot load {lib}: RooMultiPdf needs the Combine library")


def main():
    load_combine()
    rng = np.random.default_rng(777)
    x = ROOT.RooRealVar("x", "x", 1.0, 11.0)
    x.setBins(50)
    mean = ROOT.RooRealVar("mean_sig", "mean_sig", 6.0)
    width = ROOT.RooRealVar("width_sig", "width_sig", 0.3, 0.01, 5.0)
    width.setConstant(True)
    sig = ROOT.RooGaussian("sig", "sig", x, mean, width)

    p_exp = ROOT.RooRealVar("p_exp", "p_exp", -0.2, -3.0, 0.0)
    f_exp = ROOT.RooExponential("bkg_exp", "bkg_exp", x, p_exp)
    p_pow = ROOT.RooRealVar("p_pow", "p_pow", -1.0, -5.0, 0.0)
    f_pow = ROOT.RooGenericPdf("bkg_pow", "bkg_pow", "TMath::Power(@0,@1)", ROOT.RooArgList(x, p_pow))
    b0 = ROOT.RooRealVar("b0", "b0", 1.0)
    b1 = ROOT.RooRealVar("b1", "b1", 0.3, 0.0, 10.0)
    b2 = ROOT.RooRealVar("b2", "b2", 0.1, 0.0, 10.0)
    f_bern = ROOT.RooBernstein("bkg_bern", "bkg_bern", x, ROOT.RooArgList(b0, b1, b2))

    cat = ROOT.RooCategory("pdfindex", "pdfindex")
    pdfs = ROOT.RooArgList(f_exp, f_pow, f_bern)
    bkg = ROOT.RooMultiPdf("bkg", "bkg", cat, pdfs)
    bkg_norm = ROOT.RooRealVar("bkg_norm", "bkg_norm", 1000.0, 0.0, 5000.0)

    # data: power law x^-1.2 on [1, 11] by inversion, plus a Gaussian signal
    a = -1.2 + 1.0
    u = rng.random(1000)
    xb = (1.0 + u * (11.0 ** a - 1.0)) ** (1.0 / a)
    xs = rng.normal(6.0, 0.3, 25)
    data = ROOT.RooDataHist("data_obs", "data_obs", ROOT.RooArgSet(x))
    for v in np.concatenate([xb, xs]):
        if 1.0 < v < 11.0:
            x.setVal(float(v))
            data.add(ROOT.RooArgSet(x))

    w = ROOT.RooWorkspace("w", "w")
    getattr(w, "import")(sig)
    getattr(w, "import")(cat)
    getattr(w, "import")(bkg, ROOT.RooFit.RecycleConflictNodes())
    getattr(w, "import")(bkg_norm)
    getattr(w, "import")(data)
    path = os.path.join(HERE, "workspace.root")
    w.writeToFile(path, True)
    print(f"wrote {path} ({data.sumEntries():.0f} events)")


if __name__ == "__main__":
    main()
