"""Write the inputs of the pdf shape-systematic example (``shape`` lines on RooAbsPdfs).

workspace.root, RooWorkspace "w" (names follow the mumep_ana convention
``<proc>_pdf`` / ``<proc>_pdf_<syst>Up|Down`` with ``_norm`` objects next to them):

* observable x in [0, 10] with 40 bins;
* sig_pdf: RooGaussian(x, 5.0, 0.5), with the variations
  sigshift (mean 5.1 / 4.9) and sigwidth (width 0.55 / 0.46);
* bkg_pdf: RooExponential(x, -0.3), with the variation bkgslope (slope -0.28 / -0.32);
* sigf_pdf: like sig_pdf but its mean is 5 * (1 + 0.01 * sig_scale) in the nominal and
  in the Up/Down pdfs (mean offsets +-0.1), with sig_scale a ``param`` nuisance of
  card_floating.txt (floating shape parameters: only roomodel can evaluate it);
* the variations are modest on purpose: VerticalInterpPdf extrapolates the un-normalised
  pdf values linearly for |scale * theta| > 1, and large width/shift variations then make
  the morph negative where the pdf still has probability (see docs/statistics.md);
* <pdf>_norm constants for the nominal pdfs (1.0) and for every Up/Down pdf (1.1 / 0.9):
  Combine ignores the Up/Down ones (see docs/statistics.md), pymodel notes that;
* data_obs: RooDataHist of 400 background + 30 signal events (fixed seed);
  data_obs_unbinned: the same events as a RooDataSet.

workspace_hist.root, RooWorkspace "w": the same processes as fixed RooHistPdfs (bin-centre
density times width of the pdfs above, times 1000 events), same names, same data_obs.
"""

import os

import numpy as np
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
HERE = os.path.dirname(os.path.abspath(__file__))

SIG = {"": (5.0, 0.5), "_sigshiftUp": (5.1, 0.5), "_sigshiftDown": (4.9, 0.5),
       "_sigwidthUp": (5.0, 0.55), "_sigwidthDown": (5.0, 0.46)}
BKG = {"": -0.3, "_bkgslopeUp": -0.28, "_bkgslopeDown": -0.32}
NORM = {"": 1.0, "Up": 1.1, "Down": 0.9}


def _norm_value(suffix):
    return NORM["Up"] if suffix.endswith("Up") else NORM["Down"] if suffix.endswith("Down") else NORM[""]


def make_pdfs(x):
    """{name: pdf} of all nominal and varied pdfs (plus their _norm constants)."""
    out, norms, keep = {}, [], []
    for suffix, (mean, width) in SIG.items():
        m = ROOT.RooRealVar(f"mean_sig{suffix}", "", mean)
        s = ROOT.RooRealVar(f"width_sig{suffix}", "", width)
        keep += [m, s]
        out[f"sig_pdf{suffix}"] = ROOT.RooGaussian(f"sig_pdf{suffix}", "", x, m, s)
    for suffix, slope in BKG.items():
        c = ROOT.RooRealVar(f"slope_bkg{suffix}", "", slope)
        keep.append(c)
        out[f"bkg_pdf{suffix}"] = ROOT.RooExponential(f"bkg_pdf{suffix}", "", x, c)
    for name in list(out):
        n = ROOT.RooRealVar(f"{name}_norm", "", _norm_value(name))
        n.setConstant(True)
        norms.append(n)
    return out, norms, keep


def make_floating(x):
    sig_scale = ROOT.RooRealVar("sig_scale", "signal mean shift", 0.0, -5.0, 5.0)
    sig_scale.setConstant(True)  # the datacard `param` line makes it floating
    out, keep = {}, [sig_scale]
    for suffix, offset in (("", 0.0), ("_sigshiftUp", 0.1), ("_sigshiftDown", -0.1)):
        m = ROOT.RooFormulaVar(f"mean_sigf{suffix}", f"5.0*(1+0.01*@0)+{offset!r}", ROOT.RooArgList(sig_scale))
        s = ROOT.RooRealVar(f"width_sigf{suffix}", "", 0.5)
        keep += [m, s]
        out[f"sigf_pdf{suffix}"] = ROOT.RooGaussian(f"sigf_pdf{suffix}", "", x, m, s)
    for suffix, width in (("_sigwidthUp", 0.55), ("_sigwidthDown", 0.46)):
        m = ROOT.RooFormulaVar(f"mean_sigf{suffix}", "5.0*(1+0.01*@0)", ROOT.RooArgList(sig_scale))
        s = ROOT.RooRealVar(f"width_sigf{suffix}", "", width)
        keep += [m, s]
        out[f"sigf_pdf{suffix}"] = ROOT.RooGaussian(f"sigf_pdf{suffix}", "", x, m, s)
    return out, keep


def events(rng):
    u = rng.random(400)
    k = 0.3
    xb = -np.log(1 - u * (1 - np.exp(-k * 10.0))) / k
    xs = rng.normal(5.0, 0.5, 30)
    return np.concatenate([xb, xs[(xs > 0) & (xs < 10)]])


def write(path, pdfs, extra, x, xs):
    data = ROOT.RooDataHist("data_obs", "data_obs", ROOT.RooArgSet(x))
    unb = ROOT.RooDataSet("data_obs_unbinned", "data_obs_unbinned", ROOT.RooArgSet(x))
    for v in xs:
        x.setVal(float(v))
        data.add(ROOT.RooArgSet(x))
        unb.add(ROOT.RooArgSet(x))
    w = ROOT.RooWorkspace("w", "w")
    for obj in list(pdfs.values()) + extra:
        getattr(w, "import")(obj, ROOT.RooFit.RecycleConflictNodes(), ROOT.RooFit.Silence())
    getattr(w, "import")(data)
    getattr(w, "import")(unb)
    w.writeToFile(path, True)
    print(f"wrote {path} ({data.sumEntries():.0f} events)")


def histpdfs(x, pdfs):
    """Fixed RooHistPdfs with the bin-centre density * width of ``pdfs`` (times 1000)."""
    nset = ROOT.RooArgSet(x)
    edges = [x.getBinning().binLow(i) for i in range(x.getBins())] + [x.getMax()]
    out, keep = {}, []
    for name, pdf in pdfs.items():
        dh = ROOT.RooDataHist(f"{name}_hist", "", ROOT.RooArgSet(x))
        for lo, hi in zip(edges[:-1], edges[1:]):
            x.setVal(0.5 * (lo + hi))
            dh.set(ROOT.RooArgSet(x), 1000.0 * pdf.getVal(nset) * (hi - lo))
        keep.append(dh)
        out[name] = ROOT.RooHistPdf(name, "", ROOT.RooArgSet(x), dh)
    return out, keep


def main():
    rng = np.random.default_rng(777001)
    x = ROOT.RooRealVar("x", "x", 0.0, 10.0)
    x.setBins(40)
    xs = events(rng)
    pdfs, norms, keep = make_pdfs(x)
    fpdfs, fkeep = make_floating(x)
    write(os.path.join(HERE, "workspace.root"), {**pdfs, **fpdfs}, norms, x, xs)
    hpdfs, hkeep = histpdfs(x, pdfs)
    write(os.path.join(HERE, "workspace_hist.root"), hpdfs, norms, x, xs)
    del keep, fkeep, hkeep


if __name__ == "__main__":
    main()
