"""Write workspace.root with RooWorkspace "w" for a binned parametric fit:

* observable x in [0, 10] with 40 bins;
* signal: RooGaussian(x, mean_sig, 0.5) with mean_sig = 5 * (1 + 0.01 * sig_scale)
  (a RooFormulaVar); the datacard makes sig_scale a Gaussian nuisance with ``param 0 1``;
* background: RooExponential(x, -0.3) with a constant slope;
* data_obs: a RooDataHist filled from 400 background + 20 signal events (fixed seed).
"""

import os

import numpy as np
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    rng = np.random.default_rng(424242)
    x = ROOT.RooRealVar("x", "x", 0.0, 10.0)
    x.setBins(40)
    sig_scale = ROOT.RooRealVar("sig_scale", "signal mean shift (sigma units)", 0.0, -5.0, 5.0)
    sig_scale.setConstant(True)  # the datacard `param` line makes it floating
    mean = ROOT.RooFormulaVar("mean_sig", "5.0*(1+0.01*@0)", ROOT.RooArgList(sig_scale))
    width = ROOT.RooRealVar("width_sig", "width_sig", 0.5, 0.01, 5.0)
    width.setConstant(True)
    sig = ROOT.RooGaussian("sig", "sig", x, mean, width)
    slope = ROOT.RooRealVar("slope_bkg", "slope_bkg", -0.3)
    bkg = ROOT.RooExponential("bkg", "bkg", x, slope)

    # 400 background events from exp(-0.3 x) truncated to [0, 10], 20 signal events
    u = rng.random(400)
    k = 0.3
    xb = -np.log(1 - u * (1 - np.exp(-k * 10.0))) / k
    xs = rng.normal(5.0, 0.5, 20)
    xs = xs[(xs > 0) & (xs < 10)]
    data = ROOT.RooDataHist("data_obs", "data_obs", ROOT.RooArgSet(x))
    for v in np.concatenate([xb, xs]):
        x.setVal(float(v))
        data.add(ROOT.RooArgSet(x))

    w = ROOT.RooWorkspace("w", "w")
    getattr(w, "import")(sig)
    getattr(w, "import")(bkg, ROOT.RooFit.RecycleConflictNodes())
    getattr(w, "import")(data)
    path = os.path.join(HERE, "workspace.root")
    w.writeToFile(path, True)
    print(f"wrote {path} ({data.sumEntries():.0f} events)")


if __name__ == "__main__":
    main()
