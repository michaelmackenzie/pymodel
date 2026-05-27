#!/usr/bin/env python3
"""Generate a ROOT file with a RooFit workspace containing signal and
background PDFs, used as shape inputs for the roomodel simple_shapes_card.txt
example.

Run from the examples/roomodel directory:
    python simple_shapes.py
"""
import sys

import numpy as np


def _get_root():
    import ROOT

    try:
        if not getattr(ROOT, "_pymodel_roofit_quiet", False):
            ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)
            ROOT._pymodel_roofit_quiet = True
    except Exception:
        pass

    return ROOT


OBS_MIN = 100.0
OBS_MAX = 110.0
RATES = {"sig": 12.0, "bkg": 80.0}
RNG_SEED = 42


def build_shapes_workspace():
    ROOT = _get_root()
    ws = ROOT.RooWorkspace("shapes_ws")

    mass = ROOT.RooRealVar("mass", "mass", OBS_MIN, OBS_MAX)

    # Signal: narrow Gaussian at 105
    sig_mu = ROOT.RooRealVar("sig_mu", "sig_mu", 105.0, OBS_MIN, OBS_MAX)
    sig_sigma = ROOT.RooRealVar("sig_sigma", "sig_sigma", 0.35, 0.01, 2.0)
    sig_mu.setConstant(True)
    sig_sigma.setConstant(True)
    sig_pdf = ROOT.RooGaussian("sig", "Signal Gaussian", mass, sig_mu, sig_sigma)

    # Background: falling exponential
    bkg_lam = ROOT.RooRealVar("bkg_lam", "bkg_lam", -0.25, -3.0, -0.001)
    bkg_lam.setConstant(True)
    bkg_pdf = ROOT.RooExponential("bkg", "Background Exponential", mass, bkg_lam)

    for obj in (mass, sig_mu, sig_sigma, sig_pdf, bkg_lam, bkg_pdf):
        getattr(ws, "import")(obj)

    # Generate observed toy data (sig + bkg mixture)
    rng = np.random.default_rng(RNG_SEED)
    n_sig = int(rng.poisson(RATES["sig"]))
    n_bkg = int(rng.poisson(RATES["bkg"]))

    sig_obs = rng.normal(105.0, 0.35, n_sig)
    sig_obs = sig_obs[(sig_obs >= OBS_MIN) & (sig_obs <= OBS_MAX)]
    bkg_obs = rng.exponential(4.0, n_bkg * 3)
    bkg_obs = OBS_MAX - bkg_obs
    bkg_obs = bkg_obs[(bkg_obs >= OBS_MIN) & (bkg_obs <= OBS_MAX)][:n_bkg]

    all_obs = np.concatenate([sig_obs, bkg_obs])
    rng.shuffle(all_obs)

    ws_mass = ws.var("mass")
    data_obs = ROOT.RooDataSet("data_obs", "Observed data", ROOT.RooArgSet(ws_mass))
    for val in all_obs:
        ws_mass.setVal(float(val))
        data_obs.add(ROOT.RooArgSet(ws_mass))
    getattr(ws, "import")(data_obs)

    return ws


def write_shapes_file(output_file="simple_shapes.root"):
    ROOT = _get_root()
    ws = build_shapes_workspace()
    tf = ROOT.TFile.Open(output_file, "RECREATE")
    if tf is None or tf.IsZombie():
        raise RuntimeError(f"Could not create {output_file}")
    ws.Write()
    tf.Close()
    print(f"Wrote shapes workspace: {output_file}")


if __name__ == "__main__":
    write_shapes_file()
