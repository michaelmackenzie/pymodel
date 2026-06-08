#!/usr/bin/env python3
"""Generate a ROOT file with a RooFit workspace containing binned signal and
background PDFs, used as shape inputs for the roomodel simple_shapes_binned_card.txt
example.

Run from the examples/roomodel directory:
    python simple_shapes_binned.py
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


NBINS = 20
OBS_MIN = 100.0
OBS_MAX = 110.0
BIN_EDGES = np.linspace(OBS_MIN, OBS_MAX, NBINS + 1)
RATES = {"sig": 12.0, "bkg": 80.0}
RNG_SEED = 12345


def build_shapes_workspace():
    ROOT = _get_root()
    ws = ROOT.RooWorkspace("shapes_ws")

    mass = ROOT.RooRealVar("mass", "mass", OBS_MIN, OBS_MAX)
    mass.setBins(NBINS)

    # Signal: narrow Gaussian at 105 (constant parameters)
    sig_mu = ROOT.RooRealVar("sig_mu", "sig_mu", 105.0, OBS_MIN, OBS_MAX)
    sig_sigma = ROOT.RooRealVar("sig_sigma", "sig_sigma", 0.35, 0.01, 2.0)
    sig_mu.setConstant(True)
    sig_sigma.setConstant(True)
    sig_pdf = ROOT.RooGaussian("sig", "Signal Gaussian", mass, sig_mu, sig_sigma)

    # Background: falling exponential (constant parameter)
    bkg_lam = ROOT.RooRealVar("bkg_lam", "bkg_lam", -0.25, -3.0, -0.001)
    bkg_lam.setConstant(True)
    bkg_pdf = ROOT.RooExponential("bkg", "Background Exponential", mass, bkg_lam)

    for obj in (mass, sig_mu, sig_sigma, sig_pdf, bkg_lam, bkg_pdf):
        getattr(ws, "import")(obj)

    # Generate observed toy data (sig + bkg mixture) with Poisson bin counts
    rng = np.random.default_rng(RNG_SEED)

    # Precompute PDF integrals per bin
    sig_counts = np.zeros(NBINS)
    bkg_counts = np.zeros(NBINS)
    for i in range(NBINS):
        mass.setRange("bin%d" % i, BIN_EDGES[i], BIN_EDGES[i + 1])
        sig_integral = sig_pdf.createIntegral(ROOT.RooArgSet(mass), ROOT.RooFit.Range("bin%d" % i))
        bkg_integral = bkg_pdf.createIntegral(ROOT.RooArgSet(mass), ROOT.RooFit.Range("bin%d" % i))
        sig_counts[i] = rng.poisson(RATES["sig"] * sig_integral.getVal())
        bkg_counts[i] = rng.poisson(RATES["bkg"] * bkg_integral.getVal())

    # Create TH1 histograms from counts
    sig_hist = ROOT.TH1D("sig_hist", "sig_hist", NBINS, BIN_EDGES)
    bkg_hist = ROOT.TH1D("bkg_hist", "bkg_hist", NBINS, BIN_EDGES)
    obs_hist = ROOT.TH1D("obs_hist", "obs_hist", NBINS, BIN_EDGES)
    # Use a fixed observed dataset:
    obs_counts = [16, 10,  3,  5,  5,  5,  6,  4,  4, 11,  2,  4,  2,  1,  2,  1,  1,  0,  0,  0]

    for i in range(NBINS):
        sig_hist.SetBinContent(i + 1, float(sig_counts[i]))
        bkg_hist.SetBinContent(i + 1, float(bkg_counts[i]))
        obs_hist.SetBinContent(i + 1, obs_counts[i])

    # Create RooDataHist from histograms
    data_obs_sig = ROOT.RooDataHist("data_obs_sig", "Signal data", ROOT.RooArgList(mass), sig_hist)
    data_obs_bkg = ROOT.RooDataHist("data_obs_bkg", "Background data", ROOT.RooArgList(mass), bkg_hist)
    data_obs = ROOT.RooDataHist("data_obs", "Observed data", ROOT.RooArgList(mass), obs_hist)

    for obj in (data_obs_sig, data_obs_bkg, data_obs):
        getattr(ws, "import")(obj)

    return ws


def write_shapes_file(output_file="simple_shapes_binned.root"):
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
