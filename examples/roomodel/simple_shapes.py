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
RATES = {"sig": 12.0, "bkg": 800.0}
RNG_SEED = 42


def build_shapes_workspace():
    ROOT = _get_root()
    ws = ROOT.RooWorkspace("shapes_ws")

    mass = ROOT.RooRealVar("mass", "mass", OBS_MIN, OBS_MAX)

    # Signal: narrow Gaussian at 105
    sig_mu = ROOT.RooRealVar("sig_mu", "sig_mu", 105.0, OBS_MIN, OBS_MAX)
    sig_sigma = ROOT.RooRealVar("sig_sigma", "sig_sigma", 0.35, 0.01, 2.0)
    sig_mu_offset = ROOT.RooRealVar("sig_mu_offset", "sig_mu_offset", 0., -7., 7.);
    sig_mu.setConstant(True)
    sig_sigma.setConstant(True)
    sig_mu_eff = ROOT.RooFormulaVar("sig_mu_eff", "@0 + 0.1*@1", ROOT.RooArgList(sig_mu, sig_mu_offset));
    sig_pdf = ROOT.RooGaussian("sig", "Signal Gaussian", mass, sig_mu_eff, sig_sigma)

    # Background: falling exponential
    bkg_lam = ROOT.RooRealVar("bkg_lam", "bkg_lam", -0.25, -3.0, -0.001)
    bkg_lam.setConstant(True)
    bkg_pdf = ROOT.RooExponential("bkg", "Background Exponential", mass, bkg_lam)

    # Create a total model to generate toy data with
    sig_yield = ROOT.RooRealVar("sig_yield", "Signal yield", RATES["sig"])
    bkg_yield = ROOT.RooRealVar("bkg_yield", "Background yield", RATES["bkg"])
    r         = ROOT.RooRealVar("r", "r", 1., -100., 100.)
    sig_yield_eff = ROOT.RooFormulaVar("sig_yield_eff", "@0*@1", ROOT.RooArgList(r, sig_yield))
    total_pdf = ROOT.RooAddPdf("total_pdf", "Total PDF",
                               ROOT.RooArgList(sig_pdf, bkg_pdf),
                               ROOT.RooArgList(sig_yield_eff, bkg_yield))
    rng = np.random.default_rng(RNG_SEED)
    n_data = int(rng.poisson(RATES["sig"] + RATES["bkg"]))
    data_obs = total_pdf.generate(ROOT.RooArgSet(mass), n_data)
    data_obs.SetName("data_obs")
    
    # Import the model to the workspace
    for obj in (mass, sig_mu, sig_sigma, sig_pdf, bkg_lam, bkg_pdf, data_obs):
        getattr(ws, "import")(obj)

    # Plot the model
    sig_mu_offset.setConstant(True)
    total_pdf.fitTo(data_obs)
    frame = mass.frame()
    data_obs.plotOn(frame)
    total_pdf.plotOn(frame)
    total_pdf.plotOn(frame, ROOT.RooFit.Components("sig_pdf"))
    total_pdf.plotOn(frame, ROOT.RooFit.Components("bkg_pdf"))
    ROOT.gROOT.SetBatch(True)
    c = ROOT.TCanvas()
    frame.Draw()
    c.SaveAs("simple_shapes.png")

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
