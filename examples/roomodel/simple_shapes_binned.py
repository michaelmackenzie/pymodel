#!/usr/bin/env python3
"""Binned simple-shapes workspace for the roomodel backend.

The signal is a Gaussian (mu=105, sigma=0.35, yield=12) and the background
is an Exponential (lam=-0.25, yield=80) over the mass range [100, 110] split
into 20 uniform bins.  Per-bin expected counts are computed analytically so
they are identical to those used by the zmodel and hfmodel binned examples.

Run from the examples/roomodel directory:
    python simple_shapes_binned.py
"""
import numpy as np
from scipy import special


def _get_root():
    import ROOT
    try:
        if not getattr(ROOT, "_pymodel_roofit_quiet", False):
            ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)
            ROOT._pymodel_roofit_quiet = True
    except Exception:
        pass
    return ROOT


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------
NBINS = 20
OBS_MIN = 100.0
OBS_MAX = 110.0
BIN_EDGES = np.linspace(OBS_MIN, OBS_MAX, NBINS + 1)

# ---------------------------------------------------------------------------
# Physics parameters (shared with zmodel / hfmodel examples)
# ---------------------------------------------------------------------------
SIG_MU    = 105.0
SIG_SIGMA = 0.35
BKG_LAM   = -0.25
SIG_YIELD = 12.0
BKG_YIELD = 80.0

# Fixed observed counts – identical across all three backends
OBS_COUNTS = [10, 9, 8, 7, 6, 5, 5, 4, 5, 8, 8, 3, 2, 2, 2, 2, 1, 1, 1, 1]


def _gauss_bin_frac(lo, hi, mu, sigma):
    return 0.5 * (special.erf((hi - mu) / (sigma * np.sqrt(2)))
                - special.erf((lo - mu) / (sigma * np.sqrt(2))))


def _exp_bin_frac(lo, hi, lam, total_lo, total_hi):
    norm = (np.exp(lam * total_lo) - np.exp(lam * total_hi)) / (-lam)
    return (np.exp(lam * lo) - np.exp(lam * hi)) / (-lam) / norm


# Absolute expected counts per bin
SIG_COUNTS = np.array([
    _gauss_bin_frac(BIN_EDGES[i], BIN_EDGES[i + 1], SIG_MU, SIG_SIGMA) * SIG_YIELD
    for i in range(NBINS)
])
BKG_COUNTS = np.array([
    _exp_bin_frac(BIN_EDGES[i], BIN_EDGES[i + 1], BKG_LAM, OBS_MIN, OBS_MAX) * BKG_YIELD
    for i in range(NBINS)
])

# Clip tiny values so ROOT histograms are well-behaved
SIG_COUNTS = np.clip(SIG_COUNTS, 1e-6, None)
BKG_COUNTS = np.clip(BKG_COUNTS, 1e-6, None)


def build_shapes_workspace():
    ROOT = _get_root()
    ws = ROOT.RooWorkspace("shapes_ws")

    mass = ROOT.RooRealVar("mass", "mass", OBS_MIN, OBS_MAX)
    mass.setBins(NBINS)

    # Signal: Gaussian with fixed parameters
    sig_mu    = ROOT.RooRealVar("sig_mu",    "sig_mu",    SIG_MU,    OBS_MIN, OBS_MAX)
    sig_sigma = ROOT.RooRealVar("sig_sigma", "sig_sigma", SIG_SIGMA, 0.01, 2.0)
    sig_mu.setConstant(True)
    sig_sigma.setConstant(True)
    sig_pdf = ROOT.RooGaussian("sig", "Signal Gaussian", mass, sig_mu, sig_sigma)

    # Background: Exponential with fixed parameter.
    # RooExponential evaluates exp(lam * x).  With x ~ 105 and lam = -0.25
    # this gives exp(-26) ≈ 0 (numerical underflow).  Scale lam so that the
    # product lam * OBS_MIN starts at 0: use lam_eff = BKG_LAM / OBS_MIN so
    # the normalization matches the analytic bin fracs computed with BKG_LAM.
    # Equivalently, set lam to a value that produces the same relative shape
    # over [0, OBS_MAX - OBS_MIN].  The simplest approach: use a GenericPdf
    # with the explicitly shifted formula.
    bkg_lam = ROOT.RooRealVar("bkg_lam", "bkg_lam", BKG_LAM, -3.0, -0.001)
    bkg_lam.setConstant(True)
    # Use GenericPdf with formula exp(lam*(mass - OBS_MIN))
    bkg_pdf = ROOT.RooGenericPdf(
        "bkg", "exp(@1*(@0 - %.6f))" % OBS_MIN,
        ROOT.RooArgList(mass, bkg_lam)
    )

    for obj in (mass, sig_mu, sig_sigma, sig_pdf, bkg_lam, bkg_pdf):
        getattr(ws, "import")(obj)

    # Build TH1 histograms from the analytically computed counts
    sig_hist = ROOT.TH1D("sig_hist", "sig_hist", NBINS, BIN_EDGES)
    bkg_hist = ROOT.TH1D("bkg_hist", "bkg_hist", NBINS, BIN_EDGES)
    obs_hist = ROOT.TH1D("obs_hist", "obs_hist", NBINS, BIN_EDGES)

    for i in range(NBINS):
        sig_hist.SetBinContent(i + 1, float(SIG_COUNTS[i]))
        bkg_hist.SetBinContent(i + 1, float(BKG_COUNTS[i]))
        obs_hist.SetBinContent(i + 1, float(OBS_COUNTS[i]))

    # Import RooDataHist objects into the workspace
    data_obs_sig = ROOT.RooDataHist("data_obs_sig", "Signal hist",     ROOT.RooArgList(mass), sig_hist)
    data_obs_bkg = ROOT.RooDataHist("data_obs_bkg", "Background hist", ROOT.RooArgList(mass), bkg_hist)
    data_obs     = ROOT.RooDataHist("data_obs",     "Observed data",   ROOT.RooArgList(mass), obs_hist)

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
