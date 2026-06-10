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

def build_shapes_workspace():
    ROOT = _get_root()
    ws = ROOT.RooWorkspace("shapes_ws")

    mass = ROOT.RooRealVar("mass", "mass", OBS_MIN, OBS_MAX)
    mass.setBins(NBINS)


    # Histogrammed PDFs
    sig_data = np.array([
        _gauss_bin_frac(BIN_EDGES[i], BIN_EDGES[i + 1], SIG_MU, SIG_SIGMA)
        for i in range(NBINS)
    ])
    bkg_data = np.array([
        _exp_bin_frac(BIN_EDGES[i], BIN_EDGES[i + 1], BKG_LAM, OBS_MIN, OBS_MAX)
        for i in range(NBINS)
    ])

    # Create TH1s
    bkg_hist = ROOT.TH1D("bkg_hist", "bkg_hist", NBINS, BIN_EDGES)
    sig_hist = ROOT.TH1D("sig_hist", "sig_hist", NBINS, BIN_EDGES)
    obs_hist = ROOT.TH1D("obs_hist", "obs_hist", NBINS, BIN_EDGES)

    dx = (OBS_MAX - OBS_MIN) / NBINS
    for i in range(NBINS):
        bkg_hist.SetBinContent(i + 1, bkg_data[i]) # / (BIN_EDGES[i+1] - BIN_EDGES[i]))
        sig_hist.SetBinContent(i + 1, sig_data[i]) # / (BIN_EDGES[i+1] - BIN_EDGES[i]))    
        obs_hist.SetBinContent(i + 1, float(OBS_COUNTS[i]))

    # Create PDFs
    data_sig = ROOT.RooDataHist("data_sig", "Signal Data", ROOT.RooArgList(mass), sig_hist)
    sig_pdf = ROOT.RooHistPdf("sig", "Signal PDF", mass, data_sig)

    data_bkg = ROOT.RooDataHist("data_bkg", "Background Data", ROOT.RooArgList(mass), bkg_hist)
    bkg_pdf = ROOT.RooHistPdf("bkg", "Background PDF", mass, data_bkg)

    # Observed data
    data_obs = ROOT.RooDataHist("data_obs", "Observed data", ROOT.RooArgList(mass), obs_hist)

    # Import to the workspace
    for obj in (mass, sig_pdf, bkg_pdf, data_obs):
        getattr(ws, "import")(obj)

    # Do a test fit
    yield_sig = ROOT.RooRealVar("yield_sig", "Signal yield", 12.)
    yield_bkg = ROOT.RooRealVar("yield_bkg", "Background yield", 80.)
    yield_sig.setConstant(True)
    yield_bkg.setConstant(True)
    r = ROOT.RooRealVar("r", "Signal modifier", 1., -5., 5.)
    yield_sig_eff = ROOT.RooFormulaVar("yield_sig_eff", "@0*@1", ROOT.RooArgList(r, yield_sig))
    sig_pdf.setInterpolationOrder(0)
    bkg_pdf.setInterpolationOrder(0)
    # pdf = ROOT.RooRealSumPdf("pdf", "Total PDF",
    #                          ROOT.RooArgList(sig_pdf, bkg_pdf),
    #                          ROOT.RooArgList(yield_sig_eff, yield_bkg),
    #                          False)
    pdf = ROOT.RooAddPdf("pdf", "Total PDF",
                         ROOT.RooArgList(sig_pdf, bkg_pdf),
                         ROOT.RooArgList(yield_sig_eff, yield_bkg))
    pdf.fitTo(data_obs,
              ROOT.RooFit.Extended(True),
              ROOT.RooFit.Binned(True),                      # Forces binned evaluation
              ROOT.RooFit.DataError(ROOT.RooAbsData.Poisson) # Forces Poisson weights
              )

    ROOT.gROOT.SetBatch(True)
    frame = mass.frame()
    data_obs.plotOn(frame)
    pdf.plotOn(frame)
    pdf.plotOn(frame, ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.Components("sig"))
    pdf.plotOn(frame, ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.Components("bkg"))
    c = ROOT.TCanvas()
    frame.Draw()
    c.SaveAs("simple_shapes_binned.png")

    sig_data = [ sig_hist.GetBinContent(i+1) for i in range(NBINS) ]
    bkg_data = [ bkg_hist.GetBinContent(i+1) for i in range(NBINS) ]
    ndata = sum(OBS_COUNTS)
    nfit = yield_bkg.getVal() + r.getVal()*yield_sig.getVal()
    print(r)
    print(f'N(data) = {ndata}, N(fit) = {nfit}')
    
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
