#!/usr/bin/env python3
"""Generate a ROOT file with a two-channel RooFit workspace for roomodel.

Run from examples/roomodel:
    python simple_shapes_two_channel.py
"""

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
CHANNEL_RATES = {
    "barrel": {"sig": 12.0, "bkg": 80.0},
    "endcap": {"sig": 6.0, "bkg": 200.0},
}
RNG_SEED = 123


def _generate_channel_observations(rng, n_sig, n_bkg):
    sig_obs = rng.normal(105.0, 0.35, n_sig)
    sig_obs = sig_obs[(sig_obs >= OBS_MIN) & (sig_obs <= OBS_MAX)]

    bkg_obs = rng.exponential(4.0, max(1, n_bkg * 3))
    bkg_obs = OBS_MAX - bkg_obs
    bkg_obs = bkg_obs[(bkg_obs >= OBS_MIN) & (bkg_obs <= OBS_MAX)][:n_bkg]

    all_obs = np.concatenate([sig_obs, bkg_obs])
    rng.shuffle(all_obs)
    return all_obs


def build_shapes_workspace():
    ROOT = _get_root()
    ws = ROOT.RooWorkspace("shapes_ws")

    mass = ROOT.RooRealVar("mass", "mass", OBS_MIN, OBS_MAX)

    sig_mu = ROOT.RooRealVar("sig_mu", "sig_mu", 105.0, OBS_MIN, OBS_MAX)
    sig_sigma = ROOT.RooRealVar("sig_sigma", "sig_sigma", 0.35, 0.01, 2.0)
    sig_mu.setConstant(True)
    sig_sigma.setConstant(True)
    sig_pdf = ROOT.RooGaussian("sig", "Signal Gaussian", mass, sig_mu, sig_sigma)

    bkg_lam = ROOT.RooRealVar("bkg_lam", "bkg_lam", -0.25, -3.0, -0.001)
    bkg_lam.setConstant(True)
    bkg_pdf = ROOT.RooExponential("bkg", "Background Exponential", mass, bkg_lam)

    for obj in (mass, sig_mu, sig_sigma, sig_pdf, bkg_lam, bkg_pdf):
        getattr(ws, "import")(obj)

    rng = np.random.default_rng(RNG_SEED)
    ws_mass = ws.var("mass")

    for channel, rates in CHANNEL_RATES.items():
        n_sig = int(rng.poisson(rates["sig"]))
        n_bkg = int(rng.poisson(rates["bkg"]))
        obs = _generate_channel_observations(rng, n_sig, n_bkg)

        data = ROOT.RooDataSet(f"{channel}__data_obs", f"{channel} observed", ROOT.RooArgSet(ws_mass))
        for val in obs:
            ws_mass.setVal(float(val))
            data.add(ROOT.RooArgSet(ws_mass))
        getattr(ws, "import")(data)

    return ws


def write_shapes_file(output_file="simple_shapes_two_channel.root"):
    ROOT = _get_root()
    ws = build_shapes_workspace()
    tf = ROOT.TFile.Open(output_file, "RECREATE")
    if tf is None or tf.IsZombie():
        raise RuntimeError(f"Could not create {output_file}")
    ws.Write()
    tf.Close()
    print(f"Wrote two-channel shapes workspace: {output_file}")


if __name__ == "__main__":
    write_shapes_file()
