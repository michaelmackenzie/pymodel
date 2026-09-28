"""Write the inputs of the two-dimensional (p, t0) example, in the style of the 2D path of
mumep_ana's build_model.C (``do_2d_fit_``): every process is a RooHistPdf over (obs_p, obs_t)
built from the product of independent momentum and time histograms, and data_obs is a 2D
RooDataHist over the same two observables.

workspace.root, RooWorkspace "w":

* observables obs_p in [102, 106] (20 bins of 0.2) and obs_t in [600, 1600] with the
  variable binning T_EDGES (9 bins: the bin areas differ, which exercises the bin-area
  factors of the 2D likelihood);
* <proc>_pdf for sig, dio, cosmic: RooHistPdf(obs_p, obs_t) of the TH2D
  rate * p_x(i) * p_t(j) (build_model.C make_independent_2d_hist), with the 1D fractions the
  exact bin integrals of
      sig     p: Gaussian(104.9, 0.4)            t: exp(-t / 864)
      dio     p: exp(-(p - 102) / 0.8)           t: exp(-t / 864)
      cosmic  p: 1 + 0.05 (p - 102)              t: flat
* shape variations <proc>_pdf_<syst>Up/Down (RooHistPdfs, same construction) for
  card_syst.txt: ``pscale`` shifts the momentum of sig and dio by +-0.05, ``tslope`` changes
  the cosmic time slope (flat -> 1 -+ 2e-4 (t - 1100)).  Combine's text2workspace fails on
  this card (see card_syst.txt), so pymodel refuses it;
* sigf_pdf for card_param.txt: RooProdPdf(RooGaussian(obs_p; mean_p, 0.4),
  RooExponential(obs_t; -1/864)) with mean_p = 104.9 + 0.1 * sig_pshift and sig_pshift a
  constant workspace variable that the card's ``param`` line makes floating;
* data_obs: 2D RooDataHist of fixed-seed events (dio + cosmic + the signal at r = 1) drawn
  from the continuous shapes; data_obs_unbinned: the same events as a 2D RooDataSet;
  data_obs_p: their momenta only (1D RooDataHist over obs_p), for the refusal check of a 2D
  model with 1D data (what the old 2D mumep workspaces had).

th2.root: TH2D templates and data of the same model, for the refusal check of card_th2.txt
(Combine's ShapeTools accepts only TH1 histograms in shapes lines).
"""

import math
import os

import numpy as np
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
HERE = os.path.dirname(os.path.abspath(__file__))

P_EDGES = np.linspace(102.0, 106.0, 21)
T_EDGES = np.array([600.0, 650.0, 700.0, 800.0, 900.0, 1000.0, 1150.0, 1300.0, 1450.0, 1600.0])
RATES = {"sig": 5.0, "dio": 20.0, "cosmic": 10.0}
TAU = 864.0


def gauss_cdf(x, mean, sigma):
    return 0.5 * (1.0 + math.erf((x - mean) / (sigma * math.sqrt(2.0))))


def fractions(cdf, edges):
    c = np.array([cdf(e) for e in edges])
    f = np.diff(c)
    return f / f.sum()


def p_fractions(proc, shift=0.0):
    if proc in ("sig", "sigf"):
        return fractions(lambda x: gauss_cdf(x, 104.9 + shift, 0.4), P_EDGES)
    if proc == "dio":
        return fractions(lambda x: -0.8 * math.exp(-(x - shift - 102.0) / 0.8), P_EDGES)
    return fractions(lambda x: (x - 102.0) + 0.025 * (x - 102.0) ** 2, P_EDGES)


def t_fractions(proc, slope=0.0):
    if proc in ("sig", "dio"):
        return fractions(lambda t: -TAU * math.exp(-t / TAU), T_EDGES)
    # 1 + slope (t - 1100): integral t + slope (t - 1100)^2 / 2
    return fractions(lambda t: t + 0.5 * slope * (t - 1100.0) ** 2, T_EDGES)


VARIATIONS = {  # suffix -> (process, p shift, t slope)
    "_pscaleUp": ({"sig", "dio"}, 0.05, 0.0), "_pscaleDown": ({"sig", "dio"}, -0.05, 0.0),
    "_tslopeUp": ({"cosmic"}, 0.0, -2e-4), "_tslopeDown": ({"cosmic"}, 0.0, 2e-4),
}


def th2(name, proc, shift=0.0, slope=0.0):
    """rate * p_x * p_t as a TH2D (build_model.C make_independent_2d_hist)."""
    h = ROOT.TH2D(name, name, len(P_EDGES) - 1, P_EDGES, len(T_EDGES) - 1, T_EDGES)
    px, pt = p_fractions(proc, shift), t_fractions(proc, slope)
    for i in range(len(px)):
        for j in range(len(pt)):
            v = RATES[proc] * px[i] * pt[j]
            h.SetBinContent(i + 1, j + 1, v)
            h.SetBinError(i + 1, j + 1, math.sqrt(v))
    h.SetDirectory(0)
    return h


def sample(proc, n, rng):
    """n (p, t) events of a process from its continuous shapes (independent p and t)."""
    if proc == "sig":
        p = rng.normal(104.9, 0.4, 10 * n + 10)
        p = p[(p > P_EDGES[0]) & (p < P_EDGES[-1])][:n]
    elif proc == "dio":
        u = rng.random(n)
        p = 102.0 - 0.8 * np.log(1.0 - u * (1.0 - math.exp(-4.0 / 0.8)))
    else:  # density 1 + 0.05 (p - 102) on [0, 4]: accept-reject
        q = rng.uniform(0.0, 4.0, 4 * n + 10)
        q = q[rng.random(len(q)) < (1.0 + 0.05 * q) / 1.2][:n]
        p = 102.0 + q
    if proc in ("sig", "dio"):
        u = rng.random(n)
        a, b = math.exp(-T_EDGES[0] / TAU), math.exp(-T_EDGES[-1] / TAU)
        t = -TAU * np.log(a - u * (a - b))
    else:
        t = rng.uniform(T_EDGES[0], T_EDGES[-1], n)
    return np.column_stack([p, t])


def events(rng):
    parts = [sample(proc, int(rng.poisson(RATES[proc])), rng) for proc in ("dio", "cosmic", "sig")]
    return np.concatenate(parts)


def observables():
    p = ROOT.RooRealVar("obs_p", "momentum", 104.0, P_EDGES[0], P_EDGES[-1])
    p.setBins(len(P_EDGES) - 1)
    t = ROOT.RooRealVar("obs_t", "t0", 1100.0, T_EDGES[0], T_EDGES[-1])
    t.setBinning(ROOT.RooBinning(len(T_EDGES) - 1, T_EDGES))
    return p, t


def main():
    rng = np.random.default_rng(20260928)
    xs = events(rng)
    p, t = observables()
    vars_ = ROOT.RooArgList(p, t)
    w = ROOT.RooWorkspace("w", "w")
    imp = getattr(w, "import")
    keep = []
    for proc in RATES:
        for suffix, (procs, shift, slope) in [("", (set(RATES), 0.0, 0.0))] + list(VARIATIONS.items()):
            if proc not in procs:
                continue
            name = f"{proc}_pdf{suffix}"
            h = th2(f"{name}_th2", proc, shift, slope)
            dh = ROOT.RooDataHist(f"{name}_2d_data_hist", "", vars_, h)
            pdf = ROOT.RooHistPdf(name, "", ROOT.RooArgSet(p, t), dh)
            keep += [h, dh, pdf]
            imp(pdf, ROOT.RooFit.RecycleConflictNodes(), ROOT.RooFit.Silence())
    # parametric signal with a floating momentum shift (card_param.txt)
    shift = ROOT.RooRealVar("sig_pshift", "signal momentum shift", 0.0, -5.0, 5.0)
    shift.setConstant(True)
    mean = ROOT.RooFormulaVar("mean_p_sigf", "104.9+0.1*@0", ROOT.RooArgList(shift))
    width = ROOT.RooRealVar("width_p_sigf", "", 0.4)
    slope = ROOT.RooRealVar("slope_t_sigf", "", -1.0 / TAU)
    gp = ROOT.RooGaussian("sigf_p_pdf", "", p, mean, width)
    et = ROOT.RooExponential("sigf_t_pdf", "", t, slope)
    sigf = ROOT.RooProdPdf("sigf_pdf", "", ROOT.RooArgList(gp, et))
    keep += [shift, mean, width, slope, gp, et, sigf]
    imp(sigf, ROOT.RooFit.RecycleConflictNodes(), ROOT.RooFit.Silence())
    # data
    data = ROOT.RooDataHist("data_obs", "data_obs", vars_)
    unb = ROOT.RooDataSet("data_obs_unbinned", "data_obs_unbinned", ROOT.RooArgSet(p, t))
    for pv, tv in xs:
        p.setVal(float(pv))
        t.setVal(float(tv))
        data.add(ROOT.RooArgSet(p, t))
        unb.add(ROOT.RooArgSet(p, t))
    data_p = ROOT.RooDataHist("data_obs_p", "data_obs_p", ROOT.RooArgList(p))
    for pv, _ in xs:
        p.setVal(float(pv))
        data_p.add(ROOT.RooArgSet(p))
    imp(data)
    imp(unb)
    imp(data_p)
    path = os.path.join(HERE, "workspace.root")
    w.writeToFile(path, True)
    print(f"wrote {path} ({data.sumEntries():.0f} events)")
    # TH2 inputs (refused by Combine and pymodel)
    f = ROOT.TFile(os.path.join(HERE, "th2.root"), "RECREATE")
    for proc in RATES:
        th2(proc, proc).Write()
    hd = ROOT.TH2D("data_obs", "data_obs", len(P_EDGES) - 1, P_EDGES, len(T_EDGES) - 1, T_EDGES)
    for pv, tv in xs:
        hd.Fill(pv, tv)
    hd.Write()
    f.Close()
    del keep


if __name__ == "__main__":
    main()
