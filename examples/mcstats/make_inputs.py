"""Write mcstats.root: low-statistics weighted-MC TH1 templates for the autoMCStats example.

Every template is built from a list of (MC event count, weight) per bin, so contents are
sum(w) and the TH1 errors are sqrt(sum(w^2)) (TH1::Sumw2).  The bins are chosen to exercise
every branch of Combine's CMSHistErrorPropagator::setupBinPars (see README.md):

ch1 (8 bins on [0, 8], ``ch1 autoMCStats 10``: threshold 10, signal excluded from n_eff)
  bins 0-3  n_eff(total) > 10          -> one Gaussian parameter per bin (prop_binch1_bin<i>)
  bin 4     n_eff = 9                  -> per process: bkg1 (12 MC events) Gaussian, bkg2 (1 event)
                                          Poisson, sig Gaussian (many small-weight events)
  bin 5     n_eff = 5                  -> bkg1 (6 events) and bkg2 (1 event) Poisson
  bin 6     n_eff = 4                  -> bkg1 (12 events) Gaussian; bkg2 has one +2.0 and one -1.7
                                          weight (content 0.3 < error 2.6): "Poisson not viable",
                                          Gaussian; sig (4 events) Poisson
  bin 7     only bkg1 (4 events)       -> bkg1 Poisson; bkg2 and sig are empty (content 0,
                                          error 0) and get no parameter
ch2 (5 bins on [0, 5], ``ch2 autoMCStats 5 1``: threshold 5, signal included in n_eff)
  a background shape systematic (hist-mode 1 morphing of CMSHistFunc) and signal-dominated
  low-statistics bins.

data_obs is a Poisson fluctuation of the nominal expectation with a fixed seed.
"""

import os

import numpy as np
import ROOT

ROOT.gROOT.SetBatch(True)
HERE = os.path.dirname(os.path.abspath(__file__))

# per bin: list of (number of MC events, weight)
CH1 = {
    "sig": [[(0, 0.05)], [(4, 0.05)], [(20, 0.05)], [(60, 0.05)], [(60, 0.05)], [(20, 0.05)], [(4, 0.05)],
            [(0, 0.05)]],
    "bkg1": [[(60, 0.5)], [(45, 0.5)], [(30, 0.5)], [(20, 0.5)], [(12, 0.5)], [(6, 0.5)], [(12, 0.5)], [(4, 0.5)]],
    "bkg2": [[(5, 2.0)], [(4, 2.0)], [(3, 2.0)], [(2, 2.0)], [(1, 2.0)], [(1, 2.0)], [(1, 2.0), (1, -1.7)], []],
}
CH2 = {
    "sig": [[(2, 0.3)], [(10, 0.3)], [(25, 0.3)], [(10, 0.3)], [(2, 0.3)]],
    "bkg": [[(40, 1.0)], [(25, 1.0)], [(12, 1.0)], [(4, 1.0)], [(2, 1.0)]],
}
# bkg shape systematic in ch2: multiplicative tilt of the contents (errors scaled alike)
TILT_UP = np.array([1.10, 1.05, 1.00, 0.92, 0.85])
TILT_DOWN = np.array([0.93, 0.97, 1.00, 1.06, 1.12])


def th1(name, content, sumw2, lo, hi):
    h = ROOT.TH1D(name, name, len(content), lo, hi)
    h.Sumw2()
    for i, (c, s) in enumerate(zip(content, sumw2)):
        h.SetBinContent(i + 1, float(c))
        h.SetBinError(i + 1, float(np.sqrt(s)))
    return h


def from_events(bins):
    content = np.array([sum(n * w for n, w in b) for b in bins])
    sumw2 = np.array([sum(n * w * w for n, w in b) for b in bins])
    return content, sumw2


def main():
    rng = np.random.default_rng(20260927)
    out = ROOT.TFile(os.path.join(HERE, "mcstats.root"), "RECREATE")
    for ch, procs in (("ch1", CH1), ("ch2", CH2)):
        d = out.mkdir(ch)
        d.cd()
        nb = len(next(iter(procs.values())))
        total = np.zeros(nb)
        hists = []
        for p, bins in procs.items():
            c, s = from_events(bins)
            total += c
            hists.append(th1(p, c, s, 0.0, float(nb)))
            if ch == "ch2" and p == "bkg":
                hists.append(th1("bkg_bkg_shapeUp", c * TILT_UP, s * TILT_UP ** 2, 0.0, float(nb)))
                hists.append(th1("bkg_bkg_shapeDown", c * TILT_DOWN, s * TILT_DOWN ** 2, 0.0, float(nb)))
        data = rng.poisson(total).astype(float)
        hists.append(th1("data_obs", data, data, 0.0, float(nb)))
        for h in hists:
            h.Write()
    out.Close()
    print(f"wrote {os.path.join(HERE, 'mcstats.root')}")


if __name__ == "__main__":
    main()
