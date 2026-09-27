"""Write templates.root: TH1 templates for two channels (ch1, ch2) with a background shape
systematic (bkg_shape, both channels, changes the normalisation) and a signal width
systematic (sig_width, ch1 only, normalisation unchanged so Combine drops its norm part).

Layout: <channel>/<process>, <channel>/<process>_<syst>Up|Down, <channel>/data_obs.
data_obs is a Poisson fluctuation of bkg + 1 x sig with a fixed seed.
"""

import os

import numpy as np
import ROOT

ROOT.gROOT.SetBatch(True)
HERE = os.path.dirname(os.path.abspath(__file__))
NBINS, LO, HI = 10, 0.0, 10.0
EDGES = np.linspace(LO, HI, NBINS + 1)
CENTRES = 0.5 * (EDGES[:-1] + EDGES[1:])


def gauss(mean, sigma, total):
    from scipy.stats import norm

    w = norm.cdf(EDGES[1:], mean, sigma) - norm.cdf(EDGES[:-1], mean, sigma)
    return total * w / w.sum()


def expo(slope, total):
    w = np.exp(slope * CENTRES)
    return total * w / w.sum()


def th1(name, values):
    h = ROOT.TH1D(name, name, NBINS, LO, HI)
    for i, v in enumerate(values):
        h.SetBinContent(i + 1, float(v))
    return h


def main():
    rng = np.random.default_rng(20260927)
    out = ROOT.TFile(os.path.join(HERE, "templates.root"), "RECREATE")
    config = {"ch1": dict(s=10.0, b=100.0, slope=-0.25), "ch2": dict(s=6.0, b=200.0, slope=-0.15)}
    for ch, c in config.items():
        d = out.mkdir(ch)
        d.cd()
        sig = gauss(5.0, 1.0, c["s"])
        bkg = expo(c["slope"], c["b"])
        hists = [th1("sig", sig), th1("bkg", bkg)]
        # background tilt: Up is flatter and 5% larger, Down steeper and 3% smaller
        hists.append(th1("bkg_bkg_shapeUp", expo(c["slope"] * 0.8, c["b"] * 1.05)))
        hists.append(th1("bkg_bkg_shapeDown", expo(c["slope"] * 1.2, c["b"] * 0.97)))
        if ch == "ch1":
            hists.append(th1("sig_sig_widthUp", gauss(5.0, 1.2, c["s"])))
            hists.append(th1("sig_sig_widthDown", gauss(5.0, 0.8, c["s"])))
        hists.append(th1("data_obs", rng.poisson(sig + bkg)))
        for h in hists:
            h.Write()
    out.Close()
    print(f"wrote {os.path.join(HERE, 'templates.root')}")


if __name__ == "__main__":
    main()
