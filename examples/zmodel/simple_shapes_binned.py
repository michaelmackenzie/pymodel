"""Binned simple-shapes payload for the zmodel backend.

The signal is a Gaussian (mu=105, sigma=0.35, yield=12) and the background
is an Exponential (lam=-0.25, yield=80) over the mass range [100, 110] split
into 20 uniform bins.  The normalised per-bin fractions are computed
analytically (no random sampling) so they are identical to those used by the
hfmodel and roomodel binned examples.

Run from the examples/zmodel directory:
    python simple_shapes_binned.py
"""
import dill
import numpy as np
from scipy import special
import zfit

# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------
NBINS = 20
OBS_MIN = 100.0
OBS_MAX = 110.0
BIN_EDGES = np.linspace(OBS_MIN, OBS_MAX, NBINS + 1)

# ---------------------------------------------------------------------------
# Physics parameters (shared with hfmodel / roomodel examples)
# ---------------------------------------------------------------------------
SIG_MU    = 105.0
SIG_SIGMA = 0.35
BKG_LAM   = -0.25
RATES     = {"sig": 12.0, "bkg": 80.0}

# Fixed observed counts
OBS_COUNTS = [10, 9, 8, 7, 6, 5, 5, 4, 5, 8, 8, 3, 2, 2, 2, 2, 1, 1, 1, 1]

# ---------------------------------------------------------------------------
# Analytic per-bin fractions (normalised PDFs)
# ---------------------------------------------------------------------------
def _gauss_bin_frac(lo, hi, mu, sigma):
    return 0.5 * (special.erf((hi - mu) / (sigma * np.sqrt(2)))
                - special.erf((lo - mu) / (sigma * np.sqrt(2))))


def _exp_bin_frac(lo, hi, lam, total_lo, total_hi):
    norm = (np.exp(lam * total_lo) - np.exp(lam * total_hi)) / (-lam)
    return (np.exp(lam * lo) - np.exp(lam * hi)) / (-lam) / norm


SIG_FRACS = np.array([
    _gauss_bin_frac(BIN_EDGES[i], BIN_EDGES[i + 1], SIG_MU, SIG_SIGMA)
    for i in range(NBINS)
])
BKG_FRACS = np.array([
    _exp_bin_frac(BIN_EDGES[i], BIN_EDGES[i + 1], BKG_LAM, OBS_MIN, OBS_MAX)
    for i in range(NBINS)
])

# ---------------------------------------------------------------------------
# zfit PDFs (continuous; zmodel can use with binned data via --fit-mode binned)
# ---------------------------------------------------------------------------
obs = zfit.Space("mass", limits=(OBS_MIN, OBS_MAX))

sig_mu_param    = zfit.Parameter("sig_mu",    SIG_MU,    floating=False)
sig_sigma_param = zfit.Parameter("sig_sigma", SIG_SIGMA, floating=False)
sig_pdf = zfit.pdf.Gauss(obs=obs, mu=sig_mu_param, sigma=sig_sigma_param, name="sig_pdf")

bkg_lambda_param = zfit.Parameter("bkg_lambda", BKG_LAM, floating=False)
bkg_pdf = zfit.pdf.Exponential(obs=obs, lam=bkg_lambda_param, name="bkg_pdf")

# ---------------------------------------------------------------------------
# Payload
# ---------------------------------------------------------------------------

def make_shape_payload():
    return {
        "shapes": {
            "sig": sig_pdf,
            "bkg": bkg_pdf,
        },
        "rates": dict(RATES),
        # data_obs stores per-bin counts together with the bin edges so that
        # zmodel can reconstruct pseudo-events at bin centers for the unbinned
        # likelihood, or use counts directly for a binned likelihood.
        "data_obs": {
            "values": list(OBS_COUNTS),
            "bin_edges": BIN_EDGES.tolist(),
        },
    }


def write_shape_payload(output_file="simple_shapes_binned.pkl"):
    payload = make_shape_payload()
    with open(output_file, "wb") as handle:
        dill.dump(payload, handle)
    print(f"Wrote shape payload: {output_file}")


if __name__ == "__main__":
    write_shape_payload()

    # Perform a test fit
    binning = zfit.binned.RegularBinning(
        len(OBS_COUNTS),
        OBS_MIN,
        OBS_MAX,
        name="mass"
    )
    obs_binned = zfit.Space("mass", binning=binning)
    data_obs = zfit.data.BinnedData.from_tensor(
        space=obs_binned,
        values=OBS_COUNTS
    )

    # Define the total floating signal yield directly as the parameter target
    yield_sig_fit = zfit.Parameter("yield_sig_fit", RATES["sig"], lower=0.0, upper=60.0, step_size=0.1)
    yield_bkg_fixed = zfit.Parameter("yield_bkg", RATES["bkg"], floating=False)

    # Convert the exact, analytical numpy bin fractions into zfit Binned Data elements
    sig_hist_data = zfit.data.BinnedData.from_tensor(space=obs_binned, values=SIG_FRACS * RATES["sig"])
    bkg_hist_data = zfit.data.BinnedData.from_tensor(space=obs_binned, values=BKG_FRACS * RATES["bkg"])

    # Instantiating a HistogramPDF
    sig_extended = zfit.pdf.HistogramPDF(
        data=sig_hist_data,
        extended=yield_sig_fit
    )
    bkg_extended = zfit.pdf.HistogramPDF(
        data=bkg_hist_data,
        extended=yield_bkg_fixed
    )

    # Assemble the combination model
    model = zfit.pdf.BinnedSumPDF(
        pdfs=[sig_extended, bkg_extended],
        obs=obs_binned
    )

    # Execute pure vector Poisson optimization
    loss = zfit.loss.ExtendedBinnedNLL(model=model, data=data_obs)
    minimizer = zfit.minimize.Minuit()

    # Minuit discovers yield_sig_fit automatically
    result = minimizer.minimize(loss)

    print(result)

    # Reconstruct the rate multiplier
    best_fit_r = yield_sig_fit.value() / RATES["sig"]
    print(f"\nBest fit signal modifier r: {best_fit_r:.4f}")
