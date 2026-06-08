import sys
import pickle
import dill
import numpy as np
import zfit

nbins = 20
obs = zfit.Space("mass", limits=(100.0, 110.0), binning=nbins)

sig_mu = zfit.Parameter("sig_mu", 105.0, 103.0, 107.0)
sig_sigma = zfit.Parameter("sig_sigma", 0.35, 0.02, 2.0)
sig_pdf = zfit.pdf.Gauss(obs=obs, mu=sig_mu, sigma=sig_sigma, name="sig_pdf")

bkg_lambda = zfit.Parameter("bkg_lambda", -0.25, -3.0, -0.001)
bkg_pdf = zfit.pdf.Exponential(obs=obs, lam=bkg_lambda, name="bkg_pdf")

RATES = {
    "sig": 12.0,
    "bkg": 80.0,
}
RNG_SEED = 12345


def make_data_obs():
    # Use a fixed observed dataset:
    obs_counts = [16, 10,  3,  5,  5,  5,  6,  4,  4, 11,  2,  4,  2,  1,  2,  1,  1,  0,  0,  0]
    return obs_counts


def make_shape_payload():
    return {
        "shapes": {
            "sig": sig_pdf,
            "bkg": bkg_pdf,
        },
        "rates": dict(RATES),
        "data_obs": {
            "values": make_data_obs(),
        },
    }


def write_shape_payload(output_file="simple_shapes_binned.pkl"):
    payload = make_shape_payload()
    with open(output_file, "wb") as handle:
        dill.dump(payload, handle)
    sys.stdout.write(f"Wrote shape payload: {output_file}\n")
    sys.stdout.flush()


if __name__ == "__main__":
    write_shape_payload()
