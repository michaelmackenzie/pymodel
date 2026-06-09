"""Binned simple-shapes workspace for the hfmodel backend.

The signal is a Gaussian (mu=105, sigma=0.35, yield=12) and the background
is an Exponential (lam=-0.25, yield=80) over the mass range [100, 110] split
into 20 uniform bins.  Per-bin expected counts are computed analytically so
they are identical to those used by the zmodel and roomodel binned examples.

Run from the examples/hfmodel directory:
    python simple_shapes_binned.py
"""
import json
import numpy as np
from scipy import special

# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------
NBINS = 20
OBS_MIN = 100.0
OBS_MAX = 110.0
BIN_EDGES = np.linspace(OBS_MIN, OBS_MAX, NBINS + 1)

# ---------------------------------------------------------------------------
# Physics parameters (shared with zmodel / roomodel examples)
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


# Absolute expected counts per bin at nominal signal strength mu=1
sig_data = np.array([
    _gauss_bin_frac(BIN_EDGES[i], BIN_EDGES[i + 1], SIG_MU, SIG_SIGMA) * SIG_YIELD
    for i in range(NBINS)
])
bkg_data = np.array([
    _exp_bin_frac(BIN_EDGES[i], BIN_EDGES[i + 1], BKG_LAM, OBS_MIN, OBS_MAX) * BKG_YIELD
    for i in range(NBINS)
])

# pyhf requires all template bins to be > 0; clip tiny values
sig_data = np.clip(sig_data, 1e-6, None)
bkg_data = np.clip(bkg_data, 1e-6, None)

workspace_dict = {
    "channels": [
        {
            "name": "demo",
            "samples": [
                {
                    "name": "sig",
                    # Absolute expected counts at mu=1; normfactor "mu" scales this
                    "data": sig_data.tolist(),
                    "modifiers": [
                        {"name": "mu",        "type": "normfactor", "data": None},
                        {"name": "lumi",      "type": "normsys",
                         "data": {"hi": 1.05, "lo": 1.0 / 1.05}},
                    ],
                },
                {
                    "name": "bkg",
                    # Absolute expected counts; no normfactor (bkg is fixed)
                    "data": bkg_data.tolist(),
                    "modifiers": [
                        {"name": "lumi",      "type": "normsys",
                         "data": {"hi": 1.05, "lo": 1.0 / 1.05}},
                        {"name": "bkg_norm",  "type": "normsys",
                         "data": {"hi": 1.10, "lo": 1.0 / 1.10}},
                    ],
                },
            ],
        }
    ],
    "observations": [
        {"name": "demo", "data": OBS_COUNTS}
    ],
    "measurements": [
        {
            "name": "physics_model",
            "config": {
                "poi": "mu",
                "parameters": [],
            },
        }
    ],
    "version": "1.0.0",
}

filename = "simple_shapes_binned_workspace.json"
with open(filename, "w") as f:
    json.dump(workspace_dict, f, indent=2)
print(f"Wrote workspace: {filename}")
