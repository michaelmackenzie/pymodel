import json
import numpy as np

nbins = 20
bin_edges = np.linspace(100.0, 110.0, nbins + 1)
bin_centers = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2.0

sig_mu = 105.0
sig_sigma = 0.35
sig_yield = 12.0
bkg_lambda = -0.25
bkg_yield = 80.0
RNG_SEED = 12345

# Compute signal and background shapes (normalized PDFs)
sig_shape = np.exp(-((bin_centers - sig_mu) / sig_sigma) ** 2 / 2)
sig_shape /= sig_shape.sum()

bkg_shape = np.exp(bkg_lambda * (bin_centers - 100.0))
bkg_shape /= bkg_shape.sum()

# Use a fixed observed dataset:
obs_counts = [16, 10,  3,  5,  5,  5,  6,  4,  4, 11,  2,  4,  2,  1,  2,  1,  1,  0,  0,  0]

workspace_dict = {
    "channels": [
        {
            "name": "demo",
            "samples": [
                {
                    "name": "sig",
                    "data": sig_shape.tolist(),
                    "modifiers": [
                        {"name": "mu", "type": "normfactor", "data": None}
                    ],
                },
                {
                    "name": "bkg",
                    "data": bkg_shape.tolist(),
                    "modifiers": [],
                },
            ],
        }
    ],
    "observations": [
        {"name": "demo", "data": obs_counts}
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
    json.dump(workspace_dict, f, indent=4)
