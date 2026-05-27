import json
import numpy as np

nbins = 20
bin_edges = np.linspace(0, 100, nbins + 1)
bin_centers = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2.0

bkg_shape = np.exp(-bin_centers / 25.0)
sig_shape = np.exp(-((bin_centers - 50.0) / 6.0) ** 2)
bkg_shape /= sum(bkg_shape)
sig_shape /= sum(sig_shape)
obs_shape = np.random.default_rng(42).poisson(100*(bkg_shape + 0.1 * sig_shape)).astype(float)


workspace_dict = {
    "channels": [
        {
            "name": "demo",
            "samples": [
                {
                    "name": "signal",
                    "data": sig_shape.tolist()
                },
                {
                    "name": "background",
                    "data": bkg_shape.tolist()
                }
            ]
        }
    ],
    "observations": [
        {"name": "demo", "data": obs_shape.tolist()}
    ],
    "measurements": [
        {"name": "physics_search", "config": {"poi": "mu", "parameters": []}}
    ],
    "version": "1.0.0"
}

filename = "simple_shapes_workspace.json"
with open(filename, "w") as f:
    json.dump(workspace_dict, f, indent=4)
