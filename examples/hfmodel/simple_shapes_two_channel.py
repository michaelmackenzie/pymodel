import json
from pathlib import Path
import numpy as np
import pyhf


pyhf.set_backend("numpy")


nbins = 20
bin_edges = np.linspace(0, 100, nbins + 1)
bin_centers = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2.0

# Category 1: SR_high_pt
bkg_shape_1 = 1000.0 * np.exp(-bin_centers / 25.0)
sig_shape_1 = 80.0 * np.exp(-((bin_centers - 50.0) / 6.0) ** 2)
obs_shape_1 = np.random.default_rng(42).poisson(bkg_shape_1 + 0.8 * sig_shape_1).astype(float)

# Category 2: SR_low_pt
bkg_shape_2 = 2500.0 * np.exp(-bin_centers / 35.0)
sig_shape_2 = 120.0 * np.exp(-((bin_centers - 55.0) / 10.0) ** 2)
obs_shape_2 = np.random.default_rng(43).poisson(bkg_shape_2 + 0.5 * sig_shape_2).astype(float)

# Shape systematic variations for signal
sig_shape_1_hi = (sig_shape_1 * 1.15).tolist()
sig_shape_1_lo = (sig_shape_1 * 0.85).tolist()
sig_shape_2_hi = (sig_shape_2 * 1.10).tolist()
sig_shape_2_lo = (sig_shape_2 * 0.90).tolist()

# Per-bin statistical uncertainty for background
bkg_uncorr_err_1 = (bkg_shape_1 * 0.05).tolist()
bkg_uncorr_err_2 = (bkg_shape_2 * 0.05).tolist()


workspace_dict = {
    "channels": [
        {
            "name": "SR_high_pt",
            "samples": [
                {
                    "name": "signal",
                    "data": sig_shape_1.tolist(),
                    "modifiers": [
                        {"name": "mu", "type": "normfactor", "data": None},
                        {
                            "name": "signal_shape_syst",
                            "type": "histosys",
                            "data": {"hi_data": sig_shape_1_hi, "lo_data": sig_shape_1_lo},
                        }
                    ],
                },
                {
                    "name": "background",
                    "data": bkg_shape_1.tolist(),
                    "modifiers": [
                    ],
                },
            ],
        },
        {
            "name": "SR_low_pt",
            "samples": [
                {
                    "name": "signal",
                    "data": sig_shape_2.tolist(),
                    "modifiers": [
                        {"name": "mu", "type": "normfactor", "data": None},
                        {
                            "name": "signal_shape_syst",
                            "type": "histosys",
                            "data": {"hi_data": sig_shape_2_hi, "lo_data": sig_shape_2_lo},
                        }
                    ],
                },
                {
                    "name": "background",
                    "data": bkg_shape_2.tolist(),
                    "modifiers": [
                    ],
                },
            ],
        },
    ],
    "observations": [
        {"name": "SR_high_pt", "data": obs_shape_1.tolist()},
        {"name": "SR_low_pt", "data": obs_shape_2.tolist()},
    ],
    "measurements": [
        {"name": "physics_search", "config": {"poi": "mu", "parameters": []}}
    ],
    "version": "1.0.0",
}


if __name__ == "__main__":
    workspace = pyhf.Workspace(workspace_dict)
    model = workspace.model()
    observations = workspace.data(model)

    print(f"  channels: {model.config.channels}")
    print(f"     nbins: {model.config.channel_nbins}")
    print(f"   samples: {model.config.samples}")
    print(f" modifiers: {model.config.modifiers}")
    print(f"parameters: {model.config.parameters}")
    print(f"  nauxdata: {model.config.nauxdata}")
    print(f"data size : {len(observations)}")

    out_path = Path(__file__).resolve().parent / "simple_shapes_two_channel_workspace.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(workspace_dict, handle, indent=2)
    print(f"Successfully serialized and saved workspace to: {out_path}")
