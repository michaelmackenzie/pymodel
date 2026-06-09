import __main__
import json
import os
from typing import Any, Dict, Optional
import pickle
import dill

import zfit

from zmodel.utilities import FitModel


# ===========================================================================


def _inject_hs3_helpers():
    """Inject helpers into __main__ for zfit HS3 JSON serialization."""
    __main__.zfit = zfit
    __main__.znp = zfit.z.numpy
    __main__.zk = zfit.z


def save_fit_model_bundle(fit_model: FitModel, output_file: str, card=None, card_dir: Optional[str] = None):
    # card and card_dir are accepted for API compatibility but are ignored
    # since the HS3 model is already self-contained
    hs3_payload = zfit.hs3.dumps(fit_model.model)
    try:
        json.dumps(hs3_payload)
        hs3_json_payload = hs3_payload
    except TypeError:
        hs3_json_payload = None

    # Extract bin edges from observed values if available
    binned_edges_by_channel = {}
    if fit_model.observed_values_by_channel:
        import numpy as np
        for channel, values in fit_model.observed_values_by_channel.items():
            if values is not None and len(values) > 0:
                # Infer bin edges from unique values (bin centers)
                unique_vals = np.unique(values)
                if len(unique_vals) > 1:
                    # Estimate bin width from unique values
                    diffs = np.diff(np.sort(unique_vals))
                    bin_width = np.median(diffs) if len(diffs) > 0 else (max(values) - min(values)) / max(1, len(unique_vals) - 1)
                    low, high = float(np.min(values) - bin_width/2), float(np.max(values) + bin_width/2)
                    n_bins = int(round((high - low) / bin_width))
                    edges = np.linspace(low, high, n_bins + 1).tolist()
                    binned_edges_by_channel[channel] = edges

    bundle = {
        "format": "fit_model_bundle_v1",
        "fit_metadata": {
            "process_names": list(fit_model.process_names),
            "signal_process": fit_model.signal_process,
            "signal_nominal_yield": fit_model.signal_nominal_yield,
            "channels": list(fit_model.channels),
            "channel_obs": fit_model.channel_obs,
            "channel_obs_ranges": fit_model.channel_obs_ranges,
            "binned_edges_by_channel": binned_edges_by_channel,
            "has_extended_model": fit_model.model.is_extended if hasattr(fit_model.model, 'is_extended') else False,
        },
        "hs3_model": hs3_json_payload,
    }
    # Note: zmodel's hs3_model is already self-contained with all PDFs and parameters.
    # No "card" block is saved anymore since the model doesn't need external references.
    
    # Preserve observed data if available (for binned/unbinned analyses with observed datasets)
    if fit_model.data is not None:
        bundle["observed_data"] = fit_model.data

    ext = os.path.splitext(output_file)[1].lower()
    if ext == ".json":
        with open(output_file, "w", encoding="utf-8") as handle:
            json.dump(bundle, handle, indent=2)
        return

    with open(output_file, "wb") as handle:
        dill.dump(bundle, handle)


def _choose_top_model(distributions: Dict[str, Any]):
    if not distributions:
        raise ValueError("No distributions found in loaded HS3 payload")
    if len(distributions) == 1:
        return next(iter(distributions.values()))

    for name, model in distributions.items():
        if name.startswith("model_"):
            return model

    return next(iter(distributions.values()))


def _fit_model_from_hs3_payload(hs3_payload: Dict[str, Any], fit_metadata: Optional[Dict[str, Any]] = None, observed_data: Any = None):
    _inject_hs3_helpers()
    loaded = zfit.hs3.loads(hs3_payload)
    distributions = loaded.get("distributions", {})
    constraints = list(loaded.get("constraints", {}).values())
    model = _choose_top_model(distributions)

    signal_process = None
    if fit_metadata is not None:
        signal_process = fit_metadata.get("signal_process")

    if signal_process is None:
        for param in model.get_params():
            if is_signal_strength_poi(param.name):
                # Strip "mu_" prefix to recover the process name, or leave None
                # for the bare "mu" case (single-process default).
                signal_process = param.name[3:] if param.name.startswith("mu_") else None
                break

    fit_model_obj = FitModel(
        obs=model.space,
        obs_range=tuple(float(x) for x in model.space.limit1d),
        shapes={},
        yields={},
        extended_pdfs={},
        model=model,
        data=observed_data,
        process_names=list((fit_metadata or {}).get("process_names", [])),
        signal_process=signal_process,
        constraints=constraints,
        loss=None,
        result=None,
        signal_nominal_yield=(fit_metadata or {}).get("signal_nominal_yield"),
        channels=list((fit_metadata or {}).get("channels", [])),
        channel_obs=(fit_metadata or {}).get("channel_obs", {}),
        channel_obs_ranges=(fit_metadata or {}).get("channel_obs_ranges", {}),
    )
    # Store bin edges for reconstruction of binned spaces
    fit_model_obj._binned_edges_by_channel = (fit_metadata or {}).get("binned_edges_by_channel", {})
    return fit_model_obj


def load_fit_model(model_file: str) -> FitModel:
    payload = None
    with open(model_file, "rb") as handle:
        try:
            # Try dill first (supports complex objects like zfit.Data)
            payload = dill.load(handle)
        except Exception:
            try:
                handle.seek(0)
                payload = pickle.load(handle)
            except Exception:
                handle.seek(0)
                payload = json.load(handle)

    if payload.get("format") == "fit_model_bundle_v1":
        # Always reconstruct from the self-contained HS3 payload.
        # Any legacy "card" block referencing external shape files is ignored.
        hs3_payload = payload.get("hs3_model")
        if hs3_payload is None:
            raise ValueError("Saved bundle is missing HS3 model payload")
        observed_data = payload.get("observed_data")
        return _fit_model_from_hs3_payload(hs3_payload, payload.get("fit_metadata"), observed_data)

    raise ValueError(f"Unsupported model file format in {model_file}")
