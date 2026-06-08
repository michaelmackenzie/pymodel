import __main__
import json
import os
from typing import Any, Dict, Optional
import pickle

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

    bundle = {
        "format": "fit_model_bundle_v1",
        "fit_metadata": {
            "process_names": list(fit_model.process_names),
            "signal_process": fit_model.signal_process,
            "signal_nominal_yield": fit_model.signal_nominal_yield,
        },
        "hs3_model": hs3_json_payload,
    }
    # Note: zmodel's hs3_model is already self-contained with all PDFs and parameters.
    # No "card" block is saved anymore since the model doesn't need external references.

    ext = os.path.splitext(output_file)[1].lower()
    if ext == ".json":
        with open(output_file, "w", encoding="utf-8") as handle:
            json.dump(bundle, handle, indent=2)
        return

    with open(output_file, "wb") as handle:
        pickle.dump(bundle, handle)


def _choose_top_model(distributions: Dict[str, Any]):
    if not distributions:
        raise ValueError("No distributions found in loaded HS3 payload")
    if len(distributions) == 1:
        return next(iter(distributions.values()))

    for name, model in distributions.items():
        if name.startswith("model_"):
            return model

    return next(iter(distributions.values()))


def _fit_model_from_hs3_payload(hs3_payload: Dict[str, Any], fit_metadata: Optional[Dict[str, Any]] = None):
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

    return FitModel(
        obs=model.space,
        obs_range=tuple(float(x) for x in model.space.limit1d),
        shapes={},
        yields={},
        extended_pdfs={},
        model=model,
        data=None,
        process_names=list((fit_metadata or {}).get("process_names", [])),
        signal_process=signal_process,
        constraints=constraints,
        loss=None,
        result=None,
        signal_nominal_yield=(fit_metadata or {}).get("signal_nominal_yield"),
    )


def load_fit_model(model_file: str) -> FitModel:
    payload = None
    with open(model_file, "rb") as handle:
        try:
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
        return _fit_model_from_hs3_payload(hs3_payload, payload.get("fit_metadata"))

    raise ValueError(f"Unsupported model file format in {model_file}")
