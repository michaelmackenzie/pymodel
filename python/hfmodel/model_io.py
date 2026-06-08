import json
from typing import Any, Dict, Optional

import pyhf

from hfmodel.utilities import FitModel


BUNDLE_FORMAT = "fit_model_bundle_v2_pyhf"


def save_fit_model_bundle(fit_model: FitModel, output_file: str):
    bundle: Dict[str, Any] = {
        "format": BUNDLE_FORMAT,
        "workspace": fit_model.workspace,
        "fit_metadata": {
            "process_names": list(fit_model.process_names),
            "process_ids": list(fit_model.process_ids),
            "signal_processes": list(fit_model.signal_processes),
            "channels": list(fit_model.channels),
            "term_channels": dict(fit_model.term_channels),
            "term_processes": dict(fit_model.term_processes),
            "observed_counts_by_channel": dict(fit_model.observed_counts_by_channel),
            "measurement_name": fit_model.measurement_name,
            "poi_name": fit_model.poi_name,
        },
    }

    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(bundle, handle, indent=2)


def _fit_model_from_workspace_payload(workspace_payload: Dict[str, Any], fit_metadata: Optional[Dict[str, Any]] = None):
    workspace = pyhf.Workspace(workspace_payload)
    measurement_name = None
    if fit_metadata is not None:
        measurement_name = fit_metadata.get("measurement_name")

    if measurement_name is not None:
        model = workspace.model(measurement_name=measurement_name)
    else:
        model = workspace.model()

    data = workspace.data(model)

    poi_name = model.config.poi_name
    if fit_metadata is not None and fit_metadata.get("poi_name"):
        poi_name = fit_metadata["poi_name"]

    return FitModel(
        workspace=workspace_payload,
        model=model,
        data=data,
        process_names=list((fit_metadata or {}).get("process_names", [])),
        process_ids=list((fit_metadata or {}).get("process_ids", [])),
        signal_processes=list((fit_metadata or {}).get("signal_processes", [])),
        channels=list((fit_metadata or {}).get("channels", workspace.channels)),
        term_channels=dict((fit_metadata or {}).get("term_channels", {})),
        term_processes=dict((fit_metadata or {}).get("term_processes", {})),
        observed_counts_by_channel=dict((fit_metadata or {}).get("observed_counts_by_channel", {})),
        measurement_name=measurement_name,
        poi_name=poi_name,
    )


def load_fit_model(model_file: str) -> FitModel:
    with open(model_file, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if payload.get("format") == BUNDLE_FORMAT:
        # Always reconstruct from the self-contained workspace payload.
        # Any legacy "card" block referencing external shape files is ignored.
        workspace_payload = payload.get("workspace")
        if workspace_payload is None:
            raise ValueError("Saved bundle is missing workspace payload")
        return _fit_model_from_workspace_payload(workspace_payload, payload.get("fit_metadata"))

    # Backward compatibility with raw pyhf workspace JSON.
    if "channels" in payload and "measurements" in payload and "version" in payload:
        return _fit_model_from_workspace_payload(payload, {})

    raise ValueError(f"Unsupported model file format in {model_file}")
