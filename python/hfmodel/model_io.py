import json
import os
from dataclasses import asdict
from typing import Any, Dict, Optional

import pyhf

from hfmodel.utilities import FitModel


BUNDLE_FORMAT = "fit_model_bundle_v2_pyhf"


def _serialize_card(card, card_dir: Optional[str]) -> Dict[str, Any]:
    payload = asdict(card)
    if card_dir is None:
        return payload

    resolved_specs = []
    for shape_spec in payload.get("shape_specs", []):
        shape_file = shape_spec["file"]
        if not os.path.isabs(shape_file):
            shape_file = os.path.abspath(os.path.join(card_dir, shape_file))
        resolved_specs.append(
            {
                "process": shape_spec["process"],
                "channel": shape_spec["channel"],
                "file": shape_file,
            }
        )
    payload["shape_specs"] = resolved_specs

    resolved_data_obs = {}
    for channel, obs_file in payload.get("data_obs_files", {}).items():
        if os.path.isabs(obs_file):
            resolved_data_obs[channel] = obs_file
        else:
            resolved_data_obs[channel] = os.path.abspath(os.path.join(card_dir, obs_file))
    payload["data_obs_files"] = resolved_data_obs
    return payload


def _deserialize_card(card_payload: Dict[str, Any]):
    from hfmodel.build_model_from_text import CardSpec, ShapeSpec, UncertaintySpec, ConstraintSpec

    return CardSpec(
        shape_specs=[ShapeSpec(**item) for item in card_payload.get("shape_specs", [])],
        is_counting=bool(card_payload.get("is_counting", False)),
        channels=list(card_payload.get("channels", [])),
        bin_names=list(card_payload.get("bin_names", [])),
        process_names=list(card_payload.get("process_names", [])),
        process_ids=list(card_payload.get("process_ids", [])),
        rates=list(card_payload.get("rates", [])),
        uncertainties=[UncertaintySpec(**item) for item in card_payload.get("uncertainties", [])],
        observations=dict(card_payload.get("observations", {})),
        data_obs_files=dict(card_payload.get("data_obs_files", {})),
        category=card_payload.get("category"),
        observation_count=card_payload.get("observation_count"),
        param_constraints=[ConstraintSpec(**item) for item in card_payload.get("param_constraints", [])],
    )


def save_fit_model_bundle(fit_model: FitModel, output_file: str, card=None, card_dir: Optional[str] = None):
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

    if card is not None:
        bundle["card"] = _serialize_card(card, card_dir)

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
        card_payload = payload.get("card")
        if card_payload is not None:
            from hfmodel.build_model_from_text import build_model_from_card

            card = _deserialize_card(card_payload)
            return build_model_from_card(card, os.path.dirname(os.path.abspath(model_file)))

        workspace_payload = payload.get("workspace")
        if workspace_payload is None:
            raise ValueError("Saved bundle is missing workspace payload")
        return _fit_model_from_workspace_payload(workspace_payload, payload.get("fit_metadata"))

    # Backward compatibility with raw pyhf workspace JSON.
    if "channels" in payload and "measurements" in payload and "version" in payload:
        return _fit_model_from_workspace_payload(payload, {})

    raise ValueError(f"Unsupported model file format in {model_file}")
