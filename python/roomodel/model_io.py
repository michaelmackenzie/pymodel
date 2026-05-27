import json
import os
from typing import Dict

from roomodel.utilities import FitModel


def _get_root():
    import ROOT

    return ROOT


def _read_metadata(root_file_path: str) -> Dict:
    ROOT = _get_root()
    tf = ROOT.TFile.Open(root_file_path)
    if tf is None or tf.IsZombie():
        raise ValueError(f"Could not open ROOT file: {root_file_path}")
    try:
        payload_obj = tf.Get("pymodel_metadata")
        if payload_obj is None:
            return {}
        text = payload_obj.GetTitle()
        if not text:
            return {}
        return json.loads(str(text))
    finally:
        tf.Close()


def _write_metadata(tf, metadata: Dict) -> None:
    ROOT = _get_root()
    payload = json.dumps(metadata, sort_keys=True)
    named = ROOT.TNamed("pymodel_metadata", payload)
    tf.WriteObject(named, "pymodel_metadata")


def save_fit_model_bundle(payload: Dict, output_file: str) -> str:
    ROOT = _get_root()
    output_file = os.path.abspath(output_file)
    workspace = payload["workspace"]
    metadata = payload.get("metadata", {})

    tf = ROOT.TFile.Open(output_file, "RECREATE")
    if tf is None or tf.IsZombie():
        raise ValueError(f"Could not create ROOT file: {output_file}")
    try:
        tf.WriteObject(workspace, workspace.GetName())
        _write_metadata(tf, metadata)
        tf.Write()
    finally:
        tf.Close()
    return output_file


def load_fit_model(model_file: str) -> FitModel:
    model_file = os.path.abspath(model_file)
    workspace, metadata = load_workspace_and_metadata(model_file)

    return FitModel(
        model_file=model_file,
        workspace_name=str(workspace.GetName()),
        model_name=str(metadata.get("model_name", "simPdf")),
        data_name=metadata.get("data_name"),
        channels=list(metadata.get("channels", [])),
        process_names=list(metadata.get("process_names", [])),
        signal_processes=list(metadata.get("signal_processes", [])),
        observed_counts_by_channel=dict(metadata.get("observed_counts_by_channel", {})),
        poi_name=metadata.get("poi_name"),
        metadata=metadata,
    )


def load_workspace_and_metadata(model_file: str):
    ROOT = _get_root()
    model_file = os.path.abspath(model_file)
    tf = ROOT.TFile.Open(model_file)
    if tf is None or tf.IsZombie():
        raise ValueError(f"Could not open model ROOT file: {model_file}")

    workspace = None
    try:
        for key in tf.GetListOfKeys() or []:
            obj = tf.Get(key.GetName())
            if obj is not None and obj.InheritsFrom("RooWorkspace"):
                workspace = obj
                break
    finally:
        tf.Close()

    if workspace is None:
        raise ValueError(f"No RooWorkspace found in model file: {model_file}")

    metadata = _read_metadata(model_file)
    return workspace, metadata
