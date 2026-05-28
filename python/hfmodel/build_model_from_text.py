import copy
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyhf

from backends.path_bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path(__file__)

from backends.card_parser import (
    CardSpec,
    parse_model_card as parse_common_model_card,
)
from backends.builder_common import (
    kind_token,
    make_term_mappings,
    make_term_names,
    resolve_shape_file_for_term,
)
from hfmodel.model_io import save_fit_model_bundle
from hfmodel.utilities import FitModel


def parse_model_card(card_path: str) -> CardSpec:
    return parse_common_model_card(
        card_path,
        shape_extension=".json",
        shape_description="a JSON workspace file",
    )


def _load_workspace_payload(file_path: str) -> Dict:
    with open(file_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Workspace file '{file_path}' does not contain a JSON object")
    if "channels" not in payload or "observations" not in payload:
        raise ValueError(f"Workspace file '{file_path}' is not a valid pyhf workspace")
    return payload


def _normalize_pdf(values: List[float], context: str) -> Tuple[np.ndarray, float]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{context} has empty distribution")
    total = float(np.sum(arr))
    if total <= 0.0:
        raise ValueError(f"{context} has non-positive normalization ({total})")
    return arr / total, total


def _common_process_aliases(process: str) -> List[str]:
    aliases = [process]
    lowered = process.lower()
    if lowered == "sig":
        aliases.append("signal")
    if lowered == "signal":
        aliases.append("sig")
    if lowered == "bkg":
        aliases.append("background")
    if lowered == "background":
        aliases.append("bkg")
    return aliases


def _find_sample_block(channel_block: Dict, process: str) -> Optional[Dict]:
    by_name = {sample.get("name"): sample for sample in channel_block.get("samples", [])}
    for alias in _common_process_aliases(process):
        if alias in by_name:
            return by_name[alias]

    lowered_map = {
        str(sample.get("name", "")).lower(): sample
        for sample in channel_block.get("samples", [])
    }
    for alias in _common_process_aliases(process):
        if alias.lower() in lowered_map:
            return lowered_map[alias.lower()]

    return None


def _resolve_workspace_payloads(card: CardSpec, card_dir: str):
    payloads: Dict[str, Dict] = {}
    term_payloads = []
    for process, channel in zip(card.process_names, card.bin_names):
        rel_path = resolve_shape_file_for_term(card, process, channel)
        full_path = rel_path if os.path.isabs(rel_path) else os.path.join(card_dir, rel_path)
        full_path = os.path.abspath(full_path)
        if full_path not in payloads:
            payloads[full_path] = _load_workspace_payload(full_path)
        term_payloads.append((full_path, payloads[full_path]))
    return term_payloads, payloads


def _find_channel_block(workspace_payload: Dict, channel: str) -> Dict:
    for channel_block in workspace_payload.get("channels", []):
        if channel_block.get("name") == channel:
            return channel_block
    raise ValueError(f"Workspace does not contain channel '{channel}'")


def _find_observation_block(workspace_payload: Dict, channel: str) -> Optional[Dict]:
    for obs_block in workspace_payload.get("observations", []):
        if obs_block.get("name") == channel:
            return obs_block
    return None


def _parse_norm_pair(raw_value: str) -> Tuple[float, float]:
    if "/" in raw_value:
        hi_str, lo_str = raw_value.split("/", 1)
        hi = float(hi_str)
        lo = float(lo_str)
        return hi, lo

    value = float(raw_value)
    if value <= 0.0:
        raise ValueError(f"Invalid normalization uncertainty value '{raw_value}'")
    return value, 1.0 / value


def _apply_rate_to_sample_data(sample_data: List[float], target_rate: Optional[float]) -> List[float]:
    normalized, source_total = _normalize_pdf(sample_data, "Nominal sample")
    target_total = source_total if target_rate is None else float(target_rate)
    if target_total < 0.0:
        raise ValueError(f"Cannot use negative target rate {target_total}")
    return (normalized * target_total).tolist()


def _rescale_shape_modifier_data(modifier: Dict, scale_factor: float):
    if modifier.get("type") != "histosys":
        return
    if abs(scale_factor - 1.0) < 1e-12:
        return

    data = modifier.get("data", {})
    if not isinstance(data, dict):
        return

    if "hi_data" in data:
        hi_arr = np.asarray(data["hi_data"], dtype=float)
        if hi_arr.size == 0:
            raise ValueError(f"histosys hi_data ({modifier.get('name', 'unknown')}) has empty distribution")
        data["hi_data"] = (hi_arr * scale_factor).tolist()
    if "lo_data" in data:
        lo_arr = np.asarray(data["lo_data"], dtype=float)
        if lo_arr.size == 0:
            raise ValueError(f"histosys lo_data ({modifier.get('name', 'unknown')}) has empty distribution")
        data["lo_data"] = (lo_arr * scale_factor).tolist()
    modifier["data"] = data


def _rescale_staterror_modifier_data(modifier: Dict, scale_factor: float):
    if modifier.get("type") != "staterror":
        return
    if abs(scale_factor - 1.0) < 1e-12:
        return
    data = modifier.get("data")
    if data is None:
        return
    arr = np.asarray(data, dtype=float)
    modifier["data"] = (arr * scale_factor).tolist()


def _build_counting_workspace(card: CardSpec) -> Dict:
    channel_samples: Dict[str, List[Dict]] = {channel: [] for channel in card.channels}
    signal_processes = {
        process
        for process, proc_id in zip(card.process_names, card.process_ids)
        if proc_id <= 0
    }
    term_samples: List[Dict] = []

    for process, process_id, channel, rate in zip(
        card.process_names,
        card.process_ids,
        card.bin_names,
        card.rates,
    ):
        nominal = float(1.0 if rate is None else rate)
        sample = {
            "name": process,
            "data": [nominal],
            "modifiers": [],
        }

        if process_id <= 0:
            sample["modifiers"].append(
                {"name": "mu", "type": "normfactor", "data": None}
            )

        channel_samples[channel].append(sample)
        term_samples.append(sample)

    # Attach uncertainty modifiers after all samples are created so indexing is stable.
    for idx, sample in enumerate(term_samples):

        for unc in card.uncertainties:
            raw_value = unc.values[idx]
            if raw_value == "-":
                continue
            kind = kind_token(unc.kind)
            if kind == "shape":
                raise ValueError(
                    f"Shape uncertainty '{unc.name}' is not supported in counting mode"
                )
            hi, lo = _parse_norm_pair(raw_value)
            sample["modifiers"].append(
                {
                    "name": unc.name,
                    "type": "normsys",
                    "data": {"hi": hi, "lo": lo},
                }
            )

    channels = [
        {"name": name, "samples": samples}
        for name, samples in channel_samples.items()
    ]

    observations = []
    for channel in card.channels:
        if channel in card.observations:
            value = float(card.observations[channel])
        elif card.observation_count is not None and len(card.channels) == 1:
            value = float(card.observation_count)
        else:
            value = float(
                np.sum(
                    [
                        float(item["data"][0])
                        for item in channel_samples[channel]
                    ]
                )
            )
        observations.append({"name": channel, "data": [value]})

    measurement = {
        "name": "measurement",
        "config": {
            "poi": "mu" if signal_processes else "rate",
            "parameters": [],
        },
    }

    workspace = {
        "channels": channels,
        "observations": observations,
        "measurements": [measurement],
        "version": "1.0.0",
    }

    return workspace


def _build_shape_workspace(card: CardSpec, card_dir: str) -> Dict:
    term_payloads, payload_cache = _resolve_workspace_payloads(card, card_dir)

    built_channels: Dict[str, Dict] = {}
    obs_by_channel: Dict[str, List[float]] = {}

    for idx, (process, process_id, channel, rate) in enumerate(
        zip(card.process_names, card.process_ids, card.bin_names, card.rates)
    ):
        workspace_path, workspace_payload = term_payloads[idx]
        channel_block = _find_channel_block(workspace_payload, channel)

        source_sample = _find_sample_block(channel_block, process)
        if source_sample is None:
            raise ValueError(
                f"Workspace '{workspace_path}' channel '{channel}' has no sample '{process}'"
            )

        built_channel = built_channels.setdefault(channel, {"name": channel, "samples": []})

        source_sample_data = [float(x) for x in source_sample.get("data", [])]
        if not source_sample_data:
            raise ValueError(
                f"Workspace '{workspace_path}' sample '{process}' in '{channel}' has empty data"
            )
        source_total = float(np.sum(np.asarray(source_sample_data, dtype=float)))
        sample_data = _apply_rate_to_sample_data(source_sample_data, rate)
        target_total = float(np.sum(np.asarray(sample_data, dtype=float)))
        scale_factor = 1.0
        if source_total > 0.0:
            scale_factor = target_total / source_total

        raw_modifiers = copy.deepcopy(source_sample.get("modifiers", []))
        # Build rate uncertainties from card only: keep shape and MC-stat modifiers from input workspace.
        modifiers = []
        for mod in raw_modifiers:
            mod_type = mod.get("type")
            if mod_type == "normsys":
                continue
            if mod_type == "histosys":
                # Preserve rate impact from shape variations by applying only
                # the nominal scale factor, not per-template renormalization.
                _rescale_shape_modifier_data(mod, scale_factor)
            if mod_type == "staterror":
                _rescale_staterror_modifier_data(mod, scale_factor)
            modifiers.append(mod)

        modifier_names = {(mod.get("name"), mod.get("type")) for mod in modifiers}

        if process_id <= 0 and ("mu", "normfactor") not in modifier_names:
            modifiers.append({"name": "mu", "type": "normfactor", "data": None})
            modifier_names.add(("mu", "normfactor"))

        for unc in card.uncertainties:
            raw_value = unc.values[idx]
            if raw_value == "-":
                continue
            kind = kind_token(unc.kind)
            if kind in ("lnN", "gs"):
                hi, lo = _parse_norm_pair(raw_value)
                key = (unc.name, "normsys")
                if key not in modifier_names:
                    modifiers.append(
                        {
                            "name": unc.name,
                            "type": "normsys",
                            "data": {"hi": hi, "lo": lo},
                        }
                    )
                    modifier_names.add(key)
            else:
                # shape uncertainties are expected to be provided by histosys in workspace JSON
                key = (unc.name, "histosys")
                if key not in modifier_names:
                    raise ValueError(
                        f"Shape uncertainty '{unc.name}' for {process}/{channel} is missing histosys in '{workspace_path}'"
                    )

        # Keep only one sample per process per channel in the combined workspace.
        existing_names = {sample.get("name") for sample in built_channel["samples"]}
        if process in existing_names:
            raise ValueError(f"Duplicate process '{process}' in channel '{channel}'")

        built_channel["samples"].append(
            {
                "name": process,
                "data": sample_data,
                "modifiers": modifiers,
            }
        )

    for channel in card.channels:
        obs_file = card.data_obs_files.get(channel, card.data_obs_files.get("*"))
        observation = None

        if obs_file is not None:
            obs_path = obs_file if os.path.isabs(obs_file) else os.path.join(card_dir, obs_file)
            obs_path = os.path.abspath(obs_path)
            payload = payload_cache.get(obs_path)
            if payload is None:
                payload = _load_workspace_payload(obs_path)
                payload_cache[obs_path] = payload
            observation = _find_observation_block(payload, channel)

        if observation is None:
            # Fall back to any mapped process payload for the same channel.
            for _, payload in term_payloads:
                candidate = _find_observation_block(payload, channel)
                if candidate is not None:
                    observation = candidate
                    break

        if observation is None:
            raise ValueError(
                f"No observation data found for channel '{channel}'. Add 'shapes data_obs {channel} <workspace.json>'."
            )

        obs_values = [float(x) for x in observation.get("data", [])]
        if not obs_values:
            raise ValueError(f"Observation for channel '{channel}' is empty")
        obs_by_channel[channel] = obs_values

        if channel in card.observations:
            expected = float(card.observations[channel])
            found = float(np.sum(np.asarray(obs_values, dtype=float)))
            if not np.isclose(expected, found, atol=0.5):
                raise ValueError(
                    f"Observation count mismatch for channel '{channel}': card={expected}, workspace={found}"
                )

    workspace = {
        "channels": [built_channels[channel] for channel in card.channels],
        "observations": [
            {"name": channel, "data": obs_by_channel[channel]}
            for channel in card.channels
        ],
        "measurements": [
            {
                "name": "measurement",
                "config": {
                    "poi": "mu",
                    "parameters": [],
                },
            }
        ],
        "version": "1.0.0",
    }

    return workspace


def build_model_from_card(card: CardSpec, card_dir: str):
    if card.param_constraints:
        raise ValueError("'param' constraints are not yet supported in pyhf card builder")

    term_names = make_term_names(card.process_names, card.bin_names)
    term_channels, term_processes = make_term_mappings(
        term_names,
        card.process_names,
        card.bin_names,
    )

    if card.is_counting:
        workspace_dict = _build_counting_workspace(card)
    else:
        workspace_dict = _build_shape_workspace(card, card_dir)

    workspace = pyhf.Workspace(workspace_dict)
    model = workspace.model(measurement_name="measurement")
    data = workspace.data(model)

    observed_counts_by_channel: Dict[str, float] = {
        obs["name"]: float(np.sum(np.asarray(obs["data"], dtype=float)))
        for obs in workspace_dict["observations"]
    }

    signal_processes = [
        process
        for process, proc_id in zip(card.process_names, card.process_ids)
        if proc_id <= 0
    ]

    return FitModel(
        workspace=workspace_dict,
        model=model,
        data=data,
        process_names=list(card.process_names),
        process_ids=list(card.process_ids),
        signal_processes=list(dict.fromkeys(signal_processes)),
        channels=list(card.channels),
        term_channels=term_channels,
        term_processes=term_processes,
        observed_counts_by_channel=observed_counts_by_channel,
        measurement_name="measurement",
        poi_name=model.config.poi_name,
    )


def build_and_save_model_from_card_file(input_card: str, output_file: str) -> str:
    card_path = os.path.abspath(input_card)
    card_dir = os.path.dirname(card_path)

    card = parse_model_card(card_path)
    fit_model = build_model_from_card(card, card_dir)

    output_path = os.path.abspath(output_file)
    if not output_path.lower().endswith(".json"):
        output_path = f"{output_path}.json"
    save_fit_model_bundle(fit_model, output_path, card=card, card_dir=card_dir)
    return output_path
