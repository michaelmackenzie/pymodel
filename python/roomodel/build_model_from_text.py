import os
import pathlib
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backends.builder_common import resolve_shape_file_for_term
from backends.card_parser import CardSpec, parse_model_card as parse_common_model_card
from roomodel.model_io import save_fit_model_bundle


def _get_root():
    import ROOT

    try:
        if not getattr(ROOT, "_pymodel_roofit_quiet", False):
            ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)
            ROOT._pymodel_roofit_quiet = True
    except Exception:
        pass

    return ROOT


def parse_model_card(card_path: str) -> CardSpec:
    return parse_common_model_card(
        card_path,
        shape_extension=".root",
        shape_description="a ROOT file",
    )


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


def _first_workspace_from_file(root_path: str):
    ROOT = _get_root()
    tf = ROOT.TFile.Open(root_path)
    if tf is None or tf.IsZombie():
        raise ValueError(f"Could not open ROOT shapes file '{root_path}'")
    try:
        for key in tf.GetListOfKeys() or []:
            obj = tf.Get(key.GetName())
            if obj is not None and obj.InheritsFrom("RooWorkspace"):
                ws = obj.Clone(obj.GetName()) if hasattr(obj, "Clone") else obj
                ws.SetName(str(obj.GetName()))
                return ws
    finally:
        tf.Close()
    raise ValueError(f"No RooWorkspace found in shapes file '{root_path}'")


def _iter_roo_collection(collection):
    if collection is None:
        return
    try:
        for obj in collection:
            yield obj
    except TypeError:
        return


def _pdf_by_name_or_aliases(workspace, process_name: str):
    names = _common_process_aliases(process_name)
    for name in list(names):
        names.append(name.lower())
        names.append(name.upper())
    seen = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        pdf = workspace.pdf(name)
        if pdf is not None:
            return pdf

    # Fallback: exact lowercase compare against all PDFs.
    target = process_name.lower()
    for obj in _iter_roo_collection(workspace.allPdfs()):
        if str(obj.GetName()).lower() == target:
            return obj
    return None


def _data_obs_by_channel(workspace, channel: str):
    candidates = [
        f"{channel}__data_obs",
        f"data_obs__{channel}",
        "data_obs",
    ]
    for name in candidates:
        data = workspace.data(name)
        if data is not None:
            return data
    return None


def _make_obs_var(name: str, lower: float = 0.0, upper: float = 1.0):
    ROOT = _get_root()
    return ROOT.RooRealVar(name, name, lower, upper)


def _make_signal_strength_var(ws, signal_processes: List[str], mu_min: float = 0.0, mu_max: float = 10.0):
    """Create the primary signal-strength POI if a signal process exists."""
    ROOT = _get_root()
    if not signal_processes:
        return None
    poi_name = f"mu_{signal_processes[0]}"
    poi = ROOT.RooRealVar(poi_name, poi_name, 1.0, float(mu_min), float(mu_max))
    getattr(ws, "import")(poi)
    return poi


def _physical_mu_min_from_card(card: CardSpec, signal_processes: List[str]) -> float:
    lower_bounds = []
    for channel in card.channels:
        sig_rate, bkg_rate = _channel_rate_split(card, channel, signal_processes)
        if sig_rate > 0.0:
            lower_bounds.append(-float(bkg_rate) / float(sig_rate))
    if not lower_bounds:
        return 0.0
    # Keep total expected yield slightly positive in every channel.
    return float(max(lower_bounds) + 1.0e-6)


def _channel_rate_split(card: CardSpec, channel: str, signal_processes: List[str]) -> Tuple[float, float]:
    sig_rate = 0.0
    bkg_rate = 0.0
    signal_set = set(signal_processes)
    for process, bin_name, rate in zip(card.process_names, card.bin_names, card.rates):
        if bin_name != channel:
            continue
        val = float(rate or 0.0)
        if process in signal_set:
            sig_rate += val
        else:
            bkg_rate += val
    return sig_rate, bkg_rate


def _build_counting_workspace(card: CardSpec):
    ROOT = _get_root()
    ws = ROOT.RooWorkspace("workspace")

    channel_cat = ROOT.RooCategory("channel", "channel")
    for channel in card.channels:
        channel_cat.defineType(channel)

    sim_pdf = ROOT.RooSimultaneous("simPdf", "simPdf", channel_cat)

    observed_counts = {}
    signal_processes = [
        process
        for process, pid in zip(card.process_names, card.process_ids)
        if int(pid) <= 0
    ]
    poi = _make_signal_strength_var(ws, signal_processes, mu_min=_physical_mu_min_from_card(card, signal_processes))

    for channel in card.channels:
        obs = _make_obs_var(f"count_obs_{channel}", 0.0, 1.0)
        getattr(ws, "import")(obs)

        sig_rate, bkg_rate = _channel_rate_split(card, channel, signal_processes)
        total_rate = sig_rate + bkg_rate

        center = ROOT.RooConstVar(f"center_total__{channel}", f"center_total__{channel}", 0.5)
        sigma = ROOT.RooConstVar(f"sigma_total__{channel}", f"sigma_total__{channel}", 0.15)
        getattr(ws, "import")(center)
        getattr(ws, "import")(sigma)
        channel_pdf = ROOT.RooGaussian(f"pdf_total__{channel}", f"pdf_total__{channel}", obs, center, sigma)
        getattr(ws, "import")(channel_pdf)

        if poi is not None:
            sig_rate_const = ROOT.RooRealVar(f"sig_rate__{channel}", f"sig_rate__{channel}", float(sig_rate), -1.0e12, 1.0e12)
            bkg_rate_const = ROOT.RooRealVar(f"bkg_rate__{channel}", f"bkg_rate__{channel}", float(bkg_rate), -1.0e12, 1.0e12)
            sig_rate_const.setConstant(True)
            bkg_rate_const.setConstant(True)
            getattr(ws, "import")(sig_rate_const)
            getattr(ws, "import")(bkg_rate_const)
            yield_total = ROOT.RooFormulaVar(
                f"yield_total__{channel}",
                "@0*@1 + @2",
                ROOT.RooArgList(poi, sig_rate_const, bkg_rate_const),
            )
            getattr(ws, "import")(yield_total)
        else:
            yield_total = ROOT.RooRealVar(f"yield_total__{channel}", f"yield_total__{channel}", total_rate, 0.0, 1.0e9)
            getattr(ws, "import")(yield_total)

        channel_ext_pdf = ROOT.RooExtendPdf(
            f"ext_pdf_total__{channel}", f"ext_pdf_total__{channel}",
            channel_pdf, yield_total,
        )
        getattr(ws, "import")(channel_ext_pdf, ROOT.RooFit.RecycleConflictNodes())

        sim_pdf.addPdf(channel_ext_pdf, channel)

        obs_val = float(card.observations.get(channel, card.observation_count or 0.0))
        observed_counts[channel] = obs_val

    getattr(ws, "import")(sim_pdf, ROOT.RooFit.RecycleConflictNodes())

    return ws, "simPdf", None, observed_counts, signal_processes


def _resolve_shape_terms(card: CardSpec, card_dir: str) -> List[Tuple[str, str, str]]:
    terms = []
    for process, channel in zip(card.process_names, card.bin_names):
        rel_path = resolve_shape_file_for_term(card, process, channel)
        full = rel_path if os.path.isabs(rel_path) else os.path.join(card_dir, rel_path)
        terms.append((process, channel, os.path.abspath(full)))
    return terms


def _extract_obs_from_pdf(pdf, workspace):
    obs_set = pdf.getObservables(workspace.allVars())
    for obj in obs_set:
        if obj.InheritsFrom("RooRealVar"):
            return obj
    return None


def _build_shape_workspace(card: CardSpec, card_dir: str):
    ROOT = _get_root()
    ws = ROOT.RooWorkspace("workspace")

    channel_cat = ROOT.RooCategory("channel", "channel")
    for channel in card.channels:
        channel_cat.defineType(channel)

    signal_processes = [
        process
        for process, pid in zip(card.process_names, card.process_ids)
        if int(pid) <= 0
    ]
    signal_set = set(signal_processes)
    poi = _make_signal_strength_var(ws, signal_processes, mu_min=_physical_mu_min_from_card(card, signal_processes))

    by_channel = {ch: {"obs_name": None, "min": None, "max": None, "terms": []} for ch in card.channels}
    observed_counts = {ch: 0.0 for ch in card.channels}

    shapes_cache: Dict[str, object] = {}

    for process, pid, channel, rate, (_, _, root_path) in zip(
        card.process_names,
        card.process_ids,
        card.bin_names,
        card.rates,
        _resolve_shape_terms(card, card_dir),
    ):
        if root_path not in shapes_cache:
            shapes_cache[root_path] = _first_workspace_from_file(root_path)
        src_ws = shapes_cache[root_path]

        src_pdf = _pdf_by_name_or_aliases(src_ws, process)
        if src_pdf is None:
            raise ValueError(
                f"Could not resolve PDF for process '{process}' in shapes file '{root_path}'"
            )

        obs = _extract_obs_from_pdf(src_pdf, src_ws)
        if obs is None:
            raise ValueError(f"PDF '{src_pdf.GetName()}' has no RooRealVar observable")

        ch_info = by_channel[channel]
        obs_name = str(obs.GetName())
        if ch_info["obs_name"] is None:
            ch_info["obs_name"] = obs_name
            ch_info["min"] = float(obs.getMin())
            ch_info["max"] = float(obs.getMax())
        elif ch_info["obs_name"] != obs_name:
            raise ValueError(
                f"Channel '{channel}' mixes observables '{ch_info['obs_name']}' and '{obs_name}'"
            )

        ch_info["terms"].append(
            {
                "process": str(process),
                "rate": float(rate if rate is not None else 1.0),
                "root_path": root_path,
                "pdf_name": str(src_pdf.GetName()),
            }
        )

        src_data = _data_obs_by_channel(src_ws, channel)
        if src_data is not None:
            try:
                observed_counts[channel] = float(src_data.sumEntries())
            except Exception:
                pass

    sim_pdf = ROOT.RooSimultaneous("simPdf", "simPdf", channel_cat)
    for channel in card.channels:
        channel_info = by_channel[channel]
        if channel_info["min"] is None or channel_info["max"] is None or not channel_info["terms"]:
            raise ValueError(f"No terms for channel '{channel}'")

        obs_name = str(channel_info["obs_name"])
        ws_obs = ws.var(obs_name)
        if ws_obs is None:
            ws_obs = _make_obs_var(obs_name, channel_info["min"], channel_info["max"])
            getattr(ws, "import")(ws_obs)

        ch_pdf_list = ROOT.RooArgList()
        ch_yield_list = ROOT.RooArgList()

        sig_rate_total = 0.0
        bkg_rate_total = 0.0
        for term in channel_info["terms"]:
            process = term["process"]
            root_path = term["root_path"]
            src_pdf_name = term["pdf_name"]
            rate_val = float(term["rate"])

            src_ws = shapes_cache[root_path]
            src_pdf = src_ws.pdf(src_pdf_name)
            if src_pdf is None:
                raise ValueError(
                    f"Could not reload PDF '{src_pdf_name}' for process '{process}' in '{root_path}'"
                )

            getattr(ws, "import")(src_pdf, ROOT.RooFit.RecycleConflictNodes())
            term_pdf = ws.pdf(src_pdf_name)
            if term_pdf is None:
                raise ValueError(f"Failed to import shape PDF '{src_pdf_name}'")

            yield_name = f"yield_{process}__{channel}"
            if process in signal_set and poi is not None:
                rate_const = ROOT.RooRealVar(
                    f"rate_{process}__{channel}",
                    f"rate_{process}__{channel}",
                    rate_val,
                    -1.0e12,
                    1.0e12,
                )
                rate_const.setConstant(True)
                getattr(ws, "import")(rate_const)
                term_yield = ROOT.RooFormulaVar(
                    yield_name,
                    "@0*@1",
                    ROOT.RooArgList(poi, rate_const),
                )
                getattr(ws, "import")(term_yield)
                sig_rate_total += rate_val
            else:
                term_yield = ROOT.RooRealVar(
                    yield_name,
                    yield_name,
                    rate_val,
                    -1.0e12,
                    1.0e12,
                )
                term_yield.setConstant(True)
                getattr(ws, "import")(term_yield)
                bkg_rate_total += rate_val

            term_yield_obj = ws.function(yield_name)
            if term_yield_obj is None:
                term_yield_obj = ws.var(yield_name)
            if term_yield_obj is None:
                raise ValueError(f"Failed to resolve yield object '{yield_name}'")

            ch_pdf_list.add(term_pdf)
            ch_yield_list.add(term_yield_obj)

        sig_rate_const = ROOT.RooRealVar(f"sig_rate__{channel}", f"sig_rate__{channel}", sig_rate_total, -1.0e12, 1.0e12)
        bkg_rate_const = ROOT.RooRealVar(f"bkg_rate__{channel}", f"bkg_rate__{channel}", bkg_rate_total, -1.0e12, 1.0e12)
        sig_rate_const.setConstant(True)
        bkg_rate_const.setConstant(True)
        getattr(ws, "import")(sig_rate_const)
        getattr(ws, "import")(bkg_rate_const)

        ch_pdf = ROOT.RooAddPdf(
            f"pdf_total__{channel}",
            f"pdf_total__{channel}",
            ch_pdf_list,
            ch_yield_list,
        )
        getattr(ws, "import")(ch_pdf, ROOT.RooFit.RecycleConflictNodes())

        sim_pdf.addPdf(ch_pdf, channel)

    getattr(ws, "import")(sim_pdf, ROOT.RooFit.RecycleConflictNodes())

    return ws, "simPdf", None, observed_counts, signal_processes


def build_model_from_card(card: CardSpec, card_dir: str):
    if card.is_counting:
        ws, model_name, data_name, observed_counts, signal_processes = _build_counting_workspace(card)
    else:
        ws, model_name, data_name, observed_counts, signal_processes = _build_shape_workspace(card, card_dir)

    poi_name = "mu"
    if signal_processes:
        poi_name = f"mu_{signal_processes[0]}"

    metadata = {
        "format": "fit_model_bundle_v1_roomodel",
        "workspace_name": str(ws.GetName()),
        "model_name": model_name,
        "data_name": data_name,
        "channels": list(card.channels),
        "process_names": list(card.process_names),
        "process_ids": list(card.process_ids),
        "signal_processes": list(signal_processes),
        "observed_counts_by_channel": dict(observed_counts),
        "poi_name": poi_name,
    }

    return {
        "workspace": ws,
        "metadata": metadata,
    }


def build_and_save_model_from_card_file(card_path: str, output_file: str) -> str:
    card_path = os.path.abspath(card_path)
    output_file = os.path.abspath(output_file)

    card = parse_model_card(card_path)
    card_dir = os.path.dirname(card_path)
    payload = build_model_from_card(card, card_dir)
    return save_fit_model_bundle(payload, output_file)
