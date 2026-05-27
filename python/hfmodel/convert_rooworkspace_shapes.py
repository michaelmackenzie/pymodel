#!/usr/bin/env python3
import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from hfmodel.model_io import load_fit_model


class ConversionError(RuntimeError):
    pass


@dataclass
class WorkspaceConversionResult:
    workspace_name: str
    output_file: str
    n_channels: int
    n_samples: int


_ROOT_LOGGING_CONFIGURED = False


def _configure_root_logging() -> None:
    global _ROOT_LOGGING_CONFIGURED
    if _ROOT_LOGGING_CONFIGURED:
        return

    import ROOT

    try:
        ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
    except Exception:
        pass

    try:
        ROOT.gErrorIgnoreLevel = ROOT.kError
    except Exception:
        pass

    _ROOT_LOGGING_CONFIGURED = True


def _iter_roo_collection(collection) -> Iterable:
    if collection is None:
        return

    try:
        iterator = iter(collection)
    except TypeError:
        create_iterator = getattr(collection, "createIterator", None)
        if not callable(create_iterator):
            return

        iterator = create_iterator()
        while True:
            obj = iterator.Next()
            if not obj:
                break
            yield obj
        return

    for obj in iterator:
        yield obj


def _collect_workspaces(root_file) -> List:
    workspaces = []

    def _scan_directory(directory):
        for key in directory.GetListOfKeys():
            name = key.GetName()
            obj = directory.Get(name)
            if obj is None:
                continue
            if obj.InheritsFrom("RooWorkspace"):
                workspaces.append(obj)
            elif obj.InheritsFrom("TDirectory"):
                _scan_directory(obj)

    _scan_directory(root_file)
    return workspaces


def _sanitize_name(name: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in str(name))


def _resolve_root_path(card_dir: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(card_dir, path))


def _parse_bin_edges(edges_text: Optional[str]) -> Optional[np.ndarray]:
    if edges_text is None:
        return None
    values = [item.strip() for item in str(edges_text).split(",") if item.strip()]
    if len(values) < 2:
        raise ConversionError("--bin-edges must provide at least two comma-separated values")
    edges = np.asarray([float(item) for item in values], dtype=float)
    if np.any(~np.isfinite(edges)):
        raise ConversionError("--bin-edges contains non-finite values")
    if np.any(np.diff(edges) <= 0.0):
        raise ConversionError("--bin-edges must be strictly increasing")
    return edges


def _axis_edges_from_th1(hist_obj) -> np.ndarray:
    axis = hist_obj.GetXaxis()
    n_bins = int(hist_obj.GetNbinsX())
    lows = [axis.GetBinLowEdge(1 + idx) for idx in range(n_bins)]
    highs = [axis.GetBinUpEdge(n_bins)]
    return np.asarray(lows + highs, dtype=float)


def _counts_from_th1(hist_obj) -> np.ndarray:
    n_bins = int(hist_obj.GetNbinsX())
    return np.asarray([hist_obj.GetBinContent(1 + idx) for idx in range(n_bins)], dtype=float)


def _pdf_observable(pdf, workspace):
    observables = pdf.getObservables(workspace.allVars())
    obs_vars = [obj for obj in _iter_roo_collection(observables) if obj.InheritsFrom("RooRealVar")]
    if len(obs_vars) != 1:
        obs_names = [obj.GetName() for obj in obs_vars]
        raise ConversionError(
            f"PDF '{pdf.GetName()}' ({pdf.ClassName()}) in workspace '{workspace.GetName()}' "
            f"has {len(obs_vars)} observables {obs_names}; only 1D PDFs are supported"
        )
    return obs_vars[0]


def _histogram_from_pdf(pdf, workspace, default_bins: Optional[np.ndarray], n_bins: Optional[int]):
    import ROOT

    obs_var = _pdf_observable(pdf, workspace)
    class_name = str(pdf.ClassName())

    if class_name == "RooHistPdf":
        roo_datahist = pdf.dataHist()
        if roo_datahist is None:
            raise ConversionError(
                f"RooHistPdf '{pdf.GetName()}' in workspace '{workspace.GetName()}' has no data histogram"
            )
        try:
            hist_obj = roo_datahist.createHistogram(obs_var.GetName())
        except Exception:
            hist_obj = roo_datahist.createHistogram(pdf.GetName(), obs_var)
        if hist_obj is None:
            raise ConversionError(
                f"Could not create histogram for RooHistPdf '{pdf.GetName()}' in workspace '{workspace.GetName()}'"
            )
        edges = _axis_edges_from_th1(hist_obj)
        counts = _counts_from_th1(hist_obj)
        return edges, counts

    if default_bins is not None:
        n_bin = int(len(default_bins) - 1)
        binning = ROOT.RooFit.Binning(n_bin, np.asarray(default_bins, dtype=float))
    else:
        if n_bins is None:
            raise ConversionError(
                "Unbinned PDF conversion requires either --bin-edges or --bins"
            )
        low = float(obs_var.getMin())
        high = float(obs_var.getMax())
        if not np.isfinite(low) or not np.isfinite(high) or not low < high:
            raise ConversionError(
                f"Observable '{obs_var.GetName()}' for PDF '{pdf.GetName()}' has invalid range [{low}, {high}]"
            )
        n_bin = int(n_bins)
        default_bins = np.linspace(low, high, n_bin + 1, dtype=float)
        binning = ROOT.RooFit.Binning(n_bin, low, high)

    hist_name = f"{_sanitize_name(workspace.GetName())}_{_sanitize_name(pdf.GetName())}_hist"
    hist_obj = pdf.createHistogram(hist_name, obs_var, binning)
    if hist_obj is None:
        raise ConversionError(
            f"Could not create histogram for PDF '{pdf.GetName()}' in workspace '{workspace.GetName()}'"
        )

    counts = _counts_from_th1(hist_obj)
    return np.asarray(default_bins, dtype=float), counts


def _extract_observation_counts(workspace, channel_name: str, edges: np.ndarray) -> np.ndarray:
    data_obs = workspace.data("data_obs")
    if not data_obs:
        for dataset in _iter_roo_collection(workspace.allData()):
            name = str(dataset.GetName())
            if name.endswith("data_obs") or name.endswith("__data_obs"):
                data_obs = dataset
                break
    if not data_obs:
        return np.zeros(len(edges) - 1, dtype=float)

    is_datahist = False
    try:
        is_datahist = bool(data_obs.InheritsFrom("RooDataHist"))
    except Exception:
        is_datahist = False

    if is_datahist:
        obs_set = data_obs.get()
        obs_vars = [obj for obj in _iter_roo_collection(obs_set) if obj.InheritsFrom("RooRealVar")]
        if len(obs_vars) != 1:
            raise ConversionError(
                f"data_obs in workspace '{workspace.GetName()}' is not 1D"
            )
        obs = obs_vars[0]
        try:
            hist_obj = data_obs.createHistogram(obs.GetName())
        except Exception:
            hist_obj = data_obs.createHistogram(f"{channel_name}_data_obs", obs)
        if hist_obj is None:
            raise ConversionError(
                f"Could not create histogram from data_obs in workspace '{workspace.GetName()}'"
            )
        hist_edges = _axis_edges_from_th1(hist_obj)
        if len(hist_edges) != len(edges) or not np.allclose(hist_edges, edges, rtol=1e-6, atol=1e-9):
            raise ConversionError(
                f"data_obs binning in workspace '{workspace.GetName()}' does not match sample binning"
            )
        return _counts_from_th1(hist_obj)

    obs_set = data_obs.get()
    obs_vars = [obj for obj in _iter_roo_collection(obs_set) if obj.InheritsFrom("RooRealVar")]
    if len(obs_vars) != 1:
        raise ConversionError(
            f"data_obs in workspace '{workspace.GetName()}' is not 1D"
        )
    obs_name = obs_vars[0].GetName()
    values = []
    for idx in range(int(data_obs.numEntries())):
        row = data_obs.get(idx)
        val_obj = row.find(obs_name)
        values.append(float(val_obj.getVal()))
    counts, _ = np.histogram(np.asarray(values, dtype=float), bins=np.asarray(edges, dtype=float))
    return counts.astype(float)


def _make_sample_from_counts(name: str, counts: np.ndarray) -> Dict:
    # Keep only a normfactor placeholder for likely signal-like names.
    lower_name = name.lower()
    modifiers = []
    if "sig" in lower_name or "signal" in lower_name:
        modifiers.append({"name": "mu", "type": "normfactor", "data": None})
    return {
        "name": name,
        "data": [float(x) for x in np.asarray(counts, dtype=float)],
        "modifiers": modifiers,
    }


def convert_root_workspaces_to_pyhf(
    root_path: str,
    output_dir: str,
    output_prefix: str,
    bins: Optional[int],
    bin_edges: Optional[np.ndarray],
    workspace_prefix: bool,
) -> List[WorkspaceConversionResult]:
    import ROOT

    _configure_root_logging()

    root_file = ROOT.TFile.Open(root_path)
    if root_file is None or root_file.IsZombie():
        raise ConversionError(f"Could not open ROOT file '{root_path}'")

    workspaces = _collect_workspaces(root_file)
    if not workspaces:
        raise ConversionError(f"No RooWorkspace objects found in '{root_path}'")

    os.makedirs(output_dir, exist_ok=True)

    channels = []
    observations = []
    workspace_name_mapping = {}
    total_samples = 0

    for ws in workspaces:
        ws_name = str(ws.GetName())
        channel_name = _sanitize_name(ws_name)
        prefix = f"{channel_name}__" if workspace_prefix or len(workspaces) > 1 else ""
        workspace_name_mapping[ws_name] = prefix

        sample_entries = []
        channel_edges = None

        for pdf in _iter_roo_collection(ws.allPdfs()):
            sample_name = f"{prefix}{pdf.GetName()}" if prefix else str(pdf.GetName())
            edges, counts = _histogram_from_pdf(
                pdf=pdf,
                workspace=ws,
                default_bins=bin_edges,
                n_bins=bins,
            )

            if channel_edges is None:
                channel_edges = np.asarray(edges, dtype=float)
            else:
                if len(edges) != len(channel_edges) or not np.allclose(edges, channel_edges, rtol=1e-6, atol=1e-9):
                    raise ConversionError(
                        f"Channel '{channel_name}' has inconsistent sample binning; "
                        "all PDFs in a pyhf channel must share one binning"
                    )

            sample_entries.append(_make_sample_from_counts(sample_name, counts))

        if not sample_entries:
            raise ConversionError(f"Workspace '{ws_name}' contains no PDFs")

        total_samples += len(sample_entries)
        channels.append({"name": channel_name, "samples": sample_entries})

        if channel_edges is None:
            raise ConversionError(f"Workspace '{ws_name}' had no valid sample histograms")
        obs_counts = _extract_observation_counts(ws, channel_name=channel_name, edges=channel_edges)
        if np.sum(obs_counts) <= 0.0:
            # If no data_obs is present, use sum of nominal templates.
            obs_counts = np.sum(
                np.asarray([s["data"] for s in sample_entries], dtype=float),
                axis=0,
            )
        observations.append({"name": channel_name, "data": [float(x) for x in obs_counts]})

    workspace_payload = {
        "channels": channels,
        "observations": observations,
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
        "workspace_name_mapping": workspace_name_mapping,
    }

    output_file = os.path.join(output_dir, f"{output_prefix}.json")
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(workspace_payload, handle, indent=2)

    root_file.Close()

    return [
        WorkspaceConversionResult(
            workspace_name=",".join(ws.GetName() for ws in workspaces),
            output_file=output_file,
            n_channels=len(channels),
            n_samples=total_samples,
        )
    ]


def _collect_histosys_variations(sample_block: Dict) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    variations = {}
    for mod in sample_block.get("modifiers", []):
        if mod.get("type") != "histosys":
            continue
        name = str(mod.get("name"))
        data = mod.get("data") or {}
        hi = np.asarray(data.get("hi_data", []), dtype=float).reshape(-1)
        lo = np.asarray(data.get("lo_data", []), dtype=float).reshape(-1)
        if hi.size and lo.size and hi.size == lo.size:
            variations[name] = (hi, lo)
    return variations


def export_pyhf_to_root(
    input_model: str,
    output_root: str,
    workspace_name: str,
    observable_min: float,
    observable_max: Optional[float],
):
    import ROOT

    _configure_root_logging()

    fit_model = load_fit_model(os.path.abspath(input_model))
    ws_payload = fit_model.workspace

    channels = ws_payload.get("channels", [])
    observations = {
        item.get("name"): np.asarray(item.get("data", []), dtype=float)
        for item in ws_payload.get("observations", [])
    }

    if not channels:
        raise ConversionError("Input pyhf workspace has no channels")

    if observable_max is None:
        max_bins = max(len(ch.get("samples", [{}])[0].get("data", [])) for ch in channels if ch.get("samples"))
        observable_max = observable_min + float(max_bins)

    workspace = ROOT.RooWorkspace(workspace_name)
    ws_import = getattr(workspace, "import")

    for channel in channels:
        channel_name = str(channel.get("name"))
        samples = channel.get("samples", [])
        if not samples:
            continue

        n_bins = len(np.asarray(samples[0].get("data", []), dtype=float).reshape(-1))
        if n_bins <= 0:
            raise ConversionError(f"Channel '{channel_name}' has empty sample data")

        edges = np.linspace(float(observable_min), float(observable_max), n_bins + 1, dtype=float)
        obs_name = f"{_sanitize_name(channel_name)}_obs"
        obs_var = ROOT.RooRealVar(obs_name, obs_name, 0.5 * (edges[0] + edges[-1]), edges[0], edges[-1])
        obs_var.setBins(n_bins)
        ws_import(obs_var)

        sample_sum = np.zeros(n_bins, dtype=float)

        for sample in samples:
            sample_name = str(sample.get("name"))
            nominal = np.asarray(sample.get("data", []), dtype=float).reshape(-1)
            if nominal.size != n_bins:
                raise ConversionError(
                    f"Channel '{channel_name}' sample '{sample_name}' has bin count {nominal.size}, expected {n_bins}"
                )

            sample_sum += nominal

            th1_name = f"{_sanitize_name(channel_name)}__{_sanitize_name(sample_name)}_hist"
            th1 = ROOT.TH1D(th1_name, th1_name, n_bins, edges)
            for idx, val in enumerate(nominal, start=1):
                th1.SetBinContent(idx, float(val))

            datahist_name = f"{_sanitize_name(channel_name)}__{_sanitize_name(sample_name)}_datahist"
            roo_data_hist = ROOT.RooDataHist(datahist_name, datahist_name, ROOT.RooArgList(obs_var), th1)
            roo_pdf_name = f"{_sanitize_name(channel_name)}__{_sanitize_name(sample_name)}"
            roo_pdf = ROOT.RooHistPdf(roo_pdf_name, roo_pdf_name, ROOT.RooArgSet(obs_var), roo_data_hist)
            ws_import(roo_data_hist)
            ws_import(roo_pdf)

            for syst_name, (hi_data, lo_data) in _collect_histosys_variations(sample).items():
                if hi_data.size != n_bins or lo_data.size != n_bins:
                    continue
                for suffix, var_data in (("Up", hi_data), ("Down", lo_data)):
                    th1_var_name = (
                        f"{_sanitize_name(channel_name)}__{_sanitize_name(sample_name)}_"
                        f"{_sanitize_name(syst_name)}{suffix}_hist"
                    )
                    th1_var = ROOT.TH1D(th1_var_name, th1_var_name, n_bins, edges)
                    for idx, val in enumerate(var_data, start=1):
                        th1_var.SetBinContent(idx, float(val))

                    datahist_var_name = (
                        f"{_sanitize_name(channel_name)}__{_sanitize_name(sample_name)}_"
                        f"{_sanitize_name(syst_name)}{suffix}_datahist"
                    )
                    roo_data_hist_var = ROOT.RooDataHist(
                        datahist_var_name,
                        datahist_var_name,
                        ROOT.RooArgList(obs_var),
                        th1_var,
                    )
                    roo_pdf_var_name = (
                        f"{_sanitize_name(channel_name)}__{_sanitize_name(sample_name)}_"
                        f"{_sanitize_name(syst_name)}{suffix}"
                    )
                    roo_pdf_var = ROOT.RooHistPdf(
                        roo_pdf_var_name,
                        roo_pdf_var_name,
                        ROOT.RooArgSet(obs_var),
                        roo_data_hist_var,
                    )
                    ws_import(roo_data_hist_var)
                    ws_import(roo_pdf_var)

        obs_data = observations.get(channel_name)
        if obs_data is None or obs_data.size != n_bins:
            obs_data = sample_sum

        th1_obs_name = f"{_sanitize_name(channel_name)}__data_obs_hist"
        th1_obs = ROOT.TH1D(th1_obs_name, th1_obs_name, n_bins, edges)
        for idx, val in enumerate(np.asarray(obs_data, dtype=float), start=1):
            th1_obs.SetBinContent(idx, float(val))

        data_obs_name = f"{_sanitize_name(channel_name)}__data_obs"
        roo_data_obs = ROOT.RooDataHist(data_obs_name, data_obs_name, ROOT.RooArgList(obs_var), th1_obs)
        ws_import(roo_data_obs)

    output_path = os.path.abspath(output_root)
    root_file = ROOT.TFile.Open(output_path, "RECREATE")
    if root_file is None or root_file.IsZombie():
        raise ConversionError(f"Could not create ROOT file '{output_path}'")

    root_file.cd()
    workspace.Write(workspace_name)
    root_file.Close()

    return output_path, workspace_name


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Convert between Combine RooWorkspace ROOT files and hfmodel/pyhf workspace JSON. "
            "ROOT -> JSON creates binned pyhf templates. JSON -> ROOT creates RooHistPdf objects."
        )
    )
    parser.add_argument("input", help="Input ROOT file or hfmodel/pyhf JSON file")
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Output directory for ROOT -> JSON conversion",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output prefix for ROOT -> JSON conversion (default: input basename)",
    )
    parser.add_argument(
        "--workspace-prefix",
        action="store_true",
        help="Prefix sample names by workspace name when converting ROOT -> JSON",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="Number of bins for unbinned RooFit PDFs when converting ROOT -> JSON (default: 40)",
    )
    parser.add_argument(
        "--bin-edges",
        type=str,
        default=None,
        help="Optional explicit comma-separated bin edges for unbinned PDFs, e.g. '100,105,110'",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Output ROOT path for JSON -> ROOT conversion",
    )
    parser.add_argument(
        "--workspace-name",
        type=str,
        default="workspace",
        help="RooWorkspace name for JSON -> ROOT conversion (default: workspace)",
    )
    parser.add_argument(
        "--obs-min",
        type=float,
        default=0.0,
        help="Observable lower edge for JSON -> ROOT histogram axis (default: 0)",
    )
    parser.add_argument(
        "--obs-max",
        type=float,
        default=None,
        help="Observable upper edge for JSON -> ROOT histogram axis (default: obs-min + nBins)",
    )
    return parser


def main():
    args = _build_parser().parse_args()

    input_path = os.path.abspath(args.input)
    bin_edges = _parse_bin_edges(args.bin_edges)

    if args.output_root is not None:
        output_root = os.path.abspath(args.output_root)
        if not output_root.lower().endswith(".root"):
            raise ConversionError("--output-root must end with .root")

        output_path, ws_name = export_pyhf_to_root(
            input_model=input_path,
            output_root=output_root,
            workspace_name=args.workspace_name,
            observable_min=float(args.obs_min),
            observable_max=args.obs_max,
        )
        print(f"Wrote RooWorkspace '{ws_name}' to {output_path}")
        return

    output_dir = os.path.abspath(args.output_dir)
    output_prefix = args.output_prefix
    if output_prefix is None:
        output_prefix = os.path.splitext(os.path.basename(input_path))[0]

    results = convert_root_workspaces_to_pyhf(
        root_path=input_path,
        output_dir=output_dir,
        output_prefix=output_prefix,
        bins=args.bins,
        bin_edges=bin_edges,
        workspace_prefix=bool(args.workspace_prefix),
    )

    for result in results:
        print(
            f"Converted {result.workspace_name} -> {result.output_file} "
            f"(channels={result.n_channels}, samples={result.n_samples})"
        )


if __name__ == "__main__":
    main()
