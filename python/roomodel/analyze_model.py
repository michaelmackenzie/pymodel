import json
import os
import time
from statistics import NormalDist

import numpy as np

from backends.path_bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path(__file__)

from backends.analysis_common import is_signal_strength_poi, normalize_output_path, resolve_dataset_mode
from backends.analysis_reporting import (
    add_fit_quality,
    add_poi_distributions,
    distribution_summary,
    init_ensemble_report,
    maybe_plot_summary_artifacts,
    print_runtime_summary,
    resolve_output_or_default,
    save_and_print_ensemble_report,
    save_and_print_snapshot,
)
from backends.analysis_console import print_limit_summary_lines
from backends.print_model_helpers import print_model_info
from roomodel.analyze_plotting import plot_summary_artifacts
from roomodel.build_model_from_text import build_and_save_model_from_card_file
from roomodel.model_io import load_fit_model, load_workspace_and_metadata


def _get_root():
    import ROOT

    try:
        if not getattr(ROOT, "_pymodel_roofit_quiet", False):
            ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)
            ROOT._pymodel_roofit_quiet = True
    except Exception:
        pass

    return ROOT


def _iter_roo_collection(collection):
    if collection is None:
        return
    try:
        for obj in collection:
            yield obj
    except TypeError:
        return


def _load_analysis_model(model_file=None, input_card=None):
    if model_file:
        return load_fit_model(model_file)

    if not input_card:
        raise ValueError("Either --model-file or --input-card is required")

    card_path = os.path.abspath(input_card)
    temp_model = os.path.abspath("roomodel_from_card.root")
    build_and_save_model_from_card_file(card_path, temp_model)
    return load_fit_model(temp_model)


def _find_sim_pdf_in_prodpdf(prod_pdf, workspace):
    """Return the RooSimultaneous (or RooProdPdf channel model) embedded inside a
    RooProdPdf constraint wrapper, or *prod_pdf* itself if none is found."""
    try:
        components = prod_pdf.pdfList()
        for comp in components:
            if comp is None:
                continue
            cname = str(comp.ClassName()) if hasattr(comp, "ClassName") else ""
            if "RooSimultaneous" in cname or "RooProdPdf" == cname:
                return comp
    except Exception:
        pass
    return prod_pdf


def _workspace_objects(fit_model):
    ws, metadata = load_workspace_and_metadata(fit_model.model_file)
    model = ws.pdf(fit_model.model_name)
    if model is None or not bool(model):
        raise ValueError(f"Model PDF '{fit_model.model_name}' not found in workspace '{ws.GetName()}'")
    # If the top-level PDF is a constraint wrapper (RooProdPdf named constrainedPdf)
    # unwrap it so that the rest of the code sees the RooSimultaneous/channel PDF
    # for observable resolution, data generation, and CLs scan — while the full
    # constrained PDF is returned separately for fitting.
    model_class = str(model.ClassName()) if hasattr(model, "ClassName") else ""
    if "RooProdPdf" in model_class and fit_model.model_name == "constrainedPdf":
        inner_model = _find_sim_pdf_in_prodpdf(model, ws)
    else:
        inner_model = model
    data = ws.data(fit_model.data_name) if fit_model.data_name else None
    return ws, metadata, model, data, inner_model


def _resolve_poi_var(workspace, fit_model):
    poi_name = fit_model.poi_name
    if poi_name:
        poi = workspace.var(poi_name)
        if poi is not None and bool(poi):
            return poi

    for var in _iter_roo_collection(workspace.allVars()):
        try:
            if not bool(var.isConstant()):
                return var
        except Exception:
            continue
    return None


def _resolve_obs_var(dataset_or_ws, workspace):
    if dataset_or_ws is not None:
        try:
            argset = dataset_or_ws.get()
            for obj in argset:
                if obj.InheritsFrom("RooRealVar"):
                    return workspace.var(obj.GetName())
        except Exception:
            pass

    for var in _iter_roo_collection(workspace.allVars()):
        name = str(var.GetName())
        if name.startswith("obs_") or name.startswith("count_obs"):
            return var

    for var in _iter_roo_collection(workspace.allVars()):
        name = str(var.GetName())
        try:
            is_const = bool(var.isConstant())
        except Exception:
            is_const = True
        if is_const:
            continue
        if is_signal_strength_poi(name) or name.startswith("yield_") or name.startswith("rate_"):
            continue
        return var
    return None


def _unwrap_extended_pdf(pdf):
    """Return the inner shape PDF from a RooExtendPdf, or the PDF itself."""
    try:
        if "RooExtendPdf" in str(pdf.ClassName()):
            for server in pdf.servers():
                if server.InheritsFrom("RooAbsPdf"):
                    return server
    except Exception:
        pass
    return pdf


def _model_observable_set(workspace, model, dataset, obs_var=None):
    ROOT = _get_root()

    if model is None or not bool(model):
        return None

    if dataset is not None:
        try:
            return dataset.get()
        except Exception:
            pass

    model_name = str(model.ClassName()) if hasattr(model, "ClassName") else ""

    # Counting models can expose count observables directly in the workspace.
    # Prefer these explicitly to avoid accidental POI inclusion.
    try:
        count_obs = ROOT.RooArgSet()
        for var in _iter_roo_collection(workspace.allVars()):
            if str(var.GetName()).startswith("count_obs_"):
                count_obs.add(var)
        if count_obs.getSize() > 0:
            return count_obs
    except Exception:
        pass

    # For RooSimultaneous, collect shape observables from each channel's inner PDF.
    # Unwrap RooExtendPdf so yield variables are not included as per-event observables.
    if "RooSimultaneous" in model_name:
        try:
            obs_set = ROOT.RooArgSet()
            index_cat = model.indexCat()
            obs_set.add(index_cat)
            all_vars = workspace.allVars()
            all_obs = model.getObservables(all_vars)
            for obj in _iter_roo_collection(all_obs):
                if not obj.InheritsFrom("RooRealVar"):
                    continue
                name = str(obj.GetName())
                if (
                    is_signal_strength_poi(name)
                    or name.startswith("rate_")
                    or name.startswith("yield_")
                    or name.startswith("sig_")
                    or name.startswith("bkg_")
                    or name.startswith("nuis_")
                    or name.startswith("theta_")
                ):
                    continue
                obs_set.add(obj)
            if obs_set.getSize() > 0:
                return obs_set
        except Exception:
            pass

    # For simple PDFs, unwrap any RooExtendPdf wrapper before querying observables.
    inner_pdf = _unwrap_extended_pdf(model)
    try:
        all_vars = workspace.allVars()
        shape_obs = inner_pdf.getObservables(all_vars)
        if shape_obs is not None and shape_obs.getSize() > 0:
            # Exclude common model parameter names from observables. In composite
            # models, RooFit can report POIs/nuisances in getObservables().
            filtered = ROOT.RooArgSet()
            for obj in shape_obs:
                name = str(obj.GetName())
                if (
                    is_signal_strength_poi(name)
                    or name.startswith("rate_")
                    or name.startswith("yield_")
                    or name.startswith("sig_")
                    or name.startswith("bkg_")
                    or name.startswith("nuis_")
                    or name.startswith("theta_")
                ):
                    continue
                filtered.add(obj)
            if filtered.getSize() > 0:
                return filtered
    except Exception:
        pass

    fallback = ROOT.RooArgSet()
    if obs_var is not None:
        fallback.add(obs_var)
    index_cat = getattr(model, "indexCat", None)
    if callable(index_cat):
        try:
            fallback.add(index_cat())
        except Exception:
            pass
    return fallback if fallback.getSize() > 0 else None


def _capture_nominal_parameter_state(workspace, model, dataset_hint, obs_var=None):
    state = {}

    # Snapshot workspace real vars directly; parameter extraction from
    # RooSimultaneous can be backend-dependent and may return empty sets.
    for obj in _iter_roo_collection(workspace.allVars()):
        try:
            if not obj.InheritsFrom("RooRealVar"):
                continue
            name = str(obj.GetName())
            state[name] = {
                "value": float(obj.getVal()),
                "constant": bool(obj.isConstant()),
            }
        except Exception:
            continue
    return state


def _restore_parameter_state(workspace, state):
    for name, cfg in (state or {}).items():
        var = workspace.var(name)
        if var is None or not bool(var):
            continue
        try:
            var.setVal(float(cfg.get("value")))
        except Exception:
            pass
        try:
            var.setConstant(bool(cfg.get("constant")))
        except Exception:
            pass


def _dataset_plot_payload(dataset, obs_var, fit_mode):
    ROOT = _get_root()
    if dataset is None or obs_var is None:
        return {}

    obs_name = str(obs_var.GetName())
    if fit_mode == "binned":
        try:
            hist = dataset.createHistogram(obs_name)
            axis = hist.GetXaxis()
            n_bins = int(hist.GetNbinsX())
            edges = [axis.GetBinLowEdge(1 + i) for i in range(n_bins)] + [axis.GetBinUpEdge(n_bins)]
            counts = [hist.GetBinContent(1 + i) for i in range(n_bins)]
            return {
                "mode": "binned",
                "obs_name": obs_name,
                "edges": [float(x) for x in edges],
                "counts": [float(x) for x in counts],
            }
        except Exception:
            pass

    values = []
    try:
        n_entries = int(dataset.numEntries())
        for idx in range(n_entries):
            row = dataset.get(idx)
            val_obj = row.find(obs_name)
            if val_obj is not None:
                values.append(float(val_obj.getVal()))
    except Exception:
        values = []

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    hist_edges = []
    hist_counts = []
    if arr.size > 0:
        n_bins = max(10, min(80, int(np.sqrt(arr.size) * 2)))
        counts, edges = np.histogram(arr, bins=n_bins)
        hist_edges = [float(x) for x in edges]
        hist_counts = [float(x) for x in counts]

    return {
        "mode": "unbinned",
        "obs_name": obs_name,
        "values": values,
        "edges": hist_edges,
        "counts": hist_counts,
    }


def _to_binned_data(dataset, obs_var, bins):
    ROOT = _get_root()
    if dataset is None or obs_var is None:
        return dataset

    hist = dataset.createHistogram(
        "tmp_binned_hist",
        obs_var,
        ROOT.RooFit.Binning(int(bins), float(obs_var.getMin()), float(obs_var.getMax())),
    )
    arglist = ROOT.RooArgList(obs_var)
    data_hist = ROOT.RooDataHist("tmp_datahist", "tmp_datahist", arglist, hist)
    return data_hist


def _fit_component_plot_payload(workspace, model, fit_model, fit_data, obs_var, dataset_plot, binned_bins):
    ROOT = _get_root()
    if obs_var is None or fit_data is None:
        return None

    model_name = str(model.ClassName()) if hasattr(model, "ClassName") else ""
    plot_pdf = _unwrap_extended_pdf(model)
    channel_name = None
    if "RooSimultaneous" in model_name:
        try:
            index_cat = model.indexCat()
            channel_name = index_cat.getLabel()
            if not channel_name:
                for state in index_cat:
                    channel_name = str(state.first)
                    break
            if channel_name:
                channel_pdf = model.getPdf(channel_name)
                if channel_pdf is not None:
                    plot_pdf = _unwrap_extended_pdf(channel_pdf)
        except Exception:
            return None
    else:
        # For non-simultaneous models, extract channel name from yield/rate variable naming
        import re
        for var in _iter_roo_collection(workspace.allVars()):
            name = str(var.GetName())
            match = re.search(r'(?:yield|rate)_\w+__(\w+)', name)
            if match:
                channel_name = match.group(1)
                break
        if not channel_name:
            # Try to find from functions
            for func in _iter_roo_collection(workspace.allFunctions()):
                name = str(func.GetName())
                match = re.search(r'(?:yield|rate)_\w+__(\w+)', name)
                if match:
                    channel_name = match.group(1)
                    break

    n_bins = int(binned_bins)
    edges = np.asarray(dataset_plot.get("edges", []), dtype=float)
    if edges.size >= 2:
        n_bins = int(edges.size - 1)
        lo = float(edges[0])
        hi = float(edges[-1])
    else:
        lo = float(obs_var.getMin())
        hi = float(obs_var.getMax())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    n_bins = max(5, min(200, n_bins))
    # Sample more densely than histogram bins for visually smooth component curves.
    n_plot_points = max(300, 6 * n_bins)
    grid = np.linspace(lo, hi, n_plot_points)
    # Keep normalization in data-count units using the effective histogram bin width.
    bin_width = (hi - lo) / n_bins

    process_names = list(dict.fromkeys(fit_model.process_names or []))
    signal_set = set(fit_model.signal_processes or [])

    def _get_yield_obj(name):
        obj = workspace.function(name)
        if obj is None or not bool(obj):
            obj = workspace.var(name)
        return obj if (obj is not None and bool(obj)) else None

    try:
        norm_set = ROOT.RooArgSet(obs_var)
        signal_curve = np.zeros_like(grid, dtype=float)
        background_curve = np.zeros_like(grid, dtype=float)
        total_curve = np.zeros_like(grid, dtype=float)

        for process in process_names:
            if not channel_name:
                continue
            # Try the per-channel renamed PDF name: {process}_{channel}
            term_pdf = workspace.pdf(f"{process}_{channel_name}")
            if term_pdf is None or not bool(term_pdf):
                # Try shape naming convention
                term_pdf = workspace.pdf(f"shape_{process}__{channel_name}")
            if term_pdf is None or not bool(term_pdf):
                # Try just process name
                term_pdf = workspace.pdf(str(process))
            term_yield = _get_yield_obj(f"yield_{process}__{channel_name}")
            if term_yield is None or not bool(term_yield):
                term_yield = _get_yield_obj(f"rate_{process}__{channel_name}")
            if term_pdf is None or not bool(term_pdf) or term_yield is None:
                continue

            yld = float(term_yield.getVal())
            comp_vals = np.zeros_like(grid, dtype=float)
            for idx, x in enumerate(grid):
                obs_var.setVal(float(x))
                comp_vals[idx] = float(term_pdf.getVal(norm_set)) * yld * bin_width

            total_curve += comp_vals
            if process in signal_set:
                signal_curve += comp_vals
            else:
                background_curve += comp_vals

        if np.all(total_curve <= 0.0):
            # Fallback for models without per-process component naming.
            pdf_vals = np.zeros_like(grid, dtype=float)
            for idx, x in enumerate(grid):
                obs_var.setVal(float(x))
                pdf_vals[idx] = float(plot_pdf.getVal(norm_set))
            n_events = float(fit_data.sumEntries()) if hasattr(fit_data, "sumEntries") else float(fit_data.numEntries())
            total_curve = pdf_vals * max(n_events, 1.0) * bin_width
            background_curve = total_curve.copy()
            signal_curve = np.zeros_like(total_curve)
    except Exception:
        return None

    return {
        "obs_name": str(obs_var.GetName()),
        "channel": channel_name,
        "x": [float(x) for x in grid],
        "total": [float(v) for v in total_curve],
        "signal": [float(v) for v in signal_curve],
        "background": [float(v) for v in background_curve],
    }


def _resolve_channel_obs_var(workspace, channel_pdf):
    if channel_pdf is None or not bool(channel_pdf):
        return None

    try:
        obs = channel_pdf.getObservables(workspace.allVars())
    except Exception:
        obs = None

    for obj in _iter_roo_collection(obs):
        try:
            if not obj.InheritsFrom("RooRealVar"):
                continue
            name = str(obj.GetName())
            if (
                is_signal_strength_poi(name)
                or name.startswith("rate_")
                or name.startswith("yield_")
                or name.startswith("sig_")
                or name.startswith("bkg_")
                or name.startswith("nuis_")
                or name.startswith("theta_")
            ):
                continue
            var = workspace.var(name)
            if var is not None and bool(var):
                return var
        except Exception:
            continue
    return None


def _channel_plot_payloads(workspace, model, fit_model, fit_data, fit_mode, binned_bins):
    model_name = str(model.ClassName()) if hasattr(model, "ClassName") else ""
    if "RooSimultaneous" not in model_name:
        return []

    ROOT = _get_root()
    index_cat = model.indexCat()
    cat_name = str(index_cat.GetName())
    payloads = []

    for state in index_cat:
        channel = str(state.first)
        channel_pdf = model.getPdf(channel)
        if channel_pdf is None or not bool(channel_pdf):
            continue

        obs_var = _resolve_channel_obs_var(workspace, channel_pdf)
        if obs_var is None:
            continue

        try:
            channel_data = fit_data.reduce(ROOT.RooFit.Cut(f"{cat_name}=={cat_name}::{channel}"))
        except Exception:
            channel_data = None
        if channel_data is None or not bool(channel_data):
            continue

        ds_payload = _dataset_plot_payload(channel_data, obs_var, fit_mode)
        comp_payload = _fit_component_plot_payload(
            workspace,
            channel_pdf,
            fit_model,
            channel_data,
            obs_var,
            ds_payload,
            int(binned_bins),
        )

        payloads.append(
            {
                "channel": channel,
                "dataset_plot": ds_payload,
                "fit_components": comp_payload,
            }
        )

    return payloads


def _delta_nll_scan_payload(model, fit_data, poi_var, n_points=31, full_range=False, poi_scan_min=None):
    ROOT = _get_root()
    if fit_data is None or poi_var is None or not bool(poi_var):
        return None

    try:
        can_extend = bool(model.canBeExtended())
    except Exception:
        can_extend = False

    nll_opts = [ROOT.RooFit.Offset(True)]
    if can_extend:
        nll_opts.append(ROOT.RooFit.Extended(True))
    try:
        nll = model.createNLL(fit_data, *nll_opts)
    except Exception:
        return None
    if nll is None or not bool(nll):
        return None

    poi_name = str(poi_var.GetName())
    poi_val = float(poi_var.getVal())
    poi_min = float(poi_var.getMin())
    poi_max = float(poi_var.getMax())
    try:
        poi_err = abs(float(poi_var.getError()))
    except Exception:
        poi_err = 0.0

    scan_lo = poi_min
    scan_hi = poi_max
    if (not bool(full_range)) and poi_err > 0.0:
        scan_lo = max(poi_min, poi_val - 4.0 * poi_err)
        scan_hi = min(poi_max, poi_val + 4.0 * poi_err)
    if poi_scan_min is not None and np.isfinite(float(poi_scan_min)):
        scan_lo = max(float(poi_scan_min), scan_lo)
    if scan_hi <= scan_lo:
        scan_lo, scan_hi = poi_min, poi_max
        if poi_scan_min is not None and np.isfinite(float(poi_scan_min)):
            scan_lo = max(float(poi_scan_min), scan_lo)
    if scan_hi <= scan_lo:
        return None

    xs = np.linspace(scan_lo, scan_hi, max(11, int(n_points)))
    y_raw = [float("nan")] * len(xs)
    status = [1] * len(xs)

    has_free_nuisance = False
    param_snapshot = []
    nuisance_vars = []
    try:
        params = model.getParameters(fit_data.get())
        for var in _iter_roo_collection(params):
            if not var.InheritsFrom("RooRealVar"):
                continue
            try:
                param_snapshot.append((var, float(var.getVal()), bool(var.isConstant())))
            except Exception:
                pass
            if str(var.GetName()) == poi_name:
                continue
            if not bool(var.isConstant()):
                nuisance_vars.append(var)
                has_free_nuisance = True
    except Exception:
        has_free_nuisance = True

    old_const = bool(poi_var.isConstant())
    old_val = poi_val
    last_success_nuisance = None
    scan_order = sorted(range(len(xs)), key=lambda i: abs(float(xs[i]) - poi_val))
    try:
        for idx in scan_order:
            x = float(xs[idx])
            # Reset parameters before each scan point so one failed point does
            # not poison subsequent minimizations.
            for var, val0, const0 in param_snapshot:
                try:
                    var.setVal(float(val0))
                except Exception:
                    pass
                try:
                    var.setConstant(bool(const0))
                except Exception:
                    pass

            # Warm start nuisance parameters from the most recent successful
            # profiled point to improve convergence continuity across the scan.
            if last_success_nuisance:
                for var in nuisance_vars:
                    name = str(var.GetName())
                    if name not in last_success_nuisance:
                        continue
                    try:
                        var.setVal(float(last_success_nuisance[name]))
                    except Exception:
                        pass

            poi_var.setVal(float(x))
            poi_var.setConstant(True)
            if has_free_nuisance:
                minim = ROOT.RooMinimizer(nll)
                minim.setPrintLevel(-1)
                minim.setStrategy(0)
                fit_status = int(minim.migrad())
            else:
                fit_status = 0
            status[idx] = fit_status
            try:
                nll_val = float(nll.getVal())
            except Exception:
                nll_val = float("nan")
            y_raw[idx] = nll_val
            if fit_status == 0 and np.isfinite(nll_val):
                snapshot = {}
                for var in nuisance_vars:
                    try:
                        snapshot[str(var.GetName())] = float(var.getVal())
                    except Exception:
                        pass
                if snapshot:
                    last_success_nuisance = snapshot
    except Exception:
        return None
    finally:
        for var, val0, const0 in param_snapshot:
            try:
                var.setVal(float(val0))
            except Exception:
                pass
            try:
                var.setConstant(bool(const0))
            except Exception:
                pass
        poi_var.setVal(old_val)
        poi_var.setConstant(old_const)

    if not y_raw:
        return None

    finite = np.asarray(y_raw, dtype=float)
    mask = np.isfinite(finite)
    if not np.any(mask):
        return None
    y_min = float(np.min(finite[mask]))
    y = [float(v - y_min) if np.isfinite(v) else float("nan") for v in y_raw]
    x_out = []
    y_out = []
    status_out = []
    for xv, yv, sv in zip(xs, y, status):
        if not np.isfinite(yv):
            continue
        x_out.append(float(xv))
        y_out.append(float(yv))
        status_out.append(int(sv))
    if not y_out:
        return None
    return {
        "poi_name": poi_name,
        "x": x_out,
        "delta_nll": y_out,
        "status": status_out,
    }


def _interp_first_crossing(x_vals, y_vals, threshold):
    if not x_vals or not y_vals or len(x_vals) != len(y_vals):
        return None
    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return None
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return None

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    for i in range(1, len(x)):
        x0 = float(x[i - 1])
        x1 = float(x[i])
        y0 = float(y[i - 1])
        y1 = float(y[i])
        if y0 > threshold and y1 <= threshold:
            if y1 == y0:
                return x1
            t = (threshold - y0) / (y1 - y0)
            return float(x0 + t * (x1 - x0))
    return None


def _compute_asimov_sigma(workspace, model, inner_model, fit_model, obs_var, fit_mode, binned_bins):
    """Fit the mu=0 Asimov dataset and return the POI uncertainty sigma_A.

    This gives the sensitivity measure used in the full asymptotic CLs formula
    (Cowan et al. 2011): q_mu_A = (mu / sigma_A)^2.

    Returns sigma_A (float > 0) or None on failure.
    """
    ROOT = _get_root()

    # Build mu=0 Asimov dataset from inner_model (unwrapped from constraint wrapper)
    poi_var = _resolve_poi_var(workspace, fit_model)
    if poi_var is None or not bool(poi_var):
        return None

    # Snapshot current state
    nominal_state = _capture_nominal_parameter_state(workspace, model, None, obs_var=obs_var)

    try:
        inner_model_name = str(inner_model.ClassName()) if hasattr(inner_model, "ClassName") else ""
        # Try counting path first — works for RooProdPdf of Poisson channels.
        # Pass poi_var + poi_val=0 so the function sets mu=0 internally and
        # restores it, keeping the workspace consistent afterwards.
        asimov = _build_asimov_for_counting(workspace, poi_var=poi_var, poi_val=0.0)
        if asimov is None or not bool(asimov):
            if "RooSimultaneous" in inner_model_name:
                asimov = _build_asimov_for_simultaneous(workspace, inner_model)
                if asimov is None or not bool(asimov):
                    obs_set = _model_observable_set(workspace, inner_model, None, obs_var=obs_var)
                    if obs_set is None:
                        return None
                    asimov = inner_model.generateBinned(obs_set, ROOT.RooFit.ExpectedData(True))
            else:
                if obs_var is None:
                    return None
                asimov = inner_model.generateBinned(
                    ROOT.RooArgSet(obs_var),
                    ROOT.RooFit.Binning(int(binned_bins), float(obs_var.getMin()), float(obs_var.getMax())),
                    ROOT.RooFit.ExpectedData(True),
                )

        if asimov is None or not bool(asimov):
            return None

        # Restore state then free-fit the Asimov to get sigma_A
        _restore_parameter_state(workspace, nominal_state)
        poi_var.setConstant(False)

        can_extend = False
        try:
            can_extend = bool(model.canBeExtended())
        except Exception:
            pass

        fit_opts = [ROOT.RooFit.Save(True), ROOT.RooFit.PrintLevel(-1), ROOT.RooFit.Strategy(0)]
        if can_extend:
            fit_opts.append(ROOT.RooFit.Extended(True))

        res = model.fitTo(asimov, *fit_opts)
        if res is None or not bool(res):
            return None

        sigma_a = None
        try:
            sigma_a = float(poi_var.getError())
        except Exception:
            pass

        if sigma_a is None or not np.isfinite(sigma_a) or sigma_a <= 0.0:
            return None

        return sigma_a

    except Exception:
        return None
    finally:
        _restore_parameter_state(workspace, nominal_state)


def _apply_cls_summary_from_scan(summary, alpha, poi_min_limit=0.0, sigma_asimov=None):
    scan = summary.get("delta_nll_scan") or {}
    x = np.asarray(scan.get("x", []), dtype=float)
    y = np.asarray(scan.get("delta_nll", []), dtype=float)
    if x.size == 0 or y.size != x.size:
        summary["cls_error"] = "missing delta_nll_scan"
        return

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        summary["cls_error"] = "empty delta_nll_scan"
        return

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # One-sided asymptotic CLs (Cowan et al. 2011, eqs. 12-13).
    #
    # Test statistic: q_mu = max(0, 2*DeltaNLL)
    #
    # Full formula using the Asimov sensitivity q_mu_A = (mu/sigma_A)^2:
    #   p_sb = 1 - Phi(sqrt(q_mu))
    #   p_b  = 1 - Phi(sqrt(q_mu) - sqrt(q_mu_A))
    #   CLs  = p_sb / p_b
    #
    # When sigma_asimov is unavailable we fall back to the simplified form
    #   p_b = Phi(sqrt(q_mu))
    # which assumes sqrt(q_mu_A) = sqrt(q_mu), i.e. the data equals the Asimov.
    # The full formula gives the correct expected CLs band and a more accurate
    # observed limit when the data deviates from the background expectation.
    q_mu = np.clip(2.0 * y, 0.0, None)
    sqrt_q_mu = np.sqrt(q_mu)
    norm = NormalDist()
    p_sb = np.asarray([1.0 - norm.cdf(float(v)) for v in sqrt_q_mu], dtype=float)

    if sigma_asimov is not None and np.isfinite(float(sigma_asimov)) and float(sigma_asimov) > 0.0:
        # q_mu_A(mu) = (mu / sigma_A)^2 varies across the scan
        sqrt_q_mu_A = x / float(sigma_asimov)   # = mu / sigma_A, element-wise
        p_b = np.asarray(
            [1.0 - norm.cdf(float(sq) - float(sa))
             for sq, sa in zip(sqrt_q_mu, sqrt_q_mu_A)],
            dtype=float,
        )
    else:
        # Simplified fallback: p_b = Phi(sqrt(q_mu))
        p_b = np.asarray([norm.cdf(float(v)) for v in sqrt_q_mu], dtype=float)

    # Protect against p_b == 0; CLs -> 0 there.
    cls_obs = np.where(p_b > 0.0, p_sb / p_b, 0.0)

    pos = x >= float(poi_min_limit)
    x_pos = x[pos]
    cls_pos = cls_obs[pos]
    limit = _interp_first_crossing(x_pos.tolist(), cls_pos.tolist(), float(alpha))
    if limit is None and x_pos.size > 0:
        if float(cls_pos[-1]) <= float(alpha):
            limit = float(x_pos[-1])

    summary["cls_alpha"] = float(alpha)
    summary["cls_scan_points"] = int(x.size)
    summary["cls_scan_max"] = float(np.max(x))
    if sigma_asimov is not None and np.isfinite(float(sigma_asimov)) and float(sigma_asimov) > 0.0:
        summary["cls_sigma_asimov"] = float(sigma_asimov)
    summary["cls_curve"] = {
        "pois": [float(v) for v in x.tolist()],
        "observed": [float(v) for v in cls_obs.tolist()],
        "expected_median": [float("nan")] * int(x.size),
        "expected_band": [],
    }
    if limit is not None and np.isfinite(limit):
        summary["cls_observed"] = float(limit)
        summary["yield_upper_limit"] = float(limit)
    else:
        summary["cls_error"] = "no-threshold-crossing"


def _q_mu_observed(summary):
    """Return the observed test statistic q_mu_obs = 2*DeltaNLL at each scan point.

    Returns (x, q_obs) arrays, or (None, None) if the scan is unavailable.
    """
    scan = summary.get("delta_nll_scan") or {}
    x = np.asarray(scan.get("x", []), dtype=float)
    y = np.asarray(scan.get("delta_nll", []), dtype=float)
    if x.size == 0 or y.size != x.size:
        return None, None
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        return None, None
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    q_obs = np.clip(2.0 * y, 0.0, None)
    return x, q_obs


def _compute_q_mu_for_dataset(workspace, model, inner_model, data, fit_model,
                               poi_var, mu_test, fit_mode, binned_bins, obs_var):
    """Fit *data* with POI free and POI fixed at *mu_test*; return q_mu = max(0, 2*DNLL).

    Returns float or NaN on failure.
    """
    ROOT = _get_root()

    can_extend = False
    try:
        can_extend = bool(model.canBeExtended())
    except Exception:
        pass

    fit_opts_free = [ROOT.RooFit.Save(True), ROOT.RooFit.PrintLevel(-1), ROOT.RooFit.Strategy(0)]
    if can_extend:
        fit_opts_free.append(ROOT.RooFit.Extended(True))

    old_const = bool(poi_var.isConstant())
    old_val = float(poi_var.getVal())

    try:
        # Free fit
        poi_var.setConstant(False)
        res_free = model.fitTo(data, *fit_opts_free)
        if res_free is None or not bool(res_free):
            return float("nan")
        nll_free = float(res_free.minNll())
        mu_hat = float(poi_var.getVal())

        # One-sided: q_mu = 0 when mu_hat > mu_test
        if mu_hat > float(mu_test):
            return 0.0

        # Fixed-POI fit
        poi_var.setVal(float(mu_test))
        poi_var.setConstant(True)
        fit_opts_fix = [ROOT.RooFit.Save(True), ROOT.RooFit.PrintLevel(-1), ROOT.RooFit.Strategy(0)]
        if can_extend:
            fit_opts_fix.append(ROOT.RooFit.Extended(True))
        res_fix = model.fitTo(data, *fit_opts_fix)
        if res_fix is None or not bool(res_fix):
            return float("nan")
        nll_fix = float(res_fix.minNll())

        q = 2.0 * (nll_fix - nll_free)
        return max(0.0, float(q)) if np.isfinite(q) else float("nan")

    except Exception:
        return float("nan")
    finally:
        poi_var.setVal(old_val)
        poi_var.setConstant(old_const)


def _apply_cls_summary_from_toys(
    summary, alpha, workspace, model, inner_model, fit_model,
    n_toys_per_point, obs_data, fit_mode, binned_bins, obs_var,
    poi_min_limit=0.0, seed=1234,
):
    """Compute CLs via explicit toy generation at each scan point.

    For each POI grid point mu_test:
      1. Compute q_mu_obs on the actual (observed/toy) dataset.
      2. Generate n_toys_per_point toys under H1 (mu=0, b-only) and under
         H_test (mu=mu_test, s+b) and fit each to get a distribution of q_mu.
      3. p_sb = fraction of s+b toys with q >= q_obs
         p_b  = fraction of b-only toys with q >= q_obs
         CLs  = p_sb / p_b  (with p_b clamped away from 0)

    The grid is taken from the existing delta_nll_scan so scan range and
    number of points are consistent with the asymptotic path.
    """
    x_scan, q_obs_scan = _q_mu_observed(summary)
    if x_scan is None:
        summary["cls_error"] = "missing delta_nll_scan for toy CLs"
        return

    rng = np.random.default_rng(int(seed))
    poi_var = _resolve_poi_var(workspace, fit_model)
    if poi_var is None or not bool(poi_var):
        summary["cls_error"] = "could not resolve POI for toy CLs"
        return

    nominal_state = _capture_nominal_parameter_state(workspace, model, obs_data, obs_var=obs_var)

    cls_vals = []
    p_sb_vals = []
    p_b_vals = []

    for mu_test, q_obs in zip(x_scan.tolist(), q_obs_scan.tolist()):
        # --- generate b-only toys (mu=0) ---
        _restore_parameter_state(workspace, nominal_state)
        poi_var.setVal(0.0)
        q_bonly = []
        for _ in range(int(n_toys_per_point)):
            try:
                toy = _generate_dataset(workspace, inner_model, obs_data, obs_var, fit_mode, binned_bins)
                if toy is None or not bool(toy):
                    continue
                _restore_parameter_state(workspace, nominal_state)
                poi_var.setVal(0.0)
                q = _compute_q_mu_for_dataset(
                    workspace, model, inner_model, toy, fit_model,
                    poi_var, mu_test, fit_mode, binned_bins, obs_var,
                )
                _restore_parameter_state(workspace, nominal_state)
                poi_var.setVal(0.0)
                if np.isfinite(q):
                    q_bonly.append(float(q))
            except Exception:
                pass

        # --- generate s+b toys (mu=mu_test) ---
        _restore_parameter_state(workspace, nominal_state)
        poi_var.setVal(float(mu_test))
        q_splusb = []
        for _ in range(int(n_toys_per_point)):
            try:
                toy = _generate_dataset(workspace, inner_model, obs_data, obs_var, fit_mode, binned_bins)
                if toy is None or not bool(toy):
                    continue
                _restore_parameter_state(workspace, nominal_state)
                poi_var.setVal(float(mu_test))
                q = _compute_q_mu_for_dataset(
                    workspace, model, inner_model, toy, fit_model,
                    poi_var, mu_test, fit_mode, binned_bins, obs_var,
                )
                _restore_parameter_state(workspace, nominal_state)
                poi_var.setVal(float(mu_test))
                if np.isfinite(q):
                    q_splusb.append(float(q))
            except Exception:
                pass

        # --- compute p-values ---
        n_b = len(q_bonly)
        n_sb = len(q_splusb)
        if n_b == 0 or n_sb == 0:
            cls_vals.append(float("nan"))
            p_sb_vals.append(float("nan"))
            p_b_vals.append(float("nan"))
            continue

        p_sb = float(np.sum(np.asarray(q_splusb) >= float(q_obs))) / n_sb
        p_b  = float(np.sum(np.asarray(q_bonly)  >= float(q_obs))) / n_b
        # Clamp p_b away from 0 to avoid division by zero; if no b-only toys
        # exceeded q_obs, CLs -> 0.
        cls = (p_sb / p_b) if p_b > 0.0 else 0.0

        cls_vals.append(float(cls))
        p_sb_vals.append(float(p_sb))
        p_b_vals.append(float(p_b))

    _restore_parameter_state(workspace, nominal_state)

    x_arr = x_scan
    cls_arr = np.asarray(cls_vals, dtype=float)

    pos = x_arr >= float(poi_min_limit)
    x_pos = x_arr[pos]
    cls_pos = cls_arr[pos]
    limit = _interp_first_crossing(x_pos.tolist(), cls_pos.tolist(), float(alpha))
    if limit is None and x_pos.size > 0:
        if np.isfinite(cls_pos[-1]) and float(cls_pos[-1]) <= float(alpha):
            limit = float(x_pos[-1])

    summary["cls_alpha"] = float(alpha)
    summary["cls_scan_points"] = int(x_arr.size)
    summary["cls_scan_max"] = float(np.max(x_arr))
    summary["cls_toys_per_point"] = int(n_toys_per_point)
    summary["cls_curve"] = {
        "pois": [float(v) for v in x_arr.tolist()],
        "observed": [float(v) if np.isfinite(v) else None for v in cls_arr.tolist()],
        "p_sb": [float(v) if np.isfinite(v) else None for v in p_sb_vals],
        "p_b":  [float(v) if np.isfinite(v) else None for v in p_b_vals],
        "expected_median": [float("nan")] * int(x_arr.size),
        "expected_band": [],
    }
    if limit is not None and np.isfinite(limit):
        summary["cls_observed"] = float(limit)
        summary["yield_upper_limit"] = float(limit)
    else:
        summary["cls_error"] = "no-threshold-crossing (toys)"


def _apply_feldman_cousins_summary_from_scan(summary, alpha, scan_points, n_toys, poi_min_limit=0.0):
    """Apply Feldman-Cousins confidence interval using the likelihood-ratio method.
    
    This is now a fallback for when a true Neyman-construction FC is not available.
    For the true FC with toys, use _apply_feldman_cousins_true() which requires
    access to the model and workspace.
    """
    from backends.analysis_common import compute_likelihood_interval
    
    scan = summary.get("delta_nll_scan") or {}
    x = np.asarray(scan.get("x", []), dtype=float)
    y = np.asarray(scan.get("delta_nll", []), dtype=float)
    if x.size == 0 or y.size != x.size:
        summary["feldman_cousins"] = {
            "fc_status": "failed: missing delta_nll_scan",
            "alpha": float(alpha),
        }
        return

    interval = compute_likelihood_interval(x, y, float(alpha), poi_min_limit=float(poi_min_limit))
    
    # Compute q_obs for reporting
    mask = np.isfinite(x) & np.isfinite(y)
    x_finite = x[mask]
    y_finite = y[mask]
    q_obs = np.asarray(np.clip(2.0 * y_finite, 0.0, None), dtype=float)
    
    # Compute q_crit
    z_crit = NormalDist().inv_cdf(1.0 - 0.5 * float(alpha))
    q_crit = float(z_crit * z_crit)
    
    summary["feldman_cousins"] = {
        "fc_status": "ok" if interval is not None else "no-accepted-points",
        "alpha": float(alpha),
        "fc_interval": interval,
        "scan_points": int(scan_points),
        "scan_max": float(np.max(x_finite)) if x_finite.size > 0 else float("nan"),
        "n_toys_per_point": int(n_toys),
        "grid": {
            "poi": [float(v) for v in x_finite.tolist()],
            "q_obs": [float(v) for v in q_obs.tolist()],
            "q_crit": [float(q_crit)] * int(x_finite.size),
            "toy_valid": [int(n_toys)] * int(x_finite.size),
        },
        "note": "likelihood-ratio asymptotic method (not true Feldman-Cousins)",
    }


def _apply_feldman_cousins_true(summary, alpha, scan_points, n_toys, poi_min_limit, 
                                workspace, model, data, fit_model, dataset_id=0, seed=1234):
    """Apply true Feldman-Cousins Neyman construction with toy generation per grid point.
    
    This uses the compute_feldman_cousins() algorithm from analysis_common.py which
    generates toys at each POI value and derives per-point critical values.
    
    Parameters
    ----------
    summary : dict
        Analysis summary to update with FC results.
    alpha : float
        Significance level (e.g., 0.05 for 95% CL).
    scan_points : int
        Number of grid points for the FC scan.
    n_toys : int
        Number of toy datasets per grid point.
    poi_min_limit : float
        Minimum POI value to consider.
    workspace : RooWorkspace
        The workspace containing the model and data.
    model : RooPdf
        The model PDF to use for fits and toy generation.
    data : RooDataset or RooDataHist
        The observed dataset (iteration dataset, not necessarily observed).
    fit_model : roomodel.FitModel
        Metadata about the fit model.
    dataset_id : int
        Seed component for reproducibility.
    seed : int
        Random seed.
    """
    from backends.path_bootstrap import ensure_repo_root_on_path
    ensure_repo_root_on_path(__file__)
    
    from backends.analysis_common import compute_feldman_cousins
    from backends.roomodel.analysis_backend import RooFitAnalysisBackend, RooFitAnalysisState
    
    try:
        # Resolve POI and other required objects
        poi = _resolve_poi_var(workspace, fit_model)
        if poi is None or not bool(poi):
            summary["feldman_cousins"] = {
                "fc_status": "failed: could not resolve POI",
                "alpha": float(alpha),
            }
            return
        
        poi_name = str(poi.GetName())
        
        # Create the backend adapter
        backend = RooFitAnalysisBackend(
            workspace=workspace,
            model=model,
            inner_model=model,
            poi=poi,
            poi_name=poi_name,
            fit_model=fit_model,
            observed_data=data,
        )
        
        # Create the state object with current data set to the iteration dataset
        state = RooFitAnalysisState(
            workspace=workspace,
            model=model,
            poi=poi,
            poi_name=poi_name,
            fit_model=fit_model,
            rng=np.random.default_rng(int(seed) + int(dataset_id)),
            current_data=data,
            observed_data=data,
        )
        
        # Compute Feldman-Cousins interval with toys
        fc_result = compute_feldman_cousins(
            backend=backend,
            state=state,
            alpha=float(alpha),
            scan_points=int(scan_points),
            n_toys=int(n_toys),
            scan_max=float(summary.get("delta_nll_scan", {}).get("x", [float("nan")])[-1]) or 5.0,
            dataset_id=dataset_id,
            seed=seed,
        )
        
        # Convert FCResult to summary dict
        interval = fc_result.interval
        grid = fc_result.grid or {}
        
        summary["feldman_cousins"] = {
            "fc_status": fc_result.status,
            "alpha": float(alpha),
            "fc_interval": list(interval) if interval is not None else None,
            "scan_points": fc_result.scan_points,
            "scan_max": fc_result.scan_max,
            "n_toys_per_point": fc_result.n_toys,
            "grid": {
                "poi": grid.get("poi", []),
                "q_obs": grid.get("q_obs", []),
                "q_crit": grid.get("q_crit", []),
                "p_obs": grid.get("p_obs", []),
                "toy_valid": grid.get("toy_valid", []),
            },
            "note": "true Feldman-Cousins Neyman construction with toys",
        }
    except Exception as exc:
        summary["feldman_cousins"] = {
            "fc_status": f"failed: {str(exc)[:100]}",
            "alpha": float(alpha),
        }


def _extract_fit_params(fit_result, workspace):
    """Extract floating parameter values and uncertainties from RooFit fit result.
    
    Parameters
    ----------
    fit_result : RooFitResult
        The fit result object from model.fitTo()
    workspace : RooWorkspace
        The RooWorkspace containing the parameters
        
    Returns
    -------
    tuple of (dict, dict)
        (param_values, param_uncertainties) with all floating parameters
    """
    ROOT = _get_root()
    param_values = {}
    param_uncertainties = {}
    
    if fit_result is None or not bool(fit_result):
        return param_values, param_uncertainties
    
    try:
        # Get the correlation matrix to access all parameters
        params = fit_result.floatParsFinal()
        if params is not None:
            for i in range(params.getSize()):
                var = params.at(i)
                if var is not None:
                    name = str(var.GetName())
                    try:
                        value = float(var.getVal())
                        param_values[name] = value
                    except Exception:
                        pass
                    try:
                        error = float(var.getError())
                        if error > 0:
                            param_uncertainties[name] = error
                    except Exception:
                        pass
    except Exception:
        pass
    
    return param_values, param_uncertainties


def _run_single_fit(
    workspace,
    model,
    data,
    fit_model,
    fit_mode,
    binned_bins,
    input_obs_var=None,
    collect_plot_details=False,
    limit_scan_points=None,
    limit_scan_full_range=False,
    limit_scan_poi_min=None,
    inner_model=None,
):
    ROOT = _get_root()
    obs = _resolve_obs_var(data, workspace) or input_obs_var
    fit_data = data
    is_datahist = False
    # Use inner_model (unwrapped from constraint wrapper) for class detection;
    # the constrained model (model) is used for fitting below.
    _detect_model = inner_model if inner_model is not None else model
    model_name = str(_detect_model.ClassName()) if hasattr(_detect_model, "ClassName") else ""
    is_simultaneous = "RooSimultaneous" in model_name

    if data is not None:
        try:
            is_datahist = bool(data) and bool(data.InheritsFrom("RooDataHist"))
        except Exception:
            is_datahist = False

    if fit_mode == "binned" and data is not None and bool(data) and not is_datahist and not is_simultaneous:
        fit_data = _to_binned_data(data, obs, binned_bins)

    if fit_data is None or not bool(fit_data):
        raise ValueError("No dataset available for fitting")

    can_extend = False
    try:
        can_extend = bool(model.canBeExtended())
    except Exception:
        can_extend = False

    recover_from_undef = getattr(ROOT.RooFit, "RecoverFromUndefinedRegions", None)
    fit_result = None
    best_result = None
    best_score = None
    for strategy in (0, 1, 2):
        fit_opts = [
            ROOT.RooFit.Save(True),
            ROOT.RooFit.PrintLevel(-1),
            ROOT.RooFit.Strategy(int(strategy)),
        ]
        if can_extend:
            fit_opts.append(ROOT.RooFit.Extended(True))
        if callable(recover_from_undef):
            try:
                fit_opts.append(recover_from_undef(10.0))
            except Exception:
                pass
        try:
            trial = model.fitTo(fit_data, *fit_opts)
        except Exception:
            trial = None
        if trial is None or not bool(trial):
            continue
        status_trial = int(trial.status())
        cov_trial = int(trial.covQual())
        score = (status_trial, -cov_trial)
        if best_result is None or score < best_score:
            best_result = trial
            best_score = score
        if status_trial == 0 and cov_trial >= 2:
            fit_result = trial
            break

    if fit_result is None:
        fit_result = best_result

    status = int(fit_result.status()) if (fit_result is not None and bool(fit_result)) else 1
    cov_qual = int(fit_result.covQual()) if (fit_result is not None and bool(fit_result)) else -1

    poi = _resolve_poi_var(workspace, fit_model)
    poi_fit = None
    poi_unc = None
    if poi is not None and bool(poi):
        poi_fit = float(poi.getVal())
        try:
            poi_unc = float(poi.getError())
        except Exception:
            poi_unc = None

    valid = (status == 0) and (cov_qual >= 2)
    pull = None
    truth = None
    if poi_fit is not None and poi_unc is not None and poi_unc > 0.0:
        truth = float(1.0)
        pull = float((poi_fit - truth) / poi_unc)

    nll_val = None
    if fit_result is not None and bool(fit_result):
        try:
            nll_val = float(fit_result.minNll())
        except Exception:
            nll_val = None

    dataset_plot = _dataset_plot_payload(fit_data, obs, fit_mode)
    fit_components = None
    channel_plots = None
    delta_nll_scan = None
    need_scan = bool(collect_plot_details) or (limit_scan_points is not None and int(limit_scan_points) > 0)
    if collect_plot_details:
        fit_components = _fit_component_plot_payload(
            workspace,
            _detect_model,
            fit_model,
            fit_data,
            obs,
            dataset_plot,
            int(binned_bins),
        )
        channel_plots = _channel_plot_payloads(
            workspace,
            _detect_model,
            fit_model,
            fit_data,
            fit_mode,
            int(binned_bins),
        )
        if channel_plots:
            dataset_plot = channel_plots[0].get("dataset_plot") or dataset_plot
            fit_components = channel_plots[0].get("fit_components") or fit_components
    if need_scan:
        scan_points = int(limit_scan_points) if (limit_scan_points is not None and int(limit_scan_points) > 0) else 31
        delta_nll_scan = _delta_nll_scan_payload(
            model,  # constrained model for NLL scan (correct likelihood)
            fit_data,
            poi,
            n_points=scan_points,
            full_range=bool(limit_scan_full_range),
            poi_scan_min=limit_scan_poi_min,
        )

    # Extract floating parameter values and uncertainties from fit result
    fit_params, fit_param_unc = _extract_fit_params(fit_result, workspace)

    return {
        "valid": bool(valid),
        "fit_status": status,
        "hesse_status": cov_qual,
        "poi_name": fit_model.poi_name,
        "poi_fit": poi_fit,
        "poi_unc_hesse": poi_unc,
        "poi_true": truth,
        "poi_pull": pull,
        "nll": nll_val,
        "dataset_plot": dataset_plot,
        "fit_components": fit_components,
        "channel_plots": channel_plots,
        "delta_nll_scan": delta_nll_scan,
        "fit_params": fit_params,
        "fit_param_unc": fit_param_unc,
    }


def _build_asimov_for_counting(workspace, poi_var=None, poi_val=None):
    """Build an Asimov dataset for a counting experiment.

    For counting models the observables are ``count_obs_<channel>``
    ``RooRealVar`` variables.  The Asimov dataset sets each channel's observed
    count to the **continuous expected yield** (``yield_total__<channel>``)
    evaluated at the requested parameter point — no Poisson smearing is
    applied.

    Parameters
    ----------
    workspace:
        The RooWorkspace containing the model.
    poi_var:
        Optional ``RooRealVar`` for the signal-strength POI.  When provided
        together with *poi_val*, the POI is temporarily set to *poi_val*
        before evaluating yields and restored afterwards.  This allows the
        caller to request a background-only (``poi_val=0``) or signal-plus-
        background Asimov without modifying the workspace persistently.
    poi_val:
        Float value to assign to *poi_var* during yield evaluation.  Ignored
        when *poi_var* is ``None``.

    Returns a single-entry ``RooDataSet`` with all count observables set to
    their expected (continuous) values, or ``None`` if no ``count_obs_*``
    variables exist in the workspace.
    """
    ROOT = _get_root()
    count_vars = []
    for var in _iter_roo_collection(workspace.allVars()):
        if str(var.GetName()).startswith("count_obs_"):
            count_vars.append(var)
    if not count_vars:
        return None

    # Temporarily fix POI at the requested value for yield evaluation.
    saved_poi_val = None
    saved_poi_const = None
    if poi_var is not None and poi_val is not None:
        try:
            saved_poi_val = float(poi_var.getVal())
            saved_poi_const = bool(poi_var.isConstant())
            poi_var.setVal(float(poi_val))
            poi_var.setConstant(True)
        except Exception:
            saved_poi_val = None

    try:
        count_obs_set = ROOT.RooArgSet()
        for var in count_vars:
            count_obs_set.add(var)

        for var in count_vars:
            name = str(var.GetName())
            channel = name.split("count_obs_", 1)[1] if "count_obs_" in name else ""
            mean_obj = workspace.function(f"yield_total__{channel}")
            if mean_obj is None or not bool(mean_obj):
                mean_obj = workspace.var(f"yield_total__{channel}")
            if mean_obj is not None and bool(mean_obj):
                # Use the continuous expected yield directly — RooPoisson is
                # constructed with setNoRounding(True) so non-integer n values
                # are evaluated via the gamma-function generalisation.
                var.setVal(max(float(mean_obj.getVal()), 0.0))
            # If the yield function is not found, leave the variable at its
            # current value (set from the card observation row at build time).

        asimov = ROOT.RooDataSet("asimov_count_data", "asimov_count_data", count_obs_set)
        asimov.add(count_obs_set)
        return asimov

    finally:
        # Restore POI to its state before we changed it.
        if saved_poi_val is not None and poi_var is not None:
            try:
                poi_var.setVal(saved_poi_val)
                poi_var.setConstant(saved_poi_const)
            except Exception:
                pass


def _generate_dataset(workspace, model, dataset_hint, obs_var, fit_mode, binned_bins):
    ROOT = _get_root()
    model_name = str(model.ClassName()) if hasattr(model, "ClassName") else ""

    if "RooSimultaneous" in model_name:
        toy_sim = _build_toy_for_simultaneous(workspace, model, fit_mode)
        if toy_sim is not None and bool(toy_sim):
            return toy_sim

    # Counting-mode fallback: build one-event datasets with explicit Poisson
    # draws for each count observable. This avoids RooFit generation paths that
    # can drop dimensions or return degenerate toys for composite count models.
    count_vars = []
    for var in _iter_roo_collection(workspace.allVars()):
        if str(var.GetName()).startswith("count_obs_"):
            count_vars.append(var)
    if count_vars:
        count_obs_set = ROOT.RooArgSet()
        for var in count_vars:
            count_obs_set.add(var)

        rng = np.random.default_rng()
        for var in count_vars:
            name = str(var.GetName())
            channel = name.split("count_obs_", 1)[1] if "count_obs_" in name else ""
            mean_obj = workspace.function(f"yield_total__{channel}")
            if mean_obj is None or not bool(mean_obj):
                mean_obj = workspace.var(f"yield_total__{channel}")
            if mean_obj is None or not bool(mean_obj):
                mean_val = max(float(var.getVal()), 0.0)
            else:
                mean_val = max(float(mean_obj.getVal()), 0.0)
            var.setVal(float(rng.poisson(mean_val)))

        toy = ROOT.RooDataSet("toy_count_data", "toy_count_data", count_obs_set)
        toy.add(count_obs_set)
        return toy

    resolved_obs_set = _model_observable_set(workspace, model, dataset_hint, obs_var=obs_var)
    if resolved_obs_set is None:
        raise ValueError("Could not resolve observables for toy generation")

    # For multi-observable models (e.g. channel products), always use the full
    # observable set. For single-observable models, keep legacy behavior so the
    # requested plotting observable remains the generation target.
    if obs_var is not None:
        try:
            n_obs = int(resolved_obs_set.getSize())
        except Exception:
            n_obs = 0
        if n_obs <= 1:
            obs_set = ROOT.RooArgSet(obs_var)
        else:
            obs_set = resolved_obs_set
    else:
        obs_set = resolved_obs_set

    # For RooSimultaneous, ensure the index category is present in the obs set
    if "RooSimultaneous" in model_name:
        index_cat_fn = getattr(model, "indexCat", None)
        if callable(index_cat_fn):
            try:
                cat = index_cat_fn()
                if cat is not None and not obs_set.contains(cat):
                    obs_set = ROOT.RooArgSet(obs_set)
                    obs_set.add(cat)
            except Exception:
                pass

    can_extend = False
    try:
        can_extend = bool(model.canBeExtended())
    except Exception:
        pass

    # RooSimultaneous does not accept Extended(True) or Binning(...) as command
    # args in either generate() or generateBinned() in ROOT 6.32 — they are
    # rejected as "unrecognized command: BinningSpec".  Only pass Extended(True)
    # to simple (non-Simultaneous) PDFs.
    is_simultaneous = "RooSimultaneous" in model_name
    gen_ext = [] if is_simultaneous else ([ROOT.RooFit.Extended(True)] if can_extend else [])

    if fit_mode == "binned":
        generated = None
        try:
            if is_simultaneous:
                generated = model.generateBinned(obs_set)
            elif obs_var is not None:
                generated = model.generateBinned(
                    obs_set,
                    ROOT.RooFit.Binning(int(binned_bins), float(obs_var.getMin()), float(obs_var.getMax())),
                    *gen_ext,
                )
            else:
                generated = model.generateBinned(obs_set, *gen_ext)
        except Exception:
            generated = None

        if generated is not None and bool(generated):
            return generated

    generated = model.generate(obs_set, *gen_ext)
    if generated is None or not bool(generated):
        raise ValueError("Toy generation failed for roomodel")
    return generated


def _build_toy_for_simultaneous(workspace, model, fit_mode):
    ROOT = _get_root()

    try:
        index_cat = model.indexCat()
    except Exception:
        return None

    obs_names = set()
    obs_set_union = ROOT.RooArgSet()
    channel_data = {}

    for state in index_cat:
        ch_name = str(state.first)
        ch_pdf = model.getPdf(ch_name)
        if ch_pdf is None or not bool(ch_pdf):
            continue

        obs_var = _resolve_channel_obs_var(workspace, ch_pdf)
        if obs_var is None:
            continue

        obs_name = str(obs_var.GetName())
        if obs_name not in obs_names:
            obs_names.add(obs_name)
            obs_set_union.add(obs_var)

        ch_obs_set = ROOT.RooArgSet(obs_var)
        can_extend = False
        try:
            can_extend = bool(ch_pdf.canBeExtended())
        except Exception:
            pass

        toy_ch = None
        try:
            # Do not pass Extended(True) to RooExtendPdf or RooSimultaneous channel
            # PDFs — ROOT 6.32 rejects it with "unrecognized command: BinningSpec"
            # for both generate() and generateBinned().  Extended generation is
            # handled automatically when the PDF has an extended term.
            if fit_mode == "binned":
                toy_ch = ch_pdf.generateBinned(ch_obs_set)
            else:
                toy_ch = ch_pdf.generate(ch_obs_set)
        except Exception:
            toy_ch = None

        if toy_ch is not None and bool(toy_ch):
            channel_data[ch_name] = toy_ch

    if obs_set_union.getSize() == 0 or not channel_data:
        return None

    cmd_args = [ROOT.RooFit.Index(index_cat)]
    for ch_name, toy_ch in channel_data.items():
        cmd_args.append(ROOT.RooFit.Import(ch_name, toy_ch))

    # RooDataSet/RooDataHist cmd-arg constructors accept up to 8 RooCmdArg arguments.
    if len(cmd_args) > 8:
        return None

    try:
        if fit_mode == "binned":
            return ROOT.RooDataHist("toy_sim_data", "toy_sim_data", ROOT.RooArgList(obs_set_union), *cmd_args)
        return ROOT.RooDataSet("toy_sim_data", "toy_sim_data", obs_set_union, *cmd_args)
    except Exception:
        return None


def _build_asimov_for_simultaneous(workspace, model):
    ROOT = _get_root()

    try:
        index_cat = model.indexCat()
    except Exception:
        return None

    all_vars = workspace.allVars()
    obs_names = set()
    obs_list = ROOT.RooArgList()
    channel_data = {}

    for state in index_cat:
        ch_name = state.first
        ch_pdf = model.getPdf(ch_name)
        if ch_pdf is None:
            continue

        ch_obs_set = ROOT.RooArgSet()
        try:
            raw_obs = ch_pdf.getObservables(all_vars)
        except Exception:
            raw_obs = None

        for obj in _iter_roo_collection(raw_obs):
            try:
                if not obj.InheritsFrom("RooRealVar"):
                    continue
                name = str(obj.GetName())
                if (
                    is_signal_strength_poi(name)
                    or name.startswith("rate_")
                    or name.startswith("yield_")
                    or name.startswith("sig_")
                    or name.startswith("bkg_")
                    or name.startswith("nuis_")
                    or name.startswith("theta_")
                ):
                    continue
                var = workspace.var(name)
                if var is None or not bool(var):
                    continue
                ch_obs_set.add(var)
                if name not in obs_names:
                    obs_names.add(name)
                    obs_list.add(var)
            except Exception:
                continue

        if ch_obs_set.getSize() == 0:
            continue

        try:
            dh = ch_pdf.generateBinned(ch_obs_set, ROOT.RooFit.ExpectedData(True))
        except Exception:
            dh = None
        if dh is not None and bool(dh):
            channel_data[str(ch_name)] = dh

    if obs_list.getSize() == 0 or not channel_data:
        return None

    cmd_args = [ROOT.RooFit.Index(index_cat)]
    for ch_name, dh in channel_data.items():
        cmd_args.append(ROOT.RooFit.Import(str(ch_name), dh))

    # RooDataHist cmd-arg constructor accepts up to 8 RooCmdArg arguments.
    if len(cmd_args) > 8:
        return None

    try:
        return ROOT.RooDataHist("asimov", "asimov", obs_list, *cmd_args)
    except Exception:
        return None


def _build_ensemble_evaluation_report(summaries, total_time_s):
    report = init_ensemble_report(summaries, total_time_s)
    if not summaries:
        return report

    add_fit_quality(report, summaries, include_invalid_fraction=True)
    add_poi_distributions(report, summaries)

    nll_values = [summary.get("nll") for summary in summaries if summary.get("nll") is not None]
    if nll_values:
        report["nll"] = distribution_summary(nll_values)

    return report


def _save_analysis_snapshot(output_path, fit_model, summaries, args):
    payload = {
        "format": "roomodel_analysis_snapshot_v1",
        "model_file": fit_model.model_file,
        "workspace_name": fit_model.workspace_name,
        "model_name": fit_model.model_name,
        "channels": fit_model.channels,
        "process_names": fit_model.process_names,
        "summaries": summaries,
        "config": {
            "toys": args.toys,
            "fit_mode": args.fit_mode,
            "binned_bins": args.binned_bins,
            "plot": bool(args.plot),
            "ntoys_plot": int(args.ntoys_plot),
            "set_parameters": args.set_parameters,
            "freeze_parameters": args.freeze_parameters,
            "set_parameter_ranges": args.set_parameter_ranges,
        },
    }

    final_path = normalize_output_path(output_path, ".json")
    with open(final_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return final_path


def _print_toy_poi_ranges(summaries):
    poi_values = [float(s.get("poi_fit")) for s in summaries if s.get("poi_fit") is not None]
    if not poi_values:
        print("POI summary (toys): no valid POI fit values available")
        return

    arr = np.asarray(poi_values, dtype=float)
    q16, q84 = np.percentile(arr, [16.0, 84.0])
    q025, q975 = np.percentile(arr, [2.5, 97.5])
    poi_name = next((s.get("poi_name") for s in summaries if s.get("poi_name")), "POI")
    print(f"POI summary ({poi_name}, toys={arr.size}):")
    print(f"  1 sigma central range: [{q16:.6g}, {q84:.6g}]")
    print(f"  2 sigma central range: [{q025:.6g}, {q975:.6g}]")


def _parse_name_value_csv(spec):
    if spec is None:
        return []
    text = str(spec).strip()
    if not text:
        return []
    items = []
    for token in text.split(","):
        part = token.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid parameter assignment '{part}' (expected name=value)")
        name, raw = part.split("=", 1)
        name = name.strip()
        raw = raw.strip()
        if not name or raw == "":
            raise ValueError(f"Invalid parameter assignment '{part}'")
        items.append((name, raw))
    return items


def _parse_name_list_csv(spec):
    if spec is None:
        return []
    text = str(spec).strip()
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def _apply_parameter_overrides(workspace, args):
    def _resolve_numeric_obj(name):
        var = workspace.var(name)
        if var is not None and bool(var):
            return var, "var"
        fn = workspace.function(name)
        if fn is not None and bool(fn):
            return fn, "func"
        return None, None

    # 1) Apply ranges first, then values, then freeze to mirror common fit workflows.
    for name, raw_range in _parse_name_value_csv(getattr(args, "set_parameter_ranges", None)):
        var = workspace.var(name)
        if var is None or not bool(var):
            raise ValueError(f"Unknown variable in --set-parameter-ranges: '{name}'")
        if ":" not in raw_range:
            raise ValueError(f"Invalid range '{raw_range}' for '{name}' (expected min:max)")
        lo_str, hi_str = raw_range.split(":", 1)
        lo = float(lo_str.strip())
        hi = float(hi_str.strip())
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            raise ValueError(f"Invalid range for '{name}': {raw_range}")
        var.setRange(float(lo), float(hi))
        try:
            cur = float(var.getVal())
            if cur < lo:
                var.setVal(float(lo))
            elif cur > hi:
                var.setVal(float(hi))
        except Exception:
            pass
        print(f"Applied parameter range: {name}=[{lo:.6g}, {hi:.6g}]")

    for name, raw_value in _parse_name_value_csv(getattr(args, "set_parameters", None)):
        obj, kind = _resolve_numeric_obj(name)
        if obj is None:
            raise ValueError(f"Unknown variable in --set-parameters: '{name}'")
        value = float(raw_value)
        if kind == "var":
            obj.setVal(value)
        else:
            # RooConstVar and RooFormulaVar are read-only from Python.
            # Allow overriding const/rate-like functions by creating a replacement RooRealVar.
            obj_name = str(obj.GetName())
            if not (
                obj_name.startswith("yield_")
                or obj_name.startswith("rate_")
                or obj_name.startswith("sig_rate__")
                or obj_name.startswith("bkg_rate__")
            ):
                raise ValueError(
                    f"--set-parameters cannot modify non-variable function '{name}'"
                )
            ROOT = _get_root()
            replacement = ROOT.RooRealVar(obj_name, obj_name, value, -1.0e12, 1.0e12)
            getattr(workspace, "import")(replacement, ROOT.RooFit.RecycleConflictNodes())
        print(f"Applied parameter value: {name}={value:.6g}")

    for name in _parse_name_list_csv(getattr(args, "freeze_parameters", None)):
        var = workspace.var(name)
        if var is None or not bool(var):
            raise ValueError(f"Unknown variable in --freeze-parameters: '{name}'")
        var.setConstant(True)
        print(f"Froze parameter: {name}")


def _enforce_physical_poi_bounds(workspace, fit_model):
    # Intentionally no-op: POI should float freely within its configured
    # workspace bounds, with no analysis-time bound tightening.
    return


def run_analysis_cli(args):
    fit_model = _load_analysis_model(model_file=args.model_file, input_card=args.input_card)

    # Construct set_parameter_ranges to include --poi-min and --poi-max if provided
    set_ranges_spec = getattr(args, "set_parameter_ranges", None) or ""
    poi_name = getattr(fit_model, "poi_name", None)
    if poi_name:
        poi_ranges = []
        if getattr(args, "poi_min", None) is not None:
            poi_ranges.append(f"{poi_name}={args.poi_min}")
        if getattr(args, "poi_max", None) is not None:
            if poi_ranges:
                poi_ranges[-1] += f":{args.poi_max}"
            else:
                # No poi_min but have poi_max; use a very low lower bound
                poi_ranges.append(f"{poi_name}=-1e12:{args.poi_max}")
        if poi_ranges:
            if set_ranges_spec:
                set_ranges_spec = f"{set_ranges_spec},{','.join(poi_ranges)}"
            else:
                set_ranges_spec = ",".join(poi_ranges)
    
    # Temporarily set args.set_parameter_ranges to include POI bounds
    original_ranges = getattr(args, "set_parameter_ranges", None)
    args.set_parameter_ranges = set_ranges_spec or None

    ws, metadata, model, observed_data, inner_model = _workspace_objects(fit_model)
    _apply_parameter_overrides(ws, args)
    
    # Restore original args value
    args.set_parameter_ranges = original_ranges
    _enforce_physical_poi_bounds(ws, fit_model)
    # Use inner_model (the RooSimultaneous / channel PDF, unwrapped from any
    # constraint wrapper) for observable resolution and dataset generation.
    obs_var = _resolve_obs_var(observed_data, ws)
    if obs_var is None:
        raise ValueError("Could not resolve observable from workspace")

    nominal_param_state = _capture_nominal_parameter_state(ws, inner_model, observed_data, obs_var=obs_var)

    has_observed_data = observed_data is not None
    use_observed_data, use_asimov_data, n_toys = resolve_dataset_mode(args.toys, has_observed_data)

    if args.fit_mode == "auto":
        observed_is_hist = False
        if observed_data is not None:
            try:
                observed_is_hist = bool(observed_data) and bool(observed_data.InheritsFrom("RooDataHist"))
            except Exception:
                observed_is_hist = False
        mode = "binned" if observed_is_hist else "unbinned"
    else:
        mode = args.fit_mode

    summaries = []
    total_start = time.perf_counter()
    n_plot = max(0, int(args.ntoys_plot)) if bool(args.plot) else 0
    cls_points = int(args.cls_scan_points) if args.cls_scan_points is not None else 25
    if bool(args.cls_smart_scan):
        cls_points = max(cls_points, 41)
    fc_points = int(args.fc_scan_points) if args.fc_scan_points is not None else 21
    limit_scan_points = max(cls_points if args.cls is not None else 0, fc_points if args.feldman_cousins is not None else 0)
    limit_scan_full_range = (args.cls is not None) or bool(args.cls_smart_scan) or (args.feldman_cousins is not None)
    limit_poi_min = float(args.limit_poi_min)

    cls_toys = int(getattr(args, "cls_toys", 0) or 0)

    # Compute Asimov sigma once for the full asymptotic CLs formula.
    # Only needed when cls_toys == 0 (asymptotic path).
    sigma_asimov = None
    if args.cls is not None and cls_toys == 0:
        sigma_asimov = _compute_asimov_sigma(
            ws, model, inner_model, fit_model, obs_var, mode, int(args.binned_bins)
        )
        if sigma_asimov is not None:
            print(f"Asimov sigma (sensitivity): {sigma_asimov:.6g}")
        else:
            print("Warning: could not compute Asimov sigma; using simplified p_b = Phi(sqrt(q_mu))")
        _restore_parameter_state(ws, nominal_param_state)

    if use_observed_data:
        summary = _run_single_fit(
            ws,
            model,
            observed_data,
            fit_model,
            mode,
            int(args.binned_bins),
            input_obs_var=obs_var,
            collect_plot_details=(n_plot > 0),
            limit_scan_points=limit_scan_points,
            limit_scan_full_range=limit_scan_full_range,
            limit_scan_poi_min=limit_poi_min,
            inner_model=inner_model,
        )
        if args.cls is not None:
            if cls_toys > 0:
                _apply_cls_summary_from_toys(
                    summary, float(args.cls),
                    ws, model, inner_model, fit_model,
                    cls_toys, observed_data, mode, int(args.binned_bins), obs_var,
                    poi_min_limit=limit_poi_min,
                    seed=int(args.seed) if hasattr(args, "seed") else 1234,
                )
            else:
                _apply_cls_summary_from_scan(summary, float(args.cls), poi_min_limit=limit_poi_min, sigma_asimov=sigma_asimov)
        if args.feldman_cousins is not None:
            _apply_feldman_cousins_true(
                summary,
                float(args.feldman_cousins),
                int(fc_points),
                int(args.fc_toys),
                limit_poi_min,
                ws,
                model,
                observed_data,
                fit_model,
                dataset_id=0,
                seed=int(args.seed) if hasattr(args, "seed") else 1234,
            )
        summary["dataset_id"] = 0
        summary["observed_fit"] = True
        summaries.append(summary)

        poi_name = summary.get("poi_name") or "POI"
        poi_fit = summary.get("poi_fit")
        poi_unc = summary.get("poi_unc_hesse")
        valid = summary.get("valid", False)
        status = "valid  " if valid else "invalid"
        if poi_fit is None:
            print(f"Observed data: {status}, {poi_name}=n/a        +- n/a")
        elif poi_unc is None:
            print(f"Observed data: {status}, {poi_name}={float(poi_fit):<10.6g} +- n/a")
        else:
            print(f"Observed data: {status}, {poi_name}={float(poi_fit):<10.6g} +- {float(poi_unc):<10.6g}")
    elif use_asimov_data:
        _restore_parameter_state(ws, nominal_param_state)
        ROOT = _get_root()
        # Counting experiments: build the Asimov dataset directly from expected
        # yields rather than via generateBinned, which fails for RooProdPdf of
        # Poisson channels (no event count / not extended).
        # Generate at mu=0 (background-only Asimov) so that a free fit of the
        # Asimov data recovers mu≈0 as expected for sensitivity studies.
        _asimov_poi = _resolve_poi_var(ws, fit_model)
        asimov = _build_asimov_for_counting(ws, poi_var=_asimov_poi, poi_val=0.0)
        is_counting_asimov = asimov is not None and bool(asimov)
        if not is_counting_asimov:
            # Shape model path — use generateBinned with ExpectedData.
            # Use inner_model (unwrapped from constraint wrapper) for observable
            # resolution and Asimov generation — constraint PDFs have no observables.
            obs_set = _model_observable_set(ws, inner_model, observed_data, obs_var=obs_var)
            if obs_set is None:
                raise ValueError("Could not resolve observables for Asimov generation")
            inner_model_name = str(inner_model.ClassName()) if hasattr(inner_model, "ClassName") else ""
            # Ensure the index category is in obs_set for RooSimultaneous
            if "RooSimultaneous" in inner_model_name:
                index_cat_fn = getattr(inner_model, "indexCat", None)
                if callable(index_cat_fn):
                    try:
                        cat = index_cat_fn()
                        if cat is not None and not obs_set.contains(cat):
                            obs_set = ROOT.RooArgSet(obs_set)
                            obs_set.add(cat)
                    except Exception:
                        pass
            if "RooSimultaneous" in inner_model_name:
                asimov = _build_asimov_for_simultaneous(ws, inner_model)
                if asimov is None or not bool(asimov):
                    asimov = inner_model.generateBinned(obs_set, ROOT.RooFit.ExpectedData(True))
            elif obs_var is not None:
                try:
                    n_obs = int(obs_set.getSize())
                except Exception:
                    n_obs = 0
                if n_obs > 1:
                    asimov = inner_model.generateBinned(obs_set, ROOT.RooFit.ExpectedData(True))
                else:
                    asimov = inner_model.generateBinned(
                        obs_set,
                        ROOT.RooFit.Binning(int(args.binned_bins), float(obs_var.getMin()), float(obs_var.getMax())),
                        ROOT.RooFit.ExpectedData(True),
                    )
            else:
                asimov = inner_model.generateBinned(obs_set, ROOT.RooFit.ExpectedData(True))
        # Counting Asimov is an unbinned single-entry RooDataSet; use unbinned
        # fit mode so _run_single_fit does not try to histogram integer counts.
        asimov_fit_mode = "unbinned" if is_counting_asimov else "binned"
        summary = _run_single_fit(
            ws,
            model,
            asimov,
            fit_model,
            asimov_fit_mode,
            int(args.binned_bins),
            input_obs_var=obs_var,
            collect_plot_details=(n_plot > 0),
            limit_scan_points=limit_scan_points,
            limit_scan_full_range=limit_scan_full_range,
            limit_scan_poi_min=limit_poi_min,
            inner_model=inner_model,
        )
        if args.cls is not None:
            if cls_toys > 0:
                _apply_cls_summary_from_toys(
                    summary, float(args.cls),
                    ws, model, inner_model, fit_model,
                    cls_toys, asimov, "binned", int(args.binned_bins), obs_var,
                    poi_min_limit=limit_poi_min,
                    seed=int(args.seed) if hasattr(args, "seed") else 1234,
                )
            else:
                _apply_cls_summary_from_scan(summary, float(args.cls), poi_min_limit=limit_poi_min, sigma_asimov=sigma_asimov)
        if args.feldman_cousins is not None:
            _apply_feldman_cousins_true(
                summary,
                float(args.feldman_cousins),
                int(fc_points),
                int(args.fc_toys),
                limit_poi_min,
                ws,
                model,
                asimov,
                fit_model,
                dataset_id=0,
                seed=int(args.seed) if hasattr(args, "seed") else 1234,
            )
        summary["dataset_id"] = 0
        summary["asimov_fit"] = True
        summaries.append(summary)
        status = summary.get("fit_status", -1)
        poi_name = summary.get("poi_name")
        poi_fit = summary.get("poi_fit")
        poi_unc = summary.get("poi_unc_hesse")
        if poi_name is None or poi_fit is None:
            print(f"Asimov fit: status={status}, no POI result")
        elif poi_unc is None:
            print(f"Asimov fit: status={status}, {poi_name}={float(poi_fit):<10.6g} +- n/a")
        else:
            print(f"Asimov fit: status={status}, {poi_name}={float(poi_fit):<10.6g} +- {float(poi_unc):<10.6g}")
    else:
        for idx in range(int(n_toys)):
            _restore_parameter_state(ws, nominal_param_state)
            # Use inner_model for toy generation (constraint PDFs have no observables)
            toy_data = _generate_dataset(ws, inner_model, observed_data, obs_var, mode, int(args.binned_bins))
            _restore_parameter_state(ws, nominal_param_state)
            summary = _run_single_fit(
                ws,
                model,
                toy_data,
                fit_model,
                mode,
                int(args.binned_bins),
                input_obs_var=obs_var,
                collect_plot_details=(idx < n_plot),
                limit_scan_points=limit_scan_points,
                limit_scan_full_range=limit_scan_full_range,
                limit_scan_poi_min=limit_poi_min,
                inner_model=inner_model,
            )
            if args.cls is not None:
                if cls_toys > 0:
                    _apply_cls_summary_from_toys(
                        summary, float(args.cls),
                        ws, model, inner_model, fit_model,
                        cls_toys, toy_data, mode, int(args.binned_bins), obs_var,
                        poi_min_limit=limit_poi_min,
                        seed=int(args.seed) + idx if hasattr(args, "seed") else 1234 + idx,
                    )
                else:
                    _apply_cls_summary_from_scan(summary, float(args.cls), poi_min_limit=limit_poi_min, sigma_asimov=sigma_asimov)
            if args.feldman_cousins is not None:
                _apply_feldman_cousins_true(
                    summary,
                    float(args.feldman_cousins),
                    int(fc_points),
                    int(args.fc_toys),
                    limit_poi_min,
                    ws,
                    model,
                    toy_data,
                    fit_model,
                    dataset_id=idx,
                    seed=int(args.seed) if hasattr(args, "seed") else 1234,
                )
            summary["dataset_id"] = idx
            summaries.append(summary)

            poi_name = summary.get("poi_name") or "POI"
            poi_fit = summary.get("poi_fit")
            poi_unc = summary.get("poi_unc_hesse")
            if poi_fit is None:
                print(f"Toy {idx + 1}/{int(n_toys)}: {poi_name} fit unavailable")
            elif poi_unc is None:
                print(f"Toy {idx + 1}/{int(n_toys)}: {poi_name} = {float(poi_fit):.6g} +/- n/a")
            else:
                print(f"Toy {idx + 1}/{int(n_toys)}: {poi_name} = {float(poi_fit):.6g} +/- {float(poi_unc):.6g}")

        _print_toy_poi_ranges(summaries)

    total_time_s = time.perf_counter() - total_start

    print(f"Analyzed roomodel workspace '{fit_model.workspace_name}'")
    if summaries and (args.cls is not None or args.feldman_cousins is not None):
        print_limit_summary_lines(summaries[0], include_scan_details=True, include_yield_upper=True)
    
    # Print model information if requested
    if getattr(args, "print_model", False) and summaries:
        first_summary = summaries[0]
        # Try to load the workspace for additional model info
        state = None
        try:
            state = load_workspace_and_metadata(args.model_file)
        except Exception:
            pass
        print_model_info(fit_model, first_summary, "roomodel", state=state)
    
    print_runtime_summary(summaries, total_time_s)

    maybe_plot_summary_artifacts(args, summaries, fit_model, plot_summary_artifacts)

    output = resolve_output_or_default(args.output, args.seed, ".json")
    save_and_print_ensemble_report(
        summaries=summaries,
        total_time_s=total_time_s,
        output_path=output,
        report_file=args.report_file,
        build_report_fn=_build_ensemble_evaluation_report,
    )
    save_and_print_snapshot(
        output_path=output,
        fit_model=fit_model,
        summaries=summaries,
        args=args,
        save_snapshot_fn=_save_analysis_snapshot,
    )
