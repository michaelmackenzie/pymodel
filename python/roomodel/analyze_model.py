import json
import os
import time
from statistics import NormalDist

import numpy as np

from backends.path_bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path(__file__)

from backends.analysis_common import normalize_output_path, resolve_dataset_mode
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


def _workspace_objects(fit_model):
    ws, metadata = load_workspace_and_metadata(fit_model.model_file)
    model = ws.pdf(fit_model.model_name)
    if model is None or not bool(model):
        raise ValueError(f"Model PDF '{fit_model.model_name}' not found in workspace '{ws.GetName()}'")
    data = ws.data(fit_model.data_name) if fit_model.data_name else None
    return ws, metadata, model, data


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
        if name.startswith("mu_") or name.startswith("yield_") or name.startswith("rate_"):
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
                    name.startswith("mu_")
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
                    name.startswith("mu_")
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
            term_pdf = workspace.pdf(f"shape_{process}__{channel_name}")
            if term_pdf is None or not bool(term_pdf):
                term_pdf = workspace.pdf(str(process))
            term_yield = _get_yield_obj(f"yield_{process}__{channel_name}")
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
                name.startswith("mu_")
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


def _apply_cls_summary_from_scan(summary, alpha, poi_min_limit=0.0):
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

    # One-sided asymptotic mapping using q_mu = 2*DeltaNLL.
    q_mu = np.clip(2.0 * y, 0.0, None)
    z = np.sqrt(q_mu)
    norm = NormalDist()
    cls_obs = np.asarray([1.0 - norm.cdf(float(v)) for v in z], dtype=float)

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


def _apply_feldman_cousins_summary_from_scan(summary, alpha, scan_points, n_toys, poi_min_limit=0.0):
    scan = summary.get("delta_nll_scan") or {}
    x = np.asarray(scan.get("x", []), dtype=float)
    y = np.asarray(scan.get("delta_nll", []), dtype=float)
    if x.size == 0 or y.size != x.size:
        summary["feldman_cousins"] = {
            "fc_status": "failed: missing delta_nll_scan",
            "alpha": float(alpha),
        }
        return

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        summary["feldman_cousins"] = {
            "fc_status": "failed: empty delta_nll_scan",
            "alpha": float(alpha),
        }
        return

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    limit_mask = x >= float(poi_min_limit)
    x = x[limit_mask]
    y = y[limit_mask]
    if x.size == 0:
        summary["feldman_cousins"] = {
            "fc_status": "failed: no scan points above limit-poi-min",
            "alpha": float(alpha),
        }
        return

    z_crit = NormalDist().inv_cdf(1.0 - 0.5 * float(alpha))
    q_crit = float(z_crit * z_crit)
    q_obs = np.asarray(np.clip(2.0 * y, 0.0, None), dtype=float)
    accepted = q_obs <= q_crit
    interval = None
    if np.any(accepted):
        interval = [float(np.min(x[accepted])), float(np.max(x[accepted]))]

    summary["feldman_cousins"] = {
        "fc_status": "ok" if interval is not None else "no-accepted-points",
        "alpha": float(alpha),
        "fc_interval": interval,
        "scan_points": int(scan_points),
        "scan_max": float(np.max(x)),
        "n_toys_per_point": int(n_toys),
        "grid": {
            "poi": [float(v) for v in x.tolist()],
            "q_obs": [float(v) for v in q_obs.tolist()],
            "q_crit": [float(q_crit)] * int(x.size),
            "toy_valid": [int(n_toys)] * int(x.size),
        },
        "note": "profile-likelihood approximation",
    }


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
):
    ROOT = _get_root()
    obs = _resolve_obs_var(data, workspace) or input_obs_var
    fit_data = data
    is_datahist = False
    model_name = str(model.ClassName()) if hasattr(model, "ClassName") else ""
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
            model,
            fit_model,
            fit_data,
            obs,
            dataset_plot,
            int(binned_bins),
        )
        channel_plots = _channel_plot_payloads(
            workspace,
            model,
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
            model,
            fit_data,
            poi,
            n_points=scan_points,
            full_range=bool(limit_scan_full_range),
            poi_scan_min=limit_scan_poi_min,
        )

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
    }


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

    gen_ext = [ROOT.RooFit.Extended(True)] if can_extend else []

    if fit_mode == "binned":
        generated = None
        try:
            if "RooSimultaneous" in model_name:
                generated = model.generateBinned(obs_set, *gen_ext)
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
        gen_ext = [ROOT.RooFit.Extended(True)] if can_extend else []

        toy_ch = None
        try:
            if fit_mode == "binned":
                toy_ch = ch_pdf.generateBinned(ch_obs_set, *gen_ext)
            else:
                toy_ch = ch_pdf.generate(ch_obs_set, *gen_ext)
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
                    name.startswith("mu_")
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

    ws, metadata, model, observed_data = _workspace_objects(fit_model)
    _apply_parameter_overrides(ws, args)
    _enforce_physical_poi_bounds(ws, fit_model)
    obs_var = _resolve_obs_var(observed_data, ws)
    if obs_var is None:
        raise ValueError("Could not resolve observable from workspace")

    nominal_param_state = _capture_nominal_parameter_state(ws, model, observed_data, obs_var=obs_var)

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
    limit_scan_full_range = bool(args.cls_smart_scan) or (args.feldman_cousins is not None)
    limit_poi_min = float(args.limit_poi_min)

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
        )
        if args.cls is not None:
            _apply_cls_summary_from_scan(summary, float(args.cls), poi_min_limit=limit_poi_min)
        if args.feldman_cousins is not None:
            _apply_feldman_cousins_summary_from_scan(
                summary,
                float(args.feldman_cousins),
                int(fc_points),
                int(args.fc_toys),
                poi_min_limit=limit_poi_min,
            )
        summary["dataset_id"] = 0
        summary["observed_fit"] = True
        summaries.append(summary)
    elif use_asimov_data:
        _restore_parameter_state(ws, nominal_param_state)
        ROOT = _get_root()
        obs_set = _model_observable_set(ws, model, observed_data, obs_var=obs_var)
        if obs_set is None:
            raise ValueError("Could not resolve observables for Asimov generation")
        model_name = str(model.ClassName()) if hasattr(model, "ClassName") else ""
        # Ensure the index category is in obs_set for RooSimultaneous
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
        if "RooSimultaneous" in model_name:
            asimov = _build_asimov_for_simultaneous(ws, model)
            if asimov is None or not bool(asimov):
                asimov = model.generateBinned(obs_set, ROOT.RooFit.ExpectedData(True))
        elif obs_var is not None:
            try:
                n_obs = int(obs_set.getSize())
            except Exception:
                n_obs = 0
            if n_obs > 1:
                asimov = model.generateBinned(obs_set, ROOT.RooFit.ExpectedData(True))
            else:
                asimov = model.generateBinned(
                    obs_set,
                    ROOT.RooFit.Binning(int(args.binned_bins), float(obs_var.getMin()), float(obs_var.getMax())),
                    ROOT.RooFit.ExpectedData(True),
                )
        else:
            asimov = model.generateBinned(obs_set, ROOT.RooFit.ExpectedData(True))
        summary = _run_single_fit(
            ws,
            model,
            asimov,
            fit_model,
            "binned",
            int(args.binned_bins),
            input_obs_var=obs_var,
            collect_plot_details=(n_plot > 0),
            limit_scan_points=limit_scan_points,
            limit_scan_full_range=limit_scan_full_range,
            limit_scan_poi_min=limit_poi_min,
        )
        if args.cls is not None:
            _apply_cls_summary_from_scan(summary, float(args.cls), poi_min_limit=limit_poi_min)
        if args.feldman_cousins is not None:
            _apply_feldman_cousins_summary_from_scan(
                summary,
                float(args.feldman_cousins),
                int(fc_points),
                int(args.fc_toys),
                poi_min_limit=limit_poi_min,
            )
        summary["dataset_id"] = 0
        summary["asimov_fit"] = True
        summaries.append(summary)
    else:
        for idx in range(int(n_toys)):
            _restore_parameter_state(ws, nominal_param_state)
            toy_data = _generate_dataset(ws, model, observed_data, obs_var, mode, int(args.binned_bins))
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
            )
            if args.cls is not None:
                _apply_cls_summary_from_scan(summary, float(args.cls), poi_min_limit=limit_poi_min)
            if args.feldman_cousins is not None:
                _apply_feldman_cousins_summary_from_scan(
                    summary,
                    float(args.feldman_cousins),
                    int(fc_points),
                    int(args.fc_toys),
                    poi_min_limit=limit_poi_min,
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
