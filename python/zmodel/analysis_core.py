"""
zfit analysis core.

This module retains the zfit-specific operations that cannot be abstracted:

  - Runtime / graph-mode configuration
  - Binned-space and binned-model construction (zfit.Space, VariableBinning, …)
  - Dataset format helpers (toy, Asimov, observed → zfit Data / BinnedData)
  - Loss construction (ExtendedBinnedNLL / ExtendedUnbinnedNLL)
  - POI / signal-parameter discovery on the FitModel
  - Profile-scan helpers that require direct access to the zfit loss object

Everything that is backend-agnostic (CLs scan, Feldman-Cousins, NLL profile
scan, expected-quantile extraction, pull computation, the main analysis loop)
now lives in ``backends/analysis_common.py`` and is invoked via the
``ZfitAnalysisBackend`` adapter in ``zmodel/analysis_backend.py``.
"""

import time

import numpy as np
import zfit
from scipy.optimize import minimize_scalar

from backends.zfit_parameter_utils import (
    all_params_list as _all_params,
    capture_fit_model_parameter_values as _capture_fit_model_parameter_values,
    capture_parameter_values as _capture_parameter_values,
    channel_models as _channel_models,
    find_parameter_by_name,
    restore_parameter_values as _restore_parameter_values,
)
from backends.analysis_common import (
    is_signal_strength_poi,
    resolve_data_mode as _resolve_data_mode,
    run_analysis_common,
)
from zmodel.utilities import AsymptoticCalculator, POI


# ===========================================================================
# Runtime configuration
# ===========================================================================

def is_likely_counting_model(fit_model):
    if fit_model.obs_range == (0.0, 1.0):
        return True
    obs_name = None
    if getattr(fit_model.obs, "obs", None):
        obs_name = fit_model.obs.obs[0]
    return obs_name == "count_obs"


def configure_runtime(graph_mode, fit_model, toys):
    if graph_mode == "on":
        zfit.run.set_graph_mode(True)
        return
    if graph_mode == "off":
        zfit.run.set_graph_mode(False)
        return

    use_graph = not (is_likely_counting_model(fit_model) or toys <= 5)
    zfit.run.set_graph_mode(use_graph)


# ===========================================================================
# Binned-data helpers
# ===========================================================================

def _is_binned_dataset(dataset):
    if dataset is None:
        return False

    data_space = getattr(dataset, "space", None)
    if data_space is not None and getattr(data_space, "binned", False):
        return True

    has_values = callable(getattr(dataset, "values", None))
    has_value = callable(getattr(dataset, "value", None))
    return has_values and not has_value


def _has_histogram_input_data(fit_model):
    data = getattr(fit_model, "data", None)
    if isinstance(data, dict):
        return any(_is_binned_dataset(entry) for entry in data.values())
    return _is_binned_dataset(data)


def _native_binned_space_from_data(dataset):
    if not _is_binned_dataset(dataset):
        return None
    return getattr(dataset, "space", None)


def _resolve_fit_mode(fit_mode, fit_model):
    if fit_mode == "auto":
        if is_likely_counting_model(fit_model) or _has_histogram_input_data(fit_model):
            return "binned"
        return "unbinned"
    return fit_mode


def _build_counting_binned_space(fit_model):
    obs_name = "count_obs"
    if getattr(fit_model.obs, "obs", None):
        obs_name = fit_model.obs.obs[0]

    low, high = fit_model.obs_range
    edges = np.array([float(low), float(high)], dtype=float)
    binning = zfit.binned.VariableBinning(edges, name=obs_name)
    return zfit.Space(obs_name, binning=binning)


def _binning_edges_as_float_array(binning):
    edges = getattr(binning, "edges", binning)

    if isinstance(edges, (list, tuple)):
        if len(edges) == 1:
            edges = edges[0]
        else:
            raise ValueError("Binned fits currently support only 1D observables")

    if hasattr(edges, "numpy"):
        edges = edges.numpy()

    arr = np.asarray(edges, dtype=float)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim != 1:
        raise ValueError("Binned fits currently support only 1D observables")

    return arr


def _build_binned_space(fit_model, bins):
    native_space = _native_binned_space_from_data(getattr(fit_model, "data", None))
    if native_space is not None:
        return native_space

    if is_likely_counting_model(fit_model):
        return _build_counting_binned_space(fit_model)

    obs_names = getattr(fit_model.obs, "obs", None)
    if not obs_names or len(obs_names) != 1:
        raise ValueError("Binned fits currently support only 1D observables")

    low, high = fit_model.obs_range
    edges = np.linspace(float(low), float(high), int(bins) + 1)
    binning = zfit.binned.VariableBinning(edges, name=obs_names[0])
    return zfit.Space(obs_names[0], binning=binning)


def _build_channel_binned_spaces(fit_model, bins):
    channel_obs = getattr(fit_model, "channel_obs", {}) or {}
    channel_ranges = getattr(fit_model, "channel_obs_ranges", {}) or {}
    channel_data = getattr(fit_model, "data", {}) or {}
    spaces = {}
    for channel, obs_space in channel_obs.items():
        native_space = None
        if isinstance(channel_data, dict):
            native_space = _native_binned_space_from_data(channel_data.get(channel))
        if native_space is not None:
            spaces[channel] = native_space
            continue

        obs_names = getattr(obs_space, "obs", None)
        if not obs_names or len(obs_names) != 1:
            raise ValueError("Binned fits currently support only 1D observables per channel")

        low, high = channel_ranges.get(channel, tuple(float(x) for x in obs_space.limit1d))
        edges = np.linspace(float(low), float(high), int(bins) + 1)
        binning = zfit.binned.VariableBinning(edges, name=obs_names[0])
        spaces[channel] = zfit.Space(obs_names[0], binning=binning)
    return spaces


def _build_channel_binned_models(fit_model, channel_binned_spaces):
    channel_models = _channel_models(fit_model)
    return {
        channel: model.to_binned(channel_binned_spaces[channel])
        for channel, model in channel_models.items()
    }


# ===========================================================================
# Toy / Asimov / observed dataset construction
# ===========================================================================

def _make_binned_toy_data(model, binned_space):
    edges = _binning_edges_as_float_array(binned_space.binning)

    is_binned_model = isinstance(model, zfit.core.binnedpdf.BaseBinnedPDF)
    if is_binned_model:
        try:
            expected_counts = np.asarray(model.values(), dtype=float).reshape(-1)
        except Exception:
            rel_counts = np.asarray(model.rel_counts(model.space), dtype=float).reshape(-1)
            total_yield = 1.0
            get_yield = getattr(model, "get_yield", None)
            if callable(get_yield):
                try:
                    total_yield = float(get_yield().value())
                except Exception:
                    total_yield = 1.0
            expected_counts = rel_counts * total_yield
        expected_counts = np.clip(
            np.nan_to_num(expected_counts, nan=0.0, posinf=0.0, neginf=0.0),
            0.0,
            None,
        )
        counts = np.random.poisson(expected_counts).astype(float)
        centers = 0.5 * (edges[:-1] + edges[1:])
        values = np.repeat(centers, counts.astype(int)).astype(float)
        data = zfit.data.BinnedData.from_tensor(space=binned_space, values=counts)
        return data, values, edges, counts

    sample = model.sample(n="auto")
    values = np.asarray(sample.value(), dtype=float).reshape(-1)
    counts, _ = np.histogram(values, bins=edges)
    data = zfit.data.BinnedData.from_tensor(space=binned_space, values=counts.astype(float))
    return data, values, edges, counts.astype(float)


def _guess_binned_space_for_model(fit_model, model, channel=None, bins=40):
    data = getattr(fit_model, "data", None)
    if channel is not None and isinstance(data, dict):
        native = _native_binned_space_from_data(data.get(channel))
        if native is not None:
            return native
    else:
        native = _native_binned_space_from_data(data)
        if native is not None:
            return native

    model_space = getattr(model, "space", None)
    if model_space is not None and getattr(model_space, "binned", False):
        return model_space

    if channel is not None:
        channel_obs = (getattr(fit_model, "channel_obs", {}) or {}).get(channel)
        if channel_obs is not None and getattr(channel_obs, "binned", False):
            return channel_obs

    low, high = None, None
    if channel is not None:
        channel_ranges = getattr(fit_model, "channel_obs_ranges", {}) or {}
        if channel in channel_ranges:
            low, high = channel_ranges[channel]
    if low is None or high is None:
        low, high = getattr(fit_model, "obs_range", (0.0, 1.0))

    obs_name = "obs"
    obs_names = getattr(model_space, "obs", None)
    if obs_names:
        obs_name = obs_names[0]
    edges = np.linspace(float(low), float(high), int(bins) + 1)
    return zfit.Space(obs_name, binning=zfit.binned.VariableBinning(edges, name=obs_name))


def _make_unbinned_toy_data_from_binned_model(fit_model, model, channel=None, bins=40):
    binned_space = _guess_binned_space_for_model(fit_model, model, channel=channel, bins=bins)
    if hasattr(model, "to_binned"):
        binned_model = model.to_binned(binned_space)
    else:
        binned_model = model

    expected_counts = np.asarray(binned_model.values(), dtype=float).reshape(-1)
    expected_counts = np.clip(np.nan_to_num(expected_counts, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    toy_counts = np.random.poisson(expected_counts).astype(float)

    edges = _binning_edges_as_float_array(binned_space.binning)
    centers = 0.5 * (edges[:-1] + edges[1:])
    values = np.repeat(centers, toy_counts.astype(int))
    unbinned_data = _values_to_unbinned_dataset(values, getattr(model, "space", None))
    return unbinned_data, values.astype(float), edges, toy_counts


def _expected_counts_by_channel(fit_model):
    channel_expectations = {}
    term_channels = getattr(fit_model, "term_channels", {}) or {}
    for term_name, yield_param in getattr(fit_model, "yields", {}).items():
        channel = term_channels.get(term_name)
        if channel is None:
            continue
        channel_expectations[channel] = channel_expectations.get(channel, 0.0) + float(yield_param.value())
    return channel_expectations


def _build_toy_data(fit_model, resolved_fit_mode, binned_space, is_counting):
    channel_models = _channel_models(fit_model)

    if channel_models:
        if resolved_fit_mode == "binned":
            if not isinstance(binned_space, dict):
                raise ValueError("Expected per-channel binned spaces for channel-based binned fit")

            channel_data = {}
            channel_binned = {}
            for channel, model in channel_models.items():
                if channel not in binned_space:
                    raise ValueError(f"Missing binned space for channel '{channel}'")
                data, values, edges, counts = _make_binned_toy_data(model, binned_space[channel])
                channel_data[channel] = data
                channel_binned[channel] = {
                    "edges": edges.tolist(),
                    "counts": counts.tolist(),
                    "values": values.tolist(),
                }
            dataset_plot = {"mode": "binned", "channel_binned": channel_binned}
            return channel_data, None, dataset_plot

        channel_data = {}
        channel_values = {}
        for channel, model in channel_models.items():
            try:
                sample = model.sample(n="auto")
                values = np.asarray(sample.value(), dtype=float).reshape(-1)
                channel_data[channel] = sample
            except Exception:
                sample, values, _edges, _counts = _make_unbinned_toy_data_from_binned_model(
                    fit_model,
                    model,
                    channel=channel,
                )
                channel_data[channel] = sample
            channel_values[channel] = values
        dataset_plot = {"mode": "unbinned", "channel_values": channel_values}
        return channel_data, None, dataset_plot

    if resolved_fit_mode == "binned":
        if is_counting:
            expected = float(fit_model.model.get_yield().value())
            channel_expectations = _expected_counts_by_channel(fit_model)
            channel_counts = {}
            if channel_expectations:
                for channel, value in channel_expectations.items():
                    channel_counts[channel] = float(np.random.poisson(max(0.0, value)))
                toy_count = int(sum(channel_counts.values()))
            else:
                toy_count = int(np.random.poisson(expected))
            low, high = fit_model.obs_range
            edges = np.array([float(low), float(high)], dtype=float)
            counts = np.array([float(toy_count)], dtype=float)
            data = zfit.data.BinnedData.from_tensor(space=binned_space, values=counts)
            dataset_plot = {"mode": "binned", "edges": edges, "counts": counts}
            if channel_counts:
                dataset_plot["channel_counts"] = channel_counts
            return data, toy_count, dataset_plot

        data, values, edges, counts = _make_binned_toy_data(fit_model.model, binned_space)
        dataset_plot = {
            "mode": "binned",
            "edges": edges,
            "counts": counts,
            "values": values,
        }
        return data, None, dataset_plot

    try:
        data = fit_model.model.sample(n="auto")
        values = np.asarray(data.value(), dtype=float).reshape(-1)
    except Exception:
        data, values, _edges, _counts = _make_unbinned_toy_data_from_binned_model(
            fit_model,
            fit_model.model,
        )
    dataset_plot = {"mode": "unbinned", "values": values}
    return data, None, dataset_plot


def _build_asimov_binned_data(binned_model, binned_space, fit_model):
    if isinstance(binned_model, dict):
        data = {}
        channel_binned = {}
        for channel, model in binned_model.items():
            channel_data = model.to_binneddata()
            expected_counts = np.asarray(model.values(), dtype=float).reshape(-1)
            edges = _binning_edges_as_float_array(binned_space[channel].binning)
            data[channel] = channel_data
            channel_binned[channel] = {
                "edges": edges.tolist(),
                "counts": expected_counts.tolist(),
            }
        dataset_plot = {
            "mode": "binned",
            "channel_binned": channel_binned,
            "asimov": True,
        }
        return data, channel_binned, dataset_plot

    data = binned_model.to_binneddata()
    expected_counts = np.asarray(binned_model.values(), dtype=float).reshape(-1)
    edges = _binning_edges_as_float_array(binned_space.binning)
    dataset_plot = {
        "mode": "binned",
        "edges": edges,
        "counts": expected_counts,
        "asimov": True,
    }
    channel_expectations = _expected_counts_by_channel(fit_model)
    if channel_expectations:
        dataset_plot["channel_counts"] = channel_expectations
    return data, expected_counts, dataset_plot


# ===========================================================================
# Loss construction
# ===========================================================================

def _build_loss(fit_model, resolved_fit_mode, binned_model, data):
    channel_models = _channel_models(fit_model)

    if channel_models:
        if not isinstance(data, dict):
            raise ValueError("Expected per-channel dataset dictionary for channel-based model")

        combined_loss = None
        for index, (channel, model) in enumerate(channel_models.items()):
            if channel not in data:
                raise ValueError(f"Missing dataset for channel '{channel}'")
            constraints = fit_model.constraints if index == 0 else []
            if resolved_fit_mode == "binned":
                if not isinstance(binned_model, dict) or channel not in binned_model:
                    raise ValueError(f"Missing binned model for channel '{channel}'")
                loss = zfit.loss.ExtendedBinnedNLL(
                    model=binned_model[channel],
                    data=data[channel],
                    constraints=constraints,
                )
            else:
                loss = zfit.loss.ExtendedUnbinnedNLL(
                    model=model,
                    data=data[channel],
                    constraints=constraints,
                )
            combined_loss = loss if combined_loss is None else combined_loss + loss
        return combined_loss

    if resolved_fit_mode == "binned":
        return zfit.loss.ExtendedBinnedNLL(
            model=binned_model,
            data=data,
            constraints=fit_model.constraints,
        )

    return zfit.loss.ExtendedUnbinnedNLL(
        model=fit_model.model,
        data=data,
        constraints=fit_model.constraints,
    )


# ===========================================================================
# Parameter-of-interest discovery
# ===========================================================================

def _resolve_process_key(process_map, process_name):
    if not process_map or process_name is None:
        return None

    if process_name in process_map:
        return process_name

    suffixed = [name for name in process_map if name.startswith(f"{process_name}__")]
    if len(suffixed) == 1:
        return suffixed[0]

    if "__" in process_name:
        base_name = process_name.split("__", 1)[0]
        if base_name in process_map:
            return base_name

    return None


def _find_signal_parameter(fit_model):
    if fit_model.signal_process is not None:
        target = f"mu_{fit_model.signal_process}"
        for param in _all_params(fit_model):
            if param.name == target:
                return param

    for param in _all_params(fit_model):
        if is_signal_strength_poi(param.name):
            return param

    if fit_model.signal_process and getattr(fit_model, "yields", None):
        matched_key = _resolve_process_key(fit_model.yields, fit_model.signal_process)
        if matched_key is not None:
            return fit_model.yields[matched_key]

    return None


def _resolve_poi_parameter(fit_model, poi_name=None, promote_poi=False):
    if poi_name is not None:
        poi_param = find_parameter_by_name(fit_model, poi_name)
        if poi_param is None:
            raise ValueError(f"Could not find parameter '{poi_name}' in model")
        if promote_poi and hasattr(poi_param, "floating"):
            poi_param.floating = True
        return poi_param

    poi_param = _find_signal_parameter(fit_model)
    if poi_param is not None and promote_poi and hasattr(poi_param, "floating"):
        poi_param.floating = True
    return poi_param


# ===========================================================================
# Scan-range defaults (kept here for compatibility; also available from
# backends/analysis_common.py via the generic helpers)
# ===========================================================================

def _default_scan_max(signal_param, fit_model):
    if signal_param is None:
        return None
    if is_signal_strength_poi(signal_param.name):
        return 5.0
    if fit_model.signal_nominal_yield is not None:
        return max(50.0, 3.0 * fit_model.signal_nominal_yield)
    return 50.0


def _default_cls_scan_points(fit_model, resolved_fit_mode, cls_scan_points):
    if cls_scan_points is not None:
        if cls_scan_points < 3:
            raise ValueError("--cls-scan-points must be >= 3")
        return int(cls_scan_points)

    if resolved_fit_mode == "binned" and is_likely_counting_model(fit_model):
        return 9
    return 25


# ===========================================================================
# Observed / Asimov dataset helpers (zfit-specific format conversions)
# ===========================================================================

def _observed_dataset_to_values(dataset, fallback_space=None):
    value_method = getattr(dataset, "value", None)
    if callable(value_method):
        return np.asarray(value_method(), dtype=float).reshape(-1)

    values_method = getattr(dataset, "values", None)
    if callable(values_method):
        values = np.asarray(values_method(), dtype=float).reshape(-1)
        data_space = getattr(dataset, "space", fallback_space)
        obs_names = tuple(getattr(data_space, "obs", ()) or ()) if data_space is not None else ()
        has_binning = False
        if len(obs_names) == 1 and data_space is not None:
            try:
                _ = data_space.binning[obs_names[0]].edges
                has_binning = True
            except Exception:
                has_binning = False

        if has_binning:
            edges = np.asarray(data_space.binning[obs_names[0]].edges, dtype=float)
            centers = 0.5 * (edges[:-1] + edges[1:])
            counts = np.maximum(np.rint(values).astype(int), 0)
            return np.repeat(centers, counts)

        return values

    return np.array([float(dataset)], dtype=float)


def _values_to_unbinned_dataset(values, obs_space):
    values = np.asarray(values, dtype=float).reshape(-1)
    unbinned_space = obs_space
    if obs_space is not None and hasattr(obs_space, "obs"):
        try:
            unbinned_space = zfit.Space(obs=obs_space.obs, limits=obs_space.limits)
        except Exception:
            unbinned_space = obs_space

    n_obs = len(getattr(unbinned_space, "obs", ()) or ()) if unbinned_space is not None else 1
    if n_obs <= 1:
        array = values.reshape(-1, 1)
    else:
        array = values.reshape(-1, n_obs)
    return zfit.Data.from_numpy(obs=unbinned_space, array=array)


def _build_observed_input(fit_model, resolved_fit_mode, binned_space):
    channel_models = _channel_models(fit_model)

    if channel_models and resolved_fit_mode == "binned":
        if not isinstance(fit_model.data, dict):
            raise ValueError("Observed data for channel-based binned fit must be a per-channel dictionary")
        if not isinstance(binned_space, dict):
            raise ValueError("Expected per-channel binned spaces for channel-based binned fit")

        channel_data = {}
        channel_binned = {}
        for channel, dataset in fit_model.data.items():
            if channel not in binned_space:
                raise ValueError(f"Missing binned space for channel '{channel}'")
            edges = _binning_edges_as_float_array(binned_space[channel].binning)
            values = _observed_dataset_to_values(dataset, binned_space[channel])
            counts, _ = np.histogram(values, bins=edges)
            channel_data[channel] = zfit.data.BinnedData.from_tensor(
                space=binned_space[channel],
                values=counts.astype(float),
            )
            channel_binned[channel] = {
                "edges": edges.tolist(),
                "counts": counts.astype(float).tolist(),
                "values": values.tolist(),
            }

        dataset_plot = {
            "mode": "binned",
            "channel_binned": channel_binned,
            "observed": True,
        }
        return channel_data, None, dataset_plot

    if resolved_fit_mode == "binned":
        edges = _binning_edges_as_float_array(binned_space.binning)
        if hasattr(fit_model.data, "value") or hasattr(fit_model.data, "values"):
            observed_values = _observed_dataset_to_values(fit_model.data, binned_space)
            counts, _ = np.histogram(observed_values, bins=edges)
            if hasattr(fit_model.data, "to_binned"):
                data = fit_model.data.to_binned(binned_space)
            else:
                data = zfit.data.BinnedData.from_tensor(space=binned_space, values=counts.astype(float))
        else:
            observed_count = float(fit_model.data)
            observed_values = np.array([observed_count], dtype=float)
            counts = np.array([observed_count], dtype=float)
            data = zfit.data.BinnedData.from_tensor(space=binned_space, values=counts)

        dataset_plot = {
            "mode": "binned",
            "edges": edges,
            "counts": counts.astype(float),
            "values": observed_values,
            "observed": True,
        }
        if getattr(fit_model, "observed_counts_by_channel", None):
            dataset_plot["channel_counts"] = {
                k: float(v)
                for k, v in fit_model.observed_counts_by_channel.items()
            }
        return data, None, dataset_plot

    data = fit_model.data
    if channel_models:
        if not isinstance(data, dict):
            raise ValueError("Observed data for mixed-observable channels must be a per-channel dictionary")
        channel_data = {}
        channel_values = {}
        for channel, dataset in data.items():
            values = _observed_dataset_to_values(dataset)
            channel_values[channel] = values
            if hasattr(dataset, "value") and not hasattr(dataset, "values"):
                channel_data[channel] = dataset
            else:
                channel_data[channel] = _values_to_unbinned_dataset(values, channel_models[channel].space)
        dataset_plot = {"mode": "unbinned", "channel_values": channel_values, "observed": True}
        return channel_data, None, dataset_plot

    observed_values = _observed_dataset_to_values(data)
    unbinned_data = data
    if not hasattr(data, "value") or hasattr(data, "values"):
        unbinned_data = _values_to_unbinned_dataset(observed_values, fit_model.model.space)
    dataset_plot = {"mode": "unbinned", "values": observed_values, "observed": True}
    if getattr(fit_model, "observed_values_by_channel", None):
        dataset_plot["channel_values"] = {
            k: np.asarray(v, dtype=float).reshape(-1)
            for k, v in fit_model.observed_values_by_channel.items()
        }
    return unbinned_data, None, dataset_plot


def _build_iteration_input(
    fit_model,
    resolved_fit_mode,
    binned_model,
    binned_space,
    is_counting,
    data_mode,
):
    if data_mode == "observed":
        return _build_observed_input(fit_model, resolved_fit_mode, binned_space)
    if data_mode == "asimov":
        return _build_asimov_binned_data(binned_model, binned_space, fit_model)
    return _build_toy_data(
        fit_model=fit_model,
        resolved_fit_mode=resolved_fit_mode,
        binned_space=binned_space,
        is_counting=is_counting,
    )


# ===========================================================================
# Main entry point
# ===========================================================================

def run_analysis(
    fit_model,
    toys,
    use_observed_data=False,
    use_asimov_data=False,
    cls_alpha=None,
    signal_strength=None,
    scan_max=None,
    fit_mode="auto",
    binned_bins=40,
    cls_scan_points=None,
    cls_smart_scan=False,
    profile_scan=False,
    poi_name=None,
    promote_poi=False,
    poi_scan_points=41,
    poi_scan_max=None,
    progress_callback=None,
    feldman_cousins_alpha=None,
    feldman_cousins_scan_points=21,
    feldman_cousins_n_toys=100,
    feldman_cousins_scan_max=None,
    checkpoint_freq=None,
    checkpoint_path=None,
    existing_results=None,
    resume_from_index=0,
    compute_nll_scan=False,
    nll_scan_points=121,
    ntoy_plots=0,
):
    """Run the zfit analysis, delegating backend-agnostic work to the common orchestrator.

    The function:
      1. Resolves fit mode, binned spaces/models, and the POI parameter.
      2. Constructs a ``ZfitAnalysisState`` and a ``ZfitAnalysisBackend`` adapter.
      3. Delegates to ``run_analysis_common`` from ``backends/analysis_common.py``
         for all CLs / FC / NLL-scan / loop logic.
    """
    from zmodel.analysis_backend import ZfitAnalysisState, ZfitAnalysisBackend

    # ------------------------------------------------------------------
    # 1. Setup
    # ------------------------------------------------------------------
    resolved_fit_mode = _resolve_fit_mode(fit_mode, fit_model)
    is_counting = is_likely_counting_model(fit_model)

    signal_param = _find_signal_parameter(fit_model)
    if cls_alpha is not None and signal_param is None:
        raise ValueError("Could not identify a signal parameter for CLs evaluation")
    cls_points = _default_cls_scan_points(fit_model, resolved_fit_mode, cls_scan_points)

    poi_param = _resolve_poi_parameter(fit_model, poi_name=poi_name, promote_poi=promote_poi)
    if poi_param is None:
        raise ValueError("Could not identify a parameter of interest")

    minimizer = zfit.minimize.Minuit()
    binned_space = None
    binned_model = None
    if resolved_fit_mode == "binned":
        if _channel_models(fit_model):
            binned_space = _build_channel_binned_spaces(fit_model, binned_bins)
            binned_model = _build_channel_binned_models(fit_model, binned_space)
        else:
            binned_space = _build_binned_space(fit_model, binned_bins)
            binned_model = fit_model.model.to_binned(binned_space)

    if use_asimov_data and resolved_fit_mode != "binned":
        raise ValueError("--toys -1 is only supported for binned fits")

    if checkpoint_freq is not None and checkpoint_freq < 1:
        raise ValueError("checkpoint_freq must be >= 1")
    if int(nll_scan_points) < 3:
        raise ValueError("nll_scan_points must be >= 3")

    data_mode = _resolve_data_mode(use_observed_data, use_asimov_data)
    resume_index = int(resume_from_index)
    if resume_index < 0:
        raise ValueError("resume_from_index must be >= 0")

    # ------------------------------------------------------------------
    # 2. Build the initial dataset and loss so the state is fully initialised
    # ------------------------------------------------------------------
    initial_data, _count, _data_plot = _build_iteration_input(
        fit_model=fit_model,
        resolved_fit_mode=resolved_fit_mode,
        binned_model=binned_model,
        binned_space=binned_space,
        is_counting=is_counting,
        data_mode=data_mode,
    )
    initial_loss = _build_loss(
        fit_model=fit_model,
        resolved_fit_mode=resolved_fit_mode,
        binned_model=binned_model,
        data=initial_data,
    )

    # POI bounds for scan-range heuristics
    poi_lower = getattr(poi_param, "lower", None)
    poi_upper = getattr(poi_param, "upper", None)
    poi_bounds = None
    if poi_lower is not None and poi_upper is not None:
        try:
            poi_bounds = (float(poi_lower), float(poi_upper))
        except Exception:
            pass

    poi_is_ss = is_signal_strength_poi(poi_param.name)

    # ------------------------------------------------------------------
    # 3. Construct state and adapter
    # ------------------------------------------------------------------
    state = ZfitAnalysisState(
        fit_model=fit_model,
        resolved_fit_mode=resolved_fit_mode,
        binned_model=binned_model,
        binned_space=binned_space,
        is_counting=is_counting,
        minimizer=minimizer,
        signal_param=signal_param if signal_param is not None else poi_param,
        poi_param=poi_param,
        current_data=initial_data,
        current_loss=initial_loss,
        _signal_nominal_yield=fit_model.signal_nominal_yield,
        _poi_is_signal_strength=poi_is_ss,
    )
    backend = ZfitAnalysisBackend.from_state(state)

    # ------------------------------------------------------------------
    # 4. Override set_data to also store the dataset_plot dict for
    #    reporting purposes (the common orchestrator doesn't know about plots)
    # ------------------------------------------------------------------

    summaries = run_analysis_common(
        backend=backend,
        state=state,
        toys=int(toys),
        data_mode=data_mode,
        cls_alpha=cls_alpha,
        signal_strength=signal_strength,
        cls_scan_points=cls_points,
        cls_smart_scan=bool(cls_smart_scan),
        poi_scan_max=poi_scan_max if poi_scan_max is not None else scan_max,
        poi_bounds=poi_bounds,
        feldman_cousins_alpha=feldman_cousins_alpha,
        feldman_cousins_scan_points=int(feldman_cousins_scan_points),
        feldman_cousins_n_toys=int(feldman_cousins_n_toys),
        feldman_cousins_scan_max=feldman_cousins_scan_max,
        compute_nll_scan=bool(compute_nll_scan),
        nll_scan_points=int(nll_scan_points),
        progress_callback=progress_callback,
        checkpoint_freq=checkpoint_freq,
        checkpoint_path=checkpoint_path,
        existing_results=list(existing_results) if existing_results else [],
        resume_from_index=resume_index,
    )

    # ------------------------------------------------------------------
    # 5. Enrich summaries with zfit-specific dataset_plot info
    #    (regenerate for each summary using the stored data_mode)
    # ------------------------------------------------------------------
    _enrich_summaries_with_dataset_plots(
        summaries=summaries,
        fit_model=fit_model,
        resolved_fit_mode=resolved_fit_mode,
        binned_model=binned_model,
        binned_space=binned_space,
        is_counting=is_counting,
        data_mode=data_mode,
        ntoy_plots=int(ntoy_plots),
        resume_index=resume_index,
    )

    return summaries


def _enrich_summaries_with_dataset_plots(
    summaries,
    fit_model,
    resolved_fit_mode,
    binned_model,
    binned_space,
    is_counting,
    data_mode,
    ntoy_plots,
    resume_index,
):
    """Backfill dataset_plot entries for the first *ntoy_plots* summaries.

    The common orchestrator does not carry plot metadata; we regenerate it
    here from the stored parameter snapshot for the first ntoy_plots datasets.
    This is a best-effort enrichment that does not re-run the fit.
    """
    # For observed/Asimov we can build the plot dict without re-running.
    # For toys the plot data was not saved; we skip enrichment beyond what
    # the common orchestrator already stored.
    if data_mode == "observed":
        try:
            _data, _count, dataset_plot = _build_observed_input(
                fit_model, resolved_fit_mode, binned_space
            )
            for summary in summaries:
                if "dataset_plot" not in summary:
                    summary["dataset_plot"] = dataset_plot
        except Exception:
            pass
    elif data_mode == "asimov":
        try:
            _data, _counts, dataset_plot = _build_asimov_binned_data(
                binned_model, binned_space, fit_model
            )
            for summary in summaries:
                if "dataset_plot" not in summary:
                    summary["dataset_plot"] = dataset_plot
        except Exception:
            pass
