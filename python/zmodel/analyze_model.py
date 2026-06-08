import dill
import multiprocessing as mp
import os
import time
import warnings

# Reduce TensorFlow C++ logging noise before zfit/tensorflow import.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_VLOG_LEVEL", "3")
os.environ.setdefault("AUTOGRAPH_VERBOSITY", "0")

import numpy as np
import tensorflow as tf
import zfit

from backends.path_bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path(__file__)

from backends.analysis_console import print_dataset_summary_header, print_limit_summary_lines
from backends.analysis_common import (
    checkpoint_mismatches,
    load_analysis_model,
    resolve_data_mode,
    resolve_dataset_mode,
)
from backends.zfit_parameter_utils import find_parameter_by_name
from backends.analysis_reporting import (
    add_fit_quality,
    add_poi_distributions,
    distribution_summary,
    init_ensemble_report,
    print_runtime_summary,
    resolve_output_or_default,
    save_and_print_ensemble_report,
    save_and_print_snapshot,
    maybe_plot_summary_artifacts,
)

from zmodel.build_model_from_text import build_model_from_card, parse_model_card
from zmodel.model_io import load_fit_model
from zmodel.analysis_core import configure_runtime, run_analysis
from zmodel.analysis_overrides import apply_parameter_overrides
from zmodel.analyze_plotting import plot_summary_artifacts


# Also silence python-side TensorFlow and absl warning emitters.
tf.get_logger().setLevel("ERROR")
try:
    from absl import logging as absl_logging

    absl_logging.set_verbosity("error")
except Exception:
    pass

try:
    tf.config.optimizer.set_experimental_options({"loop_optimization": False})
except Exception:
    pass

# Suppress known non-fatal runtime warnings that clutter command-line output.
warnings.filterwarnings(
    "ignore",
    message=r"Called analytic integral to test if available, but unknown error occured:.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Flow currently not fully supported\. Values outside the edges are all 0\.",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Exception occurred, parameter values are not reset and in an arbitrary, last used state\..*",
    category=RuntimeWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"This SumPDF is built with fracs .* and all pdf are extended:.*",
)


def _load_analysis_model(model_file=None, input_card=None):
    return load_analysis_model(
        model_file=model_file,
        input_card=input_card,
        load_fit_model_fn=load_fit_model,
        parse_model_card_fn=parse_model_card,
        build_model_from_card_fn=build_model_from_card,
    )


def _print_dataset_summary(summary, is_observed_fit=False):
    print_dataset_summary_header(
        summary,
        is_observed_fit=is_observed_fit,
        fit_precision=".3g",
        unc_precision=".3g",
        unconstrained_text="unconstrained",
        asimov_label="Asimov data",
        observed_label="Observed data",
        include_dataset_plot_flags=True,
    )
    # if "count" in summary:
    #     print(f"  Toy count: {summary['count']}")
    if "poi_hat" in summary:
        print(f"  POI ({summary['poi_name']}) profiled best fit: {summary['poi_hat']:.6f}")
        print(
            f"  POI scan range: [{summary['poi_scan_low']:.6f}, {summary['poi_scan_high']:.6f}] "
            f"with {summary['poi_scan_points']} points"
        )
    print_limit_summary_lines(
        summary,
        include_scan_details=True,
        include_expected_error=True,
        include_yield_upper=True,
        feldman_status_prefix="Feldman-Cousins",
        expected_label="CLs expected (asymptotic, b-only fit)",
        expected_precision=".4f",
    )


def _save_analysis_snapshot(output_path, fit_model, summaries, args):
    payload = {
        "format": "analyze_model_snapshot_v1",
        "fit_model": fit_model,
        "input_data": fit_model.data,
        "summaries": summaries,
        "config": {
            "model_file": args.model_file,
            "input_card": args.input_card,
            "toys": args.toys,
            "fit_mode": args.fit_mode,
            "binned_bins": args.binned_bins,
            "graph_mode": args.graph_mode,
            "cls_alpha": args.cls,
            "signal_strength": args.signal_strength,
            "scan_max": args.scan_max,
            "cls_scan_points": args.cls_scan_points,
            "cls_smart_scan": args.cls_smart_scan,
            "profile_scan": args.profile_scan,
            "poi_name": args.poi_name,
            "promote_poi": args.promote_poi,
            "poi_scan_points": args.poi_scan_points,
            "poi_scan_max": args.poi_scan_max,
            "feldman_cousins": args.feldman_cousins,
            "feldman_cousins_scan_points": args.fc_scan_points,
            "feldman_cousins_n_toys": args.fc_toys,
            "feldman_cousins_scan_max": args.fc_scan_max,
            "report_file": args.report_file,
            "nll_scan_points": args.nll_scan_points,
            "ntoys_plot": args.ntoys_plot,
            "set_parameters": args.set_parameters,
            "freeze_parameters": args.freeze_parameters,
            "set_parameter_ranges": args.set_parameter_ranges,
        },
    }

    final_path = os.path.abspath(output_path)
    with open(final_path, "wb") as handle:
        dill.dump(payload, handle)
    return final_path


def _build_ensemble_evaluation_report(summaries, total_time_s):
    report = init_ensemble_report(summaries, total_time_s)

    if not summaries:
        return report

    add_fit_quality(report, summaries, include_invalid_fraction=True)
    add_poi_distributions(report, summaries)

    coverage_values = []
    for summary in summaries:
        truth = summary.get("poi_true")
        fit = summary.get("poi_fit")
        unc = summary.get("poi_unc_hesse")
        if truth is None or fit is None or unc is None:
            continue
        truth = float(truth)
        fit = float(fit)
        unc = float(unc)
        if not (np.isfinite(truth) and np.isfinite(fit) and np.isfinite(unc) and unc > 0.0):
            continue
        coverage_values.append((truth, fit, unc))

    if coverage_values:
        within_1sigma = 0
        within_95pct = 0
        for truth, fit, unc in coverage_values:
            if abs(fit - truth) <= unc:
                within_1sigma += 1
            if abs(fit - truth) <= 1.96 * unc:
                within_95pct += 1
        n_cov = len(coverage_values)
        report["coverage"] = {
            "n": int(n_cov),
            "within_1sigma": float(within_1sigma / n_cov),
            "within_95pct": float(within_95pct / n_cov),
        }

    cls_obs = [summary.get("cls_observed") for summary in summaries if "cls_observed" in summary]
    cls_yield = [summary.get("yield_upper_limit") for summary in summaries if "yield_upper_limit" in summary]
    cls_failures = [summary for summary in summaries if "cls_error" in summary]
    if cls_obs or cls_yield or cls_failures:
        report["cls"] = {
            "observed_limit": distribution_summary(cls_obs),
            "yield_upper_limit": distribution_summary(cls_yield),
            "n_failures": int(len(cls_failures)),
            "failure_fraction": float(len(cls_failures) / len(summaries)),
        }

    fc_entries = [summary.get("feldman_cousins") for summary in summaries if "feldman_cousins" in summary]
    if fc_entries:
        fc_ok = 0
        fc_fail = 0
        fc_widths = []
        for entry in fc_entries:
            if not isinstance(entry, dict):
                fc_fail += 1
                continue
            status = str(entry.get("fc_status", "")).lower()
            if "ok" in status:
                fc_ok += 1
            else:
                fc_fail += 1
            interval = entry.get("fc_interval")
            if isinstance(interval, (list, tuple)) and len(interval) == 2:
                low, high = interval
                if low is not None and high is not None:
                    low = float(low)
                    high = float(high)
                    if np.isfinite(low) and np.isfinite(high):
                        fc_widths.append(high - low)

        report["feldman_cousins"] = {
            "n_evaluated": int(len(fc_entries)),
            "n_ok": int(fc_ok),
            "n_non_ok": int(fc_fail),
            "width": distribution_summary(fc_widths),
        }

    return report


def _split_dataset_ranges(n_datasets, n_jobs):
    n_jobs = max(1, min(int(n_jobs), int(n_datasets)))
    base = n_datasets // n_jobs
    rem = n_datasets % n_jobs
    ranges = []
    start = 0
    for worker_idx in range(n_jobs):
        size = base + (1 if worker_idx < rem else 0)
        end = start + size
        if start < end:
            ranges.append((worker_idx, start, end))
        start = end
    return ranges


def _run_parallel_worker(task):
    worker_index = int(task["worker_index"])
    start_index = int(task["start_index"])
    end_index = int(task["end_index"])

    fit_model = _load_analysis_model(model_file=task.get("model_file"), input_card=task.get("input_card"))
    apply_parameter_overrides(
        fit_model,
        set_values_spec=task.get("set_parameters"),
        set_ranges_spec=task.get("set_parameter_ranges"),
        freeze_spec=task.get("freeze_parameters"),
    )

    zfit.settings.set_seed(int(task["seed"]) + worker_index)
    configure_runtime(task["graph_mode"], fit_model, end_index - start_index)

    summaries = run_analysis(
        fit_model,
        toys=end_index,
        use_observed_data=False,
        use_asimov_data=False,
        cls_alpha=task.get("cls_alpha"),
        signal_strength=task.get("signal_strength"),
        scan_max=task.get("scan_max"),
        fit_mode=task["fit_mode"],
        binned_bins=int(task["binned_bins"]),
        cls_scan_points=task.get("cls_scan_points"),
        cls_smart_scan=bool(task.get("cls_smart_scan", False)),
        profile_scan=bool(task.get("profile_scan", False)),
        poi_name=task.get("poi_name"),
        promote_poi=bool(task.get("promote_poi", False)),
        poi_scan_points=int(task.get("poi_scan_points", 41)),
        poi_scan_max=task.get("poi_scan_max"),
        feldman_cousins_alpha=task.get("feldman_cousins_alpha"),
        feldman_cousins_scan_points=int(task.get("fc_scan_points", 21)),
        feldman_cousins_n_toys=int(task.get("fc_toys", 100)),
        feldman_cousins_scan_max=task.get("fc_scan_max"),
        progress_callback=None,
        checkpoint_freq=None,
        checkpoint_path=None,
        existing_results=None,
        resume_from_index=start_index,
        compute_nll_scan=bool(task.get("compute_nll_scan", False)),
        nll_scan_points=int(task.get("nll_scan_points", 121)),
        ntoy_plots=int(task.get("ntoys_plot", 0)),
    )
    return summaries



def run_analysis_cli(args):
    fit_model = _load_analysis_model(model_file=args.model_file, input_card=args.input_card)

    # Construct set_ranges_spec to include --poi-min and --poi-max if provided
    set_ranges_spec = args.set_parameter_ranges or ""
    poi_name = getattr(fit_model, "poi_name", None)
    if poi_name is None and hasattr(fit_model, "model"):
        # Try to infer POI name from signal parameter
        from zmodel.analysis_core import _resolve_poi_parameter
        poi_param = _resolve_poi_parameter(fit_model)
        if poi_param:
            poi_name = poi_param.name
    
    # Append POI bounds to the ranges spec if provided
    # For signal strength parameters (mu_*), default minimum is 0
    # For other parameters promoted to POI, use their existing bounds
    poi_ranges = []
    
    # Determine if this is a signal strength parameter
    poi_is_signal_strength = poi_name is not None and poi_name.startswith("mu_")
    
    # Get existing parameter bounds if POI is a model parameter
    default_poi_min = None
    default_poi_max = None
    if poi_name is not None:
        poi_param = find_parameter_by_name(fit_model, poi_name) if hasattr(fit_model, 'model') else None
        if poi_param is not None:
            default_poi_min = getattr(poi_param, 'lower', None)
            default_poi_max = getattr(poi_param, 'upper', None)
    
    # Set POI minimum
    poi_min = getattr(args, "poi_min", None)
    if poi_min is not None:
        poi_ranges.append(f"{poi_name}={poi_min}")
    elif poi_is_signal_strength:
        # Default minimum for signal strength parameters is 0
        poi_ranges.append(f"{poi_name}=0")
    elif default_poi_min is not None:
        # Use existing parameter bound for promoted POI
        try:
            poi_ranges.append(f"{poi_name}={float(default_poi_min)}")
        except (ValueError, TypeError):
            pass  # If conversion fails, don't set a default
    
    # Set POI maximum
    poi_max = getattr(args, "poi_max", None)
    if poi_max is not None:
        if poi_ranges:
            poi_ranges[-1] += f":{poi_max}"
        else:
            # Only poi_max provided; need to construct with explicit minimum
            if poi_is_signal_strength:
                poi_ranges.append(f"{poi_name}=0:{poi_max}")
            elif default_poi_min is not None:
                try:
                    poi_ranges.append(f"{poi_name}={float(default_poi_min)}:{poi_max}")
                except (ValueError, TypeError):
                    poi_ranges.append(f"{poi_name}=:{poi_max}")
            else:
                poi_ranges.append(f"{poi_name}=:{poi_max}")
    
    if poi_ranges:
        if set_ranges_spec:
            set_ranges_spec = f"{set_ranges_spec},{','.join(poi_ranges)}"
        else:
            set_ranges_spec = ",".join(poi_ranges)

    apply_parameter_overrides(
        fit_model,
        set_values_spec=args.set_parameters,
        set_ranges_spec=set_ranges_spec,
        freeze_spec=args.freeze_parameters,
    )

    zfit.settings.set_seed(args.seed)

    has_observed_data = hasattr(fit_model, "data") and fit_model.data is not None
    use_observed_data, use_asimov_data, n_toys = resolve_dataset_mode(args.toys, has_observed_data)

    n_jobs = int(getattr(args, "jobs", 1) or 1)
    if n_jobs < 1:
        raise ValueError("--jobs must be >= 1")
    if int(args.nll_scan_points) < 3:
        raise ValueError("--nll-scan-points must be >= 3")

    configure_runtime(args.graph_mode, fit_model, n_toys)
    total_start = time.perf_counter()

    # Load checkpoint if resuming
    existing_results = []
    resume_from_index = 0
    if args.resume_from:
        try:
            with open(args.resume_from, "rb") as f:
                checkpoint = dill.load(f)
                expected_checkpoint_config = {
                    "data_mode": resolve_data_mode(use_observed_data, use_asimov_data),
                    "fit_mode": args.fit_mode,
                    "cls_alpha": args.cls,
                    "signal_strength": args.signal_strength,
                    "scan_max": args.scan_max,
                    "cls_smart_scan": bool(args.cls_smart_scan),
                    "profile_scan": bool(args.profile_scan),
                    "poi_name": args.poi_name,
                    "poi_scan_points": int(args.poi_scan_points),
                    "poi_scan_max": args.poi_scan_max,
                    "feldman_cousins_alpha": args.feldman_cousins,
                    "feldman_cousins_scan_points": int(args.fc_scan_points),
                    "feldman_cousins_n_toys": int(args.fc_toys),
                    "feldman_cousins_scan_max": args.fc_scan_max,
                    "compute_nll_scan": bool(args.plot),
                    "nll_scan_points": int(args.nll_scan_points),
                    "ntoys_plot": int(args.ntoys_plot if args.plot else 0),
                }
                mismatches = checkpoint_mismatches(checkpoint, expected_checkpoint_config)
                if mismatches:
                    mismatch_text = ", ".join(
                        [f"{k}: checkpoint={old!r}, current={new!r}" for k, old, new in mismatches]
                    )
                    raise ValueError(
                        "Checkpoint is incompatible with current analysis settings: "
                        f"{mismatch_text}"
                    )

                existing_results = checkpoint.get("summaries", [])
                resume_from_index = len(existing_results)
                if resolve_data_mode(use_observed_data, use_asimov_data) == "toy":
                    print(f"Resumed from checkpoint: {len(existing_results)} toys already completed")
                else:
                    print(f"Resumed from checkpoint: {len(existing_results)} datasets already completed")
                if resume_from_index >= n_toys:
                    print(f"Already completed all {n_toys} datasets. Skipping analysis.")
                    summaries = existing_results
        except Exception as e:
            print(f"Warning: could not load checkpoint {args.resume_from}: {e}")

    if not hasattr(args, "resume_from") or not args.resume_from or resume_from_index < n_toys:
        can_parallelize = (
            n_jobs > 1
            and not use_observed_data
            and not use_asimov_data
            and n_toys > 1
            and not args.resume_from
            and args.checkpoint_freq is None
        )

        if can_parallelize:
            worker_ranges = _split_dataset_ranges(n_toys, n_jobs)
            tasks = []
            for worker_index, start_index, end_index in worker_ranges:
                tasks.append(
                    {
                        "worker_index": worker_index,
                        "start_index": start_index,
                        "end_index": end_index,
                        "model_file": args.model_file,
                        "input_card": args.input_card,
                        "set_parameters": args.set_parameters,
                        "freeze_parameters": args.freeze_parameters,
                        "set_parameter_ranges": args.set_parameter_ranges,
                        "seed": int(args.seed),
                        "graph_mode": args.graph_mode,
                        "cls_alpha": args.cls,
                        "signal_strength": args.signal_strength,
                        "scan_max": args.scan_max,
                        "fit_mode": args.fit_mode,
                        "binned_bins": int(args.binned_bins),
                        "cls_scan_points": args.cls_scan_points,
                        "cls_smart_scan": bool(args.cls_smart_scan),
                        "profile_scan": bool(args.profile_scan),
                        "poi_name": args.poi_name,
                        "promote_poi": bool(args.promote_poi),
                        "poi_scan_points": int(args.poi_scan_points),
                        "poi_scan_max": args.poi_scan_max,
                        "feldman_cousins_alpha": args.feldman_cousins,
                        "fc_scan_points": int(args.fc_scan_points),
                        "fc_toys": int(args.fc_toys),
                        "fc_scan_max": args.fc_scan_max,
                        "compute_nll_scan": bool(args.plot and start_index < args.ntoys_plot),
                        "nll_scan_points": int(args.nll_scan_points),
                        "ntoys_plot": int(args.ntoys_plot if args.plot else 0),
                    }
                )

            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=len(tasks)) as pool:
                worker_results = pool.map(_run_parallel_worker, tasks)

            summaries = []
            for chunk in worker_results:
                summaries.extend(chunk)
            summaries.sort(key=lambda item: int(item.get("dataset_id", 0)))
            for summary in summaries:
                _print_dataset_summary(summary)
        else:
            if n_jobs > 1 and (use_observed_data or use_asimov_data):
                print("Parallel processing is only applied to generated toy datasets; running sequentially.")
            if n_jobs > 1 and args.resume_from:
                print("Parallel processing is disabled when --resume-from is used; running sequentially.")
            if n_jobs > 1 and args.checkpoint_freq is not None:
                print("Parallel processing is disabled when --checkpoint-freq is set; running sequentially.")

            summaries = run_analysis(
                fit_model,
                toys=n_toys,
                use_observed_data=use_observed_data,
                use_asimov_data=use_asimov_data,
                cls_alpha=args.cls,
                signal_strength=args.signal_strength,
                scan_max=args.scan_max,
                fit_mode=args.fit_mode,
                binned_bins=args.binned_bins,
                cls_scan_points=args.cls_scan_points,
                cls_smart_scan=args.cls_smart_scan,
                profile_scan=args.profile_scan,
                poi_name=args.poi_name,
                promote_poi=args.promote_poi,
                poi_scan_points=args.poi_scan_points,
                poi_scan_max=args.poi_scan_max,
                feldman_cousins_alpha=args.feldman_cousins,
                feldman_cousins_scan_points=args.fc_scan_points,
                feldman_cousins_n_toys=args.fc_toys,
                feldman_cousins_scan_max=args.fc_scan_max,
                progress_callback=_print_dataset_summary,
                checkpoint_freq=args.checkpoint_freq,
                checkpoint_path=args.output + ".checkpoint" if args.checkpoint_freq else None,
                existing_results=existing_results,
                resume_from_index=resume_from_index,
                compute_nll_scan=args.plot,
                nll_scan_points=args.nll_scan_points,
                ntoy_plots=args.ntoys_plot if args.plot else 0
            )
        total_time_s = time.perf_counter() - total_start
    else:
        total_time_s = 0

    print(f"Analyzed model: {fit_model.model.name}")
    if args.cls is not None and summaries:
        first = summaries[0]
        if "cls_observed" in first:
            print(f"CLs observed upper limit (alpha={args.cls:g}): {first['cls_observed']:.4f}")
            if "cls_expected_quantiles" in first:
                q = first["cls_expected_quantiles"]
                print(
                    "CLs expected (asymptotic, b-only fit): "
                    f"2.5%={q['2.5%']:.4f}, 16%={q['16%']:.4f}, 50%={q['50%']:.4f}, "
                    f"84%={q['84%']:.4f}, 97.5%={q['97.5%']:.4f}"
                )
        elif "cls_error" in first:
            print(f"CLs failed (alpha={args.cls:g}): {first['cls_error']}")

    maybe_plot_summary_artifacts(args, summaries, fit_model, plot_summary_artifacts)

    print_runtime_summary(summaries, total_time_s)

    output_pkl = resolve_output_or_default(args.output, args.seed, ".pkl")

    save_and_print_ensemble_report(
        summaries=summaries,
        total_time_s=total_time_s,
        output_path=output_pkl,
        report_file=args.report_file,
        build_report_fn=_build_ensemble_evaluation_report,
    )

    save_and_print_snapshot(
        output_path=output_pkl,
        fit_model=fit_model,
        summaries=summaries,
        args=args,
        save_snapshot_fn=_save_analysis_snapshot,
    )
