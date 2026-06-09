import json
import os
import time

import numpy as np
import pyhf

from backends.path_bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path(__file__)

from backends.analysis_console import print_dataset_summary_header, print_limit_summary_lines
from backends.analysis_common import load_analysis_model, normalize_output_path, resolve_dataset_mode
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
from backends.print_model_helpers import print_model_info

from hfmodel.analysis_core import configure_runtime, run_analysis
from hfmodel.analysis_overrides import apply_parameter_overrides
from hfmodel.analyze_plotting import plot_summary_artifacts
from hfmodel.build_model_from_text import build_model_from_card, parse_model_card
from hfmodel.model_io import load_fit_model


def _load_analysis_model(model_file=None, input_card=None):
    return load_analysis_model(
        model_file=model_file,
        input_card=input_card,
        load_fit_model_fn=load_fit_model,
        parse_model_card_fn=parse_model_card,
        build_model_from_card_fn=build_model_from_card,
    )


def _configure_pyhf_backend(backend_name):
    backend = str(backend_name or "scipy").strip().lower()
    if backend not in {"scipy", "minuit", "jax"}:
        raise ValueError(f"Unsupported --backend value: {backend_name}")

    try:
        pyhf.set_backend("numpy", backend)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to configure pyhf backend '{backend}'. Ensure required dependencies are installed."
        ) from exc

    return backend


def _print_dataset_summary(summary, is_observed_fit=False):
    print_dataset_summary_header(
        summary,
        is_observed_fit=is_observed_fit,
        fit_precision=".4g",
        unc_precision=".4g",
        asimov_label="Asimov",
        observed_label="Observed",
    )
    print_limit_summary_lines(
        summary,
        include_scan_details=False,
        include_expected_error=False,
        include_yield_upper=False,
        feldman_status_prefix="Feldman-Cousins status",
        expected_label="CLs expected",
        expected_precision=".4g",
        feldman_interval_precision=".4g",
    )



def _build_ensemble_evaluation_report(summaries, total_time_s):
    report = init_ensemble_report(summaries, total_time_s)

    if not summaries:
        return report

    add_fit_quality(report, summaries, include_invalid_fraction=False)
    add_poi_distributions(report, summaries)

    cls_obs = [summary.get("cls_observed") for summary in summaries if summary.get("cls_observed") is not None]
    if cls_obs:
        report["cls"] = {
            "observed_limit": distribution_summary(cls_obs),
            "n_failures": int(sum(1 for summary in summaries if "cls_error" in summary)),
        }

    return report


def _save_analysis_snapshot(output_path, fit_model, summaries, args):
    snapshot = {
        "format": "hfmodel_analysis_snapshot_v2",
        "workspace": fit_model.workspace,
        "model_metadata": {
            "channels": fit_model.channels,
            "process_names": fit_model.process_names,
            "process_ids": fit_model.process_ids,
            "signal_processes": fit_model.signal_processes,
            "measurement_name": fit_model.measurement_name,
            "poi_name": fit_model.poi_name,
        },
        "observed_counts_by_channel": fit_model.observed_counts_by_channel,
        "summaries": summaries,
        "config": {
            "model_file": args.model_file,
            "input_card": args.input_card,
            "toys": args.toys,
            "jobs": args.jobs,
            "cls_alpha": args.cls,
            "signal_strength": args.signal_strength,
            "poi_scan_max": args.poi_scan_max,
            "cls_scan_points": args.cls_scan_points,
            "plot": bool(args.plot),
            "plot_dir": args.plot_dir,
            "ntoys_plot": args.ntoys_plot,
            "set_parameters": args.set_parameters,
            "freeze_parameters": args.freeze_parameters,
            "set_parameter_ranges": args.set_parameter_ranges,
            "backend": args.backend,
            "hessian_method": args.hessian_method,
        },
    }

    final_path = normalize_output_path(output_path, ".json")
    with open(final_path, "w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, indent=2)
    return final_path


def run_analysis_cli(args):
    if args.resume_from:
        raise ValueError(
            "--resume-from is not implemented in the pyhf analysis backend yet. "
            "Run without resume or use checkpoint JSON outputs as external bookkeeping."
        )

    backend = _configure_pyhf_backend(args.backend)
    print(f"Using pyhf backend: {backend}")

    fit_model = _load_analysis_model(model_file=args.model_file, input_card=args.input_card)

    # Construct set_ranges_spec to include --poi-min and --poi-max if provided
    set_ranges_spec = args.set_parameter_ranges or ""
    poi_name = getattr(fit_model, "poi_name", None)
    
    # Append POI bounds to the ranges spec if provided
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

    apply_parameter_overrides(
        fit_model,
        set_values_spec=args.set_parameters,
        set_ranges_spec=set_ranges_spec,
        freeze_spec=args.freeze_parameters,
    )

    has_observed_data = hasattr(fit_model, "data") and fit_model.data is not None
    use_observed_data, use_asimov_data, n_toys = resolve_dataset_mode(args.toys, has_observed_data)

    if int(getattr(args, "jobs", 1) or 1) > 1:
        print("Note: pyhf analysis currently runs sequentially; --jobs is not yet used.")

    configure_runtime()

    total_start = time.perf_counter()
    summaries = run_analysis(
        fit_model,
        toys=n_toys,
        use_observed_data=use_observed_data,
        use_asimov_data=use_asimov_data,
        cls_alpha=args.cls,
        signal_strength=args.signal_strength,
        cls_scan_points=args.cls_scan_points,
        cls_smart_scan=args.cls_smart_scan,
        poi_scan_max=args.poi_scan_max,
        feldman_cousins_alpha=args.feldman_cousins,
        feldman_cousins_scan_points=args.fc_scan_points,
        feldman_cousins_n_toys=args.fc_toys,
        feldman_cousins_scan_max=args.fc_scan_max,
        progress_callback=_print_dataset_summary,
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_path=(f"{args.output}.checkpoint.json" if args.checkpoint_freq else None),
        existing_results=[],
        resume_from_index=0,
        compute_nll_scan=args.plot,
        nll_scan_points=args.nll_scan_points,
        seed=args.seed,
        backend_name=args.backend,
        hessian_method=args.hessian_method,
    )
    total_time_s = time.perf_counter() - total_start

    print(f"Analyzed pyhf workspace model with channels: {', '.join(fit_model.channels)}")
    if (not use_observed_data) and (not use_asimov_data) and len(summaries) > 1:
        poi_values = np.asarray([item.get("poi_fit") for item in summaries], dtype=float)
        poi_values = poi_values[np.isfinite(poi_values)]
        if poi_values.size:
            p2p5, p16, p84, p97p5 = np.percentile(poi_values, [2.5, 16.0, 84.0, 97.5])
            print(
                "Toy POI bounds: "
                f"1 sigma [{p16:.4g}, {p84:.4g}], "
                f"2 sigma [{p2p5:.4g}, {p97p5:.4g}]"
            )
    
    # Print model information if requested
    if getattr(args, "print_model", False) and summaries:
        first_summary = summaries[0]
        print_model_info(fit_model, first_summary, "hfmodel")
    
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
