import json
import os
import time

import numpy as np
import pyhf

from hfmodel.analysis_core import configure_runtime, run_analysis
from hfmodel.analysis_overrides import apply_parameter_overrides
from hfmodel.analyze_plotting import plot_summary_artifacts
from hfmodel.build_model_from_text import build_model_from_card, parse_model_card
from hfmodel.model_io import load_fit_model


def _load_analysis_model(model_file=None, input_card=None):
    if model_file is not None:
        return load_fit_model(os.path.abspath(model_file))

    card_path = os.path.abspath(input_card)
    card = parse_model_card(card_path)
    return build_model_from_card(card, os.path.dirname(card_path))


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
    poi_label = summary.get("poi_name", "poi")
    poi_fit = summary.get("poi_fit")
    poi_unc = summary.get("poi_unc_hesse")
    fit_text = f"{poi_fit:.4g}" if poi_fit is not None else "n/a"
    unc_text = f"{poi_unc:.4g}" if poi_unc is not None else "n/a"
    status_text = "valid" if summary.get("valid") else "invalid"

    if summary.get("asimov_fit"):
        label = "Asimov"
    elif is_observed_fit or summary.get("observed_fit"):
        label = "Observed"
    else:
        label = f"Toy {int(summary.get('dataset_id', 0)):3d}"

    print(
        f"{label}: {status_text:<7}, {poi_label}={fit_text:<10} +- {unc_text:<10}, "
        f"time={summary.get('dataset_time_s', float('nan')):.4f}s"
    )

    if "cls_observed" in summary and summary.get("cls_observed") is not None:
        print(f"  CLs observed upper limit: {summary['cls_observed']:.4f}")
    if "cls_expected_quantiles" in summary:
        q = summary["cls_expected_quantiles"]
        print(
            "  CLs expected: "
            f"2.5%={q.get('2.5%'):.4g}, 16%={q.get('16%'):.4g}, 50%={q.get('50%'):.4g}, "
            f"84%={q.get('84%'):.4g}, 97.5%={q.get('97.5%'):.4g}"
        )
    if "cls_error" in summary:
        print(f"  CLs failed: {summary['cls_error']}")
    if isinstance(summary.get("feldman_cousins"), dict):
        fc = summary["feldman_cousins"]
        if fc.get("fc_interval") is not None:
            print(f"  Feldman-Cousins interval: {[float(f'{x:.4g}') for x in fc.get('fc_interval')]}")
        elif fc.get("fc_status"):
            print(f"  Feldman-Cousins status: {fc.get('fc_status')}")



def _distribution_summary(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None

    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p16": float(np.percentile(arr, 16)),
        "p84": float(np.percentile(arr, 84)),
    }


def _build_ensemble_evaluation_report(summaries, total_time_s):
    report = {
        "n_datasets": int(len(summaries)),
        "runtime": {
            "total_time_s": float(total_time_s),
            "average_time_s": float(total_time_s / len(summaries)) if summaries else None,
        },
    }

    if not summaries:
        return report

    valid_flags = [bool(summary.get("valid", False)) for summary in summaries]
    n_valid = int(sum(valid_flags))
    report["fit_quality"] = {
        "n_valid": n_valid,
        "n_invalid": int(len(summaries) - n_valid),
        "valid_fraction": float(n_valid / len(summaries)),
    }

    report["poi_name"] = summaries[0].get("poi_name", "poi")
    report["poi_fit"] = _distribution_summary([summary.get("poi_fit") for summary in summaries])
    report["poi_unc_hesse"] = _distribution_summary([summary.get("poi_unc_hesse") for summary in summaries])
    report["poi_pull"] = _distribution_summary([summary.get("poi_pull") for summary in summaries])

    cls_obs = [summary.get("cls_observed") for summary in summaries if summary.get("cls_observed") is not None]
    if cls_obs:
        report["cls"] = {
            "observed_limit": _distribution_summary(cls_obs),
            "n_failures": int(sum(1 for summary in summaries if "cls_error" in summary)),
        }

    return report


def _save_ensemble_report(report, output, report_file=None):
    if report_file:
        output_path = os.path.abspath(report_file)
    else:
        base, _ = os.path.splitext(os.path.abspath(output))
        output_path = f"{base}_ensemble_report.json"

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    return output_path


def _snapshot_path(output_path):
    abs_out = os.path.abspath(output_path)
    if abs_out.lower().endswith(".json"):
        return abs_out
    base, _ = os.path.splitext(abs_out)
    return f"{base}.json"


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
            "set_parameters": args.set_parameters,
            "freeze_parameters": args.freeze_parameters,
            "set_parameter_ranges": args.set_parameter_ranges,
            "backend": args.backend,
            "hessian_method": args.hessian_method,
        },
    }

    final_path = _snapshot_path(output_path)
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
    apply_parameter_overrides(
        fit_model,
        set_values_spec=args.set_parameters,
        set_ranges_spec=args.set_parameter_ranges,
        freeze_spec=args.freeze_parameters,
    )

    has_observed_data = hasattr(fit_model, "data") and fit_model.data is not None
    if args.toys is None:
        use_observed_data = has_observed_data
        use_asimov_data = False
        n_toys = 1
    elif args.toys == -1:
        use_observed_data = False
        use_asimov_data = True
        n_toys = 1
    elif args.toys < -1:
        raise ValueError("Only --toys -1 is supported as special Asimov mode")
    else:
        use_observed_data = False
        use_asimov_data = False
        n_toys = int(args.toys)

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
    if summaries:
        print(f"Average time per dataset: {total_time_s / len(summaries):.4f}s")
    print(f"Total execution time: {total_time_s:.4f}s")

    if args.plot:
        plot_summary_artifacts(
            summaries=summaries,
            fit_model=fit_model,
            plot_dir=os.path.abspath(args.plot_dir),
            binned_bins=args.binned_bins,
        )
        print(f"Saved plots to: {os.path.abspath(args.plot_dir)}")

    output = args.output or f"analysis_output_{args.seed}.json"

    ensemble_report = _build_ensemble_evaluation_report(summaries=summaries, total_time_s=total_time_s)
    report_path = _save_ensemble_report(
        report=ensemble_report,
        output=output,
        report_file=args.report_file,
    )
    print(f"Saved ensemble evaluation report to: {report_path}")

    snapshot_path = _save_analysis_snapshot(
        output_path=output,
        fit_model=fit_model,
        summaries=summaries,
        args=args,
    )
    print(f"Saved analysis snapshot to: {snapshot_path}")
