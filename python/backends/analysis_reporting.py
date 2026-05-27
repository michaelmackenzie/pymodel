import json
import os

import numpy as np


def distribution_summary(values):
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


def init_ensemble_report(summaries, total_time_s):
    return {
        "n_datasets": int(len(summaries)),
        "runtime": {
            "total_time_s": float(total_time_s),
            "average_time_s": float(total_time_s / len(summaries)) if summaries else None,
        },
    }


def add_fit_quality(report, summaries, include_invalid_fraction=True):
    if not summaries:
        return

    valid_flags = [bool(summary.get("valid", False)) for summary in summaries]
    n_valid = int(sum(valid_flags))
    n_total = len(summaries)
    fit_quality = {
        "n_valid": n_valid,
        "n_invalid": int(n_total - n_valid),
        "valid_fraction": float(n_valid / n_total),
    }
    if include_invalid_fraction:
        fit_quality["invalid_fraction"] = float((n_total - n_valid) / n_total)
    report["fit_quality"] = fit_quality


def add_poi_distributions(report, summaries):
    if not summaries:
        return

    report["poi_name"] = summaries[0].get("poi_name", "poi")
    report["poi_fit"] = distribution_summary([summary.get("poi_fit") for summary in summaries])
    report["poi_unc_hesse"] = distribution_summary([summary.get("poi_unc_hesse") for summary in summaries])
    report["poi_pull"] = distribution_summary([summary.get("poi_pull") for summary in summaries])


def save_ensemble_report(report, output_path, report_file=None):
    if report_file:
        final_path = os.path.abspath(report_file)
    else:
        base, _ = os.path.splitext(os.path.abspath(output_path))
        final_path = f"{base}_ensemble_report.json"

    with open(final_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    return final_path


def resolve_output_or_default(output_path, seed, extension):
    normalized_ext = extension if extension.startswith(".") else f".{extension}"
    if output_path:
        return output_path
    return f"analysis_output_{seed}{normalized_ext}"


def print_runtime_summary(summaries, total_time_s):
    if summaries:
        print(f"Average time per dataset: {total_time_s / len(summaries):.4f}s")
    print(f"Total execution time: {total_time_s:.4f}s")


def save_and_print_ensemble_report(summaries, total_time_s, output_path, report_file, build_report_fn):
    ensemble_report = build_report_fn(summaries=summaries, total_time_s=total_time_s)
    report_path = save_ensemble_report(
        report=ensemble_report,
        output_path=output_path,
        report_file=report_file,
    )
    print(f"Saved ensemble evaluation report to: {report_path}")
    return report_path


def save_and_print_snapshot(output_path, fit_model, summaries, args, save_snapshot_fn):
    snapshot_path = save_snapshot_fn(
        output_path=output_path,
        fit_model=fit_model,
        summaries=summaries,
        args=args,
    )
    print(f"Saved analysis snapshot to: {snapshot_path}")
    return snapshot_path


def maybe_plot_summary_artifacts(args, summaries, fit_model, plot_fn):
    if not bool(getattr(args, "plot", False)):
        return

    plot_dir = os.path.abspath(args.plot_dir)
    plot_fn(
        summaries=summaries,
        fit_model=fit_model,
        plot_dir=plot_dir,
        binned_bins=args.binned_bins,
        ntoys_plot=max(0, int(getattr(args, "ntoys_plot", 1))),
    )
    print(f"Saved plots to: {plot_dir}")