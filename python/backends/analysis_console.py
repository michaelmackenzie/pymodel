def _summary_has_flag(summary, key, include_dataset_plot_flags):
    if summary.get(key):
        return True
    if not include_dataset_plot_flags:
        return False
    dataset_plot = summary.get("dataset_plot", {})
    if isinstance(dataset_plot, dict):
        return bool(dataset_plot.get(key.replace("_fit", ""), False))
    return False


def print_dataset_summary_header(
    summary,
    is_observed_fit=False,
    *,
    fit_precision=".4g",
    unc_precision=".4g",
    unconstrained_text=None,
    asimov_label="Asimov",
    observed_label="Observed",
    include_dataset_plot_flags=False,
):
    poi_label = summary.get("poi_name", "poi")
    poi_fit = summary.get("poi_fit")
    poi_unc = summary.get("poi_unc_hesse")

    fit_text = f"{poi_fit:{fit_precision}}" if poi_fit is not None else "n/a"
    if poi_unc is None:
        unc_text = "n/a"
    else:
        try:
            poi_unc_val = float(poi_unc)
            if unconstrained_text is not None:
                # Infinity/NaN should be shown as unconstrained where requested.
                if not (poi_unc_val == poi_unc_val and abs(poi_unc_val) != float("inf")):
                    unc_text = unconstrained_text
                else:
                    unc_text = f"{poi_unc_val:{unc_precision}}"
            else:
                unc_text = f"{poi_unc_val:{unc_precision}}"
        except Exception:
            unc_text = unconstrained_text if unconstrained_text is not None else "n/a"

    if _summary_has_flag(summary, "asimov_fit", include_dataset_plot_flags):
        label = asimov_label
    elif is_observed_fit or _summary_has_flag(summary, "observed_fit", include_dataset_plot_flags):
        label = observed_label
    else:
        dataset_id = int(summary.get("dataset_id", 0))
        label = f"Toy {dataset_id:3d}"

    status_text = "valid" if summary.get("valid") else "invalid"
    print(
        f"{label}: {status_text:<7}, {poi_label}={fit_text:<10} +- {unc_text:<10}, "
        f"time={summary.get('dataset_time_s', float('nan')):.4f}s"
    )


def print_limit_summary_lines(
    summary,
    *,
    include_scan_details=False,
    include_expected_error=False,
    include_yield_upper=False,
    feldman_status_prefix="Feldman-Cousins status",
    expected_label="CLs expected (asymptotic, b-only fit)",
    expected_precision=".4f",
    feldman_interval_precision=None,
):
    if "cls_observed" in summary and summary.get("cls_observed") is not None:
        print(f"  CLs observed upper limit: {summary['cls_observed']:.4f}")
    if "cls_expected_median" in summary and summary.get("cls_expected_median") is not None:
        print(f"  CLs expected upper limit: {summary['cls_expected_median']:.4f}")
    if include_scan_details and "cls_scan_points" in summary:
        print(f"  CLs scan points: {summary['cls_scan_points']}")
    if include_scan_details and "cls_scan_max" in summary:
        print(f"  CLs scan max: {summary['cls_scan_max']:.4g}")
    if "cls_expected_quantiles" in summary:
        q = summary["cls_expected_quantiles"]
        def _fmt(v):
            return f"{v:{expected_precision}}" if v is not None and v == v else "n/a"
        print(
            f"  {expected_label}: "
            f"2.5%={_fmt(q.get('2.5%'))}, "
            f"16%={_fmt(q.get('16%'))}, "
            f"50%={_fmt(q.get('50%'))}, "
            f"84%={_fmt(q.get('84%'))}, "
            f"97.5%={_fmt(q.get('97.5%'))}"
        )
    if include_expected_error and "cls_expected_error" in summary:
        print(f"  CLs expected failed: {summary['cls_expected_error']}")
    if include_yield_upper and "yield_upper_limit" in summary:
        print(f"  Yield upper limit: {summary['yield_upper_limit']:.4f}")
    if "cls_error" in summary:
        print(f"  CLs failed: {summary['cls_error']}")

    if "feldman_cousins" not in summary:
        return

    fc = summary["feldman_cousins"]
    if not isinstance(fc, dict):
        return

    if "fc_interval" in fc:
        interval = fc["fc_interval"]
        if feldman_interval_precision is not None and isinstance(interval, (list, tuple)):
            interval = [float(f"{float(x):{feldman_interval_precision}}") for x in interval]
        print(f"  Feldman-Cousins interval: {interval}")
    elif "fc_status" in fc:
        print(f"  {feldman_status_prefix}: {fc['fc_status']}")
    else:
        print(f"  Feldman-Cousins: {fc}")