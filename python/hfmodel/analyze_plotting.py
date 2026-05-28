import os

import numpy as np
from backends.analyze_plotting_common import (
    plot_cls_scan as _plot_cls_scan_common,
    plot_delta_nll_scan as _plot_delta_nll_scan_common,
    plot_feldman_cousins as _plot_feldman_cousins_common,
    plot_hist as _plot_hist_common,
)


def _hist(values, title, xlabel, output_file, bins=30):
    _plot_hist_common(values=values, title=title, xlabel=xlabel, output_file=output_file, bins=bins, add_grid=False)


def _plot_first_dataset_channels(summary, plot_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = summary.get("dataset_plot", {})
    channels = payload.get("channels", {})
    if not channels:
        return

    fit_params_map = summary.get("fit_params", {}) or {}
    fit_unc_map = summary.get("fit_param_unc", {}) or {}
    poi_name = summary.get("poi_name")

    for channel, data in channels.items():
        obs = np.asarray(data.get("obs", []), dtype=float)
        total = np.asarray(data.get("total", []), dtype=float)
        bkg = np.asarray(data.get("bkg", []), dtype=float)
        sig = np.asarray(data.get("sig", []), dtype=float)
        prefit_total = np.asarray(data.get("prefit_total", []), dtype=float)
        prefit_bkg = np.asarray(data.get("prefit_bkg", []), dtype=float)
        prefit_sig = np.asarray(data.get("prefit_sig", []), dtype=float)
        x = np.arange(len(obs), dtype=float)

        # Propagate Hessian-derived parameter uncertainties to post-fit templates.
        total_low = None
        total_high = None
        bkg_low = None
        bkg_high = None
        if fit_params_map and fit_unc_map:
            # We use diagonal propagation by varying one parameter at a time by +/-1 sigma.
            total_var = np.zeros_like(total, dtype=float)
            bkg_var = np.zeros_like(bkg, dtype=float)
            base_total = total.copy()
            base_bkg = bkg.copy()

            for par_name, par_val in fit_params_map.items():
                par_unc = fit_unc_map.get(par_name)
                if par_unc is None:
                    continue
                try:
                    sigma = float(par_unc)
                    if not np.isfinite(sigma) or sigma <= 0.0:
                        continue

                    up_total = np.asarray(data.get("total_var_up", {}).get(par_name, []), dtype=float)
                    dn_total = np.asarray(data.get("total_var_down", {}).get(par_name, []), dtype=float)
                    if up_total.size == base_total.size and dn_total.size == base_total.size:
                        delta_total = np.maximum(np.abs(up_total - base_total), np.abs(dn_total - base_total))
                        total_var += np.square(delta_total)

                    if par_name == poi_name:
                        continue

                    up = np.asarray(data.get("bkg_var_up", {}).get(par_name, []), dtype=float)
                    dn = np.asarray(data.get("bkg_var_down", {}).get(par_name, []), dtype=float)
                    if up.size == base_bkg.size and dn.size == base_bkg.size:
                        delta = np.maximum(np.abs(up - base_bkg), np.abs(dn - base_bkg))
                        bkg_var += np.square(delta)
                except Exception:
                    continue

            if np.any(total_var > 0.0):
                sigma_total = np.sqrt(np.clip(total_var, 0.0, None))
                total_low = np.clip(base_total - sigma_total, 0.0, None)
                total_high = base_total + sigma_total

            if np.any(bkg_var > 0.0):
                sigma_tot = np.sqrt(np.clip(bkg_var, 0.0, None))
                bkg_low = np.clip(base_bkg - sigma_tot, 0.0, None)
                bkg_high = base_bkg + sigma_tot

        # Post-fit figure
        fig_post, ax_post = plt.subplots(figsize=(8, 5))
        ax_post.errorbar(x, obs, yerr=np.sqrt(np.clip(obs, 1.0, None)), fmt="ko", capsize=2, label="Data")
        if total.size:
            if total_low is not None and total_high is not None:
                ax_post.fill_between(
                    x,
                    total_low,
                    total_high,
                    step="mid",
                    color="#4D4D4D",
                    alpha=0.15,
                    linewidth=0.0,
                    label=r"Total fit $\pm 1\sigma$ (Hessian)",
                )
            ax_post.step(x, total, where="mid", color="black", linewidth=1.8, label="Total fit")
        if bkg.size:
            if bkg_low is not None and bkg_high is not None:
                ax_post.fill_between(
                    x,
                    bkg_low,
                    bkg_high,
                    step="mid",
                    color="#1F77B4",
                    alpha=0.20,
                    linewidth=0.0,
                    label=r"Background $\pm 1\sigma$ (Hessian)",
                )
            ax_post.step(x, bkg, where="mid", color="#1F77B4", linestyle="--", linewidth=1.6, label="Background")
        if sig.size:
            ax_post.step(x, sig, where="mid", color="#D62728", linestyle="-.", linewidth=1.6, label="Signal")

        ax_post.set_title(f"Post-fit Channel: {channel}")
        ax_post.set_xlabel("Bin index")
        ax_post.set_ylabel("Events")
        ax_post.legend(loc="best")
        ax_post.grid(alpha=0.25)

        plt.tight_layout()
        out_post = os.path.join(plot_dir, f"dataset_{summary.get('dataset_id', 0)}_{channel}_postfit.png")
        plt.savefig(out_post, dpi=150)
        plt.close(fig_post)

        # Pre-fit figure
        if prefit_total.size or prefit_bkg.size or prefit_sig.size:
            fig_pre, ax_pre = plt.subplots(figsize=(8, 5))
            ax_pre.errorbar(x, obs, yerr=np.sqrt(np.clip(obs, 1.0, None)), fmt="ko", capsize=2, label="Data")
            if prefit_total.size:
                ax_pre.step(x, prefit_total, where="mid", color="#7F7F7F", linewidth=1.8, label="Total pre-fit")
            if prefit_bkg.size:
                ax_pre.step(x, prefit_bkg, where="mid", color="#6BAED6", linestyle="--", linewidth=1.6, label="Background pre-fit")
            if prefit_sig.size:
                ax_pre.step(x, prefit_sig, where="mid", color="#FB6A4A", linestyle="-.", linewidth=1.6, label="Signal pre-fit")

            ax_pre.set_title(f"Pre-fit Channel: {channel}")
            ax_pre.set_xlabel("Bin index")
            ax_pre.set_ylabel("Events")
            ax_pre.legend(loc="best")
            ax_pre.grid(alpha=0.25)

            plt.tight_layout()
            out_pre = os.path.join(plot_dir, f"dataset_{summary.get('dataset_id', 0)}_{channel}_prefit.png")
            plt.savefig(out_pre, dpi=150)
            plt.close(fig_pre)


def _plot_delta_nll(summary, plot_dir):
    payload = summary.get("delta_nll_scan")
    if not isinstance(payload, dict):
        return

    out = os.path.join(plot_dir, f"dataset_{summary.get('dataset_id', 0)}_delta_nll.png")
    _plot_delta_nll_scan_common(
        x_values=payload.get("poi_values", []),
        y_values=payload.get("delta_nll", []),
        poi_name=payload.get("poi_name", "POI"),
        output_file=out,
        title="Profile Delta NLL Scan",
        y_label=r"$\Delta(-2\ln L)$",
        reference_lines=(
            (1.0, "#888888", "--", r"$\Delta$NLL = 1"),
            (3.84, "#BBBBBB", ":", r"$\Delta$NLL = 3.84"),
        ),
        line_color="#2E6F95",
    )


def _plot_cls_band(summary, plot_dir):
    cls_curve = summary.get("cls_curve", {})
    if not isinstance(cls_curve, dict):
        return
    out = os.path.join(plot_dir, f"dataset_{summary.get('dataset_id', 0)}_cls_band.png")
    _plot_cls_scan_common(
        pois=cls_curve.get("pois", []),
        observed=cls_curve.get("observed", []),
        expected_median=cls_curve.get("expected_median", []),
        expected_band=cls_curve.get("expected_band", []),
        output_file=out,
        poi_name=summary.get("poi_name", "POI"),
        alpha=summary.get("cls_alpha", 0.05),
        title="CLs Plot",
    )


def _plot_feldman_cousins_construction(summary, plot_dir):
    fc = summary.get("feldman_cousins")
    if not isinstance(fc, dict):
        return

    grid = fc.get("grid", {})
    if not isinstance(grid, dict):
        return

    out = os.path.join(plot_dir, f"dataset_{summary.get('dataset_id', 0)}_feldman_cousins.png")
    _plot_feldman_cousins_common(
        poi=grid.get("poi", []),
        q_obs=grid.get("q_obs", []),
        q_crit=grid.get("q_crit", []),
        output_file=out,
        poi_name=fc.get("poi_name", summary.get("poi_name", "POI")),
        interval=fc.get("fc_interval"),
        alpha=fc.get("alpha"),
    )


def plot_summary_artifacts(summaries, fit_model, plot_dir, binned_bins, ntoys_plot=1):
    _ = fit_model
    _ = binned_bins
    os.makedirs(plot_dir, exist_ok=True)

    poi_fits = [item.get("poi_fit") for item in summaries]
    poi_unc = [item.get("poi_unc_hesse") for item in summaries]
    poi_pull = [item.get("poi_pull") for item in summaries]
    cls_obs = [item.get("cls_observed") for item in summaries if item.get("cls_observed") is not None]

    _hist(
        values=poi_fits,
        title="POI Fit Distribution",
        xlabel="Fitted POI",
        output_file=os.path.join(plot_dir, "poi_fit_hist.png"),
    )
    _hist(
        values=poi_unc,
        title="POI Uncertainty Distribution",
        xlabel="POI uncertainty",
        output_file=os.path.join(plot_dir, "poi_unc_hist.png"),
    )
    _hist(
        values=poi_pull,
        title="POI Pull Distribution",
        xlabel="(fit - truth) / sigma",
        output_file=os.path.join(plot_dir, "poi_pull_hist.png"),
    )

    if cls_obs:
        _hist(
            values=cls_obs,
            title="Observed CLs Upper Limit Distribution",
            xlabel="Upper limit on POI",
            output_file=os.path.join(plot_dir, "cls_observed_hist.png"),
        )

    if summaries:
        n_plot = max(0, int(ntoys_plot))
        if n_plot == 0:
            return
        for summary in summaries[:n_plot]:
            _plot_first_dataset_channels(summary, plot_dir)
            _plot_delta_nll(summary, plot_dir)
            _plot_cls_band(summary, plot_dir)
            _plot_feldman_cousins_construction(summary, plot_dir)
