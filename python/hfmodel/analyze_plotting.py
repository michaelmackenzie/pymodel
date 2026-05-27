import os

import numpy as np


def _hist(values, title, xlabel, output_file, bins=30):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(arr, bins=max(5, min(int(bins), 80)), color="#4C78A8", alpha=0.85, edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Entries")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close(fig)


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
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = summary.get("delta_nll_scan")
    if not isinstance(payload, dict):
        return

    x = np.asarray(payload.get("poi_values", []), dtype=float)
    y = np.asarray(payload.get("delta_nll", []), dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if not np.any(valid):
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x[valid], y[valid], color="#2E6F95", linewidth=2.0, label=r"$\Delta$NLL")
    ax.axhline(1.0, color="#888888", linestyle="--", linewidth=1.0, label=r"$\Delta$NLL = 1")
    ax.axhline(3.84, color="#BBBBBB", linestyle=":", linewidth=1.0, label=r"$\Delta$NLL = 3.84")
    ax.set_xlabel(payload.get("poi_name", "POI"))
    ax.set_ylabel(r"$\Delta(-2\ln L)$")
    ax.set_title("Profile Delta NLL Scan")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    plt.tight_layout()
    out = os.path.join(plot_dir, f"dataset_{summary.get('dataset_id', 0)}_delta_nll.png")
    plt.savefig(out, dpi=150)
    plt.close(fig)


def _plot_cls_band(summary, plot_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cls_curve = summary.get("cls_curve", {})
    if not isinstance(cls_curve, dict):
        return
    ref_value = summary.get("cls_alpha", 0.05)

    pois = np.asarray(cls_curve.get("pois", []), dtype=float)
    obs = np.asarray(cls_curve.get("observed", []), dtype=float)
    exp_median = np.asarray(cls_curve.get("expected_median", []), dtype=float)
    exp_band = cls_curve.get("expected_band", [])
    if pois.size == 0 or exp_median.size != pois.size:
        return

    band = np.asarray(exp_band, dtype=float)
    has_band = band.ndim == 2 and band.shape[0] == pois.size and band.shape[1] >= 5

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.set_axisbelow(True)

    if has_band:
        low2 = band[:, 0]
        low1 = band[:, 1]
        high1 = band[:, 3]
        high2 = band[:, 4]
        valid2 = np.isfinite(pois) & np.isfinite(low2) & np.isfinite(high2)
        valid1 = np.isfinite(pois) & np.isfinite(low1) & np.isfinite(high1)
        # #4CBB17 (Deep Lime / Standard HEP Green)
        # #2CA02C (Matplotlib Tab Green)
        # (Yellow): #FFFF00 #FFE08A
        # #FFCD00 (Gold / Standard CERN Yellow)
        # #FFD700 (Web Gold)
        if np.any(valid2):
            ax.fill_between(pois[valid2], low2[valid2], high2[valid2], color="#FFD700", alpha=1., label=r"Expected $\pm2\sigma$")
        if np.any(valid1):
            ax.fill_between(pois[valid1], low1[valid1], high1[valid1], color="#4CBB17", alpha=1., label=r"Expected $\pm1\sigma$")

    valid_exp = np.isfinite(pois) & np.isfinite(exp_median)
    if np.any(valid_exp):
        ax.plot(pois[valid_exp], exp_median[valid_exp], color="black", linestyle="--", linewidth=1.8, label="Expected median")

    valid_obs = np.isfinite(pois) & np.isfinite(obs)
    if np.any(valid_obs):
        ax.plot(pois[valid_obs], obs[valid_obs], color="#1F77B4", linewidth=2.0, label="Observed")

    ax.axhline(ref_value, color="#CC4C02", linestyle=":", linewidth=1.5, label=f"$CL_s = {ref_value}$")
    ax.set_xlabel(summary.get("poi_name", "POI"))
    ax.set_ylabel(r"$CL_s$")
    ax.set_title("CLs Plot")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    plt.tight_layout()
    out = os.path.join(plot_dir, f"dataset_{summary.get('dataset_id', 0)}_cls_band.png")
    plt.savefig(out, dpi=150)
    plt.close(fig)


def _plot_feldman_cousins_construction(summary, plot_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fc = summary.get("feldman_cousins")
    if not isinstance(fc, dict):
        return

    grid = fc.get("grid", {})
    if not isinstance(grid, dict):
        return

    poi = np.asarray(grid.get("poi", []), dtype=float)
    q_obs = np.asarray(grid.get("q_obs", []), dtype=float)
    q_crit = np.asarray(grid.get("q_crit", []), dtype=float)
    if poi.size == 0:
        return

    valid_obs = np.isfinite(poi) & np.isfinite(q_obs)
    valid_crit = np.isfinite(poi) & np.isfinite(q_crit)
    if not np.any(valid_obs) and not np.any(valid_crit):
        return

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    if np.any(valid_obs):
        ax.plot(
            poi[valid_obs],
            q_obs[valid_obs],
            color="#1F77B4",
            linewidth=2.0,
            marker="o",
            markersize=3.5,
            label=r"$q_\mu^{obs}$",
        )

    if np.any(valid_crit):
        ax.plot(
            poi[valid_crit],
            q_crit[valid_crit],
            color="#D62728",
            linewidth=2.0,
            marker="s",
            markersize=3.0,
            linestyle="--",
            label=r"$q_\mu^{crit}$",
        )

    both = np.isfinite(poi) & np.isfinite(q_obs) & np.isfinite(q_crit)
    accepted = both & (q_obs <= q_crit)
    if np.any(accepted):
        ax.scatter(
            poi[accepted],
            q_obs[accepted],
            color="#2CA02C",
            s=24,
            zorder=4,
            label="Accepted grid points",
        )

    interval = fc.get("fc_interval")
    if isinstance(interval, list) and len(interval) == 2:
        lo = float(interval[0])
        hi = float(interval[1])
        if np.isfinite(lo) and np.isfinite(hi) and hi >= lo:
            ax.axvspan(lo, hi, color="#A1D99B", alpha=0.22, label="FC interval")
            ax.axvline(lo, color="#2CA02C", linestyle=":", linewidth=1.3)
            ax.axvline(hi, color="#2CA02C", linestyle=":", linewidth=1.3)

    alpha = fc.get("alpha")
    title = "Feldman-Cousins Construction"
    if alpha is not None:
        try:
            title += f" (alpha={float(alpha):.3g})"
        except Exception:
            pass
    ax.set_title(title)
    ax.set_xlabel(fc.get("poi_name", summary.get("poi_name", "POI")))
    ax.set_ylabel(r"$q_\mu$")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    plt.tight_layout()
    out = os.path.join(plot_dir, f"dataset_{summary.get('dataset_id', 0)}_feldman_cousins.png")
    plt.savefig(out, dpi=150)
    plt.close(fig)


def plot_summary_artifacts(summaries, fit_model, plot_dir, binned_bins):
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
        _plot_first_dataset_channels(summaries[0], plot_dir)
        _plot_delta_nll(summaries[0], plot_dir)
        _plot_cls_band(summaries[0], plot_dir)
        _plot_feldman_cousins_construction(summaries[0], plot_dir)
