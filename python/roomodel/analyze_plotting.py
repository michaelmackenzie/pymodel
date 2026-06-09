import os

import numpy as np
from backends.analyze_plotting_common import (
    plot_cls_scan as _plot_cls_scan_common,
    plot_delta_nll_scan as _plot_delta_nll_scan_common,
    plot_feldman_cousins as _plot_feldman_cousins_common,
    plot_hist as _plot_hist_common,
)


def _hist(values, title, xlabel, output_file, bins=30):
    _plot_hist_common(values=values, title=title, xlabel=xlabel, output_file=output_file, bins=bins, add_grid=True)


def _plot_dataset(summary, plot_dir):
    channel_payloads = summary.get("channel_plots") or []
    if channel_payloads:
        for item in channel_payloads:
            payload = item.get("dataset_plot") or {}
            mode = payload.get("mode")
            if mode not in {"binned", "unbinned"}:
                continue

            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            dataset_id = int(summary.get("dataset_id", 0))
            channel_name = str(item.get("channel") or "channel")
            components = item.get("fit_components") or {}
            comp_x = np.asarray(components.get("x", []), dtype=float)
            comp_total = np.asarray(components.get("total", []), dtype=float)
            comp_signal = np.asarray(components.get("signal", []), dtype=float) if components.get("signal") is not None else np.asarray([], dtype=float)
            comp_background = np.asarray(components.get("background", []), dtype=float)

            fig, ax = plt.subplots(figsize=(8, 5))
            if mode == "binned":
                edges = np.asarray(payload.get("edges", []), dtype=float)
                counts = np.asarray(payload.get("counts", []), dtype=float)
                if len(edges) < 2 or len(counts) != len(edges) - 1:
                    plt.close(fig)
                    continue
                centers = 0.5 * (edges[:-1] + edges[1:])
                ax.errorbar(centers, counts, yerr=np.sqrt(np.clip(counts, 0.0, None)), fmt="o", color="black", label="Data")
                ax.set_xlabel(payload.get("obs_name", "obs"))
                ax.set_ylabel("Entries")
            else:
                values = np.asarray(payload.get("values", []), dtype=float)
                values = values[np.isfinite(values)]
                edges = np.asarray(payload.get("edges", []), dtype=float)
                counts = np.asarray(payload.get("counts", []), dtype=float)

                if edges.size >= 2 and counts.size == edges.size - 1:
                    centers = 0.5 * (edges[:-1] + edges[1:])
                    ax.errorbar(centers, counts, yerr=np.sqrt(np.clip(counts, 0.0, None)), fmt="o", color="black", label="Data")
                elif values.size > 0:
                    bins = max(10, min(80, int(np.sqrt(values.size) * 2)))
                    counts, edges = np.histogram(values, bins=bins)
                    centers = 0.5 * (edges[:-1] + edges[1:])
                    ax.errorbar(centers, counts, yerr=np.sqrt(np.clip(counts, 0.0, None)), fmt="o", color="black", label="Data")
                else:
                    plt.close(fig)
                    continue

                ax.set_xlabel(payload.get("obs_name", "obs"))
                ax.set_ylabel("Entries")

            if comp_x.size > 0 and comp_background.size == comp_x.size:
                ax.plot(comp_x, comp_background, color="#54A24B", linewidth=2.0, label="Background fit")
            if comp_x.size > 0 and comp_signal.size == comp_x.size:
                ax.plot(comp_x, comp_signal, color="#E45756", linewidth=2.0, label="Signal fit")
            if comp_x.size > 0 and comp_total.size == comp_x.size:
                ax.plot(comp_x, comp_total, color="black", linewidth=2.0, linestyle="--", label="Total fit")

            title = "Observed data" if summary.get("observed_fit") else f"Dataset {dataset_id}"
            ax.set_title(f"{title} ({channel_name})")
            ax.grid(alpha=0.25)
            ax.legend(loc="best")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"dataset_{dataset_id:04d}_{channel_name}.png"), dpi=150)
            plt.close(fig)
        return

    payload = summary.get("dataset_plot", {})
    mode = payload.get("mode")
    if mode not in {"binned", "unbinned"}:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dataset_id = int(summary.get("dataset_id", 0))
    components = summary.get("fit_components") or {}
    comp_x = np.asarray(components.get("x", []), dtype=float)
    comp_total = np.asarray(components.get("total", []), dtype=float)
    comp_signal = np.asarray(components.get("signal", []), dtype=float) if components.get("signal") is not None else np.asarray([], dtype=float)
    comp_background = np.asarray(components.get("background", []), dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    if mode == "binned":
        edges = np.asarray(payload.get("edges", []), dtype=float)
        counts = np.asarray(payload.get("counts", []), dtype=float)
        if len(edges) < 2 or len(counts) != len(edges) - 1:
            plt.close(fig)
            return
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.errorbar(centers, counts, yerr=np.sqrt(np.clip(counts, 0.0, None)), fmt="o", color="black", label="Data")
        ax.set_xlabel(payload.get("obs_name", "obs"))
        ax.set_ylabel("Entries")
    else:
        values = np.asarray(payload.get("values", []), dtype=float)
        values = values[np.isfinite(values)]
        edges = np.asarray(payload.get("edges", []), dtype=float)
        counts = np.asarray(payload.get("counts", []), dtype=float)

        if edges.size >= 2 and counts.size == edges.size - 1:
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax.errorbar(centers, counts, yerr=np.sqrt(np.clip(counts, 0.0, None)), fmt="o", color="black", label="Data")
        elif values.size > 0:
            bins = max(10, min(80, int(np.sqrt(values.size) * 2)))
            counts, edges = np.histogram(values, bins=bins)
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax.errorbar(centers, counts, yerr=np.sqrt(np.clip(counts, 0.0, None)), fmt="o", color="black", label="Data")
        else:
            plt.close(fig)
            return

        ax.set_xlabel(payload.get("obs_name", "obs"))
        ax.set_ylabel("Entries")

    if comp_x.size > 0 and comp_background.size == comp_x.size:
        ax.plot(comp_x, comp_background, color="#54A24B", linewidth=2.0, label="Background fit")
    if comp_x.size > 0 and comp_signal.size == comp_x.size:
        ax.plot(comp_x, comp_signal, color="#E45756", linewidth=2.0, label="Signal fit")
    if comp_x.size > 0 and comp_total.size == comp_x.size:
        ax.plot(comp_x, comp_total, color="black", linewidth=2.0, linestyle="--", label="Total fit")

    title = "Observed data" if summary.get("observed_fit") else f"Dataset {dataset_id}"
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"dataset_{dataset_id:04d}.png"), dpi=150)
    plt.close(fig)


def _plot_delta_nll(summary, plot_dir):
    payload = summary.get("delta_nll_scan") or {}
    dataset_id = int(summary.get("dataset_id", 0))
    _plot_delta_nll_scan_common(
        x_values=payload.get("x", []),
        y_values=payload.get("delta_nll", []),
        poi_name=str(payload.get("poi_name", "POI")),
        output_file=os.path.join(plot_dir, f"delta_nll_{dataset_id:04d}.png"),
        title=f"Delta NLL Scan (Dataset {dataset_id})",
        y_label=r"$\Delta \mathrm{NLL}$",
        reference_lines=(
            (0.5, "#54A24B", "--", r"$\Delta\mathrm{NLL}=0.5$"),
            (2.0, "#E45756", ":", r"$\Delta\mathrm{NLL}=2.0$"),
        ),
        line_color="#4C78A8",
        line_label=r"$\Delta\mathrm{NLL}$",
    )


def _plot_cls_band(summary, plot_dir):
    cls_curve = summary.get("cls_curve", {})
    if not isinstance(cls_curve, dict):
        return
    dataset_id = int(summary.get("dataset_id", 0))
    _plot_cls_scan_common(
        pois=cls_curve.get("pois", []),
        observed=cls_curve.get("observed", []),
        output_file=os.path.join(plot_dir, f"dataset_{dataset_id:04d}_cls_band.png"),
        poi_name=summary.get("poi_name", "POI"),
        alpha=float(summary.get("cls_alpha", 0.05)),
        title=f"CLs Scan (Dataset {dataset_id})",
    )


def _plot_feldman_cousins_construction(summary, plot_dir):
    fc = summary.get("feldman_cousins")
    if not isinstance(fc, dict):
        return
    grid = fc.get("grid") or {}
    dataset_id = int(summary.get("dataset_id", 0))
    _plot_feldman_cousins_common(
        poi=grid.get("poi", []),
        q_obs=grid.get("q_obs", []),
        q_crit=grid.get("q_crit", []),
        output_file=os.path.join(plot_dir, f"dataset_{dataset_id:04d}_feldman_cousins.png"),
        poi_name=fc.get("poi_name", summary.get("poi_name", "POI")),
        interval=fc.get("fc_interval"),
        alpha=fc.get("alpha"),
        y_label="Profile-likelihood q",
        title_base=f"Feldman-Cousins Construction (Dataset {dataset_id})",
    )


def plot_summary_artifacts(summaries, fit_model, plot_dir, binned_bins, ntoys_plot=1):
    _ = fit_model
    _ = binned_bins
    os.makedirs(plot_dir, exist_ok=True)

    poi_fits = [item.get("poi_fit") for item in summaries]
    poi_unc = [item.get("poi_unc_hesse") for item in summaries]
    poi_pull = [item.get("poi_pull") for item in summaries]

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

    n_plot = max(0, int(ntoys_plot))
    if n_plot == 0:
        return

    for summary in summaries[:n_plot]:
        _plot_dataset(summary, plot_dir)
        _plot_delta_nll(summary, plot_dir)
        _plot_cls_band(summary, plot_dir)
        _plot_feldman_cousins_construction(summary, plot_dir)
