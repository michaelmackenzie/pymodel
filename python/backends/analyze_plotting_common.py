import numpy as np


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _finite_pair(x_values, y_values):
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return None, None
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return None, None
    return x[mask], y[mask]


def plot_hist(values, title, xlabel, output_file, bins=30, add_grid=True):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return

    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(arr, bins=max(5, min(int(bins), 80)), color="#4C78A8", alpha=0.85, edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Entries")
    if add_grid:
        ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close(fig)


def plot_delta_nll_scan(
    x_values,
    y_values,
    *,
    poi_name,
    output_file,
    title,
    y_label,
    reference_lines,
    line_color="#2E6F95",
    line_label=r"$\Delta$NLL",
):
    x, y = _finite_pair(x_values, y_values)
    if x is None:
        return

    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, y, color=line_color, linewidth=2.0, label=line_label)
    for yref, color, linestyle, label in reference_lines:
        ax.axhline(float(yref), color=color, linestyle=linestyle, linewidth=1.2, label=label)

    ax.set_xlabel(poi_name or "POI")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close(fig)


def plot_cls_scan(
    pois,
    observed,
    *,
    output_file,
    poi_name,
    alpha,
    title,
    expected_median=None,
    expected_band=None,
):
    x_obs, y_obs = _finite_pair(pois, observed)
    if x_obs is None:
        return

    x_all = np.asarray(pois, dtype=float)
    expected = np.asarray(expected_median, dtype=float) if expected_median is not None else np.asarray([], dtype=float)

    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.set_axisbelow(True)

    if expected.size == x_all.size and expected.size > 0:
        x_exp, y_exp = _finite_pair(x_all, expected)
        if x_exp is not None:
            ax.plot(x_exp, y_exp, color="black", linestyle="--", linewidth=1.8, label="Expected median")

    band = np.asarray(expected_band, dtype=float) if expected_band is not None else np.asarray([])
    has_band = band.ndim == 2 and band.shape[0] == x_all.size and band.shape[1] >= 5
    if has_band:
        low2 = band[:, 0]
        low1 = band[:, 1]
        high1 = band[:, 3]
        high2 = band[:, 4]

        valid2 = np.isfinite(x_all) & np.isfinite(low2) & np.isfinite(high2)
        valid1 = np.isfinite(x_all) & np.isfinite(low1) & np.isfinite(high1)
        if np.any(valid2):
            ax.fill_between(x_all[valid2], low2[valid2], high2[valid2], color="#FFD700", alpha=1.0, label=r"Expected $\pm2\sigma$")
        if np.any(valid1):
            ax.fill_between(x_all[valid1], low1[valid1], high1[valid1], color="#4CBB17", alpha=1.0, label=r"Expected $\pm1\sigma$")

    ax.plot(x_obs, y_obs, color="#1F77B4", linewidth=2.0, label="Observed")
    ax.axhline(float(alpha), color="#CC4C02", linestyle=":", linewidth=1.5, label=f"$CL_s = {alpha}$")
    ax.set_xlabel(poi_name or "POI")
    ax.set_ylabel(r"$CL_s$")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close(fig)


def plot_feldman_cousins(
    poi,
    q_obs,
    q_crit,
    *,
    output_file,
    poi_name,
    interval=None,
    alpha=None,
    title_base="Feldman-Cousins Construction",
    y_label=r"$q_\mu$",
):
    x_obs, y_obs = _finite_pair(poi, q_obs)
    x_crit, y_crit = _finite_pair(poi, q_crit)
    if x_obs is None and x_crit is None:
        return

    poi_arr = np.asarray(poi, dtype=float)
    q_obs_arr = np.asarray(q_obs, dtype=float)
    q_crit_arr = np.asarray(q_crit, dtype=float)

    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    if x_obs is not None:
        ax.plot(x_obs, y_obs, color="#1F77B4", linewidth=2.0, marker="o", markersize=3.5, label=r"$q_\mu^{obs}$")

    if x_crit is not None:
        ax.plot(
            x_crit,
            y_crit,
            color="#D62728",
            linewidth=2.0,
            marker="s",
            markersize=3.0,
            linestyle="--",
            label=r"$q_\mu^{crit}$",
        )

    both = np.isfinite(poi_arr) & np.isfinite(q_obs_arr) & np.isfinite(q_crit_arr)
    accepted = both & (q_obs_arr <= q_crit_arr)
    if np.any(accepted):
        ax.scatter(poi_arr[accepted], q_obs_arr[accepted], color="#2CA02C", s=24, zorder=4, label="Accepted grid points")

    if isinstance(interval, (list, tuple)) and len(interval) == 2:
        lo = float(interval[0])
        hi = float(interval[1])
        if np.isfinite(lo) and np.isfinite(hi) and hi >= lo:
            ax.axvspan(lo, hi, color="#A1D99B", alpha=0.22, label="FC interval")
            ax.axvline(lo, color="#2CA02C", linestyle=":", linewidth=1.3)
            ax.axvline(hi, color="#2CA02C", linestyle=":", linewidth=1.3)

    title = title_base
    if alpha is not None:
        try:
            title += f" (alpha={float(alpha):.3g})"
        except Exception:
            pass

    ax.set_title(title)
    ax.set_xlabel(poi_name or "POI")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close(fig)
