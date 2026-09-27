"""Plots for the CLI ``--plot`` option (matplotlib, PNG).

All plots are drawn from the same result objects that are written to the JSON result file,
so they can be remade from a result file with ``python -m inference.plots <result.json>``.
"""

import json
import math
import os
import sys

import numpy as np

# categorical order (fixed, never cycled); text stays in neutral ink
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK = "#1a1a19"
MUTED = "#6b6a63"
GRID = "#e4e3dc"
BAND_1S = "#6da7ec"
BAND_2S = "#cde2fb"


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                         "legend.frameon": False})
    return plt


def _save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    import matplotlib.pyplot as plt

    plt.close(fig)
    print(f"  plot: {path}")
    return path


def _process_colors(names):
    if len(names) > len(SERIES):
        # fold the smallest into "other" is the caller's job; here we just refuse to cycle
        raise ValueError(f"{len(names)} processes exceed the {len(SERIES)} categorical colours")
    return dict(zip(names, SERIES))


def _data_counts(ch, main):
    edges = np.asarray(ch.observable.edges)
    if main.kind == "unbinned":
        counts, _ = np.histogram(main.values, bins=edges, weights=main.weights)
        return counts
    return np.asarray(main.counts)


def plot_fit(session, fits, outdir):
    plt = _plt()
    lik = session.lik
    if len(fits) == 1:
        data, res = fits[0]
        exp_post = lik.expected_by_process(res.values)
        exp_pre = lik.expected_counts(lik.nominal_values())
        for ch in lik.model.channels:
            edges = np.asarray(ch.observable.edges)
            procs = list(exp_post[ch.name])
            order = sorted(procs, key=lambda p: float(np.sum(exp_post[ch.name][p])))  # largest on top
            colors = _process_colors(procs)
            fig, ax = plt.subplots(figsize=(6.4, 4.2))
            bottom = np.zeros(len(edges) - 1)
            for p in order:
                y = exp_post[ch.name][p]
                ax.bar(edges[:-1], y, width=np.diff(edges), bottom=bottom, align="edge", color=colors[p],
                       edgecolor="white", linewidth=1.0, label=p)
                bottom = bottom + y
            ax.stairs(exp_pre[ch.name], edges, color=INK, linestyle="--", linewidth=1.5, label="pre-fit total")
            n = _data_counts(ch, data.main[ch.name])
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax.errorbar(centers, n, yerr=np.sqrt(np.maximum(n, 0)), fmt="o", color=INK, markersize=4,
                        label=data.label)
            ax.set_xlabel(ch.observable.name)
            ax.set_ylabel("events / bin")
            ax.set_title(f"{ch.name}: post-fit (r = {res.value(lik, lik.poi):.3g})", color=INK, loc="left")
            ax.legend(fontsize=8, ncol=2)
            _save(fig, outdir, f"fit_{ch.name}.png")
        return
    rs = np.array([r.value(lik, lik.poi) for _, r in fits if r.valid])
    errs = np.array([r.errors.get(lik.poi, np.nan) for _, r in fits if r.valid])
    truth = session.args.expect_signal
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].hist(rs, bins=30, color=SERIES[0], edgecolor="white")
    axes[0].axvline(truth, color=INK, linestyle="--", linewidth=1.5)
    axes[0].set_xlabel(r"$\hat{r}$")
    axes[0].set_ylabel("toys")
    pulls = (rs - truth) / errs
    pulls = pulls[np.isfinite(pulls)]
    axes[1].hist(pulls, bins=30, color=SERIES[0], edgecolor="white", density=True)
    x = np.linspace(-4, 4, 200)
    axes[1].plot(x, np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi), color=INK, linewidth=1.5, label="N(0,1)")
    axes[1].set_xlabel(r"pull $(\hat{r} - r_{true})/\sigma$")
    axes[1].legend()
    _save(fig, outdir, "fit_toys.png")


def plot_scan(out, outdir):
    plt = _plt()
    pts = [p for p in out["points"] if p["valid"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([p["value"] for p in pts], [p["deltaNLL2"] for p in pts], color=SERIES[0], linewidth=2, marker="o",
            markersize=3)
    for level, text in ((1.0, "68%"), (3.84, "95%")):
        ax.axhline(level, color=MUTED, linestyle=":", linewidth=1)
        ax.text(ax.get_xlim()[1], level, f" {text}", color=MUTED, va="center", fontsize=8)
    ax.set_xlabel(out["param"])
    ax.set_ylabel(r"$2\Delta$NLL")
    ax.set_ylim(bottom=0)
    _save(fig, outdir, f"scan_{out['param']}.png")


def plot_asymptotic(res, outdir):
    plt = _plt()
    pts = sorted(res.points, key=lambda p: p["r"])
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy([p["r"] for p in pts], [max(p["CLs"], 1e-6) for p in pts], color=SERIES[0], marker="o",
                markersize=4, linewidth=1.5, label="observed CLs")
    ax.axhline(0.05, color=MUTED, linestyle=":", linewidth=1)
    if res.observed:
        ax.axvline(res.observed, color=INK, linewidth=1, label=f"limit {res.observed:.3g}")
    ax.set_xlabel("r")
    ax.set_ylabel("CLs")
    ax.legend()
    _save(fig, outdir, "cls_asymptotic.png")


def plot_toy_cls(res, outdir):
    plt = _plt()
    pts = [p.to_dict() for p in res.points]
    good = [p for p in pts if p["CLs"] is not None and math.isfinite(p["CLs"])]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar([p["r"] for p in good], [p["CLs"] for p in good], yerr=[p["CLs_err"] for p in good], fmt="o-",
                color=SERIES[0], markersize=4, linewidth=1.5, label="observed CLs")
    ax.axhline(0.05, color=MUTED, linestyle=":", linewidth=1)
    if res.observed:
        ax.axvline(res.observed, color=INK, linewidth=1, label=f"limit {res.observed:.3g}")
    ax.set_yscale("log")
    ax.set_xlabel("r")
    ax.set_ylabel("CLs")
    ax.legend()
    _save(fig, outdir, "cls_toys.png")


def plot_fc(res, outdir):
    plt = _plt()
    pts = [p.to_dict() for p in res.points]
    good = [p for p in pts if p["p"] is not None and math.isfinite(p["p"])]
    fig, ax = plt.subplots(figsize=(6, 4))
    if res.lower is not None and res.upper is not None:
        ax.axvspan(res.lower, res.upper, color=BAND_2S, label=f"{res.cl:.0%} CL interval")
    ax.errorbar([p["r"] for p in good], [p["p"] for p in good], yerr=[p["p_err"] for p in good], fmt="o-",
                color=SERIES[0], markersize=4, linewidth=1.5, label="p(r)")
    ax.axhline(1 - res.cl, color=MUTED, linestyle=":", linewidth=1)
    ax.set_xlabel("r")
    ax.set_ylabel("p-value")
    ax.legend()
    _save(fig, outdir, "feldman_cousins.png")


if __name__ == "__main__":
    doc = json.load(open(sys.argv[1]))
    outdir = sys.argv[2] if len(sys.argv) > 2 else "plots"
    if doc["command"] == "scan":
        plot_scan(doc["result"], outdir)
    else:
        raise SystemExit(f"re-plotting '{doc['command']}' results from JSON is not implemented")
