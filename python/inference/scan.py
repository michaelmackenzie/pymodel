"""Profile-likelihood scans (Combine MultiDimFit --algo grid, 1D) and fit summaries."""

import math
from typing import Dict, List, Optional

import numpy as np

from modelspec import ir as I

# 2*DeltaNLL thresholds for 68.27% and 95.45% two-sided intervals (1 dof)
LEVELS = {"68": 1.0, "95": 3.841458820694124}


def grid_points(lo: float, hi: float, n: int) -> np.ndarray:
    """Combine's grid convention: n points at the centres of n equal intervals."""
    width = (hi - lo) / n
    return lo + width * (np.arange(n) + 0.5)


def profile_scan(lik, fitter, data, param: str, points: np.ndarray, free=None) -> dict:
    free = free or fitter.fit(data)
    if not free.valid:
        raise RuntimeError(f"free fit failed: {free.status}")
    out = {"param": param, "best_fit": free.value(lik, param), "nll_min": free.nll, "points": [], "flags": []}
    start = free.values
    # scan outward from the best fit so each fit starts from a neighbouring solution
    best = free.value(lik, param)
    order = sorted(range(len(points)), key=lambda i: (points[i] < best, abs(points[i] - best)))
    results: Dict[int, dict] = {}
    for side in (False, True):
        start = free.values
        for i in [j for j in order if (points[j] < best) == side]:
            res = fitter.fit(data, start=start, fixed={param: float(points[i])})
            results[i] = {"value": float(points[i]), "deltaNLL2": 2.0 * (res.nll - free.nll) if res.valid else None,
                          "valid": res.valid, "status": res.status}
            if res.valid:
                start = res.values
    out["points"] = [results[i] for i in range(len(points))]
    bad = [p["value"] for p in out["points"] if not p["valid"]]
    if bad:
        out["flags"].append(f"{len(bad)} scan fits failed at {bad[:5]}{'...' if len(bad) > 5 else ''}")
    lower_neg = [p for p in out["points"] if p["valid"] and p["deltaNLL2"] < -1e-3]
    if lower_neg:
        out["flags"].append("a scan point has a lower NLL than the best fit; the free fit is not the global minimum")
    out["intervals"] = {}
    for key, level in LEVELS.items():
        lo, hi, flag = _crossings([p["value"] for p in out["points"] if p["valid"]],
                                  [p["deltaNLL2"] for p in out["points"] if p["valid"]], best, level)
        out["intervals"][key] = {"lower": lo, "upper": hi}
        if flag:
            out["flags"].append(f"{key}% interval: {flag}")
    return out


def _crossings(xs, ys, best, level):
    # the best fit itself is a point of the curve (2 DeltaNLL = 0); without it a coarse grid
    # can put a crossing on the wrong side of the minimum
    xs = np.append(np.asarray(xs, dtype=float), best)
    ys = np.append(np.asarray(ys, dtype=float), 0.0)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    lo = hi = None
    flags = []
    left = np.where(xs < best)[0]
    right = np.where(xs >= best)[0]
    for idx in left[::-1]:
        if ys[idx] >= level:
            j = idx + 1
            if j < len(xs):
                lo = xs[idx] + (xs[j] - xs[idx]) * (level - ys[idx]) / (ys[j] - ys[idx]) if ys[j] != ys[idx] else xs[idx]
            break
    for idx in right:
        if ys[idx] >= level:
            j = idx - 1
            if j >= 0:
                hi = xs[j] + (xs[idx] - xs[j]) * (level - ys[j]) / (ys[idx] - ys[j]) if ys[idx] != ys[j] else xs[idx]
            break
    # a missing crossing is only suspicious if the scan extends beyond the best fit on that
    # side (e.g. r_hat at its lower bound 0 has no lower crossing by construction)
    if lo is None and len(left):
        flags.append("no lower crossing inside the scan range")
    if hi is None and len(right) > 1:
        flags.append("no upper crossing inside the scan range")
    inside = int(np.sum((ys < level) & (xs != best)))
    if inside < 3:
        flags.append(f"only {inside} scan points inside the interval; linear interpolation is coarse, "
                     "use more points or a narrower --range")
    return lo, hi, "; ".join(flags) if flags else None


def fit_summary(lik, fit, data) -> dict:
    """Best-fit values, errors and constraint pulls (theta_hat - g) / sigma."""
    out = {"valid": fit.valid, "status": fit.status, "nll": fit.nll, "parameters": {}, "at_limit": fit.at_limit}
    for p in lik.parameters:
        entry = {"value": fit.value(lik, p.name), "role": p.role}
        if p.name in fit.errors:
            entry["error"] = fit.errors[p.name]
        if p.name in fit.minos:
            lo, hi, ok = fit.minos[p.name]
            entry["minos"] = {"lower": lo, "upper": hi, "valid": ok}
        c = p.constraint
        if c is not None and c.has_global_observable and p.role != I.ROLE_CONSTANT:
            g = data.global_obs[p.name]
            if c.kind == I.CONSTRAINT_POISSON:
                sigma = math.sqrt(max(g, 1.0))
            else:
                sigma = c.sigma_lo if entry["value"] < g else c.sigma_hi
            entry["pull"] = (entry["value"] - g) / sigma
            if p.name in fit.errors:
                entry["constraint"] = fit.errors[p.name] / sigma
        out["parameters"][p.name] = entry
    if fit.covariance is not None:
        cov = np.asarray(fit.covariance)
        d = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
        out["correlation"] = {"names": fit.cov_names, "matrix": (cov / np.outer(d, d)).tolist()}
    return out


# ----------------------------------------------------------------------------------------
# 2D scans (MultiDimFit --algo grid with two -P parameters)
# ----------------------------------------------------------------------------------------

# 2*DeltaNLL thresholds for 68.27% and 95% regions in two parameters (chi2 with 2 dof)
LEVELS_2D = {"68": 2.295748928898636, "95": 5.991464547107979}


def grid_axes_2d(ranges, points: int):
    """Combine's 2D grid: ceil(sqrt(points)) points per axis at the centres of equal intervals."""
    n = int(math.ceil(math.sqrt(points)))
    return [grid_points(lo, hi, n) for lo, hi in ranges]


def profile_scan_2d(lik, fitter, data, params, xs, ys, free=None) -> dict:
    """2*DeltaNLL on the grid xs x ys (every other parameter profiled), relative to the free fit.

    Points are fitted in order of their distance from the best fit (in units of the scan
    ranges); each starts from the parameter values of the nearest point already fitted, so the
    fits follow the valley of the likelihood.  A failed fit is retried from the best fit before
    the point is reported invalid (None, flagged)."""
    px, py = params
    for p in params:
        if p not in lik.index:
            raise KeyError(f"unknown parameter '{p}'")
    if px == py:
        raise ValueError("a 2D scan needs two different parameters")
    free = free or fitter.fit(data)
    if not free.valid:
        raise RuntimeError(f"free fit failed: {free.status}")
    best = np.array([free.value(lik, px), free.value(lik, py)])
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    span = np.array([max(np.ptp(xs), 1e-12), max(np.ptp(ys), 1e-12)])
    coords = np.array([(x, y) for x in xs for y in ys])            # Combine's order: x outer, y inner
    order = np.argsort(np.linalg.norm((coords - best) / span, axis=1), kind="stable")
    done_xy, done_vals = [], []
    results = [None] * len(coords)
    for k in order:
        x, y = coords[k]
        start = free.values
        if done_xy:
            d = np.linalg.norm((np.array(done_xy) - coords[k]) / span, axis=1)
            start = done_vals[int(np.argmin(d))]
        fixed = {px: float(x), py: float(y)}
        res = fitter.fit(data, start=start, fixed=fixed)
        if not res.valid:
            res = fitter.fit(data, start=free.values, fixed=fixed)
        results[k] = {"x": float(x), "y": float(y), "deltaNLL2": 2.0 * (res.nll - free.nll) if res.valid else None,
                      "valid": res.valid, "status": res.status}
        if res.valid:
            done_xy.append(coords[k])
            done_vals.append(res.values)
    out = {"params": [px, py], "best_fit": best.tolist(), "nll_min": free.nll,
           "axes": [xs.tolist(), ys.tolist()], "points": results, "flags": []}
    bad = [(p["x"], p["y"]) for p in results if not p["valid"]]
    if bad:
        out["flags"].append(f"{len(bad)} scan fits failed, e.g. at {bad[:3]}")
    if any(p["valid"] and p["deltaNLL2"] < -1e-3 for p in results):
        out["flags"].append("a scan point has a lower NLL than the best fit; the free fit is not the global minimum")
    z = np.full((len(xs), len(ys)), np.nan)
    for i, p in enumerate(results):
        if p["valid"]:
            z[i // len(ys), i % len(ys)] = p["deltaNLL2"]
    out["contours"] = {}
    for key, level in LEVELS_2D.items():
        lines = contour_lines(xs, ys, z, level)
        entry = {"level": level, "lines": [ln.tolist() for ln in lines]}
        if lines:
            allp = np.vstack(lines)
            entry["x_range"] = [float(allp[:, 0].min()), float(allp[:, 0].max())]
            entry["y_range"] = [float(allp[:, 1].min()), float(allp[:, 1].max())]
            closed = all(np.allclose(ln[0], ln[-1]) for ln in lines)
            if not closed:
                out["flags"].append(f"{key}% contour is not closed inside the scan range; widen --range")
        else:
            out["flags"].append(f"no {key}% contour inside the scan range")
        out["contours"][key] = entry
    inside = int(np.sum(z < LEVELS_2D["68"]))
    if inside < 4:
        out["flags"].append(f"only {inside} grid points inside the 68% region; the contours are coarse, "
                            "use more points or a narrower --range")
    return out


def contour_lines(xs, ys, z, level):
    """Contour lines of z[i, j] (at xs[i], ys[j]) at ``level``, linear interpolation between grid
    points (contourpy, the library under matplotlib's contour).  Invalid points are masked."""
    import contourpy

    zm = np.ma.masked_invalid(np.asarray(z, dtype=float).T)        # contourpy wants z[y, x]
    gen = contourpy.contour_generator(np.asarray(xs), np.asarray(ys), zm, line_type=contourpy.LineType.Separate)
    return [np.asarray(ln) for ln in gen.lines(level)]
