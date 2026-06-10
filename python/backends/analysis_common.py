"""
Backend-agnostic analysis algorithms and orchestration.

This module contains:

1. **Pure helpers** (resolve_data_mode, checkpoint_mismatches, …) – unchanged
   from the original implementation.

2. **Common statistical algorithms** that work with any backend via the
   ``AnalysisBackend`` protocol defined in ``backends/analysis_backend.py``:
     - ``interpolate_cls_crossing``  – linear interpolation of the CLs threshold
     - ``compute_cls_scan``          – full CLs scan over a POI grid
     - ``compute_cls_scan_smart``    – adaptive scan with automatic range expansion
     - ``extract_expected_cls_quantiles`` – normalise expected band into quantile dict
     - ``compute_poi_pull``          – (poi_fit - poi_true) / poi_unc
     - ``default_cls_scan_range``    – heuristic scan-range defaults
     - ``estimate_poi_unc_from_profile`` – sigma from delta-NLL = 0.5 crossing
     - ``compute_nll_profile_scan``  – delta-NLL curve for plotting
     - ``compute_feldman_cousins``   – Neyman-belt FC confidence interval

3. **``run_analysis_common``** – the backend-agnostic analysis loop (iterate
   over datasets, fit, CLs, FC, NLL scan, assemble summary dicts).
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from backends.analysis_types import CLsResult, FCResult, FitResult, NLLScanResult


# ===========================================================================
# Section 0 – Shared naming conventions
# ===========================================================================

def is_signal_strength_poi(name: str) -> bool:
    """Return True when *name* identifies a signal-strength parameter.

    A signal-strength POI is either the bare name ``"mu"`` (the default for
    models with a single signal process) or any name that starts with the
    ``"mu_"`` prefix (multi-process models or legacy names).
    """
    s = str(name)
    return s == "mu" or s.startswith("mu_")


# ===========================================================================
# Section 1 – Original helpers (unchanged public API)
# ===========================================================================

def load_analysis_model(model_file, input_card, load_fit_model_fn, parse_model_card_fn, build_model_from_card_fn):
    if model_file is not None:
        return load_fit_model_fn(os.path.abspath(model_file))

    card_path = os.path.abspath(input_card)
    card = parse_model_card_fn(card_path)
    return build_model_from_card_fn(card, os.path.dirname(card_path))


def resolve_data_mode(use_observed_data, use_asimov_data):
    if use_observed_data:
        return "observed"
    if use_asimov_data:
        return "asimov"
    return "toy"


def resolve_dataset_mode(toys, has_observed_data, *, error_suffix=""):
    if toys is None:
        return has_observed_data, False, 1
    if toys == -1:
        return False, True, 1
    if toys < -1:
        suffix = f" {error_suffix}" if error_suffix else ""
        raise ValueError(f"Only --toys -1 is supported as a special Asimov mode{suffix}")
    return False, False, int(toys)


def checkpoint_mismatches(checkpoint, expected):
    mismatches = []
    for key, expected_value in expected.items():
        if key not in checkpoint:
            mismatches.append((key, "<missing>", expected_value))
            continue
        if checkpoint.get(key) != expected_value:
            mismatches.append((key, checkpoint.get(key), expected_value))
    return mismatches


def normalize_output_path(output_path, extension):
    normalized_ext = extension if extension.startswith(".") else f".{extension}"
    abs_out = os.path.abspath(output_path)
    if abs_out.lower().endswith(normalized_ext.lower()):
        return abs_out
    base, _ = os.path.splitext(abs_out)
    return f"{base}{normalized_ext}"


# ===========================================================================
# Section 2 – Common statistical algorithms
# ===========================================================================

# ---------------------------------------------------------------------------
# 2a. CLs crossing interpolation (unified from zmodel._limit_from_curve and
#     hfmodel._interpolate_upper_limit)
# ---------------------------------------------------------------------------

def interpolate_cls_crossing(
    poi_values: np.ndarray,
    cls_values: np.ndarray,
    alpha: float,
) -> Optional[float]:
    """Linear interpolation of the POI where CLs crosses *alpha* (from above).

    Parameters
    ----------
    poi_values:
        Monotonically increasing array of POI scan points.
    cls_values:
        CLs values at each scan point (may contain NaNs).
    alpha:
        The CLs threshold (e.g. 0.05 for 95 % CL).

    Returns
    -------
    float or None
        Interpolated crossing point, or None if no crossing was found in the
        finite region of the curve.
    """
    poi_values = np.asarray(poi_values, dtype=float)
    cls_values = np.asarray(cls_values, dtype=float)

    valid = np.isfinite(cls_values)
    if not np.any(valid):
        return None

    pois = poi_values[valid]
    vals = cls_values[valid]

    if vals.size == 0:
        return None

    # Sort by POI (ascending)
    order = np.argsort(pois)
    pois = pois[order]
    vals = vals[order]

    # If the entire curve is above alpha the limit is at the last point
    if not np.any(vals <= float(alpha)):
        return float(pois[-1])

    # If the entire curve is below alpha the limit is at the first point
    if not np.any(vals > float(alpha)):
        return float(pois[0])

    # Find the first index where vals drops at or below alpha
    for idx in range(1, len(vals)):
        if vals[idx - 1] > alpha >= vals[idx]:
            x0, x1 = float(pois[idx - 1]), float(pois[idx])
            y0, y1 = float(vals[idx - 1]), float(vals[idx])
            if y1 == y0:
                return x1
            t = (alpha - y0) / (y1 - y0)
            return float(x0 + t * (x1 - x0))

    # Fallback: numerical interpolation
    return float(np.interp(float(alpha), vals[::-1], pois[::-1]))


# ---------------------------------------------------------------------------
# 2b. Default scan-range helpers (unified from both backends)
# ---------------------------------------------------------------------------

def default_cls_scan_max(
    poi_name: str,
    poi_bounds: Optional[Tuple[float, float]],
    signal_nominal_yield: Optional[float] = None,
    poi_is_signal_strength: bool = True,
) -> float:
    """Return a sensible upper bound for a CLs scan over *poi_name*.

    For signal-strength parameters (mu_*) the default is 5.0 unless a finite
    upper bound is available from the model.  For absolute-yield parameters
    the default is 3× the nominal yield (≥ 50).
    """
    if poi_bounds is not None:
        _, high = poi_bounds
        if np.isfinite(float(high)) and float(high) > 0.0:
            return float(high)

    if poi_is_signal_strength:
        return 5.0

    if signal_nominal_yield is not None and float(signal_nominal_yield) > 0.0:
        return max(50.0, 3.0 * float(signal_nominal_yield))

    return 50.0


def default_poi_scan_lower(
    poi_name: str,
    poi_value: float,
    poi_bounds: Optional[Tuple[float, float]],
    poi_is_signal_strength: bool = True,
) -> float:
    if poi_bounds is not None:
        low, _ = poi_bounds
        if np.isfinite(float(low)):
            return float(low)
    if poi_is_signal_strength:
        return 0.0
    return float(poi_value) - 5.0


def default_poi_scan_upper(
    poi_name: str,
    poi_value: float,
    poi_bounds: Optional[Tuple[float, float]],
    poi_is_signal_strength: bool = True,
    signal_nominal_yield: Optional[float] = None,
    requested_max: Optional[float] = None,
) -> float:
    if requested_max is not None:
        upper = float(requested_max)
        # Still clip to the model upper bound if finite
        if poi_bounds is not None:
            _, model_high = poi_bounds
            if np.isfinite(float(model_high)):
                upper = min(upper, float(model_high))
        return float(max(0.0, upper))

    if poi_bounds is not None:
        _, high = poi_bounds
        if np.isfinite(float(high)):
            # Leave a tiny margin from the boundary for numerical stability
            return 0.99 * float(high)

    return default_cls_scan_max(
        poi_name=poi_name,
        poi_bounds=None,
        signal_nominal_yield=signal_nominal_yield,
        poi_is_signal_strength=poi_is_signal_strength,
    )


# ---------------------------------------------------------------------------
# 2c. CLs scan (unified from zmodel._compute_cls and hfmodel._compute_cls_summary)
# ---------------------------------------------------------------------------

def compute_cls_scan(
    backend,
    state: Any,
    alpha: float,
    scan_max: float,
    scan_points: int,
) -> CLsResult:
    """Scan the POI from 0 to *scan_max* and compute CLs at each point.

    Parameters
    ----------
    backend : AnalysisBackend
        The backend adapter implementing ``hypothesis_test``.
    state :
        Backend-opaque state object.
    alpha :
        CLs threshold (e.g. 0.05).
    scan_max :
        Upper end of the POI scan range (lower end is always 0).
    scan_points :
        Number of evenly-spaced grid points.

    Returns
    -------
    CLsResult
    """
    scan_values = np.linspace(0.0, float(scan_max), int(scan_points))

    observed_cls: List[float] = []
    expected_cls_by_sigma: Dict[int, List[float]] = {s: [] for s in (-2, -1, 0, 1, 2)}
    observed_band: List[List[float]] = []  # for hfmodel-style curve storage

    for mu in scan_values:
        result = backend.hypothesis_test(state, poi_test=float(mu))
        obs = float(result.observed_cls) if np.isfinite(result.observed_cls) else float("nan")
        observed_cls.append(obs)
        for sigma in (-2, -1, 0, 1, 2):
            val = result.expected_cls.get(sigma, float("nan"))
            expected_cls_by_sigma[sigma].append(float(val) if np.isfinite(val) else float("nan"))
        # Store all 5 expected bands in order for the curve dict
        observed_band.append(
            [
                result.expected_cls.get(-2, float("nan")),
                result.expected_cls.get(-1, float("nan")),
                result.expected_cls.get(0, float("nan")),
                result.expected_cls.get(1, float("nan")),
                result.expected_cls.get(2, float("nan")),
            ]
        )

    observed_limit = interpolate_cls_crossing(scan_values, np.asarray(observed_cls), alpha)
    expected_limits = {
        sigma: interpolate_cls_crossing(scan_values, np.asarray(vals), alpha)
        for sigma, vals in expected_cls_by_sigma.items()
    }

    expected_quantiles = _build_expected_quantiles(expected_limits)

    curve = {
        "pois": scan_values.tolist(),
        "observed": [float(x) for x in observed_cls],
        "expected_median": [float(x) for x in expected_cls_by_sigma[0]],
        "expected_band": [[float(v) for v in row] for row in observed_band],
    }

    return CLsResult(
        observed_limit=observed_limit,
        expected_limit=expected_limits.get(0),
        expected_quantiles=expected_quantiles,
        scan_points=int(scan_points),
        scan_max=float(scan_max),
        curve=curve,
    )


def _build_expected_quantiles(
    expected_limits: Dict[int, Optional[float]],
) -> Dict[str, float]:
    """Convert {sigma: limit} to the standard quantile dict format."""
    q2p5 = expected_limits.get(-2)
    q16 = expected_limits.get(-1)
    q50 = expected_limits.get(0)
    q84 = expected_limits.get(1)
    q97p5 = expected_limits.get(2)

    # Fallback approximation when ±2σ bands are unavailable
    if q50 is not None and q84 is not None and q97p5 is None:
        q97p5 = float(q50) + 2.0 * (float(q84) - float(q50))
    if q50 is not None and q16 is not None and q2p5 is None:
        q2p5 = float(q50) - 2.0 * (float(q50) - float(q16))

    result: Dict[str, float] = {}
    for key, val in [("2.5%", q2p5), ("16%", q16), ("50%", q50), ("84%", q84), ("97.5%", q97p5)]:
        if val is not None:
            result[key] = float(val)
    return result


# ---------------------------------------------------------------------------
# 2d. Adaptive CLs scan (from zmodel._compute_cls_smart)
# ---------------------------------------------------------------------------

def compute_cls_scan_smart(
    backend,
    state: Any,
    alpha: float,
    scan_max: float,
    scan_points: int,
    max_expansions: int = 6,
) -> Tuple[CLsResult, float, int]:
    """Adaptive CLs scan that expands the range until the limit is well inside it.

    Expands *scan_max* by doubling up to *max_expansions* times when:
    - the observed limit falls outside 80 % of the current range, or
    - the expected +2σ band limit is not available.

    After convergence a refinement pass over a tighter range around the
    observed limit is performed for better accuracy.

    Returns
    -------
    (result, used_scan_max, used_scan_points)
    """
    upper = max(float(scan_max), 1e-6)
    points = max(int(scan_points), 9)
    result = None

    for _ in range(int(max_expansions)):
        result = compute_cls_scan(backend, state, alpha, upper, points)
        observed = result.observed_limit
        expected_p2 = result.expected_quantiles.get("97.5%")

        if observed is None:
            return result, upper, points

        observed = float(observed)
        has_expected_p2 = expected_p2 is not None
        if observed <= 0.80 * upper and has_expected_p2:
            break

        upper *= 2.0
        points = max(points + 8, 25)

    if result is None:
        result = compute_cls_scan(backend, state, alpha, upper, points)

    observed = result.observed_limit
    if observed is None:
        return result, upper, points

    observed = float(observed)
    if observed > 0.0:
        refined_upper = max(observed * 1.5, 1e-6)
        refined_points = max(points, 41)
        result = compute_cls_scan(backend, state, alpha, refined_upper, refined_points)
        return result, refined_upper, refined_points

    return result, upper, points


# ---------------------------------------------------------------------------
# 2e. Extract expected CLs quantiles from a variety of dict layouts
#     (from zmodel._extract_expected_cls_quantiles)
# ---------------------------------------------------------------------------

def extract_expected_cls_quantiles(
    cls_result: Any,
) -> Optional[Dict[str, float]]:
    """Normalise a CLs result into a standard quantile dict.

    Accepts a ``CLsResult`` dataclass, a plain dict (legacy format from either
    backend), or None.

    Returns ``{"2.5%": …, "16%": …, "50%": …, "84%": …, "97.5%": …}`` or
    None if not enough information is available.
    """
    if cls_result is None:
        return None

    # New dataclass path
    if isinstance(cls_result, CLsResult):
        q = cls_result.expected_quantiles
        if len(q) >= 3:
            return dict(q)
        return None

    # Legacy dict path (from old hfmodel inline construction)
    if not isinstance(cls_result, dict):
        return None

    expected = cls_result.get("expected")
    if isinstance(expected, dict):
        q2p5 = expected.get("2.5") or expected.get("2p5") or expected.get("-2sigma")
        q16 = expected.get("16") or expected.get("16.0") or expected.get("-1sigma")
        q50 = expected.get("50") or expected.get("50.0") or expected.get("median")
        q84 = expected.get("84") or expected.get("84.0") or expected.get("+1sigma")
        q97p5 = expected.get("97.5") or expected.get("97p5") or expected.get("+2sigma")
    elif isinstance(expected, (list, tuple)) and len(expected) >= 5:
        q2p5, q16, q50, q84, q97p5 = expected[:5]
    else:
        q2p5 = cls_result.get("expected_m2")
        q16 = cls_result.get("expected_m1")
        q50 = cls_result.get("expected")
        q84 = cls_result.get("expected_p1")
        q97p5 = cls_result.get("expected_p2")

    # Fallback approximation when ±2σ are missing
    if q50 is not None and q84 is not None and q97p5 is None:
        q97p5 = float(q50) + 2.0 * (float(q84) - float(q50))
    if q50 is not None and q16 is not None and q2p5 is None:
        q2p5 = float(q50) - 2.0 * (float(q50) - float(q16))

    values = [q2p5, q16, q50, q84, q97p5]
    if any(v is None for v in values):
        return None

    return {
        "2.5%": float(q2p5),
        "16%": float(q16),
        "50%": float(q50),
        "84%": float(q84),
        "97.5%": float(q97p5),
    }


# ---------------------------------------------------------------------------
# 2f. POI pull
# ---------------------------------------------------------------------------

def compute_poi_pull(
    poi_fit: Optional[float],
    poi_true: Optional[float],
    poi_unc: Optional[float],
) -> Optional[float]:
    """(poi_fit - poi_true) / poi_unc, or None when any input is missing/zero."""
    if poi_fit is None or poi_true is None or poi_unc is None:
        return None
    poi_fit = float(poi_fit)
    poi_true = float(poi_true)
    poi_unc = float(poi_unc)
    if not (np.isfinite(poi_fit) and np.isfinite(poi_true) and np.isfinite(poi_unc) and poi_unc > 0.0):
        return None
    return (poi_fit - poi_true) / poi_unc


# ---------------------------------------------------------------------------
# 2g. NLL profile scan (unified from zmodel._compute_nll_scan_for_plot and
#     hfmodel._compute_delta_nll_scan)
# ---------------------------------------------------------------------------

def compute_nll_profile_scan(
    backend,
    state: Any,
    poi_best: float,
    poi_unc: Optional[float],
    poi_bounds: Optional[Tuple[float, float]],
    poi_is_signal_strength: bool = True,
    signal_nominal_yield: Optional[float] = None,
    scan_max: Optional[float] = None,
    n_points: int = 121,
) -> Optional[NLLScanResult]:
    """Profile-likelihood scan of the POI for plotting.

    Scans from ``best - 5*sigma`` to ``best + 5*sigma`` when *poi_unc* is
    available; otherwise falls back to the model bounds.  Scanning proceeds
    outward from the best-fit point using warm starts to minimise artifacts.

    Returns
    -------
    NLLScanResult or None
    """
    poi_name_str = backend.poi_name(state)

    # Determine scan range
    if poi_unc is not None and np.isfinite(poi_unc) and poi_unc > 0.0:
        scan_low = poi_best - 5.0 * poi_unc
        scan_high = poi_best + 5.0 * poi_unc
    else:
        scan_low = default_poi_scan_lower(poi_name_str, poi_best, poi_bounds, poi_is_signal_strength)
        scan_high = default_poi_scan_upper(
            poi_name_str, poi_best, poi_bounds,
            poi_is_signal_strength=poi_is_signal_strength,
            signal_nominal_yield=signal_nominal_yield,
            requested_max=scan_max,
        )

    # Clip to parameter bounds
    if poi_bounds is not None:
        low_b, high_b = poi_bounds
        if np.isfinite(float(low_b)):
            scan_low = max(scan_low, float(low_b))
        if np.isfinite(float(high_b)):
            scan_high = min(scan_high, 0.99 * float(high_b))

    if not np.isfinite(scan_low) or not np.isfinite(scan_high) or scan_high <= scan_low:
        return None

    scan_values = np.linspace(scan_low, scan_high, int(n_points))
    snapshot = backend.snapshot_parameters(state)
    nll_values = np.full(int(n_points), np.nan, dtype=float)

    try:
        center_idx = int(np.argmin(np.abs(scan_values - poi_best)))

        # Scan outward from center in both directions (warm starts)
        backend.restore_parameters(state, snapshot)
        backend.set_poi_value(state, float(scan_values[center_idx]))
        fit = backend.fixed_poi_fit(state, float(scan_values[center_idx]))
        nll_values[center_idx] = fit.nll

        for idx in range(center_idx + 1, len(scan_values)):
            backend.set_poi_value(state, float(scan_values[idx]))
            fit = backend.fixed_poi_fit(state, float(scan_values[idx]))
            nll_values[idx] = fit.nll

        backend.restore_parameters(state, snapshot)
        backend.set_poi_value(state, float(scan_values[center_idx]))
        backend.fixed_poi_fit(state, float(scan_values[center_idx]))

        for idx in range(center_idx - 1, -1, -1):
            backend.set_poi_value(state, float(scan_values[idx]))
            fit = backend.fixed_poi_fit(state, float(scan_values[idx]))
            nll_values[idx] = fit.nll

    except Exception:
        return None
    finally:
        backend.restore_parameters(state, snapshot)

    nll_arr = np.asarray(nll_values, dtype=float)
    delta_nll = nll_arr - float(np.nanmin(nll_arr))

    return NLLScanResult(
        poi_name=poi_name_str,
        poi_values=scan_values.tolist(),
        delta_nll_values=delta_nll.tolist(),
    )


# ---------------------------------------------------------------------------
# 2h. POI uncertainty from delta-NLL profile crossing
#     (unified from zmodel._estimate_poi_uncertainty_from_profile and
#      hfmodel._estimate_poi_uncertainty)
# ---------------------------------------------------------------------------

def estimate_poi_unc_from_profile(
    backend,
    state: Any,
    poi_best: float,
    poi_bounds: Optional[Tuple[float, float]],
    poi_is_signal_strength: bool = True,
    signal_nominal_yield: Optional[float] = None,
    n_points: int = 41,
) -> Optional[float]:
    """Estimate POI sigma from the delta-NLL = 0.5 profile crossing.

    Returns None when the crossing cannot be located.
    """
    scan = compute_nll_profile_scan(
        backend=backend,
        state=state,
        poi_best=poi_best,
        poi_unc=None,
        poi_bounds=poi_bounds,
        poi_is_signal_strength=poi_is_signal_strength,
        signal_nominal_yield=signal_nominal_yield,
        n_points=n_points,
    )
    if scan is None:
        return None

    x = np.asarray(scan.poi_values, dtype=float)
    y = np.asarray(scan.delta_nll_values, dtype=float)
    if x.size < 5 or y.size != x.size:
        return None

    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 5:
        return None

    target = 0.5

    # A completely flat profile means the POI is unconstrained
    if np.nanmax(y) < target:
        return float("inf")

    best_idx = int(np.argmin(y))

    def _crossing(xseg, yseg):
        if xseg.size < 2:
            return None
        for i in range(xseg.size - 1):
            y0, y1 = float(yseg[i]), float(yseg[i + 1])
            if (y0 - target) == 0.0:
                return float(xseg[i])
            if (y0 - target) * (y1 - target) <= 0.0:
                if y1 == y0:
                    return float(0.5 * (float(xseg[i]) + float(xseg[i + 1])))
                t = (target - y0) / (y1 - y0)
                return float(float(xseg[i]) + t * (float(xseg[i + 1]) - float(xseg[i])))
        return None

    x_left = x[: best_idx + 1][::-1]
    y_left = y[: best_idx + 1][::-1]
    x_right = x[best_idx:]
    y_right = y[best_idx:]

    center = float(x[best_idx])
    left_cross = _crossing(x_left, y_left)
    right_cross = _crossing(x_right, y_right)

    candidates = []
    if left_cross is not None:
        candidates.append(center - left_cross)
    if right_cross is not None:
        candidates.append(right_cross - center)
    if not candidates:
        return None

    unc = float(np.nanmean(np.asarray(candidates, dtype=float)))
    if np.isfinite(unc) and unc > 0.0:
        return unc
    return None


# ---------------------------------------------------------------------------
# 2i. Feldman-Cousins (unified from zmodel._compute_feldman_cousins_for_toy
#     and hfmodel._compute_feldman_cousins_summary)
# ---------------------------------------------------------------------------

def compute_feldman_cousins(
    backend,
    state: Any,
    alpha: float,
    scan_points: int,
    n_toys: int,
    scan_max: float,
    dataset_id: int = 0,
    seed: int = 1234,
) -> FCResult:
    """Feldman-Cousins confidence interval via the Neyman construction.

    For each point on a POI grid the method:
      1. Generates *n_toys* toy datasets at that true POI value.
      2. Fits each toy to obtain the profile test statistic q_mu.
      3. Determines the critical value q_crit at level *alpha*.
      4. Checks whether q_obs (computed on the real data) satisfies q_obs <= q_crit.

    The FC interval is the union of all accepted grid points.

    Parameters
    ----------
    backend : AnalysisBackend
    state :
        Backend state.  Must already have the observed data loaded.
    alpha :
        Significance level (e.g. 0.05 for 95 % CL).
    scan_points :
        Number of grid points in [0, scan_max].
    n_toys :
        Number of toy datasets generated per grid point.
    scan_max :
        Upper bound of the POI grid.
    dataset_id, seed :
        Reproducibility seeds.
    """
    fc_alpha = float(alpha)
    if not (0.0 < fc_alpha < 1.0):
        raise ValueError("Feldman-Cousins alpha must satisfy 0 < alpha < 1")

    n_scan = max(3, int(scan_points))
    n_toys_int = max(1, int(n_toys))
    grid_max = float(scan_max)
    if not np.isfinite(grid_max) or grid_max <= 0.0:
        raise ValueError("Feldman-Cousins scan_max must be finite and > 0")

    poi_grid = np.linspace(0.0, grid_max, n_scan)
    poi_name_str = backend.poi_name(state)

    starting_snapshot = backend.snapshot_parameters(state)

    # Capture the current dataset as the "observed" data for this FC call.
    # This is the iteration dataset (real observed, Asimov, or outer-loop toy)
    # that was active before any FC toy generation begins.  We must restore it
    # before computing q_obs at each grid point because the inner toy loop
    # overwrites state.current_data with toy datasets, which would cause q_obs
    # to be evaluated on the wrong data for every grid point after the first.
    iteration_data = backend.get_current_data(state)

    q_obs_values: List[Optional[float]] = []
    q_crit_values: List[Optional[float]] = []
    p_obs_values: List[Optional[float]] = []
    toy_valid_counts: List[int] = []
    accepted: List[float] = []

    rng = np.random.default_rng(int(seed) + int(dataset_id))

    try:
        for imu, mu_test in enumerate(poi_grid):
            # --- observed q_mu at this grid point ---
            # Restore both parameters AND the iteration data so that q_obs is
            # always computed on the same dataset regardless of toy generation.
            backend.restore_parameters(state, starting_snapshot)
            backend.set_data(state, iteration_data)
            q_obs = _compute_q_mu(backend, state, float(mu_test))

            if q_obs is None:
                q_obs_values.append(None)
                q_crit_values.append(None)
                toy_valid_counts.append(0)
                continue

            # --- toy q_mu distribution at this grid point ---
            toy_q: List[float] = []
            for itoy in range(n_toys_int):
                backend.restore_parameters(state, starting_snapshot)
                backend.set_poi_value(state, float(mu_test))

                toy_data = backend.generate_toy_data(state)
                backend.set_data(state, toy_data)

                q_toy = _compute_q_mu(backend, state, float(mu_test))
                if q_toy is not None and np.isfinite(q_toy):
                    toy_q.append(float(q_toy))

            toy_valid_counts.append(len(toy_q))

            if not toy_q:
                q_obs_values.append(float(q_obs))
                q_crit_values.append(None)
                continue

            q_crit = float(np.percentile(np.asarray(toy_q, dtype=float), 100.0 * (1.0 - fc_alpha)))
            p_obs  = float(np.mean(np.asarray(toy_q, dtype=float) > float(q_obs)))
            if float(q_obs) <= q_crit:
                accepted.append(float(mu_test))

            q_obs_values.append(float(q_obs))
            q_crit_values.append(q_crit)
            p_obs_values.append(p_obs)

    finally:
        # Restore observed data and original parameters
        backend.restore_parameters(state, starting_snapshot)
        obs_data = backend.get_observed_data(state)
        backend.set_data(state, obs_data)

    interval: Optional[Tuple[float, float]] = None
    if accepted:
        interval = (float(np.min(accepted)), float(np.max(accepted)))

    status = "ok" if interval is not None else "no-accepted-points"

    return FCResult(
        interval=interval,
        status=status,
        alpha=fc_alpha,
        poi_name=poi_name_str,
        grid={
            "poi": poi_grid.tolist(),
            "q_obs": q_obs_values,
            "q_crit": q_crit_values,
            "p_obs": p_obs_values,
            "toy_valid": toy_valid_counts,
        },
        scan_points=n_scan,
        n_toys=n_toys_int,
        scan_max=float(grid_max),
    )


def _compute_q_mu(backend, state: Any, mu_test: float) -> Optional[float]:
    """Profile test statistic q_mu = max(0, NLL(mu_test) - NLL(mu_hat)).

    This is the common implementation used by the FC algorithm.
    """
    try:
        free_result = backend.fit(state)
        if not np.isfinite(free_result.nll):
            return None
        nll_hat = free_result.nll

        fixed_result = backend.fixed_poi_fit(state, mu_test)
        if not np.isfinite(fixed_result.nll):
            return None
        nll_mu = fixed_result.nll

        q = nll_mu - nll_hat
        if not np.isfinite(q):
            return None
        return float(max(0.0, q))
    except Exception:
        return None


def compute_likelihood_interval(
    poi_values: np.ndarray,
    delta_nll_values: np.ndarray,
    alpha: float,
    poi_min_limit: float = 0.0,
) -> Optional[Tuple[float, float]]:
    """Compute a confidence interval using the likelihood-ratio method (asymptotic).

    This method applies the profile-likelihood asymptotic approximation where the
    critical value q_crit is derived from the chi-square distribution as:

        q_crit = (z_critical)^2

    where z_critical is the standard normal quantile at the desired confidence level.

    The observed test statistic at each POI value is:

        q_obs = 2 * delta_nll

    An interval point is accepted if q_obs <= q_crit. The confidence interval is
    the union of all accepted points.

    Parameters
    ----------
    poi_values : np.ndarray
        Array of POI grid points (must be sorted ascending).
    delta_nll_values : np.ndarray
        Array of delta-NLL values (NLL relative to best fit).
    alpha : float
        Significance level (e.g., 0.05 for 95% CL).
    poi_min_limit : float
        Minimum POI value to consider in the interval.

    Returns
    -------
    (lower, upper) : tuple of float or None
        The confidence interval endpoints, or None if no points are accepted.

    Notes
    -----
    This is an asymptotic approximation and not the true Feldman-Cousins Neyman
    construction (which generates toys and derives per-point critical values).
    For better coverage properties, use compute_feldman_cousins() instead.
    """
    from statistics import NormalDist

    poi_values = np.asarray(poi_values, dtype=float)
    delta_nll_values = np.asarray(delta_nll_values, dtype=float)
    alpha = float(alpha)
    poi_min_limit = float(poi_min_limit)

    if poi_values.size == 0 or delta_nll_values.size != poi_values.size:
        return None

    # Mask finite values
    mask = np.isfinite(poi_values) & np.isfinite(delta_nll_values)
    if not np.any(mask):
        return None

    poi = poi_values[mask]
    delta_nll = delta_nll_values[mask]

    # Sort by POI
    order = np.argsort(poi)
    poi = poi[order]
    delta_nll = delta_nll[order]

    # Apply POI minimum limit
    limit_mask = poi >= poi_min_limit
    poi = poi[limit_mask]
    delta_nll = delta_nll[limit_mask]

    if poi.size == 0:
        return None

    # Compute critical value from chi-square distribution
    # For one-sided test: q_crit = z_crit^2
    z_crit = NormalDist().inv_cdf(1.0 - 0.5 * alpha)
    q_crit = float(z_crit * z_crit)

    # Observed test statistic: q_obs = 2 * delta_nll
    q_obs = np.asarray(np.clip(2.0 * delta_nll, 0.0, None), dtype=float)

    # Accept points where q_obs <= q_crit
    accepted = q_obs <= q_crit
    if not np.any(accepted):
        return None

    accepted_poi = poi[accepted]
    interval = (float(np.min(accepted_poi)), float(np.max(accepted_poi)))
    return interval if interval[0] < interval[1] or interval[0] == interval[1] else None


# ===========================================================================
# Section 3 – run_analysis_common: backend-agnostic analysis orchestrator
# ===========================================================================

def _write_checkpoint_json(path: Optional[str], summaries: list, config: dict) -> None:
    if path is None:
        return
    checkpoint = {"config": config, "summaries": summaries}
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(checkpoint, handle, indent=2)
    except Exception as exc:
        print(f"Warning: checkpoint write failed: {exc}")


def run_analysis_common(
    backend,
    state: Any,
    toys: int,
    data_mode: str,
    *,
    cls_alpha: Optional[float] = None,
    signal_strength: Optional[float] = None,
    cls_scan_points: Optional[int] = None,
    cls_smart_scan: bool = False,
    poi_scan_max: Optional[float] = None,
    poi_bounds: Optional[Tuple[float, float]] = None,
    feldman_cousins_alpha: Optional[float] = None,
    feldman_cousins_scan_points: int = 21,
    feldman_cousins_n_toys: int = 100,
    feldman_cousins_scan_max: Optional[float] = None,
    compute_nll_scan: bool = False,
    nll_scan_points: int = 121,
    progress_callback: Optional[Callable] = None,
    checkpoint_freq: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
    existing_results: Optional[list] = None,
    resume_from_index: int = 0,
    seed: int = 1234,
) -> List[Dict[str, Any]]:
    """Backend-agnostic analysis loop.

    Parameters
    ----------
    backend : AnalysisBackend
        The backend adapter that provides all low-level statistical operations.
    state :
        Backend-opaque context object (model + data + minimiser + …).
    toys :
        Total number of datasets (toys, observed, or Asimov iterations).
    data_mode :
        ``"toy"``, ``"observed"``, or ``"asimov"``.
    cls_alpha :
        CLs significance level; pass None to skip CLs.
    signal_strength :
        Truth-level signal strength used when generating toy data.
    cls_scan_points :
        Number of POI scan points for CLs (default chosen by the common logic).
    cls_smart_scan :
        If True use adaptive range expansion for the CLs scan.
    poi_scan_max :
        Explicit upper bound for CLs / NLL scans.  If None the backend
        heuristic is used.
    poi_bounds :
        (lower, upper) bounds of the POI parameter; used for scan ranges.
    feldman_cousins_alpha :
        Significance level for Feldman-Cousins; pass None to skip.
    feldman_cousins_scan_points, feldman_cousins_n_toys, feldman_cousins_scan_max :
        FC construction parameters.
    compute_nll_scan :
        If True compute the delta-NLL profile scan for the *first* toy/dataset.
    nll_scan_points :
        Number of points in the NLL scan.
    progress_callback :
        Optional ``callback(summary, is_observed_fit=bool)`` called after each
        dataset.
    checkpoint_freq / checkpoint_path :
        Save a JSON checkpoint every *checkpoint_freq* completed datasets.
    existing_results / resume_from_index :
        Append to *existing_results* and start iteration at *resume_from_index*.
    seed :
        Base RNG seed.

    Returns
    -------
    list[dict]
        One summary dict per dataset in iteration order.
    """
    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if int(nll_scan_points) < 3:
        raise ValueError("nll_scan_points must be >= 3")
    if resume_from_index < 0:
        raise ValueError("resume_from_index must be >= 0")
    if checkpoint_freq is not None and int(checkpoint_freq) < 1:
        raise ValueError("checkpoint_freq must be >= 1")

    # Default scan-point counts
    n_cls_points = max(3, int(cls_scan_points)) if cls_scan_points is not None else 25

    poi_name_str = backend.poi_name(state)
    poi_is_ss = backend.poi_is_signal_strength
    sig_yield = backend.signal_nominal_yield

    # Initial parameter snapshot (pre-analysis state)
    global_snapshot = backend.snapshot_parameters(state)

    summaries = list(existing_results) if existing_results else []

    checkpoint_cfg = {
        "data_mode": data_mode,
        "cls_alpha": cls_alpha,
        "signal_strength": signal_strength,
        "poi_scan_max": poi_scan_max,
        "cls_scan_points": n_cls_points,
        "cls_smart_scan": bool(cls_smart_scan),
        "feldman_cousins_alpha": feldman_cousins_alpha,
        "feldman_cousins_scan_points": int(feldman_cousins_scan_points),
        "feldman_cousins_n_toys": int(feldman_cousins_n_toys),
        "feldman_cousins_scan_max": feldman_cousins_scan_max,
        "compute_nll_scan": bool(compute_nll_scan),
        "nll_scan_points": int(nll_scan_points),
    }

    # ------------------------------------------------------------------
    # Main iteration
    # ------------------------------------------------------------------
    for dataset_id in range(int(resume_from_index), int(toys)):
        t0 = time.perf_counter()

        # Restore global best-fit / pre-analysis state for each iteration
        backend.restore_parameters(state, global_snapshot)

        # --- 1. Set truth signal strength for toy generation ---
        if signal_strength is not None and data_mode == "toy":
            backend.set_poi_value(state, float(signal_strength))

        # --- 2. Generate / retrieve dataset ---
        if data_mode == "observed":
            data = backend.get_observed_data(state)
        elif data_mode == "asimov":
            data = backend.generate_asimov_data(state)
        else:
            data = backend.generate_toy_data(state)

        backend.set_data(state, data)

        # --- 3. Fit ---
        fit_error: Optional[str] = None
        try:
            fit_result = backend.fit(state)
        except Exception as exc:
            fit_result = FitResult(
                valid=False,
                poi_value=float("nan"),
                poi_uncertainty=None,
                nll=float("nan"),
            )
            fit_error = str(exc)

        # --- 4. POI uncertainty ---
        poi_unc = fit_result.poi_uncertainty
        if poi_unc is None and fit_error is None:
            try:
                poi_unc = backend.poi_uncertainty_hesse(state, fit_result)
            except Exception:
                pass
        if (poi_unc is None or not np.isfinite(poi_unc)) and fit_error is None:
            try:
                poi_unc = estimate_poi_unc_from_profile(
                    backend=backend,
                    state=state,
                    poi_best=fit_result.poi_value,
                    poi_bounds=poi_bounds,
                    poi_is_signal_strength=poi_is_ss,
                    signal_nominal_yield=sig_yield,
                    n_points=41,
                )
            except Exception:
                pass

        # --- 5. Build core summary dict ---
        poi_true = float(signal_strength) if signal_strength is not None else None

        summary: Dict[str, Any] = {
            "dataset_id": int(dataset_id),
            "valid": bool(fit_result.valid) and fit_error is None,
            "dataset_time_s": float(time.perf_counter() - t0),
            "poi_name": poi_name_str,
            "poi_fit": float(fit_result.poi_value) if np.isfinite(fit_result.poi_value) else None,
            "poi_unc_hesse": float(poi_unc) if poi_unc is not None and np.isfinite(poi_unc) else None,
            "fit_params": dict(fit_result.param_values),
            "fit_param_unc": dict(fit_result.param_uncertainties),
            "observed_fit": data_mode == "observed",
            "asimov_fit": data_mode == "asimov",
        }

        if poi_true is not None:
            summary["poi_true"] = poi_true
        if fit_result.edm is not None:
            summary["edm"] = fit_result.edm
        if fit_error is not None:
            summary["fit_error"] = fit_error
        if fit_result.extra:
            summary["fit_status"] = fit_result.extra

        # Pull
        summary["poi_pull"] = compute_poi_pull(summary["poi_fit"], poi_true, summary["poi_unc_hesse"])

        # --- 6. NLL profile scan (first toy/dataset only when enabled) ---
        if compute_nll_scan and dataset_id == int(resume_from_index) and fit_error is None:
            try:
                nll_scan = compute_nll_profile_scan(
                    backend=backend,
                    state=state,
                    poi_best=fit_result.poi_value,
                    poi_unc=poi_unc,
                    poi_bounds=poi_bounds,
                    poi_is_signal_strength=poi_is_ss,
                    signal_nominal_yield=sig_yield,
                    scan_max=poi_scan_max,
                    n_points=int(nll_scan_points),
                )
                if nll_scan is not None:
                    summary["delta_nll_scan"] = {
                        "poi_name": nll_scan.poi_name,
                        "poi_values": nll_scan.poi_values,
                        "delta_nll": nll_scan.delta_nll_values,
                    }
            except Exception as exc:
                summary["delta_nll_scan_error"] = str(exc)

        # --- 7. CLs limit ---
        if cls_alpha is not None and fit_error is None:
            _apply_cls_to_summary(
                summary=summary,
                backend=backend,
                state=state,
                cls_alpha=float(cls_alpha),
                poi_scan_max=poi_scan_max,
                poi_bounds=poi_bounds,
                n_cls_points=n_cls_points,
                cls_smart_scan=bool(cls_smart_scan),
                poi_is_signal_strength=poi_is_ss,
                signal_nominal_yield=sig_yield,
            )

        # --- 8. Feldman-Cousins ---
        if feldman_cousins_alpha is not None and fit_error is None:
            fc_max = feldman_cousins_scan_max
            if fc_max is None:
                cls_obs = summary.get("cls_observed")
                if cls_obs is not None and np.isfinite(float(cls_obs)) and float(cls_obs) > 0.0:
                    fc_max = max(0.25, 2.0 * float(cls_obs))
                elif poi_true is not None:
                    fc_max = max(0.5, 2.0 * abs(float(poi_true)))
                else:
                    fc_max = default_cls_scan_max(
                        poi_name=poi_name_str,
                        poi_bounds=poi_bounds,
                        signal_nominal_yield=sig_yield,
                        poi_is_signal_strength=poi_is_ss,
                    )

            try:
                fc = compute_feldman_cousins(
                    backend=backend,
                    state=state,
                    alpha=float(feldman_cousins_alpha),
                    scan_points=int(feldman_cousins_scan_points),
                    n_toys=int(feldman_cousins_n_toys),
                    scan_max=float(fc_max),
                    dataset_id=int(dataset_id),
                    seed=int(seed),
                )
                summary["feldman_cousins"] = {
                    "fc_status": fc.status,
                    "alpha": fc.alpha,
                    "poi_name": fc.poi_name,
                    "fc_interval": list(fc.interval) if fc.interval is not None else None,
                    "scan_points": fc.scan_points,
                    "n_toys_per_point": fc.n_toys,
                    "scan_max": fc.scan_max,
                    "grid": fc.grid,
                }
            except Exception as exc:
                summary["feldman_cousins"] = {
                    "fc_status": "failed",
                    "alpha": float(feldman_cousins_alpha),
                    "error": str(exc),
                }

        summaries.append(summary)

        if progress_callback is not None:
            try:
                progress_callback(summary, is_observed_fit=(data_mode == "observed"))
            except TypeError:
                # Backends that don't accept is_observed_fit kwarg
                progress_callback(summary)

        # --- Checkpoint ---
        if checkpoint_freq is not None and checkpoint_path is not None:
            done = dataset_id - int(resume_from_index) + 1
            if done % int(checkpoint_freq) == 0:
                _write_checkpoint_json(checkpoint_path, summaries, checkpoint_cfg)

    if checkpoint_freq is not None and checkpoint_path is not None:
        _write_checkpoint_json(checkpoint_path, summaries, checkpoint_cfg)

    return summaries


def _apply_cls_to_summary(
    summary: Dict[str, Any],
    backend,
    state: Any,
    cls_alpha: float,
    poi_scan_max: Optional[float],
    poi_bounds: Optional[Tuple[float, float]],
    n_cls_points: int,
    cls_smart_scan: bool,
    poi_is_signal_strength: bool,
    signal_nominal_yield: Optional[float],
) -> None:
    """Run the CLs scan and write results into *summary* in-place."""
    poi_name_str = backend.poi_name(state)
    scan_upper = default_cls_scan_max(
        poi_name=poi_name_str,
        poi_bounds=poi_bounds,
        signal_nominal_yield=signal_nominal_yield,
        poi_is_signal_strength=poi_is_signal_strength,
    )
    if poi_scan_max is not None:
        scan_upper = float(poi_scan_max)
        if poi_bounds is not None:
            _, model_high = poi_bounds
            if np.isfinite(float(model_high)):
                scan_upper = min(scan_upper, float(model_high))

    snapshot_before_cls = backend.snapshot_parameters(state)

    try:
        if cls_smart_scan:
            cls_result, used_max, used_points = compute_cls_scan_smart(
                backend, state, cls_alpha, scan_upper, n_cls_points
            )
        else:
            cls_result = compute_cls_scan(backend, state, cls_alpha, scan_upper, n_cls_points)
            used_max = float(scan_upper)
            used_points = int(n_cls_points)

        if cls_result.observed_limit is not None:
            obs = float(cls_result.observed_limit)
            summary["cls_observed"] = obs
            summary["cls_scan_points"] = int(used_points)
            summary["cls_scan_max"] = float(used_max)
            if signal_nominal_yield is not None and poi_is_signal_strength:
                summary["yield_upper_limit"] = obs * float(signal_nominal_yield)
            if cls_result.curve is not None:
                summary["cls_curve"] = cls_result.curve

        if cls_result.expected_quantiles:
            summary["cls_expected_quantiles"] = cls_result.expected_quantiles
        if cls_result.expected_limit is not None:
            summary["cls_expected"] = float(cls_result.expected_limit)

    except Exception as exc:
        summary["cls_error"] = str(exc)
    finally:
        backend.restore_parameters(state, snapshot_before_cls)
