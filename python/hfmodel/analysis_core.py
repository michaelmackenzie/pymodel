"""
pyhf analysis core.

This module retains the pyhf-specific operations that cannot be abstracted:

  - Runtime configuration (no-op for pyhf)
  - MLE fitting with retry logic (pyhf.infer.mle.fit)
  - Fixed-POI fitting (pyhf.infer.mle.fixed_poi_fit)
  - Toy and Asimov data generation
  - Hessian uncertainty estimation (manual finite-diff, minuit, jax)
  - The pyhf model's channel dataset-plot payload builder
  - Parameter override application

Everything that is backend-agnostic (CLs scan, Feldman-Cousins, NLL profile
scan, expected-quantile extraction, pull computation, the main analysis loop)
now lives in ``backends/analysis_common.py`` and is invoked via the
``PyhfAnalysisBackend`` adapter in ``hfmodel/analysis_backend.py``.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyhf

from backends.analysis_common import (
    resolve_data_mode as _resolve_data_mode,
    run_analysis_common,
)


# ===========================================================================
# Runtime configuration (no-op for pyhf)
# ===========================================================================

def configure_runtime():
    return None


# ===========================================================================
# Parameter override helpers
# ===========================================================================

def _override_vectors(model, fit_model):
    init_pars = np.asarray(model.config.suggested_init(), dtype=float)
    fixed_params = list(model.config.suggested_fixed())
    par_bounds = [tuple(bound) for bound in model.config.suggested_bounds()]

    overrides = getattr(fit_model, "analysis_overrides", {}) or {}
    set_values = overrides.get("set_values", {})
    set_ranges = overrides.get("set_ranges", {})
    freeze_names = set(overrides.get("freeze", []))

    index_map = {name: idx for idx, name in enumerate(model.config.par_order)}

    for name, value in set_values.items():
        init_pars[index_map[name]] = float(value)

    for name, bounds in set_ranges.items():
        low, high = bounds
        idx = index_map[name]
        existing_low, existing_high = par_bounds[idx]
        effective_low  = float(low)  if low  is not None else float(existing_low)
        effective_high = float(high) if high is not None else float(existing_high)
        par_bounds[idx] = (effective_low, effective_high)

    for name in freeze_names:
        fixed_params[index_map[name]] = True

    return init_pars.tolist(), par_bounds, fixed_params


# ===========================================================================
# Data generation
# ===========================================================================

def _expected_data_for_pars(model, pars):
    expected = np.asarray(model.expected_data(pars), dtype=float).reshape(-1)
    return expected


def _sample_full_model_data(model, truth_pars, rng):
    draw_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
    np_state = np.random.get_state()
    try:
        np.random.seed(draw_seed)
        pdf = model.make_pdf(np.asarray(truth_pars, dtype=float))
        sampled = pdf.sample((1,))
        data = np.asarray(sampled, dtype=float).reshape(-1)
    finally:
        np.random.set_state(np_state)
    return data


def _generate_toy_data(model, truth_pars, rng):
    try:
        return _sample_full_model_data(model, truth_pars, rng)
    except Exception:
        expected = _expected_data_for_pars(model, truth_pars)
        n_main = int(model.config.nmaindata)

        main_counts = np.clip(expected[:n_main], 0.0, None)
        aux = expected[n_main:]

        toy_main = rng.poisson(main_counts).astype(float)
        if aux.size:
            return np.concatenate([toy_main, aux.astype(float)])
        return toy_main


def _asimov_data(model, truth_pars):
    return _expected_data_for_pars(model, truth_pars)


# ===========================================================================
# MLE fitting
# ===========================================================================

def _extract_fit_result(bestfit_result):
    if isinstance(bestfit_result, tuple) and len(bestfit_result) >= 2:
        bestfit = np.asarray(bestfit_result[0], dtype=float)
        maybe_unc = np.asarray(bestfit_result[1], dtype=float)
        if maybe_unc.shape == bestfit.shape:
            unc = maybe_unc
        else:
            unc = np.full(bestfit.shape, np.nan, dtype=float)
    else:
        bestfit = np.asarray(bestfit_result, dtype=float)
        unc = np.full(bestfit.shape, np.nan, dtype=float)
    return bestfit, unc


def _fit_status_dict(result_obj):
    if result_obj is None:
        return {}

    payload = {
        "success": bool(getattr(result_obj, "success", False)),
        "status": getattr(result_obj, "status", None),
        "message": str(getattr(result_obj, "message", "")),
        "fun": None,
        "x": None,
        "nit": getattr(result_obj, "nit", None),
        "nfev": getattr(result_obj, "nfev", None),
        "njev": getattr(result_obj, "njev", None),
    }

    fun = getattr(result_obj, "fun", None)
    if fun is not None:
        try:
            payload["fun"] = float(fun)
        except Exception:
            payload["fun"] = None

    x = getattr(result_obj, "x", None)
    if x is not None:
        try:
            payload["x"] = np.asarray(x, dtype=float).reshape(-1).tolist()
        except Exception:
            payload["x"] = None

    return payload


def _jittered_init_pars(base_init, bounds, fixed_params, rng, scale=0.25):
    init = np.asarray(base_init, dtype=float).copy()
    fixed = list(fixed_params)

    for idx, (low, high) in enumerate(bounds):
        if idx < len(fixed) and fixed[idx]:
            continue
        width = float(high) - float(low)
        if width <= 0.0 or not np.isfinite(width):
            continue
        shift = float(rng.normal(loc=0.0, scale=scale * width))
        init[idx] = float(np.clip(init[idx] + shift, float(low), float(high)))

    return init.tolist()


def _run_mle_fit(model, data, init_pars, par_bounds, fixed_params):
    raw = pyhf.infer.mle.fit(
        data,
        model,
        init_pars=init_pars,
        par_bounds=par_bounds,
        fixed_params=fixed_params,
        return_result_obj=True,
    )

    if isinstance(raw, tuple) and len(raw) >= 2:
        bestfit = np.asarray(raw[0], dtype=float)
        result_obj = raw[1]
        return bestfit, result_obj

    if hasattr(raw, "x"):
        bestfit = np.asarray(getattr(raw, "x"), dtype=float)
        return bestfit, raw

    return np.asarray(raw, dtype=float), None


def _fit_with_retries(model, data, init_pars, par_bounds, fixed_params, rng, max_retries=4):
    attempts = [list(init_pars)]
    for retry_idx in range(1, int(max_retries)):
        attempts.append(
            _jittered_init_pars(
                base_init=init_pars,
                bounds=par_bounds,
                fixed_params=fixed_params,
                rng=rng,
                scale=min(0.15 * retry_idx, 0.45),
            )
        )

    last_error = None
    last_status = None
    for init_try in attempts:
        try:
            bestfit, result_obj = _run_mle_fit(
                model=model,
                data=data,
                init_pars=init_try,
                par_bounds=par_bounds,
                fixed_params=fixed_params,
            )
            status_payload = _fit_status_dict(result_obj)
            success = status_payload.get("success", True)
            if result_obj is not None and not success:
                last_status = status_payload
                continue
            if not np.all(np.isfinite(bestfit)):
                last_status = status_payload
                continue
            unc = np.full(bestfit.shape, np.nan, dtype=float)
            return bestfit, unc, status_payload, result_obj
        except Exception as exc:
            last_error = str(exc)

    if last_error is None:
        last_error = "fit failed without explicit exception"
    raise RuntimeError(json.dumps({"error": last_error, "status": last_status}))


# ===========================================================================
# Hessian uncertainty estimation
# ===========================================================================

def _objective_twice_nll(model, data, pars):
    value = pyhf.infer.mle.twice_nll(np.asarray(pars, dtype=float), np.asarray(data, dtype=float), model)
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0:
        return float("nan")
    return float(arr[0])


def _numerical_hessian_uncertainties(model, data, bestfit, par_bounds, fixed_params):
    x0 = np.asarray(bestfit, dtype=float).copy()
    n = len(x0)
    fixed = list(fixed_params)

    active = [
        idx
        for idx in range(n)
        if not (idx < len(fixed) and fixed[idx])
    ]
    if not active:
        return np.full(n, np.nan, dtype=float)

    def _step(idx):
        low, high = par_bounds[idx]
        width = float(high) - float(low)
        base = max(1e-4, 1e-2 * width)
        max_sym = min(float(high) - x0[idx], x0[idx] - float(low))
        if not np.isfinite(max_sym) or max_sym <= 0.0:
            return None
        return min(base, 0.5 * max_sym)

    steps = {idx: _step(idx) for idx in active}
    active = [idx for idx in active if steps[idx] is not None]
    if not active:
        return np.full(n, np.nan, dtype=float)

    def f(point):
        try:
            return _objective_twice_nll(model=model, data=data, pars=point)
        except Exception:
            return float("nan")

    H = np.zeros((len(active), len(active)), dtype=float)
    f0 = f(x0)
    if not np.isfinite(f0):
        return np.full(n, np.nan, dtype=float)

    for ia, i in enumerate(active):
        hi = steps[i]
        xp = x0.copy(); xp[i] += hi
        xm = x0.copy(); xm[i] -= hi
        fp = f(xp)
        fm = f(xm)
        if np.isfinite(fp) and np.isfinite(fm):
            H[ia, ia] = (fp - 2.0 * f0 + fm) / (hi * hi)
        else:
            H[ia, ia] = np.nan

        for ja in range(ia + 1, len(active)):
            j = active[ja]
            hj = steps[j]
            xpp = x0.copy(); xpp[i] += hi; xpp[j] += hj
            xpm = x0.copy(); xpm[i] += hi; xpm[j] -= hj
            xmp = x0.copy(); xmp[i] -= hi; xmp[j] += hj
            xmm = x0.copy(); xmm[i] -= hi; xmm[j] -= hj
            fpp = f(xpp); fpm = f(xpm); fmp = f(xmp); fmm = f(xmm)
            if all(np.isfinite(v) for v in (fpp, fpm, fmp, fmm)):
                hij = (fpp - fpm - fmp + fmm) / (4.0 * hi * hj)
            else:
                hij = np.nan
            H[ia, ja] = hij
            H[ja, ia] = hij

    if not np.all(np.isfinite(H)):
        return np.full(n, np.nan, dtype=float)

    eps = 1e-8
    try:
        cov = np.linalg.inv(H + eps * np.eye(H.shape[0], dtype=float))
    except Exception:
        return np.full(n, np.nan, dtype=float)

    diag = np.clip(np.diag(cov), 0.0, None)
    sigma_active = np.sqrt(diag)
    sigma = np.full(n, np.nan, dtype=float)
    for k, idx in enumerate(active):
        sigma[idx] = float(sigma_active[k])
    return sigma


def _uncertainties_from_hessian_matrix(hessian_matrix, fixed_params):
    H = np.asarray(hessian_matrix, dtype=float)
    n = len(fixed_params)
    if H.shape != (n, n):
        return None
    if not np.all(np.isfinite(H)):
        return None

    active = [idx for idx in range(n) if not fixed_params[idx]]
    if not active:
        return np.full(n, np.nan, dtype=float)

    H_active = H[np.ix_(active, active)]
    if not np.all(np.isfinite(H_active)):
        return None

    eps = 1e-8
    try:
        cov = np.linalg.inv(H_active + eps * np.eye(H_active.shape[0], dtype=float))
    except Exception:
        return None

    diag = np.clip(np.diag(cov), 0.0, None)
    sigma_active = np.sqrt(diag)
    sigma = np.full(n, np.nan, dtype=float)
    for k, idx in enumerate(active):
        sigma[idx] = float(sigma_active[k])
    return sigma


def _hessian_from_minuit_result(result_obj, n_pars):
    if result_obj is None:
        return None

    minuit_obj = getattr(result_obj, "minuit", None)
    if minuit_obj is None and hasattr(result_obj, "hessian"):
        minuit_obj = result_obj
    if minuit_obj is None:
        return None

    def _as_matrix(payload):
        try:
            H = np.asarray(payload, dtype=float)
            if H.shape == (n_pars, n_pars) and np.all(np.isfinite(H)):
                return H
        except Exception:
            pass
        try:
            H = np.zeros((n_pars, n_pars), dtype=float)
            for i in range(n_pars):
                for j in range(n_pars):
                    H[i, j] = float(payload[i, j])
            if np.all(np.isfinite(H)):
                return H
        except Exception:
            return None
        return None

    hessian_attr = getattr(minuit_obj, "hessian", None)
    if callable(hessian_attr):
        try:
            H = _as_matrix(hessian_attr())
            if H is not None:
                return H
        except Exception:
            pass
    elif hessian_attr is not None:
        H = _as_matrix(hessian_attr)
        if H is not None:
            return H

    cov = _as_matrix(getattr(minuit_obj, "covariance", None))
    if cov is None:
        cov = _as_matrix(getattr(result_obj, "hess_inv", None))
    if cov is not None:
        eps = 1e-8
        try:
            return np.linalg.inv(cov + eps * np.eye(cov.shape[0], dtype=float))
        except Exception:
            return None

    return None


def _hessian_from_jax(model, data, bestfit):
    try:
        import jax
        import jax.numpy as jnp
    except Exception:
        return None

    data_jax = jnp.asarray(np.asarray(data, dtype=float))
    bestfit_jax = jnp.asarray(np.asarray(bestfit, dtype=float))

    def nll_func(pars):
        return -model.logpdf(pars, data_jax)[0]

    try:
        hessian_func = jax.jit(jax.hessian(nll_func))
        H = hessian_func(bestfit_jax)
        return np.asarray(H, dtype=float)
    except Exception:
        return None


def _estimate_hessian_uncertainties(
    model,
    data,
    bestfit,
    par_bounds,
    fixed_params,
    result_obj,
    backend_name,
    hessian_method,
):
    method = str(hessian_method or "auto").strip().lower()
    backend = str(backend_name or "scipy").strip().lower()

    if method not in {"auto", "manual", "minuit", "jax"}:
        method = "auto"

    methods = []
    if method == "manual":
        methods = ["manual"]
    elif method in {"minuit", "jax"}:
        methods = [method, "manual"]
    else:
        if backend == "minuit":
            methods = ["minuit", "manual"]
        elif backend == "jax":
            methods = ["jax", "manual"]
        else:
            methods = ["manual"]

    n_pars = len(model.config.par_order)
    for m in methods:
        if m == "minuit":
            H = _hessian_from_minuit_result(result_obj, n_pars)
            if H is None:
                continue
            sigma = _uncertainties_from_hessian_matrix(H, fixed_params)
            if sigma is not None and np.any(np.isfinite(sigma)):
                return sigma, "minuit"
            continue

        if m == "jax":
            H = _hessian_from_jax(model=model, data=data, bestfit=bestfit)
            if H is None:
                continue
            sigma = _uncertainties_from_hessian_matrix(H, fixed_params)
            if sigma is not None and np.any(np.isfinite(sigma)):
                return sigma, "jax"
            continue

        if m == "manual":
            sigma = _numerical_hessian_uncertainties(
                model=model,
                data=data,
                bestfit=bestfit,
                par_bounds=par_bounds,
                fixed_params=fixed_params,
            )
            if sigma is not None and np.any(np.isfinite(sigma)):
                return sigma, "manual"

    return None, None


def _estimate_poi_uncertainty(model, data, bestfit, init_pars, par_bounds, fixed_params):
    poi_index = int(model.config.poi_index)
    muhat = float(bestfit[poi_index])

    if poi_index >= len(par_bounds):
        return None
    low, high = par_bounds[poi_index]
    low = float(low)
    high = float(high)
    if not np.isfinite(muhat):
        return None

    span = max(high - low, 1.0)
    step = max(1e-4, 0.02 * span)

    try:
        _, base_twice_nll = pyhf.infer.mle.fixed_poi_fit(
            muhat,
            data,
            model,
            init_pars=init_pars,
            par_bounds=par_bounds,
            fixed_params=fixed_params,
            return_fitted_val=True,
        )
        base_twice_nll = float(np.asarray(base_twice_nll, dtype=float).reshape(-1)[0])
    except Exception:
        return None

    sigma_estimates = []
    for sign in (-1.0, 1.0):
        trial_mu = muhat + sign * step
        if trial_mu <= low or trial_mu >= high:
            continue
        try:
            _, trial_twice_nll = pyhf.infer.mle.fixed_poi_fit(
                float(trial_mu),
                data,
                model,
                init_pars=init_pars,
                par_bounds=par_bounds,
                fixed_params=fixed_params,
                return_fitted_val=True,
            )
        except Exception:
            continue

        trial_twice_nll = float(np.asarray(trial_twice_nll, dtype=float).reshape(-1)[0])
        delta_twice = trial_twice_nll - base_twice_nll
        if np.isfinite(delta_twice) and delta_twice > 0.0:
            sigma_estimates.append(abs(float(trial_mu - muhat)) / np.sqrt(delta_twice))

    if not sigma_estimates:
        return None
    return float(np.mean(sigma_estimates))


# ===========================================================================
# Dataset-plot payload (pyhf-specific channel slicing)
# ===========================================================================

def _channel_dataset_plot_payload(model, data, bestfit, signal_processes, prefit_pars, fit_param_unc, par_bounds):
    n_main = int(model.config.nmaindata)
    main_data = np.asarray(data[:n_main], dtype=float)

    total = np.asarray(model.expected_actualdata(bestfit), dtype=float)[:n_main]
    total_prefit = np.asarray(model.expected_actualdata(prefit_pars), dtype=float)[:n_main]

    bkg_pars = np.asarray(bestfit, dtype=float).copy()
    poi_idx = int(model.config.poi_index)
    bkg_pars[poi_idx] = 0.0
    bkg = np.asarray(model.expected_actualdata(bkg_pars), dtype=float)[:n_main]
    sig = total - bkg

    bkg_prefit_pars = np.asarray(prefit_pars, dtype=float).copy()
    bkg_prefit_pars[poi_idx] = 0.0
    bkg_prefit = np.asarray(model.expected_actualdata(bkg_prefit_pars), dtype=float)[:n_main]
    sig_prefit = total_prefit - bkg_prefit

    bkg_var_up = {}
    bkg_var_down = {}
    total_var_up = {}
    total_var_down = {}
    if fit_param_unc is not None:
        fit_unc = np.asarray(fit_param_unc, dtype=float)
        for idx, par_name in enumerate(model.config.par_order):
            if idx >= len(fit_unc) or not np.isfinite(fit_unc[idx]) or fit_unc[idx] <= 0.0:
                continue
            sigma = float(fit_unc[idx])
            low, high = par_bounds[idx]

            total_up_pars = np.asarray(bestfit, dtype=float).copy()
            total_dn_pars = np.asarray(bestfit, dtype=float).copy()
            total_up_pars[idx] = float(np.clip(total_up_pars[idx] + sigma, float(low), float(high)))
            total_dn_pars[idx] = float(np.clip(total_dn_pars[idx] - sigma, float(low), float(high)))

            total_var_up[par_name] = np.asarray(model.expected_actualdata(total_up_pars), dtype=float)[:n_main].tolist()
            total_var_down[par_name] = np.asarray(model.expected_actualdata(total_dn_pars), dtype=float)[:n_main].tolist()

            if idx == poi_idx:
                continue

            up_pars = np.asarray(bestfit, dtype=float).copy()
            dn_pars = np.asarray(bestfit, dtype=float).copy()
            up_pars[poi_idx] = 0.0
            dn_pars[poi_idx] = 0.0
            up_pars[idx] = float(np.clip(up_pars[idx] + sigma, float(low), float(high)))
            dn_pars[idx] = float(np.clip(dn_pars[idx] - sigma, float(low), float(high)))

            bkg_var_up[par_name] = np.asarray(model.expected_actualdata(up_pars), dtype=float)[:n_main].tolist()
            bkg_var_down[par_name] = np.asarray(model.expected_actualdata(dn_pars), dtype=float)[:n_main].tolist()

    payload = {"channels": {}}
    for channel_name in model.config.channels:
        slc = model.config.channel_slices[channel_name]
        channel_obs = main_data[slc]
        channel_total = total[slc]
        channel_bkg = bkg[slc]
        channel_sig = sig[slc]

        payload["channels"][channel_name] = {
            "obs": channel_obs.tolist(),
            "total": channel_total.tolist(),
            "bkg": channel_bkg.tolist(),
            "sig": channel_sig.tolist(),
            "prefit_total": total_prefit[slc].tolist(),
            "prefit_bkg": bkg_prefit[slc].tolist(),
            "prefit_sig": sig_prefit[slc].tolist(),
            "total_var_up": {k: np.asarray(v, dtype=float)[slc].tolist() for k, v in total_var_up.items()},
            "total_var_down": {k: np.asarray(v, dtype=float)[slc].tolist() for k, v in total_var_down.items()},
            "bkg_var_up": {k: np.asarray(v, dtype=float)[slc].tolist() for k, v in bkg_var_up.items()},
            "bkg_var_down": {k: np.asarray(v, dtype=float)[slc].tolist() for k, v in bkg_var_down.items()},
            "bin_index": list(range(int(len(channel_obs)))),
        }

    payload["signal_processes"] = list(signal_processes or [])
    return payload


# ===========================================================================
# Checkpoint helpers
# ===========================================================================

def _checkpoint_payload(summaries, config):
    return {
        "format": "hfmodel_analysis_checkpoint_v1",
        "config": config,
        "summaries": summaries,
    }


def _write_checkpoint(path, summaries, config):
    if path is None:
        return
    checkpoint = _checkpoint_payload(summaries=summaries, config=config)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(checkpoint, handle, indent=2)


# ===========================================================================
# Main entry point
# ===========================================================================

def run_analysis(
    fit_model,
    toys,
    use_observed_data,
    use_asimov_data,
    cls_alpha,
    signal_strength,
    cls_scan_points,
    cls_smart_scan,
    poi_scan_max,
    feldman_cousins_alpha,
    feldman_cousins_scan_points,
    feldman_cousins_n_toys,
    feldman_cousins_scan_max,
    progress_callback=None,
    checkpoint_freq=None,
    checkpoint_path=None,
    existing_results=None,
    resume_from_index=0,
    compute_nll_scan=False,
    nll_scan_points=121,
    seed=1234,
    backend_name="scipy",
    hessian_method="auto",
):
    """Run the pyhf analysis, delegating backend-agnostic work to the common orchestrator.

    The function:
      1. Applies parameter overrides and builds init_pars / par_bounds / fixed_params.
      2. Constructs a ``PyhfAnalysisState`` and a ``PyhfAnalysisBackend`` adapter.
      3. Delegates to ``run_analysis_common`` for all CLs / FC / NLL-scan / loop logic.
      4. Enriches each summary with the pyhf-specific dataset_plot payload.
    """
    from hfmodel.analysis_backend import PyhfAnalysisState, PyhfAnalysisBackend

    model = fit_model.model
    init_pars, par_bounds, fixed_params = _override_vectors(model, fit_model)

    poi_index = int(model.config.poi_index)
    poi_label = str(model.config.poi_name)

    truth_pars = np.asarray(init_pars, dtype=float).copy()
    if signal_strength is not None:
        truth_pars[poi_index] = float(signal_strength)

    data_mode = _resolve_data_mode(use_observed_data, use_asimov_data)

    # POI bounds for scan-range heuristics
    if poi_index < len(par_bounds):
        poi_bounds = (float(par_bounds[poi_index][0]), float(par_bounds[poi_index][1]))
    else:
        poi_bounds = None

    # ------------------------------------------------------------------
    # Construct state and adapter
    # ------------------------------------------------------------------
    base_rng = np.random.default_rng(int(seed))
    state = PyhfAnalysisState(
        fit_model=fit_model,
        model=model,
        init_pars=list(truth_pars),
        par_bounds=par_bounds,
        fixed_params=fixed_params,
        rng=base_rng,
        backend_name=str(backend_name or "scipy"),
        hessian_method=str(hessian_method or "auto"),
        current_data=None,
        observed_data=(np.asarray(fit_model.data, dtype=float) if fit_model.data is not None else None),
        _signal_nominal_yield=None,
        _poi_is_signal_strength=True,
    )
    backend = PyhfAnalysisBackend.from_state(state)

    # ------------------------------------------------------------------
    # Run common orchestrator
    # ------------------------------------------------------------------
    summaries = run_analysis_common(
        backend=backend,
        state=state,
        toys=int(toys),
        data_mode=data_mode,
        cls_alpha=cls_alpha,
        signal_strength=signal_strength,
        cls_scan_points=cls_scan_points,
        cls_smart_scan=bool(cls_smart_scan),
        poi_scan_max=poi_scan_max,
        poi_bounds=poi_bounds,
        feldman_cousins_alpha=feldman_cousins_alpha,
        feldman_cousins_scan_points=int(feldman_cousins_scan_points),
        feldman_cousins_n_toys=int(feldman_cousins_n_toys),
        feldman_cousins_scan_max=feldman_cousins_scan_max,
        compute_nll_scan=bool(compute_nll_scan),
        nll_scan_points=int(nll_scan_points),
        progress_callback=progress_callback,
        checkpoint_freq=checkpoint_freq,
        checkpoint_path=checkpoint_path,
        existing_results=list(existing_results or []),
        resume_from_index=int(resume_from_index),
        seed=int(seed),
    )

    # ------------------------------------------------------------------
    # Enrich summaries with pyhf-specific dataset_plot payload
    # ------------------------------------------------------------------
    _enrich_summaries_with_plots(
        summaries=summaries,
        state=state,
        model=model,
        init_pars=init_pars,
        par_bounds=par_bounds,
        fit_model=fit_model,
    )

    return summaries


def _enrich_summaries_with_plots(summaries, state, model, init_pars, par_bounds, fit_model):
    """Add channel dataset_plot payloads to each summary using best-fit values."""
    for summary in summaries:
        if "dataset_plot" in summary:
            continue

        # Reconstruct bestfit array from param_values stored in the summary
        par_names = list(model.config.par_order)
        param_vals = summary.get("fit_params", {})
        bestfit = np.asarray(
            [
                param_vals.get(name, init_pars[i])
                for i, name in enumerate(par_names)
            ],
            dtype=float,
        )

        # Reconstruct hessian_unc array from param uncertainties
        param_unc = summary.get("fit_param_unc", {})
        hessian_unc_arr = np.asarray(
            [
                param_unc.get(name, float("nan"))
                for name in par_names
            ],
            dtype=float,
        )

        # We need the data that was used for this summary.  Use observed data
        # for observed fits; for toys we fall back to state.observed_data.
        if summary.get("observed_fit") and state.observed_data is not None:
            data = state.observed_data
        elif state.current_data is not None:
            data = state.current_data
        elif state.observed_data is not None:
            data = state.observed_data
        else:
            continue

        try:
            summary["dataset_plot"] = _channel_dataset_plot_payload(
                model=model,
                data=data,
                bestfit=bestfit,
                signal_processes=getattr(fit_model, "signal_processes", []),
                prefit_pars=init_pars,
                fit_param_unc=hessian_unc_arr,
                par_bounds=par_bounds,
            )
        except Exception:
            pass
