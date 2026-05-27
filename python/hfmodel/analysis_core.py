import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyhf


def configure_runtime():
    return None


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
        par_bounds[index_map[name]] = (float(low), float(high))

    for name in freeze_names:
        fixed_params[index_map[name]] = True

    return init_pars.tolist(), par_bounds, fixed_params


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

    # Select floating parameters with finite bounds for stable finite-difference steps.
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
        # Keep symmetric shifts inside bounds.
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

    # Regularize to avoid singular inversions from numerical noise.
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

    # Some iminuit versions expose covariance/hess_inv but not hessian().
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


def _interpolate_upper_limit(scan_pois, scan_values, alpha):
    pois = np.asarray(scan_pois, dtype=float).reshape(-1)
    vals = np.asarray(scan_values, dtype=float).reshape(-1)

    valid = np.isfinite(vals)
    if not np.any(valid):
        return None

    pois = pois[valid]
    vals = vals[valid]
    if vals.size == 0:
        return None

    order = np.argsort(pois)
    pois = pois[order]
    vals = vals[order]

    above = vals > float(alpha)
    below = vals <= float(alpha)
    if not np.any(above):
        return float(pois[0])
    if not np.any(below):
        return float(pois[-1])

    crossing_index = None
    for idx in range(1, len(vals)):
        if vals[idx - 1] > alpha >= vals[idx]:
            crossing_index = idx
            break
    if crossing_index is None:
        return float(np.interp(alpha, vals[::-1], pois[::-1]))

    x0 = pois[crossing_index - 1]
    x1 = pois[crossing_index]
    y0 = vals[crossing_index - 1]
    y1 = vals[crossing_index]
    if y1 == y0:
        return float(x1)
    frac = (alpha - y0) / (y1 - y0)
    return float(x0 + frac * (x1 - x0))


def _resolve_cls_poi_scan_max(
    model,
    par_bounds,
    requested_poi_scan_max,
    cls_smart_scan=False,
    poi_fit=None,
    poi_unc=None,
):
    poi_index = int(model.config.poi_index)
    default_high = 5.0
    if poi_index < len(par_bounds):
        default_high = float(par_bounds[poi_index][1])

    if requested_poi_scan_max is not None:
        requested = float(requested_poi_scan_max)
        if np.isfinite(default_high):
            requested = min(requested, default_high)
        return float(max(0.0, requested))

    if bool(cls_smart_scan):
        if poi_fit is not None and poi_unc is not None:
            muhat = float(poi_fit)
            sigma = float(poi_unc)
            if np.isfinite(muhat) and np.isfinite(sigma) and sigma > 0.0:
                smart_high = max(0.0, muhat) + 5.0 * sigma
                if np.isfinite(default_high):
                    smart_high = min(smart_high, default_high)
                if smart_high > 0.0:
                    return float(smart_high)

    if np.isfinite(default_high):
        return float(max(0.0, default_high))
    return 5.0


def _compute_cls_summary(model, data, alpha, poi_scan_max, scan_points, init_pars, par_bounds, fixed_params):
    points = int(scan_points) if scan_points is not None else 21
    if points < 3:
        points = 3

    poi_scan_max = float(poi_scan_max)
    poi_scan = np.linspace(0.0, poi_scan_max, points)

    observed_vals = []
    expected_median_vals = []
    expected_band = []

    for poi in poi_scan:
        result = pyhf.infer.hypotest(
            float(poi),
            data,
            model,
            test_stat="qtilde",
            return_expected_set=True,
            init_pars=init_pars,
            par_bounds=par_bounds,
            fixed_params=fixed_params,
        )
        obs = float(np.asarray(result[0], dtype=float).reshape(-1)[0])
        exp = np.asarray(result[1], dtype=float).reshape(-1)

        observed_vals.append(obs)
        expected_median_vals.append(float(exp[2]))
        expected_band.append(exp.tolist())

    obs_limit = _interpolate_upper_limit(poi_scan, observed_vals, alpha)
    exp_limit = _interpolate_upper_limit(poi_scan, expected_median_vals, alpha)

    quantiles = {
        "2.5%": _interpolate_upper_limit(poi_scan, [band[0] for band in expected_band], alpha),
        "16%": _interpolate_upper_limit(poi_scan, [band[1] for band in expected_band], alpha),
        "50%": exp_limit,
        "84%": _interpolate_upper_limit(poi_scan, [band[3] for band in expected_band], alpha),
        "97.5%": _interpolate_upper_limit(poi_scan, [band[4] for band in expected_band], alpha),
    }

    return {
        "cls_observed": obs_limit,
        "cls_expected": exp_limit,
        "cls_expected_quantiles": quantiles,
        "cls_scan_points": points,
        "cls_scan_max": poi_scan_max,
        "cls_alpha" : alpha,
        "cls_curve": {
            "pois": poi_scan.tolist(),
            "observed": [float(x) for x in observed_vals],
            "expected_median": [float(x) for x in expected_median_vals],
            "expected_band": expected_band,
        },
    }


def _fit_twice_nll(model, data, init_pars, par_bounds, fixed_params, rng):
    bestfit, _unc, fit_status, _result_obj = _fit_with_retries(
        model=model,
        data=data,
        init_pars=init_pars,
        par_bounds=par_bounds,
        fixed_params=fixed_params,
        rng=rng,
        max_retries=4,
    )
    if not np.all(np.isfinite(bestfit)):
        return None, None

    nll_hat = None
    try:
        muhat = float(bestfit[int(model.config.poi_index)])
        _, nll_hat_raw = pyhf.infer.mle.fixed_poi_fit(
            muhat,
            data,
            model,
            init_pars=init_pars,
            par_bounds=par_bounds,
            fixed_params=fixed_params,
            return_fitted_val=True,
        )
        nll_hat = float(np.asarray(nll_hat_raw, dtype=float).reshape(-1)[0])
    except Exception:
        fit_fun = fit_status.get("fun") if isinstance(fit_status, dict) else None
        if fit_fun is not None and np.isfinite(float(fit_fun)):
            nll_hat = float(fit_fun)

    if nll_hat is None or not np.isfinite(nll_hat):
        return None, None
    return bestfit, nll_hat


def _profile_q_mu(model, data, mu_test, init_pars, par_bounds, fixed_params, rng):
    _, nll_hat = _fit_twice_nll(
        model=model,
        data=data,
        init_pars=init_pars,
        par_bounds=par_bounds,
        fixed_params=fixed_params,
        rng=rng,
    )
    if nll_hat is None:
        return None

    try:
        _, nll_mu_raw = pyhf.infer.mle.fixed_poi_fit(
            float(mu_test),
            data,
            model,
            init_pars=init_pars,
            par_bounds=par_bounds,
            fixed_params=fixed_params,
            return_fitted_val=True,
        )
        nll_mu = float(np.asarray(nll_mu_raw, dtype=float).reshape(-1)[0])
    except Exception:
        return None

    qmu = nll_mu - nll_hat
    if not np.isfinite(qmu):
        return None
    return float(max(0.0, qmu))


def _compute_feldman_cousins_summary(
    model,
    data,
    alpha,
    init_pars,
    par_bounds,
    fixed_params,
    scan_points,
    n_toys,
    scan_max,
    truth_pars,
    seed,
    dataset_id,
):
    poi_index = int(model.config.poi_index)
    poi_name = str(model.config.poi_name)
    low, high = par_bounds[poi_index]
    low = float(low)
    high = float(high)
    if scan_max is not None:
        high = min(high, float(scan_max))

    n_scan = max(5, int(scan_points))
    poi_grid = np.linspace(low, high, n_scan)
    accepted = []
    q_obs_values = []
    q_crit_values = []
    toy_valid_counts = []

    for imu, mu_test in enumerate(poi_grid):
        obs_rng = np.random.default_rng(int(seed) + 500000 + int(dataset_id) * 1000 + int(imu))
        q_obs = _profile_q_mu(
            model=model,
            data=data,
            mu_test=float(mu_test),
            init_pars=init_pars,
            par_bounds=par_bounds,
            fixed_params=fixed_params,
            rng=obs_rng,
        )
        if q_obs is None:
            q_obs_values.append(None)
            q_crit_values.append(None)
            toy_valid_counts.append(0)
            continue

        toy_q = []
        for itoy in range(int(n_toys)):
            toy_seed = int(seed) + 700000 + int(dataset_id) * 100000 + int(imu) * 1000 + int(itoy)
            toy_rng = np.random.default_rng(toy_seed)
            toy_truth = np.asarray(truth_pars, dtype=float).copy()
            toy_truth[poi_index] = float(mu_test)
            toy_data = _generate_toy_data(model, toy_truth, toy_rng)
            q_toy = _profile_q_mu(
                model=model,
                data=toy_data,
                mu_test=float(mu_test),
                init_pars=init_pars,
                par_bounds=par_bounds,
                fixed_params=fixed_params,
                rng=np.random.default_rng(toy_seed + 1),
            )
            if q_toy is not None and np.isfinite(q_toy):
                toy_q.append(float(q_toy))

        if not toy_q:
            q_obs_values.append(float(q_obs))
            q_crit_values.append(None)
            toy_valid_counts.append(0)
            continue

        qcrit = float(np.percentile(np.asarray(toy_q, dtype=float), 100.0 * (1.0 - float(alpha))))
        accept_mu = float(q_obs) <= qcrit
        if accept_mu:
            accepted.append(float(mu_test))

        q_obs_values.append(float(q_obs))
        q_crit_values.append(qcrit)
        toy_valid_counts.append(len(toy_q))

    interval = None
    if accepted:
        interval = [float(np.min(accepted)), float(np.max(accepted))]

    return {
        "fc_status": "ok" if interval is not None else "no-accepted-points",
        "alpha": float(alpha),
        "poi_name": poi_name,
        "fc_interval": interval,
        "scan_points": int(n_scan),
        "scan_max": float(high),
        "n_toys_per_point": int(n_toys),
        "grid": {
            "poi": [float(x) for x in poi_grid],
            "q_obs": q_obs_values,
            "q_crit": q_crit_values,
            "toy_valid": [int(x) for x in toy_valid_counts],
        },
    }


def _compute_delta_nll_scan(model, data, bestfit, init_pars, par_bounds, fixed_params, poi_scan_max, scan_points):
    poi_index = int(model.config.poi_index)
    poi_name = str(model.config.poi_name)
    low, high = par_bounds[poi_index]
    low = float(low)
    high = float(high)

    muhat = float(bestfit[poi_index]) if np.isfinite(bestfit[poi_index]) else low
    if poi_scan_max is not None:
        upper = min(float(poi_scan_max), high)
    else:
        upper = min(high, max(muhat * 2.0 + 1.0, 1.0))

    if upper <= low:
        upper = high
    n_points = int(scan_points) if scan_points is not None else 121
    n_points = max(7, n_points)

    poi_values = np.linspace(low, upper, n_points)
    twice_nll = []
    for poi in poi_values:
        try:
            _, val = pyhf.infer.mle.fixed_poi_fit(
                float(poi),
                data,
                model,
                init_pars=init_pars,
                par_bounds=par_bounds,
                fixed_params=fixed_params,
                return_fitted_val=True,
            )
            twice_nll.append(float(np.asarray(val, dtype=float).reshape(-1)[0]))
        except Exception:
            twice_nll.append(float("nan"))

    arr = np.asarray(twice_nll, dtype=float)
    valid = np.isfinite(arr)
    if np.any(valid):
        arr = arr - np.nanmin(arr)

    return {
        "poi_name": poi_name,
        "poi_values": poi_values.tolist(),
        "delta_nll": arr.tolist(),
    }


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
    model = fit_model.model
    init_pars, par_bounds, fixed_params = _override_vectors(model, fit_model)

    poi_index = int(model.config.poi_index)
    poi_label = str(model.config.poi_name)

    truth_pars = np.asarray(init_pars, dtype=float).copy()
    if signal_strength is not None:
        truth_pars[poi_index] = float(signal_strength)

    summaries = list(existing_results or [])

    checkpoint_cfg = {
        "mode": "observed" if use_observed_data else ("asimov" if use_asimov_data else "toy"),
        "cls_alpha": cls_alpha,
        "signal_strength": signal_strength,
        "poi_scan_max": poi_scan_max,
    }

    start = int(resume_from_index)
    stop = int(toys)
    for dataset_id in range(start, stop):
        t0 = time.perf_counter()
        toy_rng = np.random.default_rng(int(seed) + int(dataset_id))
        fit_rng = np.random.default_rng(int(seed) + 1000000 + int(dataset_id))

        if use_observed_data:
            data = np.asarray(fit_model.data, dtype=float)
        elif use_asimov_data:
            data = _asimov_data(model, truth_pars)
        else:
            data = _generate_toy_data(model, truth_pars, toy_rng)

        fit_status = {}
        fit_error = None
        hessian_unc = None
        hessian_source = None
        try:
            bestfit, unc, fit_status, fit_result_obj = _fit_with_retries(
                model=model,
                data=data,
                init_pars=init_pars,
                par_bounds=par_bounds,
                fixed_params=fixed_params,
                rng=fit_rng,
                max_retries=4,
            )
            hessian_unc, hessian_source = _estimate_hessian_uncertainties(
                model=model,
                data=data,
                bestfit=bestfit,
                par_bounds=par_bounds,
                fixed_params=fixed_params,
                result_obj=fit_result_obj,
                backend_name=backend_name,
                hessian_method=hessian_method,
            )
        except Exception as exc:
            bestfit = np.full(len(model.config.par_order), np.nan, dtype=float)
            unc = np.full(len(model.config.par_order), np.nan, dtype=float)
            fit_error = str(exc)
        poi_unc = None
        if poi_index < len(unc) and np.isfinite(unc[poi_index]):
            poi_unc = float(unc[poi_index])
        elif fit_error is None:
            poi_unc = _estimate_poi_uncertainty(
                model=model,
                data=data,
                bestfit=bestfit,
                init_pars=init_pars,
                par_bounds=par_bounds,
                fixed_params=fixed_params,
            )
        if poi_unc is None and hessian_unc is not None and poi_index < len(hessian_unc):
            if np.isfinite(hessian_unc[poi_index]):
                poi_unc = float(hessian_unc[poi_index])

        summary = {
            "dataset_id": int(dataset_id),
            "valid": bool(np.all(np.isfinite(bestfit))) and fit_error is None,
            "dataset_time_s": float(time.perf_counter() - t0),
            "poi_name": poi_label,
            "poi_true": float(truth_pars[poi_index]),
            "poi_fit": float(bestfit[poi_index]) if np.isfinite(bestfit[poi_index]) else None,
            "poi_unc_hesse": poi_unc,
            "fit_params": {
                name: (float(bestfit[idx]) if np.isfinite(bestfit[idx]) else None)
                for idx, name in enumerate(model.config.par_order)
            },
            "fit_param_unc": {
                name: (
                    float(hessian_unc[idx])
                    if hessian_unc is not None and idx < len(hessian_unc) and np.isfinite(hessian_unc[idx])
                    else (float(unc[idx]) if idx < len(unc) and np.isfinite(unc[idx]) else None)
                )
                for idx, name in enumerate(model.config.par_order)
            },
            "dataset_plot": _channel_dataset_plot_payload(
                model=model,
                data=data,
                bestfit=bestfit,
                signal_processes=getattr(fit_model, "signal_processes", []),
                prefit_pars=init_pars,
                fit_param_unc=hessian_unc,
                par_bounds=par_bounds,
            ),
        }

        if hessian_source is not None:
            summary["hessian_source"] = hessian_source

        if fit_status:
            summary["fit_status"] = fit_status
        if fit_error is not None:
            summary["fit_error"] = fit_error

        if summary["poi_unc_hesse"] is not None and summary["poi_unc_hesse"] > 0.0:
            summary["poi_pull"] = float((summary["poi_fit"] - summary["poi_true"]) / summary["poi_unc_hesse"])

        if cls_alpha is not None and fit_error is None:
            try:
                cls_poi_scan_max = _resolve_cls_poi_scan_max(
                    model=model,
                    par_bounds=par_bounds,
                    requested_poi_scan_max=poi_scan_max,
                    cls_smart_scan=cls_smart_scan,
                    poi_fit=(bestfit[poi_index] if np.isfinite(bestfit[poi_index]) else None),
                    poi_unc=poi_unc,
                )
                cls_summary = _compute_cls_summary(
                    model=model,
                    data=data,
                    alpha=float(cls_alpha),
                    poi_scan_max=cls_poi_scan_max,
                    scan_points=cls_scan_points,
                    init_pars=init_pars,
                    par_bounds=par_bounds,
                    fixed_params=fixed_params,
                )
                summary.update(cls_summary)
                if summary.get("cls_observed") is not None:
                    summary["yield_upper_limit"] = float(summary["cls_observed"])
            except Exception as exc:
                summary["cls_error"] = str(exc)

        if compute_nll_scan and fit_error is None and int(dataset_id) == int(start):
            try:
                summary["delta_nll_scan"] = _compute_delta_nll_scan(
                    model=model,
                    data=data,
                    bestfit=bestfit,
                    init_pars=init_pars,
                    par_bounds=par_bounds,
                    fixed_params=fixed_params,
                    poi_scan_max=poi_scan_max,
                    scan_points=nll_scan_points,
                )
            except Exception as exc:
                summary["delta_nll_scan_error"] = str(exc)

        if feldman_cousins_alpha is not None and fit_error is None:
            try:
                fc_scan_max_eff = feldman_cousins_scan_max
                if fc_scan_max_eff is None:
                    cls_obs = summary.get("cls_observed")
                    if cls_obs is not None and np.isfinite(float(cls_obs)) and float(cls_obs) > 0.0:
                        fc_scan_max_eff = max(0.25, 2.0 * float(cls_obs))
                    else:
                        fc_scan_max_eff = max(0.5, 2.0 * float(truth_pars[poi_index]))

                summary["feldman_cousins"] = _compute_feldman_cousins_summary(
                    model=model,
                    data=data,
                    alpha=float(feldman_cousins_alpha),
                    init_pars=init_pars,
                    par_bounds=par_bounds,
                    fixed_params=fixed_params,
                    scan_points=feldman_cousins_scan_points,
                    n_toys=feldman_cousins_n_toys,
                    scan_max=fc_scan_max_eff,
                    truth_pars=truth_pars,
                    seed=seed,
                    dataset_id=int(dataset_id),
                )
            except Exception as exc:
                summary["feldman_cousins"] = {
                    "fc_status": "failed",
                    "alpha": float(feldman_cousins_alpha),
                    "error": str(exc),
                }

        if use_observed_data:
            summary["observed_fit"] = True
        if use_asimov_data:
            summary["asimov_fit"] = True

        summaries.append(summary)
        if progress_callback is not None:
            progress_callback(summary, is_observed_fit=use_observed_data)

        if checkpoint_freq is not None and int(checkpoint_freq) > 0:
            done = dataset_id - start + 1
            if done % int(checkpoint_freq) == 0:
                _write_checkpoint(checkpoint_path, summaries, checkpoint_cfg)

    if checkpoint_freq is not None and int(checkpoint_freq) > 0:
        _write_checkpoint(checkpoint_path, summaries, checkpoint_cfg)

    return summaries
