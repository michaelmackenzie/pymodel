"""Nuisance-parameter impacts on the POI (Combine: ``combineTool.py -M Impacts``).

Procedure (as combineTool --doInitialFit / --doFits, MultiDimFit --algo singles / impact):

1. Initial fit: free fit of all parameters; r_hat and its 68% profile-likelihood interval
   (MultiDimFit --algo singles).  "68%" follows Combine: the crossings of
   2*DeltaNLL = chi2_1 quantile(0.68) = 0.98894 (not 1.0).
2. For every floating nuisance theta (constrained or unconstrained, e.g. rateParams; not the
   POI, frozen or discrete parameters):

   * theta_hat and its 68% interval [theta_lo, theta_hi]:
       ``errors="profile"`` (default, Combine --algo impact): crossings of the profile
       2*DeltaNLL(theta) = 0.98894, each point a fit with theta fixed and everything else
       (including r) floating, located by bracketing and Brent's method;
       ``errors="hesse"``: theta_hat -/+ the Hesse error of the initial fit.
     A crossing that does not exist inside the parameter range is replaced by the range limit
     (as Combine does) and flagged.
   * fits with theta fixed at theta_lo and at theta_hi, all other parameters floating, started
     from the initial best fit: r(theta_lo), r(theta_hi).
   * impact_hi = r(theta_hi) - r_hat, impact_lo = r(theta_lo) - r_hat,
     impact = max(|impact_hi|, |impact_lo|)  (combineTool's impact_r).

3. Pulls, as plotImpacts.py draws them: the pre-fit interval [g - s_lo, g, g + s_hi] is the 68%
   interval of the constraint term alone (Gaussian: g -/+ sigma; bifurcated Gaussian: its two
   widths; Poisson (gmN): the likelihood interval of theta - g log theta).  With
   d = theta - g divided by s_hi when d >= 0 and by s_lo otherwise,
   pull = d(theta_hat), pull_hi = d(theta_hi) - pull, pull_lo = pull - d(theta_lo), and
   constraint = (theta_hi - theta_lo) / (s_hi + s_lo).  Unconstrained parameters have no pull.

The per-nuisance work is independent, so it runs through an ``inference.parallel.Executor``
(one task per nuisance, fitter restarts seeded per parameter: results do not depend on --jobs).
"""

import math
from typing import Optional, Sequence

from scipy.optimize import brentq
from scipy.stats import chi2

from inference.parallel import PURPOSE_FIT_AT, PURPOSE_FREE_FIT, STREAM_IMPACT, ToySeeds, float_key, seeded_fitter, \
    serial, portable
from modelspec import ir as I

# Combine's "68%" level for --algo singles / impact: 2*DeltaNLL = chi2 quantile at CL = 0.68
# (FitterAlgoBase: delta68 = 0.5 * chisquared_quantile_c(1 - 0.68, 1)), not 1.0 (68.27%)
LEVEL_68 = float(chi2.ppf(0.68, 1))  # 0.98894...


class ProfileCurve:
    """2*DeltaNLL of the profile likelihood in one parameter, fits cached by value."""

    def __init__(self, lik, fitter, data, name, free, seeds: ToySeeds, key):
        self.lik, self.fitter, self.data, self.name, self.free = lik, fitter, data, name, free
        self.seeds, self.key = seeds, tuple(key)
        self.fits = {}
        self.failures = []

    def fit_at(self, x):
        x = float(x)
        if x not in self.fits:
            # start from the already-fitted value nearest to x (the best fit to begin with)
            near = min(self.fits, key=lambda v: abs(v - x), default=None)
            start = self.free.values if near is None or not self.fits[near].valid else self.fits[near].values
            with seeded_fitter(self.fitter, self.seeds, *self.key, PURPOSE_FIT_AT, float_key(x)):
                res = self.fitter.fit(self.data, start=start, fixed={self.name: x})
            if not res.valid:
                self.failures.append(f"fit with {self.name} = {x:.6g} failed ({res.status})")
            self.fits[x] = res
        return self.fits[x]

    def __call__(self, x):
        res = self.fit_at(x)
        return 2.0 * (res.nll - self.free.nll) if res.valid else math.nan

    def crossing(self, direction: int, step: float, level: float = LEVEL_68):
        """Value where the curve crosses ``level`` on one side of the best fit.
        Returns (value, at_bound)."""
        x0 = self.free.value(self.lik, self.name)
        lo, hi = self.fitter.bounds[self.name]
        bound = hi if direction > 0 else lo
        step = step if (step and math.isfinite(step) and step > 0) else max(1e-3, 0.1 * abs(x0) or 1.0)
        prev, k = x0, 0
        while True:
            x = x0 + direction * step * (2.0 ** k)
            x = min(x, hi) if direction > 0 else max(x, lo)
            y = self(x)
            if not math.isfinite(y):
                raise RuntimeError(self.failures[-1] if self.failures else f"profile of {self.name} failed")
            if y >= level:
                break
            if x == bound:
                return bound, True
            prev, k = x, k + 1
            if k > 40:
                return bound, True
        f = lambda v: self(v) - level
        a, b = (prev, x) if direction > 0 else (x, prev)
        root = brentq(f, a, b, xtol=1e-5 * step, rtol=1e-10, maxiter=100)
        if self.failures:
            raise RuntimeError(self.failures[-1])
        return float(root), False


def prefit_interval(p: I.Parameter, g: float):
    """68% interval of the constraint term alone: (g - s_lo, g, g + s_hi), or None."""
    c = p.constraint
    if c is None or c.kind == I.CONSTRAINT_FLAT:
        return None
    if c.kind == I.CONSTRAINT_GAUSS:
        return g - c.sigma_hi, g, g + c.sigma_hi
    if c.kind == I.CONSTRAINT_BIFURGAUSS:
        return g - c.sigma_lo, g, g + c.sigma_hi
    if c.kind == I.CONSTRAINT_POISSON:
        if g <= 0:
            return None
        f = lambda t: (t - g * math.log(t)) - (g - g * math.log(g)) - 0.5
        lo = brentq(f, 1e-12 * max(g, 1.0), g)
        hi = brentq(f, g, g + 20.0 * math.sqrt(g) + 20.0)
        return lo, g, hi
    raise ValueError(c.kind)


def constraint_type(p: I.Parameter) -> str:
    c = p.constraint
    if c is None or c.kind == I.CONSTRAINT_FLAT:
        return "Unconstrained"
    return {I.CONSTRAINT_GAUSS: "Gaussian", I.CONSTRAINT_BIFURGAUSS: "AsymmetricGaussian",
            I.CONSTRAINT_POISSON: "Poisson"}[c.kind]


def _scaled(x, pre):
    lo, g, hi = pre
    d = x - g
    return d / (hi - g) if d >= 0 else d / (g - lo)


def impact_task(ctx, task):
    """Impact of one nuisance (runs in a worker)."""
    lik, fitter = ctx.lik, ctx.fitter
    data, free, name = task["data"], task["free"], task["name"]
    seeds = ToySeeds(task["seed"])
    key = (STREAM_IMPACT, task["index"])
    p = lik.model.parameters[name]
    theta = free.value(lik, name)
    out = {"name": name, "type": constraint_type(p), "flags": []}
    curve = ProfileCurve(lik, fitter, data, name, free, seeds, key)
    hesse = free.errors.get(name)
    lo_b, hi_b = fitter.bounds[name]
    try:
        if task["errors"] == "hesse":
            if hesse is None:
                raise RuntimeError(f"no Hesse error for {name} in the initial fit")
            lo, hi = max(theta - hesse, lo_b), min(theta + hesse, hi_b)
            at = [lo == lo_b, hi == hi_b]
        else:
            (lo, at_lo), (hi, at_hi) = curve.crossing(-1, hesse), curve.crossing(+1, hesse)
            at = [at_lo, at_hi]
        for side, hit in zip(("lower", "upper"), at):
            if hit:
                out["flags"].append(f"{name}: no {side} 68% crossing inside its range; the range limit is used "
                                    "(as Combine does)")
        r_vals = []
        for x in (lo, hi):
            with seeded_fitter(fitter, seeds, *key, PURPOSE_FREE_FIT, float_key(x)):
                res = fitter.fit(data, start=free.values, fixed={name: x})
            if not res.valid:
                raise RuntimeError(f"fit with {name} fixed at {x:.6g} failed ({res.status})")
            r_vals.append(res.value(lik, lik.poi))
    except (RuntimeError, ValueError) as exc:
        out.update({"valid": False, "error": str(exc), "fit": [None, theta, None], "r": None, "impact": None})
        out["flags"].append(f"{name}: impact not computed: {exc}")
        return out
    r_hat = free.value(lik, lik.poi)
    out.update({"valid": True, "fit": [lo, theta, hi], "hesse_error": hesse, "r": [r_vals[0], r_hat, r_vals[1]],
                "impact_hi": r_vals[1] - r_hat, "impact_lo": r_vals[0] - r_hat,
                "impact": max(abs(r_vals[1] - r_hat), abs(r_vals[0] - r_hat)), "n_profile_fits": len(curve.fits)})
    g = data.global_obs.get(name) if p.constraint is not None and p.constraint.has_global_observable else None
    pre = prefit_interval(p, g) if g is not None else None
    if pre is not None:
        pull = _scaled(theta, pre)
        out.update({"prefit": list(pre), "pull": pull, "pull_hi": _scaled(hi, pre) - pull,
                    "pull_lo": pull - _scaled(lo, pre), "constraint": (hi - lo) / (pre[2] - pre[0])})
    else:
        out["prefit"] = [p.value, p.value, p.value]
    return out


def impact_parameters(lik, fitter, names: Optional[Sequence[str]] = None, exclude: Sequence[str] = ()):
    """Floating non-POI, non-discrete parameters (optionally restricted / excluded)."""
    cand = [p.name for p in lik.parameters if p.floating and p.name != lik.poi and p.role != I.ROLE_DISCRETE
            and p.name not in fitter.frozen]
    if names:
        unknown = [n for n in names if n not in cand]
        if unknown:
            raise ValueError(f"not floating nuisance parameters: {unknown}")
        cand = [n for n in cand if n in names]
    return [n for n in cand if n not in set(exclude)]


def impacts(lik, fitter, data, seed: int = 0, names=None, exclude=(), errors: str = "profile",
            executor=None) -> dict:
    """Initial fit, POI interval and per-nuisance impacts (see the module docstring)."""
    if errors not in ("profile", "hesse"):
        raise ValueError(f"errors must be 'profile' or 'hesse', not {errors!r}")
    seeds = ToySeeds(seed)
    executor = executor or serial(lik, fitter)
    flags = []
    with seeded_fitter(fitter, seeds, STREAM_IMPACT, 0, PURPOSE_FREE_FIT):
        free = fitter.fit(data, hesse=True)
    if not free.valid:
        raise RuntimeError(f"initial fit failed: {free.status}")
    r_hat = free.value(lik, lik.poi)
    curve = ProfileCurve(lik, fitter, data, lik.poi, free, seeds, (STREAM_IMPACT, 0))
    (r_lo, at_lo), (r_hi, at_hi) = curve.crossing(-1, free.errors.get(lik.poi)), \
        curve.crossing(+1, free.errors.get(lik.poi))
    for side, hit in (("lower", at_lo), ("upper", at_hi)):
        if hit:
            flags.append(f"{lik.poi}: no {side} 68% crossing inside [{fitter.bounds[lik.poi][0]:g}, "
                         f"{fitter.bounds[lik.poi][1]:g}]; the bound is used (widen --rmin/--rmax)")
    if lik.poi in free.at_limit:
        flags.append(f"{lik.poi} is at a bound in the initial fit ({r_hat:.4g}); impacts are truncated there "
                     "(lower --rmin, as for Combine's --rMin)")
    names = impact_parameters(lik, fitter, names, exclude)
    pdata = portable(data)
    tasks = [{"data": pdata, "free": free, "name": n, "index": 1 + lik.index[n], "errors": errors, "seed": seed}
             for n in names]
    params = executor.map(impact_task, tasks)
    for p in params:
        flags.extend(p.pop("flags"))
    params.sort(key=lambda p: -(p["impact"] if p["impact"] is not None else math.inf))
    return {"poi": lik.poi, "method": errors,
            "initial_fit": {"valid": free.valid, "status": free.status, "nll": free.nll,
                            "values": lik.values_dict(free.values)},
            "POIs": [{"name": lik.poi, "fit": [r_lo, r_hat, r_hi], "hesse_error": free.errors.get(lik.poi)}],
            "params": params, "flags": list(dict.fromkeys(flags))}
