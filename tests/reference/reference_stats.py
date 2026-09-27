"""Independent numpy/scipy reference statistics for counting and binned-template models.

This module deliberately shares NO code with pymodel (modelspec/, inference/,
stat_backends/): the formulas below were written from the Combine C++ sources
(interface/CombineMathFuncs.h, src/AsymptoticLimits.cc, python/ShapeTools.py at commit
137dbced of the mu2e_dev branch) and from the papers, so that agreement with pymodel is a
real cross-check.

Model (``RefModel``)
--------------------
* parameters: ``r`` (POI, bounded below at 0) followed by the nuisances in ``nuisances``;
* expected yield of process p in bin i::

      nu_pi = nominal_pi * r^[signal] * prod_lnN kappa**theta (or asymPow) * prod gmN alpha*n
              * prod rate parameters * (template morphing for ``shape`` systematics)

  a gmN process ignores its ``nominal`` and uses ``alpha * n`` (Combine semantics);
* NLL = sum_i (nu_i - n_i log nu_i)   (no data-only constants)
        + sum_gauss 0.5 ((theta - g)/sigma)^2  (sigma_lo below g, sigma_hi above: bifurcated)
        + sum_poisson (theta - g log theta + lgamma(g + 1))
  which is the convention of pymodel's ``Likelihood.nll`` so absolute values can be compared.

Methods
-------
* ``profile``: constrained minimisation with scipy (L-BFGS-B, central-difference gradient,
  several starts);
* ``qtilde``; ``asymptotic_limits`` = Combine AsymptoticLimits (observed: a-posteriori
  Asimov at r = 0, q-tilde branch; expected: newExpected crossings of the Asimov profile);
* ``poisson_cls_limit``: exact CLs for one counting bin without systematics;
* ``fc_interval``: Feldman-Cousins by Neyman construction with exact Poisson enumeration.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import gammaln
from scipy.stats import norm, poisson

QUANTILES = (0.025, 0.16, 0.50, 0.84, 0.975)


# ----------------------------------------------------------------------------------------
# Combine math (copied from interface/CombineMathFuncs.h)
# ----------------------------------------------------------------------------------------

def log_kappa_for_x(theta: float, log_kappa_low: float, log_kappa_high: float) -> float:
    """CombineMathFuncs.h logKappaForX."""
    if abs(theta) >= 0.5:
        return log_kappa_high if theta >= 0 else -log_kappa_low
    log_khi = log_kappa_high
    log_klo = -log_kappa_low
    avg = 0.5 * (log_khi + log_klo)
    halfdiff = 0.5 * (log_khi - log_klo)
    twox = theta + theta
    twox2 = twox * twox
    alpha = 0.125 * twox * (twox2 * (3 * twox2 - 10.0) + 15.0)
    return avg + alpha * halfdiff


def asym_pow(theta: float, kappa_low: float, kappa_high: float) -> float:
    """CombineMathFuncs.h asymPow."""
    return math.exp(log_kappa_for_x(theta, math.log(kappa_low), math.log(kappa_high)) * theta)


def smooth_step_func(x: float, smooth_region: float) -> float:
    """CombineMathFuncs.h smoothStepFunc."""
    if abs(x) >= smooth_region:
        return 1.0 if x > 0 else -1.0
    xnorm = x / smooth_region
    xnorm2 = xnorm * xnorm
    return 0.125 * xnorm * (xnorm2 * (3.0 * xnorm2 - 10.0) + 15.0)


def fast_vertical_interp(nominal, ups, downs, coefs, smooth_region):
    """CombineMathFuncs.h fastVerticalInterpHistPdf2 (unit bin widths, as text2workspace
    rebins TH1 templates onto 0..nbins).  All templates are normalised to unit sum first
    (FastVerticalInterpHistPdf2Base::initComponent)."""
    nom = np.asarray(nominal, dtype=float)
    nom = nom / nom.sum()
    out = nom.copy()
    for up, down, x in zip(ups, downs, coefs):
        hi = np.asarray(up, dtype=float) / np.sum(up)
        lo = np.asarray(down, dtype=float) / np.sum(down)
        diff = hi - lo
        summ = hi + lo - 2.0 * nom
        a = 0.5 * x
        b = smooth_step_func(x, smooth_region)
        out = out + a * (diff + b * summ)
    out = np.maximum(out, 1e-9)
    return out / out.sum()


# ----------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------

@dataclass
class RefNuisance:
    name: str
    kind: str = "gauss"        # "gauss", "poisson" (gmN), "free" (rateParam), "flat" (lnU)
    center: float = 0.0        # global observable
    sigma_lo: float = 1.0
    sigma_hi: float = 1.0
    lo: float = -7.0
    hi: float = 7.0
    start: Optional[float] = None


@dataclass
class RefProcess:
    name: str
    nominal: List[float]
    signal: bool = False
    lnN: List[Tuple[str, float, float]] = field(default_factory=list)   # asymmetric (nuis, kappa_down, kappa_up)
    sym_lnN: List[Tuple[str, float]] = field(default_factory=list)       # (nuis, kappa): kappa**theta
    gmN: Optional[Tuple[str, float]] = None                              # (nuis, alpha)
    rate_params: List[str] = field(default_factory=list)
    shapes: List[Tuple[str, List[float], List[float], float]] = field(default_factory=list)  # (nuis, up, down, scale)


@dataclass
class RefChannel:
    name: str
    observed: List[float]
    processes: List[RefProcess]


class RefModel:
    def __init__(self, channels: List[RefChannel], nuisances: List[RefNuisance], r_max: float = 20.0):
        self.channels = channels
        self.nuisances = nuisances
        self.names = ["r"] + [n.name for n in nuisances]
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.r_max = r_max
        self.bounds = [(0.0, r_max)] + [(n.lo, n.hi) for n in nuisances]

    # -- evaluation --------------------------------------------------------------------
    def nominal_point(self) -> np.ndarray:
        x = [1.0]
        for n in self.nuisances:
            x.append(n.start if n.start is not None else n.center)
        return np.array(x, dtype=float)

    def global_obs(self) -> Dict[str, float]:
        return {n.name: n.center for n in self.nuisances if n.kind in ("gauss", "poisson")}

    def process_yield(self, proc: RefProcess, x) -> np.ndarray:
        v = lambda name: x[self.idx[name]]
        if proc.gmN is not None:
            nu = np.full(len(proc.nominal), proc.gmN[1] * v(proc.gmN[0]))
        else:
            nu = np.asarray(proc.nominal, dtype=float).copy()
        f = 1.0
        if proc.signal:
            f *= v("r")
        for name, k in proc.sym_lnN:
            f *= k ** v(name)
        for name, klo, khi in proc.lnN:
            f *= asym_pow(v(name), klo, khi)
        for name in proc.rate_params:
            f *= v(name)
        if proc.shapes:
            nom = np.asarray(proc.nominal, dtype=float)
            total = nom.sum()
            ups, downs, coefs = [], [], []
            smooth = 1.0
            for name, up, down, scale in proc.shapes:
                ups.append(up)
                downs.append(down)
                coefs.append(scale * v(name))
                smooth = min(smooth, scale)
                # ShapeTools.py: normalisation effect from the template integrals
                k_up = float(np.sum(up)) / total
                k_down = float(np.sum(down)) / total
                if not (abs(k_up - 1) < 1e-3 and abs(k_down - 1) < 1e-3):
                    f *= asym_pow(v(name), k_down ** scale, k_up ** scale)
            nu = total * fast_vertical_interp(nom, ups, downs, coefs, smooth)
        return nu * f

    def expected(self, x) -> List[np.ndarray]:
        return [np.sum([self.process_yield(p, x) for p in ch.processes], axis=0) for ch in self.channels]

    def nll(self, x, data: Optional[List[np.ndarray]] = None, gobs: Optional[Dict[str, float]] = None) -> float:
        data = [np.asarray(ch.observed, dtype=float) for ch in self.channels] if data is None else data
        gobs = self.global_obs() if gobs is None else gobs
        total = 0.0
        for mu, n in zip(self.expected(x), data):
            if np.any(mu <= 0):
                if np.any((mu <= 0) & (n > 0)):
                    return math.inf
                mu = np.where(mu <= 0, 1e-300, mu)
            total += float(np.sum(mu - n * np.log(mu)))
        for nuis in self.nuisances:
            t = x[self.idx[nuis.name]]
            if nuis.kind == "gauss":
                g = gobs[nuis.name]
                s = nuis.sigma_lo if t < g else nuis.sigma_hi
                total += 0.5 * ((t - g) / s) ** 2
            elif nuis.kind == "poisson":
                g = gobs[nuis.name]
                if t <= 0:
                    return math.inf
                total += t - g * math.log(t) + gammaln(g + 1.0)
        return total

    # -- fitting ---------------------------------------------------------------------
    def profile(self, data=None, gobs=None, r_fixed: Optional[float] = None, start=None,
                r_bounds: Optional[Tuple[float, float]] = None) -> Tuple[float, np.ndarray]:
        """Minimum NLL (and the point) with r free (within r_bounds, default [0, r_max]) or fixed."""
        x0 = self.nominal_point() if start is None else np.array(start, dtype=float)
        bounds = list(self.bounds)
        if r_bounds is not None:
            bounds[0] = r_bounds
        free = list(range(len(x0)))
        if r_fixed is not None:
            free = free[1:]
            x0[0] = r_fixed
        fb = [bounds[i] for i in free]

        def f(z):
            x = x0.copy()
            x[free] = z
            val = self.nll(x, data, gobs)
            return val if math.isfinite(val) else 1e30

        def grad(z):
            g = np.zeros_like(z)
            for k in range(len(z)):
                h = 1e-6 * max(1.0, abs(z[k]))
                zp, zm = z.copy(), z.copy()
                zp[k] += h
                zm[k] -= h
                lo, hi = fb[k]
                if zm[k] < lo:
                    zm[k] = lo
                if zp[k] > hi:
                    zp[k] = hi
                g[k] = (f(zp) - f(zm)) / (zp[k] - zm[k])
            return g

        if not free:
            return f(np.array([])), x0
        best = None
        starts = [np.clip(x0[free], [b[0] for b in fb], [b[1] for b in fb])]
        rng = np.random.default_rng(12345)
        for _ in range(2):
            starts.append(np.clip(starts[0] + rng.normal(0, 0.3, len(free)), [b[0] for b in fb], [b[1] for b in fb]))
        for z0 in starts:
            res = minimize(f, z0, jac=grad, method="L-BFGS-B", bounds=fb,
                           options={"ftol": 1e-14, "gtol": 1e-9, "maxiter": 5000})
            res = minimize(f, res.x, jac=grad, method="L-BFGS-B", bounds=fb,
                           options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 5000})
            if best is None or res.fun < best.fun:
                best = res
        x = x0.copy()
        x[free] = best.x
        return float(best.fun), x

    def asimov(self, x) -> List[np.ndarray]:
        return self.expected(x)

    def matching_gobs(self, x) -> Dict[str, float]:
        return {n.name: float(x[self.idx[n.name]]) for n in self.nuisances if n.kind in ("gauss", "poisson")}


# ----------------------------------------------------------------------------------------
# Test statistic and asymptotic limits
# ----------------------------------------------------------------------------------------

def qtilde(model: RefModel, r: float, data=None, gobs=None, free=None) -> float:
    """q-tilde: 2 (NLL(r, theta_hat(r)) - NLL(r_hat, theta_hat)), r_hat >= 0, 0 if r_hat > r."""
    nll_free, x_free = free if free is not None else model.profile(data, gobs)
    if x_free[0] > r:
        return 0.0
    nll_r, _ = model.profile(data, gobs, r_fixed=r, start=x_free)
    return max(0.0, 2.0 * (nll_r - nll_free))


@dataclass
class RefAsymptotic:
    observed: Optional[float]
    expected: Dict[float, float]
    r_hat: float


def asymptotic_limits(model: RefModel, cl: float = 0.95, data=None) -> RefAsymptotic:
    """Combine AsymptoticLimits (qtilde, a-posteriori Asimov, newExpected)."""
    alpha = 1.0 - cl
    big = (0.0, 1e4)
    nll_d, x_d = model.profile(data, r_bounds=big)
    rhat = x_d[0]
    # a-posteriori Asimov: nuisances fitted to data at r = 0, global obs matching them
    _, x0 = model.profile(data, r_fixed=0.0)
    x_a = x0.copy()
    x_a[0] = 0.0
    asimov = model.asimov(x_a)
    gobs_a = model.matching_gobs(x_a)
    nll_a, x_af = model.profile(asimov, gobs_a, r_bounds=big, start=x_a)

    def cls(r):
        nll_dr, _ = model.profile(data, r_fixed=r, start=x_d)
        qmu = max(0.0, 2.0 * (nll_dr - nll_d))
        if r < rhat:
            qmu = 0.0
        nll_ar, _ = model.profile(asimov, gobs_a, r_fixed=r, start=x_af)
        qa = max(0.0, 2.0 * (nll_ar - nll_a))
        pmu = norm.sf(math.sqrt(qmu))
        one_m_pb = norm.cdf(math.sqrt(qa) - math.sqrt(qmu))
        if qmu > qa and qa > 0:
            mos = math.sqrt(qa)
            pmu = norm.sf((qmu + qa) / (2 * mos))
            one_m_pb = norm.sf((qmu - qa) / (2 * mos))
        return 0.0 if one_m_pb == 0 else pmu / one_m_pb

    lo = max(rhat, 1e-6)
    hi = max(2.0 * lo, 1.0)
    while cls(hi) > alpha:
        lo, hi = hi, 2.0 * hi
    observed = brentq(lambda r: math.log(max(cls(r), 1e-300)) - math.log(alpha), lo, hi, xtol=1e-6, rtol=1e-8)

    def dnll(r):
        return model.profile(asimov, gobs_a, r_fixed=r, start=x_af)[0] - nll_a

    expected = {}
    for pb in QUANTILES:
        level = 0.5 * (norm.ppf(pb) + norm.isf(pb * alpha)) ** 2
        lo, hi = max(x_af[0], 0.0), max(x_af[0], 0.0) + 1.0
        while dnll(hi) < level:
            lo, hi = hi, 2.0 * hi
        expected[pb] = brentq(lambda r: dnll(r) - level, lo, hi, xtol=1e-6, rtol=1e-8)
    return RefAsymptotic(observed=float(observed), expected={k: float(v) for k, v in expected.items()},
                         r_hat=float(rhat))


# ----------------------------------------------------------------------------------------
# Exact Poisson methods (one counting bin, no systematics)
# ----------------------------------------------------------------------------------------

def poisson_cls(s: float, b: float, n_obs: int) -> float:
    """CLs = P(n <= n_obs | s + b) / P(n <= n_obs | b) (n is the ordering for q-tilde)."""
    return poisson.cdf(n_obs, s + b) / poisson.cdf(n_obs, b)


def poisson_cls_limit(s_nominal: float, b: float, n_obs: int, cl: float = 0.95) -> float:
    """Exact CLs upper limit on r = s / s_nominal."""
    alpha = 1.0 - cl
    hi = 1.0
    while poisson_cls(hi, b, n_obs) > alpha:
        hi *= 2.0
    s_up = brentq(lambda s: poisson_cls(s, b, n_obs) - alpha, 0.0, hi, xtol=1e-10)
    return s_up / s_nominal


def fc_acceptance(mu: float, b: float, cl: float, n_max: Optional[int] = None) -> Tuple[int, int]:
    """Feldman-Cousins acceptance interval [n1, n2] for signal mean mu and known background b:
    rank n by R = P(n | mu + b) / P(n | max(0, n - b) + b) and add n in decreasing R until
    the summed probability reaches cl."""
    if n_max is None:
        n_max = int(mu + b + 20 + 10 * math.sqrt(mu + b + 1))
    n = np.arange(n_max + 1)
    p = poisson.pmf(n, mu + b)
    mu_best = np.maximum(0.0, n - b)
    rank = p / poisson.pmf(n, mu_best + b)
    order = np.argsort(-rank, kind="stable")
    total = 0.0
    accepted = []
    for k in order:
        accepted.append(int(n[k]))
        total += p[k]
        if total >= cl:
            break
    return min(accepted), max(accepted)


def fc_interval(n_obs: int, b: float, cl: float = 0.90, mu_max: float = 50.0, step: float = 0.005) -> Tuple[float, float]:
    """FC interval on the signal mean (FC98 used a mu step of 0.005)."""
    mus = np.arange(0.0, mu_max + step / 2, step)
    inside = []
    for mu in mus:
        n1, n2 = fc_acceptance(mu, b, cl)
        if n1 <= n_obs <= n2:
            inside.append(mu)
    if not inside:
        return math.nan, math.nan
    return float(min(inside)), float(max(inside))


def fc_coverage_counting(mu_true: float, b: float, cl: float, intervals: Dict[int, Tuple[float, float]]) -> float:
    """Exact coverage of a table of intervals {n: (lo, hi)} at mu_true."""
    cov = 0.0
    for n, (lo, hi) in intervals.items():
        if lo <= mu_true <= hi:
            cov += poisson.pmf(n, mu_true + b)
    return cov


def fc_upper_b_monotone(n_obs: int, b: float, cl: float = 0.90, b_max: float = 6.0, b_step: float = 0.005,
                        mu_window: Tuple[float, float] = (0.0, 50.0), step: float = 0.005) -> float:
    """FC98 second pathology fix: mu2(n0, b) is forced to be non-increasing in b by taking the
    largest raw upper edge over b' in [b, b_max] (FC98 scanned b in [0, 25] in steps of 0.001).
    ``mu_window`` restricts the mu scan for speed."""
    best = 0.0
    mus = np.arange(mu_window[0], mu_window[1] + step / 2, step)
    for bb in np.arange(b, b_max + b_step / 2, b_step):
        up = None
        for mu in mus:
            n1, n2 = fc_acceptance(mu, bb, cl)
            if n1 <= n_obs <= n2:
                up = mu
        if up is not None:
            best = max(best, up)
    return float(best)


# FC98 (G. Feldman, R. Cousins, Phys. Rev. D 57 (1998) 3873, arXiv:physics/9711021) Table IV,
# 90% C.L., b = 3.0, n0 = 0..10.  Checked against the arXiv text of the paper.  The table
# includes FC's two pathology compensations (non-connected intervals, and mu2 forced to be
# non-increasing in b by lengthening intervals); the second one changes n0 = 0 from the
# plain Neyman construction value 0.95 to 1.08 (reproduced by fc_upper_b_monotone).
FC98_B3_CL90 = {
    0: (0.00, 1.08), 1: (0.00, 1.88), 2: (0.00, 3.04), 3: (0.00, 4.42), 4: (0.00, 5.60),
    5: (0.00, 6.99), 6: (0.15, 8.47), 7: (0.89, 9.53), 8: (1.51, 10.99), 9: (1.88, 12.30),
    10: (2.63, 13.50),
}
