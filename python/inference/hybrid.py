"""Toy-based hypothesis tests: CLs limits (HybridNew --LHCmode LHC-limits), Feldman-Cousins
intervals (HybridNew --LHCmode LHC-feldman-cousins) and significance.

Toy construction (both modes, matching Combine's LHC modes: generateNuisances=0,
generateExternalMeasurements=1, fitNuisances=1):

* for each tested r, the nuisances are fitted to the observed data with r fixed (conditional
  MLE theta_hat(r)); s+b toys are generated at (r, theta_hat(r)) with global observables
  drawn around theta_hat(r).  With ``bypass_fit`` the pre-fit nuisances are used instead.
* background-only toys (CLs) are generated at (0, theta_hat(0)).  They do not depend on the
  tested r, so the same datasets (and their free fits) are reused at every r.

p-values use ">=": CLs+b = P(q_sb >= q_obs), CLb = P(q_b >= q_obs), and for FC
p(r) = P(t_toy >= t_obs); r is in the interval when p(r) > 1 - CL.  Every p-value carries a
binomial uncertainty, and failed toy fits are counted and reported, never silently used.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from inference.model import observed_dataset
from inference.teststat import TestStat
from inference.toys import frequentist_values, generate_at


# Test-statistic values closer than this (2*DeltaNLL units) count as equal in the ">="
# comparisons.  Discrete (counting) data give exact ties between toys and the observed data,
# but each q is the difference of two Minuit minima that are only accurate to ~1e-4 in NLL
# (EDM goal 0.002 * tolerance * errordef), so a 1e-9 tolerance split the ties at random and
# biased p-values (e.g. CLs+b for b=3, n=2, r=4: 0.010 instead of the exact 0.0296).
TIE_TOLERANCE = 1e-3


def _binomial(k, n):
    if n == 0:
        return math.nan, math.nan
    p = k / n
    return p, math.sqrt(max(p * (1 - p), 1.0 / n) / n)


@dataclass
class ToyPoint:
    r: float
    q_obs: Optional[float]
    q_sb: List[float] = field(default_factory=list)
    q_b: List[float] = field(default_factory=list)
    failed_sb: int = 0
    failed_b: int = 0
    generation_fit_valid: bool = True

    def pvalues(self, q_ref=None):
        q_ref = self.q_obs if q_ref is None else q_ref
        sb = np.asarray(self.q_sb)
        b = np.asarray(self.q_b)
        clsb, clsb_e = _binomial(int(np.sum(sb >= q_ref - TIE_TOLERANCE)), len(sb))
        clb, clb_e = _binomial(int(np.sum(b >= q_ref - TIE_TOLERANCE)), len(b))
        if not (clb > 0):
            return clsb, clsb_e, clb, clb_e, math.nan, math.nan
        cls = clsb / clb
        cls_e = cls * math.sqrt((clsb_e / clsb) ** 2 + (clb_e / clb) ** 2) if clsb > 0 else clsb_e / clb
        return clsb, clsb_e, clb, clb_e, cls, cls_e

    def to_dict(self):
        clsb, clsb_e, clb, clb_e, cls, cls_e = self.pvalues() if self.q_obs is not None else (math.nan,) * 6
        return {"r": self.r, "q_obs": self.q_obs, "n_sb": len(self.q_sb), "n_b": len(self.q_b),
                "failed_sb": self.failed_sb, "failed_b": self.failed_b, "CLsplusb": clsb, "CLsplusb_err": clsb_e,
                "CLb": clb, "CLb_err": clb_e, "CLs": cls, "CLs_err": cls_e,
                "generation_fit_valid": self.generation_fit_valid}


class _ToyEngine:
    def __init__(self, lik, fitter, kind, rng, bypass_fit=False, data=None):
        self.lik, self.fitter, self.rng = lik, fitter, rng
        self.ts = TestStat(lik, fitter, kind)
        self.bypass = bypass_fit
        self.data = data or observed_dataset(lik.model)
        self.flags: List[str] = []
        self._obs_free = None

    def observed(self, r):
        if self._obs_free is None:
            self._obs_free = self.ts.free_fit(self.data)
            if not self._obs_free.valid:
                raise RuntimeError(f"free fit to the observed data failed: {self._obs_free.status}")
        tsv = self.ts.evaluate(self.data, r, free=self._obs_free)
        if not tsv.valid:
            self.flags.append(f"observed test statistic at r={r:.6g} could not be computed")
        return tsv.value

    def generation_values(self, r):
        values, fit = frequentist_values(self.lik, self.fitter, self.data, r, self.bypass)
        ok = fit is None or fit.valid
        if not ok:
            self.flags.append(f"conditional fit to data at r={r:.6g} failed ({fit.status}); toys use its values")
        return values, ok

    def toy_stats(self, values, n, rs: Sequence[float], label):
        """Generate n toys at ``values`` and evaluate the test statistic at every r in rs.
        Returns ({r: [q...]}, {r: n_failed})."""
        out = {r: [] for r in rs}
        failed = {r: 0 for r in rs}
        for i in range(n):
            toy = generate_at(self.lik, values, self.rng, label=f"{label}_{i}")
            free = self.ts.free_fit(toy, start=values)
            for r in rs:
                tsv = self.ts.evaluate(toy, r, free=free)
                if tsv.valid:
                    out[r].append(tsv.value)
                else:
                    failed[r] += 1
        return out, failed


# ----------------------------------------------------------------------------------------
# CLs limits
# ----------------------------------------------------------------------------------------

@dataclass
class ToyLimitResult:
    observed: Optional[float]
    observed_err: Optional[float]
    expected: Dict[float, Optional[float]]
    points: List[ToyPoint]
    flags: List[str]

    def to_dict(self):
        return {"observed": self.observed, "observed_err": self.observed_err,
                "expected": {str(k): v for k, v in self.expected.items()},
                "points": [p.to_dict() for p in self.points], "flags": self.flags}


def _crossing(points, target, key):
    """Linear interpolation of log(CLs) between the first bracketing pair (Combine's grid readout).
    ``key(point) -> (cls, err)``.  Returns (limit, err, flag)."""
    pts = sorted(points, key=lambda p: p.r)
    vals = [(p.r,) + tuple(key(p)) for p in pts]
    vals = [v for v in vals if math.isfinite(v[1])]
    above = None
    for r, cls, err in vals:
        if cls > target:
            above = (r, cls, err)
        elif above is not None:
            r1, c1, e1 = above
            if cls <= 0:
                # no toy passed at the upper point: log-interpolation is impossible, so
                # interpolate linearly in CLs and flag it (refinement adds points in between)
                lim = r1 + (r - r1) * (c1 - target) / c1
                return lim, abs(r - r1) / 2, "CLs = 0 at the upper bracket point; linear interpolation used, add toys/points"
            lim = r1 + (r - r1) * math.log(target / c1) / math.log(cls / c1)
            # propagate the CLs errors through the interpolation
            slope = (math.log(cls) - math.log(c1)) / (r - r1)
            err_lim = math.hypot(e1 / c1, err / cls) / abs(slope) if slope != 0 else math.inf
            return lim, err_lim, None
    if above is None:
        return None, None, "CLs is below the target at every tested r; extend the grid downwards"
    return None, None, "CLs never falls below the target; extend the grid upwards"


def toy_cls_limit(lik, fitter, rng, grid: Sequence[float], ntoys: int, cl: float = 0.95, bypass_fit=False,
                  expected=True, refine: int = 0, data=None) -> ToyLimitResult:
    """CLs limit from toys on a grid of r values, optionally refined by ``refine`` bisection
    steps around the observed crossing (each adding one point with ``ntoys`` toys)."""
    eng = _ToyEngine(lik, fitter, "qtilde", rng, bypass_fit, data)
    target = 1.0 - cl
    b_values, b_ok = eng.generation_values(0.0)
    b_toys = []  # (toy, free fit) reused at every r
    for i in range(ntoys):
        toy = generate_at(lik, b_values, rng, label=f"b_{i}")
        b_toys.append((toy, eng.ts.free_fit(toy, start=b_values)))

    def make_point(r):
        pt = ToyPoint(r=float(r), q_obs=eng.observed(r))
        values, ok = eng.generation_values(r)
        pt.generation_fit_valid = ok and b_ok
        stats, failed = eng.toy_stats(values, ntoys, [r], "sb")
        pt.q_sb, pt.failed_sb = stats[r], failed[r]
        for toy, free in b_toys:
            tsv = eng.ts.evaluate(toy, r, free=free)
            if tsv.valid:
                pt.q_b.append(tsv.value)
            else:
                pt.failed_b += 1
        return pt

    points = [make_point(r) for r in grid]
    obs_key = lambda p: p.pvalues()[4:6] if p.q_obs is not None else (math.nan, math.nan)
    for _ in range(refine):
        lim, _, flag = _crossing(points, target, obs_key)
        if lim is None:
            break
        rs = sorted(p.r for p in points)
        lo = max([r for r in rs if r <= lim], default=None)
        hi = min([r for r in rs if r > lim], default=None)
        if lo is None or hi is None:
            break
        points.append(make_point(0.5 * (lo + hi)))
    flags = list(eng.flags)
    obs, obs_err, flag = _crossing(points, target, obs_key)
    if flag:
        flags.append(f"observed: {flag}")
    exp = {}
    if expected:
        for quant in (0.025, 0.16, 0.5, 0.84, 0.975):
            def key(p, quant=quant):
                if not p.q_b:
                    return math.nan, math.nan
                q_ref = float(np.quantile(p.q_b, 1.0 - quant))
                return p.pvalues(q_ref)[4:6]
            lim, _, flag = _crossing(points, target, key)
            exp[quant] = lim
            if flag:
                flags.append(f"expected {quant:g}: {flag}")
    for p in points:
        if p.failed_sb or p.failed_b:
            flags.append(f"r={p.r:.4g}: {p.failed_sb} s+b and {p.failed_b} b-only toy fits failed (excluded)")
    return ToyLimitResult(observed=obs, observed_err=obs_err, expected=exp, points=sorted(points, key=lambda p: p.r),
                          flags=list(dict.fromkeys(flags)))


# ----------------------------------------------------------------------------------------
# Feldman-Cousins
# ----------------------------------------------------------------------------------------

@dataclass
class FCPoint:
    r: float
    t_obs: Optional[float]
    t_toys: List[float] = field(default_factory=list)
    failed: int = 0

    def pvalue(self):
        return _binomial(int(np.sum(np.asarray(self.t_toys) >= self.t_obs - TIE_TOLERANCE)), len(self.t_toys))

    def to_dict(self):
        p, e = self.pvalue() if self.t_obs is not None else (math.nan, math.nan)
        return {"r": self.r, "t_obs": self.t_obs, "n_toys": len(self.t_toys), "failed": self.failed,
                "p": p, "p_err": e}


@dataclass
class FCResult:
    lower: Optional[float]
    upper: Optional[float]
    cl: float
    points: List[FCPoint]
    flags: List[str]

    def to_dict(self):
        return {"lower": self.lower, "upper": self.upper, "cl": self.cl,
                "points": [p.to_dict() for p in self.points], "flags": self.flags}


def _fc_interval(points, alpha, r_lo_bound):
    pts = sorted([p for p in points if p.t_obs is not None and p.t_toys], key=lambda p: p.r)
    ps = [(p.r, p.pvalue()[0]) for p in pts]
    acc = [p > alpha for _, p in ps]
    flags = []
    if not any(acc):
        return None, None, ["no tested r is inside the interval; refine the grid around the best fit"]
    first = acc.index(True)
    last = len(acc) - 1 - acc[::-1].index(True)
    if not all(acc[first:last + 1]):
        flags.append("accepted region has holes; reporting its outer envelope")

    def interp(i, j):
        (r1, p1), (r2, p2) = ps[i], ps[j]
        return r1 + (r2 - r1) * (alpha - p1) / (p2 - p1) if p2 != p1 else 0.5 * (r1 + r2)

    if first == 0:
        lower = ps[0][0]
        if ps[0][0] > r_lo_bound:
            flags.append("lowest grid point is accepted; the lower edge may be below the grid")
    else:
        lower = interp(first - 1, first)
    if last == len(ps) - 1:
        upper = None
        flags.append("highest grid point is accepted; extend the grid upwards")
    else:
        upper = interp(last, last + 1)
    return lower, upper, flags


def feldman_cousins(lik, fitter, rng, grid: Sequence[float], ntoys: int, cl: float = 0.90, bypass_fit=False,
                    refine: int = 0, data=None) -> FCResult:
    eng = _ToyEngine(lik, fitter, "tmu", rng, bypass_fit, data)
    alpha = 1.0 - cl

    def make_point(r):
        pt = FCPoint(r=float(r), t_obs=eng.observed(r))
        values, _ok = eng.generation_values(r)
        stats, failed = eng.toy_stats(values, ntoys, [r], "fc")
        pt.t_toys, pt.failed = stats[r], failed[r]
        return pt

    points = [make_point(r) for r in grid]
    for _ in range(refine):
        lower, upper, _ = _fc_interval(points, alpha, fitter.bounds[lik.poi][0])
        rs = sorted(p.r for p in points)
        added = False
        for edge in (lower, upper):
            if edge is None:
                continue
            lo = max([r for r in rs if r < edge], default=None)
            hi = min([r for r in rs if r > edge], default=None)
            if lo is not None and hi is not None:
                points.append(make_point(0.5 * (lo + hi)))
                added = True
        if not added:
            break
    lower, upper, flags = _fc_interval(points, alpha, fitter.bounds[lik.poi][0])
    flags = list(eng.flags) + flags
    for p in points:
        if p.failed:
            flags.append(f"r={p.r:.4g}: {p.failed} toy fits failed (excluded)")
    return FCResult(lower=lower, upper=upper, cl=cl, points=sorted(points, key=lambda p: p.r),
                    flags=list(dict.fromkeys(flags)))


# ----------------------------------------------------------------------------------------
# Significance
# ----------------------------------------------------------------------------------------

def significance(lik, fitter, rng=None, ntoys: int = 0, bypass_fit=False, data=None) -> dict:
    """Discovery significance from q0 (r_hat >= 0).  Asymptotic Z = sqrt(q0); with toys the
    p-value is the fraction of background-only toys with q0 >= q0_obs."""
    from scipy.stats import norm

    eng = _ToyEngine(lik, fitter, "q0", rng or np.random.default_rng(), bypass_fit, data)
    q0 = eng.observed(0.0)
    out = {"q0": q0, "flags": []}
    if q0 is None:
        out["flags"] = eng.flags
        return out
    out["asymptotic"] = {"Z": math.sqrt(q0), "p": float(norm.sf(math.sqrt(q0)))}
    free = eng._obs_free
    if free is not None and lik.poi in free.at_limit and free.values[lik.poi_index] > fitter.bounds[lik.poi][0]:
        out["flags"].append(f"best fit r = {free.values[lik.poi_index]:.4g} is at the upper POI bound, so q0 is "
                            "underestimated; raise --rmax")
    if ntoys > 0:
        values, _ok = eng.generation_values(0.0)
        stats, failed = eng.toy_stats(values, ntoys, [0.0], "b")
        qs = np.asarray(stats[0.0])
        k = int(np.sum(qs >= q0 - TIE_TOLERANCE))
        p, e = _binomial(k, len(qs))
        out["toys"] = {"p": p, "p_err": e, "Z": float(norm.isf(p)) if p > 0 else math.inf, "n_toys": len(qs),
                       "failed": failed[0.0]}
        if k == 0:
            out["flags"].append(f"no toy exceeded q0_obs; p < {1.0 / max(len(qs), 1):.3g}")
    out["flags"] += eng.flags
    return out
