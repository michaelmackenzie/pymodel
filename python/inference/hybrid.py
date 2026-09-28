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

Reproducibility and parallelism (``inference.parallel``): toy i of a stream is drawn from
SeedSequence(seed, (stream, point, i, ...)), so every toy, and every p-value built from a set
of toy indices, is identical for any number of worker processes and for any split of the toys
or grid points into jobs whose raw results are merged afterwards (``merge_toy_results``).

Adaptive toys (HybridNew --clsAcc / --rAbsAcc / --rRelAcc):

* ``cls_acc`` (CLs) / ``p_acc`` (FC): at every point, toys are added in rounds (each round
  doubles the number of toys, up to ``max_toys``) until the binomial error of CLs (p) is at
  most the target, or CLs (p) is more than 3 errors away from 1 - CL (as HybridNew does; the
  error used is the larger of the estimate and the binomial error CLs (p) would have if it were
  equal to 1 - CL, so that a low-statistics estimate far below the target does not stop early),
  or the cap is reached (flagged).
* ``r_abs_acc`` / ``r_rel_acc`` (CLs): refinement continues until the propagated error of the
  observed limit is below max(r_abs_acc, r_rel_acc * limit): a bisection point is added while
  ``refine`` steps remain and the bracket is wider than the target; after that the toys at the
  two bracketing points are doubled until the target or ``max_toys`` is reached (flagged).

Each point records its number of toys, the toy index ranges it used and why it stopped.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from inference.model import observed_dataset
from inference.parallel import (PURPOSE_FIT_AT, PURPOSE_FREE_FIT, PURPOSE_GENERATE, STREAM_B, STREAM_FC, STREAM_OBS,
                                STREAM_Q0, STREAM_SB, ToySeeds, as_seeds, float_key, seeded_fitter, serial)
from inference.teststat import TestStat
from inference.toys import frequentist_values, generate_at


# Test-statistic values closer than this (2*DeltaNLL units) count as equal in the ">="
# comparisons.  Discrete (counting) data give exact ties between toys and the observed data,
# but each q is the difference of two Minuit minima that are only accurate to ~1e-4 in NLL
# (EDM goal 0.002 * tolerance * errordef), so a 1e-9 tolerance split the ties at random and
# biased p-values (e.g. CLs+b for b=3, n=2, r=4: 0.010 instead of the exact 0.0296).
TIE_TOLERANCE = 1e-3

# observed test statistics of the same point in two merged files must agree to this
# (2*DeltaNLL units; Minuit minima are accurate to ~1e-4 in NLL)
MERGE_Q_TOLERANCE = 2e-3

TOY_RESULTS_FORMAT = "pymodel-toy-results"
TOY_RESULTS_VERSION = 1

STOP_FIXED = "fixed number of toys"
STOP_ACC = "accuracy reached"
STOP_FAR = "more than 3 sigma from 1 - CL"
STOP_CAP = "max toys reached"
STOP_NO_OBS = "observed test statistic failed"


def _binomial(k, n):
    if n == 0:
        return math.nan, math.nan
    p = k / n
    return p, math.sqrt(max(p * (1 - p), 1.0 / n) / n)


# ----------------------------------------------------------------------------------------
# toy index ranges
# ----------------------------------------------------------------------------------------

def _add_range(ranges, seed, start, stop):
    """Append [seed, start, stop) to a list of ranges, joining it to a contiguous last range."""
    if stop <= start:
        return
    if ranges and ranges[-1][0] == seed and ranges[-1][2] == start:
        ranges[-1][2] = stop
    else:
        ranges.append([seed, start, stop])


def _next_index(ranges, seed, first):
    stops = [b for s, _a, b in ranges if s == seed]
    return max(stops) if stops else first


def _n_in(ranges):
    return sum(b - a for _s, a, b in ranges)


def _overlap(r1, r2):
    """Toy index ranges present in both lists (same seed and overlapping indices)."""
    out = []
    for s1, a1, b1 in r1:
        for s2, a2, b2 in r2:
            if s1 == s2 and max(a1, a2) < min(b1, b2):
                out.append([s1, max(a1, a2), min(b1, b2)])
    return out


# ----------------------------------------------------------------------------------------
# the toy task (runs in worker processes)
# ----------------------------------------------------------------------------------------

def toy_task(ctx, task):
    """Generate toys ``start..stop-1`` of ``stream`` at ``values`` and evaluate the test
    statistic ``kind`` at every r in ``rs``.  Returns one row [q or None per r] per toy.

    Toys of streams with ``reuse`` (the b-only toys) and their free fits are cached in the
    worker, so a toy evaluated at a later r is not generated and fitted again (the cached fit
    is exactly the one a regeneration would give)."""
    lik, fitter = ctx.lik, ctx.fitter
    seeds = ToySeeds(task["seed"])
    ts = TestStat(lik, fitter, task["kind"])
    values = np.asarray(task["values"], dtype=float)
    stream = tuple(task["stream"])
    out = []
    for i in range(task["start"], task["stop"]):
        key = ("toy", task["seed"], task["kind"], stream, i, values.tobytes())
        hit = ctx.cache.get(key) if task["reuse"] else None
        if hit is None:
            toy = generate_at(lik, values, seeds.rng(*stream, i, PURPOSE_GENERATE), label=f"toy_{i}")
            with seeded_fitter(fitter, seeds, *stream, i, PURPOSE_FREE_FIT):
                free = ts.free_fit(toy, start=values)
            if task["reuse"]:
                ctx.cache[key] = (toy, free)
        else:
            toy, free = hit
        row = []
        for r in task["rs"]:
            with seeded_fitter(fitter, seeds, *stream, i, PURPOSE_FIT_AT, float_key(r)):
                row.append(ts.evaluate(toy, r, free=free).value)
        out.append(row)
    return out


class _ToyEngine:
    def __init__(self, lik, fitter, kind, seeds: ToySeeds, bypass_fit=False, data=None, executor=None):
        self.lik, self.fitter, self.seeds, self.kind = lik, fitter, seeds, kind
        self.ts = TestStat(lik, fitter, kind)
        self.bypass = bypass_fit
        self.data = data or observed_dataset(lik.model)
        self.executor = executor or serial(lik, fitter)
        self.flags: List[str] = []
        self._obs_free = None
        self._gen: Dict[float, tuple] = {}
        self.n_toys_run = 0

    def observed(self, r):
        if self._obs_free is None:
            with seeded_fitter(self.fitter, self.seeds, STREAM_OBS, PURPOSE_FREE_FIT):
                self._obs_free = self.ts.free_fit(self.data)
            if not self._obs_free.valid:
                raise RuntimeError(f"free fit to the observed data failed: {self._obs_free.status}")
        with seeded_fitter(self.fitter, self.seeds, STREAM_OBS, PURPOSE_FIT_AT, float_key(r)):
            tsv = self.ts.evaluate(self.data, r, free=self._obs_free)
        if not tsv.valid:
            self.flags.append(f"observed test statistic at r={r:.6g} could not be computed")
        return tsv.value

    def generation_values(self, r):
        if r not in self._gen:
            with seeded_fitter(self.fitter, self.seeds, STREAM_OBS, PURPOSE_GENERATE, float_key(r)):
                values, fit = frequentist_values(self.lik, self.fitter, self.data, r, self.bypass)
            ok = fit is None or fit.valid
            if not ok:
                self.flags.append(f"conditional fit to data at r={r:.6g} failed ({fit.status}); toys use its values")
            self._gen[r] = (values, ok)
        return self._gen[r]

    def run(self, specs):
        """specs: dicts with stream, values, start, stop, rs, reuse.  Returns, per spec, the
        rows of ``toy_task`` in toy-index order (split into chunks across the executor)."""
        total = sum(sp["stop"] - sp["start"] for sp in specs)
        size = self.executor.chunk_size(total)
        tasks, owner = [], []
        for k, sp in enumerate(specs):
            for a in range(sp["start"], sp["stop"], size):
                tasks.append({"seed": self.seeds.seed, "kind": self.kind, "stream": list(sp["stream"]),
                              "values": np.asarray(sp["values"], dtype=float).tolist(), "start": a,
                              "stop": min(a + size, sp["stop"]), "rs": [float(r) for r in sp["rs"]],
                              "reuse": sp["reuse"]})
                owner.append(k)
        results = self.executor.map(toy_task, tasks)
        out = [[] for _ in specs]
        for k, rows in zip(owner, results):
            out[k].extend(rows)
        self.n_toys_run += total
        return out


def _need(n_have, max_toys):
    """Next adaptive batch: double the toys, capped at max_toys."""
    return max(0, min(max(n_have, 1), max_toys - n_have))


# ----------------------------------------------------------------------------------------
# CLs limits
# ----------------------------------------------------------------------------------------

@dataclass(eq=False)  # identity hash: points are dict keys
class ToyPoint:
    r: float
    q_obs: Optional[float]
    q_sb: List[float] = field(default_factory=list)
    q_b: List[float] = field(default_factory=list)
    failed_sb: int = 0
    failed_b: int = 0
    generation_fit_valid: bool = True
    sb_toys: List[list] = field(default_factory=list)   # [seed, first, stop) toy index ranges
    b_toys: List[list] = field(default_factory=list)
    stop_reason: str = ""

    @property
    def n_sb(self):
        return len(self.q_sb) + self.failed_sb

    @property
    def n_b(self):
        return len(self.q_b) + self.failed_b

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
                "generation_fit_valid": self.generation_fit_valid, "stop_reason": self.stop_reason,
                "sb_toys": self.sb_toys, "b_toys": self.b_toys}

    def raw(self):
        return {"r": self.r, "q_obs": self.q_obs, "q_sb": list(self.q_sb), "q_b": list(self.q_b),
                "failed_sb": self.failed_sb, "failed_b": self.failed_b,
                "generation_fit_valid": self.generation_fit_valid, "sb_toys": self.sb_toys, "b_toys": self.b_toys,
                "stop_reason": self.stop_reason}

    @staticmethod
    def from_raw(d):
        return ToyPoint(r=float(d["r"]), q_obs=d["q_obs"], q_sb=[float(x) for x in d["q_sb"]],
                        q_b=[float(x) for x in d["q_b"]], failed_sb=int(d["failed_sb"]), failed_b=int(d["failed_b"]),
                        generation_fit_valid=bool(d["generation_fit_valid"]),
                        sb_toys=[list(x) for x in d["sb_toys"]], b_toys=[list(x) for x in d["b_toys"]],
                        stop_reason=d.get("stop_reason", ""))


@dataclass
class ToyLimitResult:
    observed: Optional[float]
    observed_err: Optional[float]
    expected: Dict[float, Optional[float]]
    points: List[ToyPoint]
    flags: List[str]
    cl: float = 0.95
    refinement: dict = field(default_factory=dict)
    n_toys_run: int = 0
    engine_flags: List[str] = field(default_factory=list)   # toy/fit problems (not readout flags)

    def to_dict(self):
        return {"observed": self.observed, "observed_err": self.observed_err, "cl": self.cl,
                "expected": {str(k): v for k, v in self.expected.items()},
                "points": [p.to_dict() for p in self.points], "refinement": self.refinement,
                "toys": {"n_sb": sum(p.n_sb for p in self.points), "n_b_max": max([p.n_b for p in self.points],
                                                                                  default=0),
                         "n_toy_generations_run": self.n_toys_run},
                "flags": self.flags}


def _crossing(points, target, key):
    """Linear interpolation of log(CLs) between the first bracketing pair (Combine's grid readout).
    ``key(point) -> (cls, err)``.  Returns (limit, err, flag)."""
    lim, err, flag, _ = _crossing_bracket(points, target, key)
    return lim, err, flag


def _crossing_bracket(points, target, key):
    """As ``_crossing``, plus the bracketing pair (lower point, upper point) or None."""
    pts = sorted(points, key=lambda p: p.r)
    vals = [(p,) + tuple(key(p)) for p in pts]
    vals = [v for v in vals if math.isfinite(v[1])]
    above = None
    for p, cls, err in vals:
        if cls > target:
            above = (p, cls, err)
        elif above is not None:
            p1, c1, e1 = above
            r1, r = p1.r, p.r
            if cls <= 0:
                # no toy passed at the upper point: log-interpolation is impossible, so
                # interpolate linearly in CLs and flag it (refinement adds points in between)
                lim = r1 + (r - r1) * (c1 - target) / c1
                return lim, abs(r - r1) / 2, "CLs = 0 at the upper bracket point; linear interpolation used, " \
                                             "add toys/points", (p1, p)
            lim = r1 + (r - r1) * math.log(target / c1) / math.log(cls / c1)
            # propagate the CLs errors through the interpolation
            slope = (math.log(cls) - math.log(c1)) / (r - r1)
            err_lim = math.hypot(e1 / c1, err / cls) / abs(slope) if slope != 0 else math.inf
            return lim, err_lim, None, (p1, p)
    if above is None:
        return None, None, "CLs is below the target at every tested r; extend the grid downwards", None
    return None, None, "CLs never falls below the target; extend the grid upwards", None


def _obs_key(p):
    return p.pvalues()[4:6] if p.q_obs is not None else (math.nan, math.nan)


def cls_readout(points: Sequence[ToyPoint], cl: float = 0.95, expected=True, flags=()) -> ToyLimitResult:
    """Observed and expected limits from a set of toy points (also used for merged results)."""
    target = 1.0 - cl
    flags = list(flags)
    obs, obs_err, flag = _crossing(points, target, _obs_key)
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
    for p in sorted(points, key=lambda p: p.r):
        if p.failed_sb or p.failed_b:
            flags.append(f"r={p.r:.4g}: {p.failed_sb} s+b and {p.failed_b} b-only toy fits failed (excluded)")
        if p.stop_reason.startswith(STOP_CAP):
            flags.append(f"r={p.r:.4g}: {p.stop_reason} before the accuracy target ({p.n_sb} s+b toys)")
    return ToyLimitResult(observed=obs, observed_err=obs_err, expected=exp, points=sorted(points, key=lambda p: p.r),
                          flags=list(dict.fromkeys(flags)), cl=cl)


def toy_cls_limit(lik, fitter, rng, grid: Sequence[float], ntoys: int, cl: float = 0.95, bypass_fit=False,
                  expected=True, refine: int = 0, data=None, executor=None, cls_acc: Optional[float] = None,
                  max_toys: Optional[int] = None, r_abs_acc: Optional[float] = None,
                  r_rel_acc: Optional[float] = None, toy_range: Optional[tuple] = None) -> ToyLimitResult:
    """CLs limit from toys on a grid of r values.

    ``refine`` bisection steps are added around the observed crossing (see the module
    docstring for the adaptive options).  ``rng`` is a ToySeeds / int seed / numpy Generator.
    ``toy_range = (first, stop)`` runs only those toy indices at every point, with no
    refinement or adaptive toys (one chunk of a split job; merge the saved raw results)."""
    seeds = as_seeds(rng)
    eng = _ToyEngine(lik, fitter, "qtilde", seeds, bypass_fit, data, executor)
    target = 1.0 - cl
    max_toys = max_toys or ntoys
    if max_toys < ntoys:
        raise ValueError(f"max toys per point ({max_toys}) is below the initial number of toys ({ntoys})")
    partial = toy_range is not None
    if partial and (cls_acc or r_abs_acc or r_rel_acc or refine):
        raise ValueError("a toy chunk (toy_range) runs a fixed number of toys: no adaptive toys or refinement")
    first, stop = toy_range if partial else (0, ntoys)
    r_acc_on = bool(r_abs_acc or r_rel_acc)
    b_values, b_ok = eng.generation_values(0.0)

    def add_toys(requests):
        """requests: {point: (k_sb, k_b)} -> generate and evaluate the extra toys."""
        specs, owners = [], []
        for pt, (ksb, _kb) in requests.items():
            if ksb:
                a = _next_index(pt.sb_toys, seeds.seed, first)
                values, _ok = eng.generation_values(pt.r)
                specs.append({"stream": (STREAM_SB, float_key(pt.r)), "values": values, "start": a, "stop": a + ksb,
                              "rs": [pt.r], "reuse": False})
                owners.append(("sb", [pt], a, a + ksb))
        groups = {}
        for pt, (_ksb, kb) in requests.items():
            if kb:
                a = _next_index(pt.b_toys, seeds.seed, first)
                groups.setdefault((a, a + kb), []).append(pt)
        for (a, b), pts in groups.items():
            specs.append({"stream": (STREAM_B,), "values": b_values, "start": a, "stop": b,
                          "rs": [p.r for p in pts], "reuse": True})
            owners.append(("b", pts, a, b))
        for (kind, pts, a, b), rows in zip(owners, eng.run(specs)):
            for j, pt in enumerate(pts):
                qs = [row[j] for row in rows]
                good = [q for q in qs if q is not None]
                if kind == "sb":
                    pt.q_sb.extend(good)
                    pt.failed_sb += len(qs) - len(good)
                    _add_range(pt.sb_toys, seeds.seed, a, b)
                else:
                    pt.q_b.extend(good)
                    pt.failed_b += len(qs) - len(good)
                    _add_range(pt.b_toys, seeds.seed, a, b)

    def decide(pt):
        """(stop reason, None) or (None, (k_sb, k_b)) for the next adaptive round."""
        if not cls_acc:
            return STOP_FIXED, None
        if pt.q_obs is None:
            return STOP_NO_OBS, None
        _clsb, _e, clb, _eb, cls, cls_e = pt.pvalues()
        if math.isfinite(cls) and cls_e <= cls_acc:
            return f"{STOP_ACC} (CLs error {cls_e:.3g} <= {cls_acc:g})", None
        # "far from the target" uses the larger of the error and the error CLs would have at the
        # target (a small estimate far below the target has an underestimated binomial error)
        e_target = math.sqrt(target * clb * (1 - target * clb) / max(len(pt.q_sb), 1)) / clb if clb > 0 else math.inf
        if math.isfinite(cls) and abs(cls - target) > 3 * max(cls_e, e_target):
            return f"CLs {STOP_FAR}", None
        if pt.n_sb >= max_toys and pt.n_b >= max_toys:
            return f"{STOP_CAP} ({max_toys}): CLs error {cls_e:.3g} > {cls_acc:g}", None
        return None, (_need(pt.n_sb, max_toys), _need(pt.n_b, max_toys))

    def converge(pts):
        """Initial toys for new points, then adaptive rounds until every point has stopped."""
        add_toys({pt: (stop - first, stop - first) for pt in pts})
        active = list(pts)
        while active:
            requests = {}
            for pt in active:
                reason, need = decide(pt)
                if need is None:
                    pt.stop_reason = reason
                else:
                    requests[pt] = need
            if requests:
                add_toys(requests)
            active = list(requests)

    def new_point(r):
        values, ok = eng.generation_values(float(r))
        return ToyPoint(r=float(r), q_obs=eng.observed(float(r)), generation_fit_valid=ok and b_ok)

    points = [new_point(r) for r in grid]
    converge(points)
    refinement = {"steps": 0, "r_accuracy": None, "stop_reason": "no refinement requested"}
    if partial:
        refinement["stop_reason"] = f"toy chunk [{first}, {stop}) only; merge the chunks to get the limit"
    elif refine or r_acc_on:
        for _ in range(1000):
            lim, err, _flag, bracket = _crossing_bracket(points, target, _obs_key)
            if lim is None:
                refinement["stop_reason"] = "no crossing to refine"
                break
            acc = max(r_abs_acc or 0.0, (r_rel_acc or 0.0) * abs(lim)) if r_acc_on else None
            refinement["r_accuracy"] = acc
            if acc is not None and err <= acc:
                refinement["stop_reason"] = f"r accuracy reached (error {err:.3g} <= {acc:.3g})"
                break
            lo, hi = bracket
            if refinement["steps"] < refine and (acc is None or hi.r - lo.r > acc):
                pt = new_point(0.5 * (lo.r + hi.r))
                points.append(pt)
                converge([pt])
                refinement["steps"] += 1
                continue
            if acc is None:
                refinement["stop_reason"] = f"{refine} refinement steps done"
                break
            grow = {p: (_need(p.n_sb, max_toys), _need(p.n_b, max_toys)) for p in (lo, hi)
                    if p.n_sb < max_toys or p.n_b < max_toys}
            if not grow:
                refinement["stop_reason"] = (f"{STOP_CAP} ({max_toys}) at the bracketing points before the r "
                                             f"accuracy (error {err:.3g} > {acc:.3g})")
                eng.flags.append(f"limit accuracy target {acc:.3g} not reached: error {err:.3g} with {max_toys} "
                                 "toys at the bracketing points (raise --max-toys-per-point)")
                break
            add_toys(grow)
            for p in grow:
                p.stop_reason = f"toys added for the r accuracy ({p.n_sb} s+b toys)" + \
                    (f"; {STOP_CAP}" if p.n_sb >= max_toys else "")
    flags = list(eng.flags)
    res = cls_readout(points, cl, expected, flags)
    res.refinement = refinement
    res.n_toys_run = eng.n_toys_run
    res.engine_flags = list(dict.fromkeys(eng.flags))
    return res


# ----------------------------------------------------------------------------------------
# Feldman-Cousins
# ----------------------------------------------------------------------------------------

@dataclass(eq=False)
class FCPoint:
    r: float
    t_obs: Optional[float]
    t_toys: List[float] = field(default_factory=list)
    failed: int = 0
    toys: List[list] = field(default_factory=list)   # [seed, first, stop) toy index ranges
    stop_reason: str = ""

    @property
    def n(self):
        return len(self.t_toys) + self.failed

    def pvalue(self):
        return _binomial(int(np.sum(np.asarray(self.t_toys) >= self.t_obs - TIE_TOLERANCE)), len(self.t_toys))

    def to_dict(self):
        p, e = self.pvalue() if self.t_obs is not None else (math.nan, math.nan)
        return {"r": self.r, "t_obs": self.t_obs, "n_toys": len(self.t_toys), "failed": self.failed,
                "p": p, "p_err": e, "stop_reason": self.stop_reason, "toys": self.toys}

    def raw(self):
        return {"r": self.r, "t_obs": self.t_obs, "t_toys": list(self.t_toys), "failed": self.failed,
                "toys": self.toys, "stop_reason": self.stop_reason}

    @staticmethod
    def from_raw(d):
        return FCPoint(r=float(d["r"]), t_obs=d["t_obs"], t_toys=[float(x) for x in d["t_toys"]],
                       failed=int(d["failed"]), toys=[list(x) for x in d["toys"]], stop_reason=d.get("stop_reason", ""))


@dataclass
class FCResult:
    lower: Optional[float]
    upper: Optional[float]
    cl: float
    points: List[FCPoint]
    flags: List[str]
    n_toys_run: int = 0
    engine_flags: List[str] = field(default_factory=list)

    def to_dict(self):
        return {"lower": self.lower, "upper": self.upper, "cl": self.cl,
                "points": [p.to_dict() for p in self.points],
                "toys": {"n": sum(p.n for p in self.points), "n_toy_generations_run": self.n_toys_run},
                "flags": self.flags}


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


def fc_readout(points: Sequence[FCPoint], cl: float, r_lo_bound: float, flags=()) -> FCResult:
    """Interval from a set of FC points (also used for merged results)."""
    lower, upper, iflags = _fc_interval(points, 1.0 - cl, r_lo_bound)
    flags = list(flags) + iflags
    for p in sorted(points, key=lambda p: p.r):
        if p.failed:
            flags.append(f"r={p.r:.4g}: {p.failed} toy fits failed (excluded)")
        if p.stop_reason.startswith(STOP_CAP):
            flags.append(f"r={p.r:.4g}: {p.stop_reason} before the accuracy target ({p.n} toys)")
    return FCResult(lower=lower, upper=upper, cl=cl, points=sorted(points, key=lambda p: p.r),
                    flags=list(dict.fromkeys(flags)))


def feldman_cousins(lik, fitter, rng, grid: Sequence[float], ntoys: int, cl: float = 0.90, bypass_fit=False,
                    refine: int = 0, data=None, executor=None, p_acc: Optional[float] = None,
                    max_toys: Optional[int] = None, toy_range: Optional[tuple] = None) -> FCResult:
    """Feldman-Cousins interval from toys on a grid (see the module docstring)."""
    seeds = as_seeds(rng)
    eng = _ToyEngine(lik, fitter, "tmu", seeds, bypass_fit, data, executor)
    alpha = 1.0 - cl
    max_toys = max_toys or ntoys
    if max_toys < ntoys:
        raise ValueError(f"max toys per point ({max_toys}) is below the initial number of toys ({ntoys})")
    partial = toy_range is not None
    if partial and (p_acc or refine):
        raise ValueError("a toy chunk (toy_range) runs a fixed number of toys: no adaptive toys or refinement")
    first, stop = toy_range if partial else (0, ntoys)

    def add_toys(requests):
        specs = []
        for pt, k in requests.items():
            a = _next_index(pt.toys, seeds.seed, first)
            values, _ok = eng.generation_values(pt.r)
            specs.append({"stream": (STREAM_FC, float_key(pt.r)), "values": values, "start": a, "stop": a + k,
                          "rs": [pt.r], "reuse": False})
        for (pt, k), sp, rows in zip(requests.items(), specs, eng.run(specs)):
            qs = [row[0] for row in rows]
            good = [q for q in qs if q is not None]
            pt.t_toys.extend(good)
            pt.failed += len(qs) - len(good)
            _add_range(pt.toys, seeds.seed, sp["start"], sp["stop"])

    def decide(pt):
        if not p_acc:
            return STOP_FIXED, None
        if pt.t_obs is None:
            return STOP_NO_OBS, None
        p, e = pt.pvalue()
        if e <= p_acc:
            return f"{STOP_ACC} (p error {e:.3g} <= {p_acc:g})", None
        if abs(p - alpha) > 3 * max(e, math.sqrt(alpha * (1 - alpha) / max(len(pt.t_toys), 1))):
            return f"p {STOP_FAR}", None
        if pt.n >= max_toys:
            return f"{STOP_CAP} ({max_toys}): p error {e:.3g} > {p_acc:g}", None
        return None, _need(pt.n, max_toys)

    def converge(pts):
        add_toys({pt: stop - first for pt in pts})
        active = list(pts)
        while active:
            requests = {}
            for pt in active:
                reason, need = decide(pt)
                if need is None:
                    pt.stop_reason = reason
                else:
                    requests[pt] = need
            if requests:
                add_toys(requests)
            active = list(requests)

    def new_point(r):
        eng.generation_values(float(r))
        return FCPoint(r=float(r), t_obs=eng.observed(float(r)))

    points = [new_point(r) for r in grid]
    converge(points)
    for _ in range(0 if partial else refine):
        lower, upper, _ = _fc_interval(points, alpha, fitter.bounds[lik.poi][0])
        rs = sorted(p.r for p in points)
        added = []
        for edge in (lower, upper):
            if edge is None:
                continue
            lo = max([r for r in rs if r < edge], default=None)
            hi = min([r for r in rs if r > edge], default=None)
            if lo is not None and hi is not None:
                added.append(new_point(0.5 * (lo + hi)))
        if not added:
            break
        points.extend(added)
        converge(added)
    res = fc_readout(points, cl, fitter.bounds[lik.poi][0], eng.flags)
    res.n_toys_run = eng.n_toys_run
    res.engine_flags = list(dict.fromkeys(eng.flags))
    return res


# ----------------------------------------------------------------------------------------
# raw toy results: save, load, merge (HybridNew --saveHybridResult + hadd + --readHybridResults)
# ----------------------------------------------------------------------------------------

def toy_results_doc(kind: str, points, cl: float, seed: int, r_lo_bound: float, meta: dict, flags=()) -> dict:
    """Raw per-point test-statistic arrays of a CLs ("cls") or FC ("fc") run.  ``flags`` are
    the toy/fit problems of the run (``engine_flags``), not the readout flags of a partial run."""
    if kind not in ("cls", "fc"):
        raise ValueError(kind)
    return {"format": TOY_RESULTS_FORMAT, "version": TOY_RESULTS_VERSION, "kind": kind, "cl": cl, "seed": seed,
            "test_statistic": "qtilde" if kind == "cls" else "tmu", "r_lower_bound": r_lo_bound,
            "meta": dict(meta), "flags": list(flags), "points": [p.raw() for p in sorted(points, key=lambda p: p.r)]}


def merge_toy_results(docs: Sequence[dict]):
    """Merge raw toy-result documents.  Points with the same r are combined: their toys are
    concatenated after checking that no toy (seed, index) appears twice and that the observed
    test statistics agree.  Returns (kind, points, cl, r_lower_bound, flags, info)."""
    if not docs:
        raise ValueError("nothing to merge")
    for d in docs:
        if d.get("format") != TOY_RESULTS_FORMAT:
            raise ValueError(f"not a {TOY_RESULTS_FORMAT} document (format {d.get('format')!r})")
        if d.get("version") != TOY_RESULTS_VERSION:
            raise ValueError(f"unsupported toy-results version {d.get('version')}")
    kinds = {d["kind"] for d in docs}
    if len(kinds) != 1:
        raise ValueError(f"cannot merge different kinds of toy results: {sorted(kinds)}")
    kind = kinds.pop()
    flags = []
    for key in ("cl", "r_lower_bound"):
        vals = {d[key] for d in docs}
        if len(vals) != 1:
            raise ValueError(f"the files were made with different {key}: {sorted(vals)}")
    for key in ("input", "backend", "bypass_frequentist_fit"):
        vals = {str(d["meta"].get(key)) for d in docs}
        if len(vals) != 1:
            flags.append(f"merged files differ in {key}: {sorted(vals)}")
    for d in docs:
        flags.extend(d.get("flags", []))
    cls_ = ToyPoint if kind == "cls" else FCPoint
    obs_key = "q_obs" if kind == "cls" else "t_obs"
    merged: Dict[float, object] = {}
    for d in docs:
        for raw in d["points"]:
            pt = cls_.from_raw(raw)
            if pt.r not in merged:
                merged[pt.r] = pt
                continue
            m = merged[pt.r]
            q1, q2 = getattr(m, obs_key), getattr(pt, obs_key)
            if (q1 is None) != (q2 is None) or (q1 is not None and abs(q1 - q2) > MERGE_Q_TOLERANCE):
                raise ValueError(f"r={pt.r:g}: observed test statistics differ between the files ({q1} vs {q2}); "
                                 "they were not made from the same model and data")
            if kind == "cls":
                dup = _overlap(m.sb_toys, pt.sb_toys) + _overlap(m.b_toys, pt.b_toys)
            else:
                dup = _overlap(m.toys, pt.toys)
            if dup:
                raise ValueError(f"r={pt.r:g}: toys {dup} ([seed, first, stop)) are in more than one file; "
                                 "merging them would count identical toys twice (use different --seed or --toy-chunk)")
            if kind == "cls":
                m.q_sb += pt.q_sb
                m.q_b += pt.q_b
                m.failed_sb += pt.failed_sb
                m.failed_b += pt.failed_b
                m.generation_fit_valid = m.generation_fit_valid and pt.generation_fit_valid
                m.sb_toys = sorted(m.sb_toys + pt.sb_toys)
                m.b_toys = sorted(m.b_toys + pt.b_toys)
            else:
                m.t_toys += pt.t_toys
                m.failed += pt.failed
                m.toys = sorted(m.toys + pt.toys)
            m.stop_reason = "merged"
    info = {"n_files": len(docs), "n_points": len(merged)}
    return kind, list(merged.values()), docs[0]["cl"], docs[0]["r_lower_bound"], list(dict.fromkeys(flags)), info


# ----------------------------------------------------------------------------------------
# Significance
# ----------------------------------------------------------------------------------------

def significance(lik, fitter, rng=None, ntoys: int = 0, bypass_fit=False, data=None, executor=None) -> dict:
    """Discovery significance from q0 (r_hat >= 0).  Asymptotic Z = sqrt(q0); with toys the
    p-value is the fraction of background-only toys with q0 >= q0_obs."""
    from scipy.stats import norm

    seeds = as_seeds(rng if rng is not None else np.random.default_rng())
    eng = _ToyEngine(lik, fitter, "q0", seeds, bypass_fit, data, executor)
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
        rows = eng.run([{"stream": (STREAM_Q0,), "values": values, "start": 0, "stop": ntoys, "rs": [0.0],
                         "reuse": False}])[0]
        qs = np.asarray([row[0] for row in rows if row[0] is not None])
        failed = sum(1 for row in rows if row[0] is None)
        k = int(np.sum(qs >= q0 - TIE_TOLERANCE))
        p, e = _binomial(k, len(qs))
        out["toys"] = {"p": p, "p_err": e, "Z": float(norm.isf(p)) if p > 0 else math.inf, "n_toys": len(qs),
                       "failed": failed, "toys": [[seeds.seed, 0, ntoys]]}
        if k == 0:
            out["flags"].append(f"no toy exceeded q0_obs; p < {1.0 / max(len(qs), 1):.3g}")
        if failed:
            out["flags"].append(f"{failed} toy fits failed (excluded)")
    out["flags"] += eng.flags
    return out
