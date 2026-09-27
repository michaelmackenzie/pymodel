"""Maximum-likelihood fits with iminuit, shared by every backend.

``Fitter.fit`` minimises ``Likelihood.nll`` over the floating parameters, with optional fixed
values and bound overrides.  A fit is retried with a higher Minuit strategy and perturbed
starting points before it is declared failed; the returned ``FitResult`` always carries the
status, so no caller can mistake a failed fit for a good one.  Discrete (envelope)
parameters are profiled by fitting every state and keeping the best (penalised) one.
"""

import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from iminuit import Minuit

from modelspec import ir as I


@dataclass
class FitResult:
    values: np.ndarray
    nll: float
    valid: bool
    status: str
    errors: Dict[str, float] = field(default_factory=dict)
    covariance: Optional[np.ndarray] = None
    cov_names: List[str] = field(default_factory=list)
    minos: Dict[str, tuple] = field(default_factory=dict)
    nfcn: int = 0
    at_limit: List[str] = field(default_factory=list)

    def value(self, lik, name) -> float:
        return float(self.values[lik.index[name]])


@dataclass
class FitSettings:
    strategy: int = 1
    tolerance: float = 0.01         # iminuit tol: EDM goal 0.002*tol*errordef (0.1 left flat envelope fits ~0.003 NLL above the minimum)
    max_retries: int = 3
    hesse: bool = False


class Fitter:
    def __init__(self, lik, settings: Optional[FitSettings] = None, rng: Optional[np.random.Generator] = None):
        self.lik = lik
        self.settings = settings or FitSettings()
        self.rng = rng or np.random.default_rng(0)
        self.frozen: set = {p.name for p in lik.parameters if not p.floating and p.role != I.ROLE_DISCRETE}
        self.bounds: Dict[str, tuple] = {p.name: (p.lo, p.hi) for p in lik.parameters}
        self.discretes = [p for p in lik.parameters if p.role == I.ROLE_DISCRETE]

    # ----- user-level controls ----------------------------------------------------------
    def freeze(self, names: Sequence[str]):
        for n in names:
            if n not in self.lik.index:
                raise KeyError(f"Unknown parameter '{n}'")
            self.frozen.add(n)

    def set_range(self, name: str, lo: float, hi: float):
        if name not in self.lik.index:
            raise KeyError(f"Unknown parameter '{name}'")
        self.bounds[name] = (lo, hi)

    # ----- fitting ------------------------------------------------------------------------
    def fit(self, data, start: Optional[np.ndarray] = None, fixed: Optional[Dict[str, float]] = None,
            bounds: Optional[Dict[str, tuple]] = None, hesse: Optional[bool] = None,
            minos: Sequence[str] = ()) -> FitResult:
        start = self.lik.nominal_values() if start is None else np.array(start, dtype=float)
        fixed = dict(fixed or {})
        bnds = dict(self.bounds)
        bnds.update(bounds or {})
        hesse = self.settings.hesse if hesse is None else hesse
        free_discretes = [p for p in self.discretes if p.name not in fixed and p.name not in self.frozen]
        if not free_discretes:
            return self._fit_continuous(data, start, fixed, bnds, hesse, minos)
        best = None
        for states in itertools.product(*[range(p.n_states) for p in free_discretes]):
            fx = dict(fixed)
            fx.update({p.name: float(s) for p, s in zip(free_discretes, states)})
            res = self._fit_continuous(data, start, fx, bnds, hesse, minos)
            if best is None or (res.valid and (not best.valid or res.nll < best.nll)):
                best = res
        return best

    def _fit_continuous(self, data, start, fixed, bnds, hesse, minos) -> FitResult:
        lik = self.lik
        x0 = np.array(start, dtype=float)
        for name, val in fixed.items():
            x0[lik.index[name]] = val
        inactive = set(lik.inactive_parameters(x0))
        free = [i for i, p in enumerate(lik.parameters)
                if p.floating and p.name not in fixed and p.name not in self.frozen and p.name not in inactive]
        for i in free:  # start inside the bounds
            lo, hi = bnds[lik.names[i]]
            x0[i] = min(max(x0[i], lo), hi)

        def total(xfree):
            x = x0.copy()
            x[free] = xfree
            val = lik.nll(x, data)
            return val if math.isfinite(val) else 1e30

        if not free:
            val = lik.nll(x0, data)
            ok = math.isfinite(val)
            return FitResult(values=x0, nll=val, valid=ok, status="no-free-parameters" if ok else "nll-not-finite")

        attempts = []
        strategy = self.settings.strategy
        for attempt in range(self.settings.max_retries + 1):
            xs = x0[free].copy()
            if attempt >= 2:  # perturbed restart
                for k, i in enumerate(free):
                    lo, hi = bnds[lik.names[i]]
                    step = 0.1 * (hi - lo) if math.isfinite(hi - lo) else 0.1 * max(1.0, abs(xs[k]))
                    xs[k] = min(max(xs[k] + self.rng.normal(0.0, step), lo), hi)
            m = Minuit(total, xs, name=[lik.names[i] for i in free])
            m.errordef = Minuit.LIKELIHOOD
            m.strategy = min(2, strategy + (1 if attempt >= 1 else 0))
            m.tol = self.settings.tolerance
            m.print_level = 0
            m.limits = [bnds[lik.names[i]] for i in free]
            try:
                m.migrad()
            except (RuntimeError, ValueError) as exc:
                attempts.append(f"migrad raised {exc}")
                continue
            attempts.append(m)
            if m.valid:
                break
        good = [a for a in attempts if not isinstance(a, str)]
        if not good:
            return FitResult(values=x0, nll=math.inf, valid=False, status="; ".join(attempts))
        m = min(good, key=lambda a: (not a.valid, a.fval))
        x = x0.copy()
        x[free] = np.array(m.values)
        res = FitResult(values=x, nll=float(m.fval), valid=bool(m.valid), nfcn=int(m.nfcn),
                        status="ok" if m.valid else _migrad_problem(m))
        res.at_limit = [lik.names[i] for k, i in enumerate(free) if m.fmin is not None and _at_limit(m, k)]
        if hesse and m.valid:
            try:
                m.hesse()
                res.errors = {lik.names[i]: float(m.errors[k]) for k, i in enumerate(free)}
                res.covariance = np.array(m.covariance)
                res.cov_names = [lik.names[i] for i in free]
                if not m.accurate:
                    res.status = "ok (covariance not accurate)"
            except RuntimeError as exc:
                res.status = f"hesse failed: {exc}"
        for name in minos:
            if name in res.cov_names or name in [lik.names[i] for i in free]:
                try:
                    me = m.merrors[name] if name in m.merrors else m.minos(name).merrors[name]
                    res.minos[name] = (float(me.lower), float(me.upper), bool(me.is_valid))
                except RuntimeError as exc:
                    res.minos[name] = (math.nan, math.nan, False)
                    res.status += f"; minos({name}) failed: {exc}"
        return res


def _migrad_problem(m) -> str:
    fm = m.fmin
    if fm is None:
        return "failed"
    problems = [k for k in ("is_above_max_edm", "has_reached_call_limit", "hesse_failed", "has_covariance")
                if getattr(fm, k, False) and k != "has_covariance"]
    return "invalid minimum" + (f" ({', '.join(problems)})" if problems else "")


def _at_limit(m, k) -> bool:
    lo, hi = m.limits[k]
    v = m.values[k]
    err = m.errors[k]
    return (math.isfinite(lo) and v - lo < 1e-3 * max(err, 1e-9)) or (math.isfinite(hi) and hi - v < 1e-3 * max(err, 1e-9))
