"""Asymptotic CLs upper limits, ported from Combine's AsymptoticLimits.cc (default options:
qtilde, a-posteriori Asimov, newExpected, rAbsAcc 0.0005, rRelAcc 0.005).

Observed limit (runLimit/getCLs):
  1. free fit to data (r >= 0) -> nll_D, r_hat, sigma estimate;
  2. Asimov dataset at r = 0 with the nuisances fitted to data at r = 0 and matching global
     observables (asimovDatasetWithFit); free fit to it -> nll_A;
  3. CLs(r) from q_mu = 2(nll_D(r) - nll_D) and q_A = 2(nll_A(r) - nll_A), both profiled:
         CLs+b = Phi_c(sqrt(q_mu)),  1 - CLb = Phi(sqrt(q_A) - sqrt(q_mu))
     and, for q_mu > q_A (qtilde), Phi_c((q_mu + q_A) / (2 sqrt(q_A))) and
     Phi_c((q_mu - q_A) / (2 sqrt(q_A)));
  4. bracket the CLs = 1 - CL crossing and bisect it with log-interpolation.
Expected limits (runLimitExpected, newExpected): the crossings of the profiled Asimov NLL
with 0.5 * (N + Phi_c^-1(pb * (1 - CL)))^2, N = Phi^-1(pb), for pb in the five quantiles.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from inference.model import Dataset, observed_dataset
from inference.toys import asimov_at, asimov_main, frequentist_values

QUANTILES = (0.025, 0.16, 0.50, 0.84, 0.975)
R_ABS_ACC = 0.0005
R_REL_ACC = 0.005


@dataclass
class AsymptoticResult:
    observed: Optional[float]
    expected: Dict[float, Optional[float]]
    flags: List[str] = field(default_factory=list)
    r_hat: Optional[float] = None
    points: List[dict] = field(default_factory=list)   # (r, CLs, CLs+b, CLb, q_mu, q_A) evaluations
    asimov_fit: Optional[dict] = None

    def to_dict(self) -> dict:
        return {"observed": self.observed, "expected": {str(k): v for k, v in self.expected.items()},
                "flags": self.flags, "r_hat": self.r_hat, "points": self.points, "asimov_fit": self.asimov_fit}


class AsymptoticLimits:
    def __init__(self, lik, fitter, cl: float = 0.95, bypass_frequentist_fit: bool = False):
        self.lik, self.fitter, self.cl = lik, fitter, cl
        self.bypass = bypass_frequentist_fit
        self.flags: List[str] = []

    # ----- helpers --------------------------------------------------------------------
    def _poi_bounds(self, hi):
        lo_b, hi_b = self.fitter.bounds[self.lik.poi]
        return {self.lik.poi: (0.0, max(hi, hi_b))}

    def _profile(self, data, r, start, rmax):
        res = self.fitter.fit(data, start=start, fixed={self.lik.poi: r}, bounds=self._poi_bounds(rmax))
        if not res.valid:
            self.flags.append(f"fit at r={r:.6g} on {data.label} failed: {res.status}")
        return res

    def _asimov(self, observed):
        lik = self.lik
        values, fit = frequentist_values(lik, self.fitter, observed, 0.0, self.bypass)
        info = {"bypassed": self.bypass}
        if fit is not None:
            info.update({"valid": fit.valid, "status": fit.status, "values": lik.values_dict(fit.values)})
            if not fit.valid:
                self.flags.append(f"fit to data at r=0 for the Asimov dataset failed: {fit.status}")
        values = values.copy()
        values[lik.poi_index] = 0.0
        if self.bypass:
            data = Dataset(main=asimov_main(lik, values), global_obs=observed_dataset(lik.model).global_obs,
                           label="asimov")
        else:
            data = asimov_at(lik, values)
        return data, info

    # ----- main ----------------------------------------------------------------------
    def run(self, data: Optional[Dataset] = None, expected: bool = True, observed: bool = True) -> AsymptoticResult:
        lik = self.lik
        data = data or observed_dataset(lik.model)
        self.flags = []
        rmax0 = self.fitter.bounds[lik.poi][1]
        asimov, asimov_info = self._asimov(data)
        free_a = self.fitter.fit(asimov, bounds=self._poi_bounds(rmax0))
        if not free_a.valid:
            self.flags.append(f"free fit to the Asimov dataset failed: {free_a.status}")
        result = AsymptoticResult(observed=None, expected={}, asimov_fit=asimov_info)
        if free_a.values[lik.poi_index] > 1e-3 * rmax0:
            self.flags.append(f"best fit to the Asimov dataset is r = {free_a.values[lik.poi_index]:.4g}, "
                              "not 0")
        if observed:
            result.observed = self._observed(data, asimov, free_a, rmax0, result)
        if expected:
            result.expected = self._expected(asimov, free_a, rmax0)
        result.flags = list(dict.fromkeys(self.flags))
        return result

    def _cls(self, r, data, asimov, free_d, free_a, rmax, points):
        lik = self.lik
        cond_d = self._profile(data, r, free_d.values, rmax)
        qmu = max(0.0, 2.0 * (cond_d.nll - free_d.nll))
        if r < free_d.values[lik.poi_index]:
            qmu = 0.0
        cond_a = self._profile(asimov, r, free_a.values, rmax)
        qa = max(0.0, 2.0 * (cond_a.nll - free_a.nll))
        pmu = norm.sf(math.sqrt(qmu))
        one_m_pb = norm.cdf(math.sqrt(qa) - math.sqrt(qmu))
        if qmu > qa and qa > 0:
            mos = math.sqrt(qa)
            pmu = norm.sf((qmu + qa) / (2 * mos))
            one_m_pb = norm.sf((qmu - qa) / (2 * mos))
        cls = 0.0 if one_m_pb == 0 else pmu / one_m_pb
        points.append({"r": r, "CLs": cls, "CLsplusb": pmu, "CLb": one_m_pb, "q_mu": qmu, "q_A": qa,
                       "valid": bool(cond_d.valid and cond_a.valid)})
        return cls

    def _observed(self, data, asimov, free_a, rmax0, result):
        lik = self.lik
        free_d = self.fitter.fit(data, bounds=self._poi_bounds(rmax0), hesse=True)
        if not free_d.valid:
            self.flags.append(f"free fit to data failed: {free_d.status}")
            return None
        rhat = free_d.values[lik.poi_index]
        result.r_hat = float(rhat)
        if lik.poi in free_d.at_limit and rhat > 0.5 * rmax0:
            self.flags.append(f"best fit r = {rhat:.4g} is at the upper POI bound; raise --rmax")
        r_err = max(free_d.errors.get(lik.poi, 0.0), 0.02 * (rmax0 - 0.0))
        target = 1.0 - self.cl
        rmin = max(0.0, rhat)
        rmax = rmin + 3.0 * r_err
        cls_max, cls_min = 1.0, 0.0
        bracketed = False
        for _ in range(5):
            cls = self._cls(rmax, data, asimov, free_d, free_a, rmax, result.points)
            if cls < target:
                cls_min = cls
                bracketed = True
                break
            rmax *= 2
        if not bracketed:
            self.flags.append(f"observed CLs never fell below {target:g} (last r = {rmax / 2:.4g})")
            return None
        while True:
            if cls_max < 3 * target and cls_min > 0.3 * target:
                r_cross = rmin + (rmax - rmin) * math.log(cls_max / target) / math.log(cls_max / cls_min)
                limit = 0.8 * r_cross + 0.2 * (rmax if (r_cross - rmin) < (rmax - r_cross) else rmin)
            else:
                limit = 0.5 * (rmin + rmax)
            limit_err = 0.5 * (rmax - rmin)
            cls = self._cls(limit, data, asimov, free_d, free_a, rmax, result.points)
            if cls > target:
                cls_max, rmin = cls, limit
            else:
                cls_min, rmax = cls, limit
            if limit_err <= max(R_REL_ACC * limit, R_ABS_ACC):
                return float(limit)

    def _expected(self, asimov, free_a, rmax0):
        lik = self.lik
        nll0 = free_a.nll
        alpha = 1.0 - self.cl
        start = {"x": free_a.values}

        def dnll(r):
            res = self._profile(asimov, r, start["x"], max(rmax0, 1.1 * r))
            if res.valid:
                start["x"] = res.values
            return res.nll - nll0

        out = {}
        hi_guess = max(rmax0, 1e-3)
        for pb in QUANTILES:
            n = norm.ppf(pb)
            level = 0.5 * (n + norm.isf(pb * alpha)) ** 2
            lo, hi = 0.0, None
            r = hi_guess
            for _ in range(40):  # bracket the crossing, growing r as Combine does
                if dnll(r) > level:
                    hi = r
                    break
                lo, r = r, 2.0 * r
            if hi is None:
                self.flags.append(f"expected {pb:g} quantile: no crossing found below r = {r:.4g}")
                out[pb] = None
                continue
            try:
                out[pb] = float(brentq(lambda x: dnll(x) - level, lo, hi, xtol=R_ABS_ACC, rtol=R_REL_ACC / 10))
            except ValueError as exc:
                self.flags.append(f"expected {pb:g} quantile: root finding failed ({exc})")
                out[pb] = None
            hi_guess = out[pb] if out[pb] else hi_guess
        return out
