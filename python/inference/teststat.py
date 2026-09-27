"""Profile-likelihood test statistics.

All statistics are 2 * (NLL(conditional) - NLL(unconditional)) with the POI bounded below at
its lower bound (0 by default, i.e. r_hat >= 0 as in Combine):

* ``qtilde``  (Combine "LHC", used for CLs upper limits): one-sided, q = 0 when r_hat > mu.
* ``tmu``     (Combine "PL", used for Feldman-Cousins): two-sided.
* ``q0``      (discovery): q0 = 0 when r_hat <= 0.

``TestStat.evaluate`` returns the value and the fits it used; a failed fit gives ``None`` so
that callers can count and report it rather than using a wrong number.
"""

from dataclasses import dataclass
from typing import Optional

from inference.fitting import FitResult


@dataclass
class TSValue:
    value: Optional[float]
    free: FitResult
    cond: Optional[FitResult]

    @property
    def valid(self) -> bool:
        return self.value is not None


class TestStat:
    kinds = ("qtilde", "tmu", "q0")

    def __init__(self, lik, fitter, kind: str):
        if kind not in self.kinds:
            raise ValueError(f"Unknown test statistic '{kind}'")
        self.lik, self.fitter, self.kind = lik, fitter, kind

    def free_fit(self, data, start=None) -> FitResult:
        return self.fitter.fit(data, start=start)

    def evaluate(self, data, mu: float, free: Optional[FitResult] = None, start=None) -> TSValue:
        lik = self.lik
        if free is None:
            free = self.free_fit(data, start=start)
        if not free.valid:
            return TSValue(None, free, None)
        rhat = free.values[lik.poi_index]
        if self.kind == "qtilde" and rhat > mu:
            return TSValue(0.0, free, None)
        if self.kind == "q0":
            mu = 0.0
            if rhat <= 0.0:
                return TSValue(0.0, free, None)
        cond = self.fitter.fit(data, start=free.values, fixed={lik.poi: mu})
        if not cond.valid:
            return TSValue(None, free, cond)
        q = 2.0 * (cond.nll - free.nll)
        if q < 0:
            if q < -1e-3:
                # the conditional fit found a lower minimum than the free fit: refit the free
                # fit from the conditional values
                refit = self.fitter.fit(data, start=cond.values)
                if refit.valid and refit.nll < free.nll:
                    return self.evaluate(data, mu, free=refit)
            q = 0.0
        return TSValue(q, free, cond)
