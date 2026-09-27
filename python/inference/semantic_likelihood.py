"""Pure-numpy likelihood that evaluates the ModelIR with the reference formulas of
``modelspec.semantics``.

It is not a user-facing backend.  It exists as the semantic oracle: the tests compare every
backend's ``nll_main`` and expected yields against it, parameter point by parameter point.
It supports counting channels, histogram templates (with shape systematics) and parametric
shapes whose histogram is fixed (``Shape.contents``), on binned or counting data.
"""

import math

import numpy as np

from inference.model import Likelihood, fixed_shape_total_factors
from modelspec import semantics as S


class SemanticLikelihood(Likelihood):
    backend_name = "semantic"
    # "shape:parametric" is listed because every parametric shape carries it; __init__ refuses
    # parametric shapes that have no fixed histogram (floating parameters)
    supported = {"data:count", "data:binned", "shape:counting", "shape:template", "shape:parametric",
                 "shape:parametric-histogram",
                 "norm:lnN", "norm:asym_lnN", "norm:lnU", "norm:gmN", "norm:rate_param", "syst:shape",
                 "constraint:gauss", "constraint:bifurgauss", "constraint:poisson", "constraint:flat"}

    def __init__(self, model):
        super().__init__(model)
        for ch in model.channels:
            for proc in ch.processes:
                if proc.shape.kind == "parametric" and not proc.shape.contents:
                    raise NotImplementedError(f"{ch.name}/{proc.name}: parametric shape with floating parameters")
                if proc.shape.kind == "envelope":
                    raise NotImplementedError("envelopes")
                if any(t.kind in ("formula", "ws_norm") for t in proc.norm_terms):
                    raise NotImplementedError("formula/ws_norm terms")
            if ch.data.kind == "unbinned":
                raise NotImplementedError("unbinned data")

    def _norm(self, proc, v):
        f = 1.0
        for t in proc.norm_terms:
            x = v[self.index[t.param]]
            if t.kind in ("lnN", "lnU"):
                f *= t.kappa_hi ** x
            elif t.kind == "asym_lnN":
                f *= float(S.asym_pow(x, t.kappa_lo, t.kappa_hi))
            elif t.kind == "gmN":
                f *= t.alpha * x
            elif t.kind == "rate_param":
                f *= x
        return f

    def expected_by_process(self, values):
        v = np.asarray(values, dtype=float)
        r = v[self.poi_index]
        out = {}
        for ch in self.model.channels:
            procs = {}
            for proc in ch.processes:
                rate = proc.rate
                if any(t.kind == "gmN" for t in proc.norm_terms):
                    rate = 1.0
                norm = rate * self._norm(proc, v) * (r if proc.is_signal else 1.0)
                shape = proc.shape
                if shape.kind == "counting":
                    procs[proc.name] = np.array([norm])
                elif shape.kind == "template":
                    systs = [(s.up, s.down, s.scale) for s in shape.systs]
                    thetas = [v[self.index[s.param]] for s in shape.systs]
                    procs[proc.name] = S.template_expected(shape.contents, 1.0, systs, thetas) * norm
                else:  # parametric with a fixed histogram: contents are the pdf fractions
                    # (bin-centre density * width, not renormalised, exactly as RooFit/Combine)
                    procs[proc.name] = norm * np.asarray(shape.contents, dtype=float)
            out[ch.name] = procs
        return out

    def nll_main(self, values, native):
        total = 0.0
        by_proc = self.expected_by_process(values)
        facs = fixed_shape_total_factors(self.model)
        for ch in self.model.channels:
            procs = by_proc[ch.name]
            mu = np.sum(list(procs.values()), axis=0)
            nu_tot = sum(facs[ch.name][p] * float(np.sum(y)) for p, y in procs.items())
            n = native[ch.name].counts
            if np.any(mu <= 0):
                if np.any((mu <= 0) & (n > 0)):
                    return math.inf
                mu = np.where(mu <= 0, 1e-300, mu)
            total += nu_tot - float(np.sum(n * np.log(mu)))
        return total
