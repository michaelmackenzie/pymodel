"""Pure-numpy likelihood that evaluates the ModelIR with the reference formulas of
``modelspec.semantics``.

It is not a user-facing backend.  It exists as the semantic oracle: the tests compare every
backend's ``nll_main`` and expected yields against it, parameter point by parameter point.
It supports counting channels, histogram templates (with shape systematics) and parametric
shapes whose histogram is fixed (``Shape.contents``, also with shape systematics:
``semantics.vertical_pdf_fractions`` / ``hist_pdf_fractions``), on binned or counting data, and
autoMCStats channels (``semantics.cmshist_template`` + ``semantics.bb_lite_expected``).
Multi-dimensional channels are evaluated on their flattened bins (ir.Observable).
"""

import math

import numpy as np

from inference.model import Likelihood, fixed_shape_total_factors
from modelspec import ir as I
from modelspec import semantics as S


class SemanticLikelihood(Likelihood):
    backend_name = "semantic"
    # "shape:parametric" is listed because every parametric shape carries it; __init__ refuses
    # parametric shapes that have no fixed histogram (floating parameters)
    supported = {"data:count", "data:binned", "shape:counting", "shape:template", "shape:parametric",
                 "shape:parametric-histogram",
                 "norm:lnN", "norm:asym_lnN", "norm:lnU", "norm:gmN", "norm:rate_param", "syst:shape",
                 "constraint:gauss", "constraint:bifurgauss", "constraint:poisson", "constraint:flat",
                 "mcstats:bb-lite",
                 # shape systematics on fixed pdfs (semantics.vertical_pdf_fractions / hist_pdf_fractions)
                 "syst:pdf-morph", "syst:histpdf-morph",
                 # N-D channels: the same formulas on the flattened bins (bin volumes as widths)
                 "obs:multidim"}

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

    def _mcstats_channel(self, ch, v):
        """(per-process yields, floored total) of an autoMCStats channel."""
        r = v[self.poi_index]
        coeffs, hists, errors = {}, {}, {}
        for proc in ch.processes:
            shape = proc.shape
            integral = float(np.sum(shape.contents))
            if proc.rate == 0.0 or integral <= 0.0:  # dropped by Combine / empty template: no yield, no error
                coeffs[proc.name] = 0.0
                hists[proc.name] = np.zeros(len(shape.contents))
                errors[proc.name] = np.zeros(len(shape.contents))
                continue
            h, snorm = S.cmshist_template(shape.contents, [(s.up, s.down, s.scale) for s in shape.systs],
                                          [v[self.index[s.param]] for s in shape.systs])
            rate = 1.0 if any(t.kind == "gmN" for t in proc.norm_terms) else proc.rate
            coeffs[proc.name] = rate / integral * self._norm(proc, v) * (r if proc.is_signal else 1.0) * snorm
            hists[proc.name] = h
            errors[proc.name] = np.sqrt(np.asarray(shape.sumw2, dtype=float))
        return S.bb_lite_expected(coeffs, hists, errors, ch.mcstats.params,
                                  {bp.param: v[self.index[bp.param]] for bp in ch.mcstats.params})

    def expected_by_process(self, values):
        v = np.asarray(values, dtype=float)
        r = v[self.poi_index]
        out = {}
        for ch in self.model.channels:
            if ch.mcstats is not None:
                out[ch.name] = self._mcstats_channel(ch, v)[0]
                continue
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
                elif shape.pdf_systs:
                    procs[proc.name] = norm * self._pdf_morph_fractions(ch, shape, v)
                else:  # parametric with a fixed histogram: contents are the pdf fractions
                    # (bin-centre density * width, not renormalised, exactly as RooFit/Combine)
                    procs[proc.name] = norm * np.asarray(shape.contents, dtype=float)
            out[ch.name] = procs
        return out

    def _pdf_morph_fractions(self, ch, shape, v):
        """Bin fractions of a fixed pdf with shape systematics (Combine getPdf)."""
        thetas = [v[self.index[s.param]] for s in shape.pdf_systs]
        widths = ch.observable.bin_volumes()
        if shape.pdf_morph == I.PDF_MORPH_HIST:
            return S.hist_pdf_fractions(shape.contents, [(s.up_contents, s.down_contents, s.scale)
                                                         for s in shape.pdf_systs], thetas, widths)
        return S.vertical_pdf_fractions(shape.contents, shape.raw_integral,
                                        [(s.up_contents, s.down_contents, s.up_integral, s.down_integral, s.scale)
                                         for s in shape.pdf_systs], thetas, widths, ch.bin_integration)

    def nll_main(self, values, native):
        total = 0.0
        by_proc = self.expected_by_process(values)
        facs = fixed_shape_total_factors(self.model)
        v = np.asarray(values, dtype=float)
        for ch in self.model.channels:
            procs = by_proc[ch.name]
            for proc in ch.processes:  # morphed pdfs: the extended term is the yield
                if proc.shape.pdf_systs:
                    facs[ch.name][proc.name] = 1.0 / float(np.sum(self._pdf_morph_fractions(ch, proc.shape, v)))
            n = native[ch.name].counts
            if ch.mcstats is not None:
                mu = self._mcstats_channel(ch, v)[1]  # floored at 1e-9 per bin: never <= 0
                total += float(np.sum(mu)) - float(np.sum(n * np.log(mu)))
                continue
            mu = np.sum(list(procs.values()), axis=0)
            nu_tot = sum(facs[ch.name][p] * float(np.sum(y)) for p, y in procs.items())
            if np.any(mu <= 0):
                if np.any((mu <= 0) & (n > 0)):
                    return math.inf
                mu = np.where(mu <= 0, 1e-300, mu)
            total += nu_tot - float(np.sum(n * np.log(mu)))
        return total
