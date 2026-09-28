"""pyhf ``Likelihood``: the main-measurement Poisson term of a pyhf model built from the IR.

Only pyhf's expected rates are used.  ``nll_main`` is computed here as
``nu_tot - sum(n_i log nu_i)`` (no ``lgamma(n+1)``, no constraint terms: the shared layer adds
the IR constraints), so it follows the convention in ``inference/model.py``; nu_tot differs
from sum(nu_i) only for parametric histograms evaluated at bin centres, so ``pyhf_logpdf``
(pyhf's own Poisson) differs from it by that amount for such models.  The pyhf
constraint terms and auxiliary data are only used for the native export and for
``pyhf_logpdf`` (validation).

autoMCStats channels (``_MCStats``): pyhf gives C_p(theta) * h_p(theta) before any floor.
The histosys deltas of a template sum to zero, so C_p = sum_i y_pi / integral_p exactly;
h_p is then floored at 1e-9 (CMSHistFunc) and the Barlow-Beeston-lite terms of
``semantics.bb_lite_expected`` are added (vectorised here, independently of the oracle).
"""

import math
from typing import Dict

import numpy as np

from inference.model import Dataset, Likelihood, fixed_shape_total_factors
from modelspec import ir as I
from stat_backends.hfmodel.builder import MODIFIER_SETTINGS, MORPH_FLOOR, build_spec

CMSHIST_FLOOR = 1e-9  # CMSHistFunc / CMSHistErrorPropagator CropUnderflows


class _MCStats:
    """Index arrays of the autoMCStats bin parameters of all channels (pyhf sample rows r,
    global bin columns j)."""

    def __init__(self, lik, model):
        rows = {s: i for i, s in enumerate(lik.pdf.config.samples)}
        nb = lik._template_mask.shape[1]
        self.rows, self.cols, self.integral = [], [], []   # template cells of autoMCStats channels
        self.err = np.zeros(lik._template_mask.shape)
        self.bins = np.zeros(nb, dtype=bool)                # columns of autoMCStats channels
        tot_j, tot_i, pois_r, pois_j, pois_i, pois_n, gau_r, gau_j, gau_i = ([] for _ in range(9))
        for ch in model.channels:
            if ch.mcstats is None:
                continue
            sl = lik._slices[ch.name]
            off = sl.start
            self.bins[sl] = True
            for proc in ch.processes:
                r = rows[proc.name]
                self.err[r, sl] = np.sqrt(np.asarray(proc.shape.sumw2, dtype=float))
                total = float(np.sum(proc.shape.contents))
                if proc.rate != 0.0 and total > 0.0:
                    self.rows.append(r)
                    self.cols.append(sl)
                    self.integral.append(total)
            for bp in ch.mcstats.params:
                i = lik.index[bp.param]
                if bp.kind == I.MCSTATS_TOTAL:
                    tot_j.append(off + bp.bin)
                    tot_i.append(i)
                elif bp.kind == I.MCSTATS_POISSON:
                    pois_r.append(rows[bp.process])
                    pois_j.append(off + bp.bin)
                    pois_i.append(i)
                    pois_n.append(bp.n_eff)
                else:
                    gau_r.append(rows[bp.process])
                    gau_j.append(off + bp.bin)
                    gau_i.append(i)
        arr = lambda a, t=int: np.asarray(a, dtype=t)  # noqa: E731
        self.tot_j, self.tot_i = arr(tot_j), arr(tot_i)
        self.pois_r, self.pois_j, self.pois_i, self.pois_n = arr(pois_r), arr(pois_j), arr(pois_i), arr(pois_n, float)
        self.gau_r, self.gau_j, self.gau_i = arr(gau_r), arr(gau_j), arr(gau_i)

    def apply(self, y, values):
        """y (samples x bins, pyhf rates) -> per-process yields with the CMSHistFunc floor and
        the BB-lite terms (unfloored sums, as the oracle's expected_by_process)."""
        y = np.array(y, dtype=float)
        coef = np.zeros(y.shape[0])
        for r, sl, total in zip(self.rows, self.cols, self.integral):
            c = y[r, sl].sum() / total
            coef[r] = c
            y[r, sl] = c * np.maximum(y[r, sl] / c, CMSHIST_FLOOR) if c != 0.0 else 0.0
        v = np.asarray(values, dtype=float)
        base = y.copy()
        ce = coef[:, None] * self.err
        if self.tot_j.size:
            w = ce[:, self.tot_j] ** 2
            s2 = w.sum(axis=0)
            ok = s2 > 0
            shift = np.where(ok, v[self.tot_i] / np.sqrt(np.where(ok, s2, 1.0)), 0.0)
            y[:, self.tot_j] += w * shift[None, :]
        if self.pois_r.size:
            y[self.pois_r, self.pois_j] += (v[self.pois_i] / self.pois_n - 1.0) * base[self.pois_r, self.pois_j]
        if self.gau_r.size:
            y[self.gau_r, self.gau_j] += v[self.gau_i] * ce[self.gau_r, self.gau_j]
        return y


def import_pyhf():
    try:
        import pyhf
    except ImportError as err:
        raise ImportError("hfmodel needs pyhf: run scripts/install_python_deps.sh and source setup_env.sh") from err
    pyhf.set_backend("numpy")
    return pyhf


class HFLikelihood(Likelihood):
    backend_name = "hfmodel"

    def __init__(self, model):
        super().__init__(model)
        pyhf = import_pyhf()
        self.spec = build_spec(model)
        self.notes = list(self.spec.notes)
        poi = model.poi if model.poi in self.spec.links else None
        self.pdf = pyhf.Model(self.spec.model_spec, poi_name=poi, modifier_settings=MODIFIER_SETTINGS,
                              validate=True)
        self._main = self.pdf.main_model
        cfg = self.pdf.config
        # IR vector -> pyhf vector: pars = values[src] * fac
        self._src = np.zeros(cfg.npars, dtype=int)
        self._fac = np.ones(cfg.npars)
        for name in cfg.par_order:
            link = self.spec.links[name]
            sl = cfg.par_slice(name)
            self._src[sl] = self.index[link.ir_name]
            self._fac[sl] = link.factor
        self.channel_order = list(cfg.channels)
        self._slices = {ch: cfg.channel_slices[ch] for ch in self.channel_order}
        row = {s: i for i, s in enumerate(cfg.samples)}
        self._cells = {ch.name: [(p.name, row[p.name]) for p in ch.processes] for ch in model.channels}
        self._templates = [(row[p], self._slices[ch]) for ch, p in self.spec.template_samples]
        mask = np.zeros((len(cfg.samples), sum(ch.observable.nbins for ch in model.channels)), dtype=bool)
        for r, sl in self._templates:
            mask[r, sl] = True
        self._template_mask = mask
        # extended term: nu_tot = sum over cells of y * factor (factor != 1 only for
        # parametric histograms evaluated at bin centres, see inference/model.py)
        facs = fixed_shape_total_factors(model)
        self._total_factor = np.ones(mask.shape)
        for ch, cells in self._cells.items():
            for p, r in cells:
                self._total_factor[r, self._slices[ch]] = facs[ch][p]
        # morphed RooHistPdfs: the fractions sum to 1 (extended term = yield); density crop
        widths = {ch.name: ch.observable.bin_volumes() for ch in model.channels}
        self._histpdf = [(row[p], self._slices[ch], widths[ch]) for ch, p in self.spec.histpdf_samples]
        for r, sl, _ in self._histpdf:
            self._total_factor[r, sl] = 1.0
        self._mcstats = _MCStats(self, model) if self.spec.mcstats_samples or any(
            ch.mcstats is not None for ch in model.channels) else None

    # ----- parameter / data mapping -----------------------------------------------------
    def pyhf_pars(self, values) -> np.ndarray:
        """pyhf parameter vector for IR values (gmN: gamma = n / N)."""
        return np.asarray(values, dtype=float)[self._src] * self._fac

    def ir_values(self, pars, base=None) -> np.ndarray:
        """IR value vector from a pyhf parameter vector (other IR parameters from ``base``)."""
        out = self.nominal_values() if base is None else np.array(base, dtype=float)
        out[self._src] = np.asarray(pars, dtype=float) / self._fac
        return out

    def auxdata(self, global_obs: Dict[str, float]) -> np.ndarray:
        """pyhf auxiliary data for the IR global observables (Gaussian: g; gmN Poisson: g = observed N)."""
        cfg = self.pdf.config
        out = []
        for name in cfg.auxdata_order:
            out.extend([float(global_obs[self.spec.links[name].ir_name])] * cfg.param_set(name).n_parameters)
        return np.asarray(out)

    def pyhf_data(self, data: Dataset) -> np.ndarray:
        return np.concatenate([self.prepare(data), self.auxdata(data.global_obs)])

    def pyhf_logpdf(self, values, data: Dataset) -> float:
        """pyhf's own full log-likelihood (main + constraints, with all constants)."""
        return float(self.pdf.logpdf(self.pyhf_pars(values), self.pyhf_data(data))[0])

    # ----- Likelihood interface ---------------------------------------------------------
    def prepare(self, data: Dataset):
        return np.concatenate([np.asarray(data.main[ch].counts, dtype=float) for ch in self.channel_order])

    def _by_sample(self, values) -> np.ndarray:
        y = self._main.expected_data(self.pyhf_pars(values), return_by_sample=True)
        if self._mcstats is not None:
            y = self._mcstats.apply(y, values)
        if self._templates and np.any(y[self._template_mask] <= 0):
            y = np.array(y, dtype=float)
            for r, sl in self._templates:
                # y = factor * (unit-sum morphed shape): floor the shape, renormalise, rescale.
                # factor 0 (e.g. r = 0) leaves an all-zero sample, as in the reference.
                total = y[r, sl].sum()
                if total == 0:
                    continue
                u = y[r, sl] / total
                if np.any(u <= 0):
                    u = np.where(u <= 0, MORPH_FLOOR, u)
                    y[r, sl] = u * (total / u.sum())
        if self._histpdf:
            y = self._crop_histpdfs(y)
        return y

    def _crop_histpdfs(self, y):
        """FastVerticalInterpHistPdf2: bins below a density of 1e-9 are set to it, then the
        shape is renormalised (the yield is unchanged)."""
        y = np.array(y, dtype=float)
        for r, sl, w in self._histpdf:
            total = y[r, sl].sum()
            if total == 0:
                continue
            u = y[r, sl] / total
            u = np.where(u / w < MORPH_FLOOR, MORPH_FLOOR * w, u)
            y[r, sl] = u * (total / u.sum())
        return y

    def expected_by_process(self, values):
        y = self._by_sample(values)
        return {ch: {p: np.array(y[r, self._slices[ch]]) for p, r in cells} for ch, cells in self._cells.items()}

    def nll_main(self, values, native) -> float:
        y = self._by_sample(values)
        nu = y.sum(axis=0)
        n = native
        if self._mcstats is not None:
            bb = self._mcstats.bins
            nu[bb] = np.maximum(nu[bb], CMSHIST_FLOOR)  # CMSHistErrorPropagator CropUnderflows
            nu_tot = float(np.sum(y[:, ~bb] * self._total_factor[:, ~bb])) + float(np.sum(nu[bb]))
        else:
            nu_tot = float(np.sum(y * self._total_factor))
        if np.any(nu <= 0):
            if np.any((nu <= 0) & (n > 0)):
                return math.inf
            nu = np.where(nu <= 0, 1e-300, nu)
        return nu_tot - float(np.sum(n * np.log(nu)))
