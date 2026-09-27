"""pyhf ``Likelihood``: the main-measurement Poisson term of a pyhf model built from the IR.

Only pyhf's expected rates are used.  ``nll_main`` is computed here as
``nu_tot - sum(n_i log nu_i)`` (no ``lgamma(n+1)``, no constraint terms: the shared layer adds
the IR constraints), so it follows the convention in ``inference/model.py``; nu_tot differs
from sum(nu_i) only for parametric histograms evaluated at bin centres, so ``pyhf_logpdf``
(pyhf's own Poisson) differs from it by that amount for such models.  The pyhf
constraint terms and auxiliary data are only used for the native export and for
``pyhf_logpdf`` (validation).
"""

import math
from typing import Dict

import numpy as np

from inference.model import Dataset, Likelihood, fixed_shape_total_factors
from stat_backends.hfmodel.builder import MODIFIER_SETTINGS, MORPH_FLOOR, build_spec


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
        mask = np.zeros((len(cfg.samples), sum(len(ch.observable.edges) - 1 for ch in model.channels)), dtype=bool)
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
        return y

    def expected_by_process(self, values):
        y = self._by_sample(values)
        return {ch: {p: np.array(y[r, self._slices[ch]]) for p, r in cells} for ch, cells in self._cells.items()}

    def nll_main(self, values, native) -> float:
        y = self._by_sample(values)
        nu = y.sum(axis=0)
        nu_tot = float(np.sum(y * self._total_factor))
        n = native
        if np.any(nu <= 0):
            if np.any((nu <= 0) & (n > 0)):
                return math.inf
            nu = np.where(nu <= 0, 1e-300, nu)
        return nu_tot - float(np.sum(n * np.log(nu)))
