"""zmodel likelihood: the ModelIR evaluated with zfit pdfs and zfit.z.numpy.

Everything that depends on the parameters is one TensorFlow graph of the full parameter
vector (``tf.function`` with a fixed input signature, traced once):

* yields: rate * r^[signal] * prod(norm terms), with Combine's asymPow for asymmetric lnN;
* templates: Combine's vertical morph (smooth step, clip at 1e-9, renormalise) and the
  asymPow normalisation term with kappa**scale, exactly as ``modelspec.semantics``;
* parametric shapes: zfit pdfs translated from RooFit (``roofit.py``).  On binned data the
  expected count is N * f(bin centre) * width (``bin_integration == "center"``, not
  renormalised, as Combine) or N * exact bin integral (``"integral"``).  A parametric shape
  with a fixed histogram (``Shape.contents``) uses the IR histogram directly on binned data.

nll_main (no constraints, inference/model.py convention):
    binned/count  nu_tot - sum_i n_i log nu_i, nu_tot = sum of the process totals: the full pdf
                  yield for bin-centre shapes (as Combine/RooFit), the sum of the bins otherwise;
    unbinned      nu_tot - sum_j w_j log(sum_p nu_p f_p(x_j)).
Expected yields of an unbinned channel (used for Asimov data and toy totals) follow Combine's
Asimov generation (ToyMCSamplerOpt generateWithHisto): pdf density at the bin centres times
the bin width, scaled by one common factor per channel so that the channel total equals
nu_tot.  Templates on unbinned data are step densities (content / width).
"""

import itertools
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from inference.model import Likelihood, observed_dataset
from modelspec import ir as I
from modelspec import semantics as S
from stat_backends.base import UnsupportedByBackend

CHECK_TOLERANCE = 1e-6        # max |f_zfit - f_RooFit| / max f_RooFit for translated pdfs
SAMPLE_SEGMENTS = 20000       # envelope segments for accept-reject sampling
_UID = itertools.count()


@dataclass
class _Proc:
    name: str
    is_signal: bool
    rate: float
    kind: str                      # "count", "template", "hist", "pdf"
    terms: List[Callable] = field(default_factory=list)
    # template
    nom: Optional[np.ndarray] = None
    dhi: Optional[np.ndarray] = None
    dlo: Optional[np.ndarray] = None
    syst_idx: List[int] = field(default_factory=list)
    scales: Optional[np.ndarray] = None
    vsmooth: float = 1.0
    kappas: List[tuple] = field(default_factory=list)  # (param index, kappa_lo**scale, kappa_hi**scale)
    empty: bool = False
    # fixed histogram
    contents: Optional[np.ndarray] = None
    # parametric
    pdf: object = None
    obs_label: str = ""
    roo: object = None
    roo_obs: object = None


@dataclass
class _Chan:
    name: str
    kind: str                      # "count", "binned", "unbinned"
    lo: float
    hi: float
    edges: np.ndarray
    integration: str
    procs: List[_Proc]


# ----------------------------------------------------------------------------------------
# znp versions of the reference formulas
# ----------------------------------------------------------------------------------------

def _asym_pow(theta, kappa_lo, kappa_hi):
    import zfit.z.numpy as znp

    lkhi = math.log(kappa_hi)
    lklo = -math.log(kappa_lo)
    avg = 0.5 * (lkhi + lklo)
    half = 0.5 * (lkhi - lklo)
    twox = 2.0 * theta
    twox2 = twox * twox
    alpha = 0.125 * twox * (twox2 * (3.0 * twox2 - 10.0) + 15.0)
    inner = avg + alpha * half
    outer = znp.where(theta >= 0, lkhi, lklo)
    return znp.exp(znp.where(znp.abs(theta) >= 0.5, outer, inner) * theta)


def _smooth_step(x, vsmooth):
    import zfit.z.numpy as znp

    xn = x / vsmooth
    xn2 = xn * xn
    inner = 0.125 * xn * (xn2 * (3.0 * xn2 - 10.0) + 15.0)
    return znp.where(znp.abs(x) >= vsmooth, znp.sign(x), inner)


class ZLikelihood(Likelihood):
    backend_name = "zmodel"

    def __init__(self, model: I.ModelIR, check: bool = True, xla: bool = True):
        super().__init__(model)
        import tensorflow as tf

        self.tf = tf
        self.notes: List[str] = []
        self.translation_checks: List[dict] = []
        self.classes = set()
        self._uid = next(_UID)
        self._zparams: Dict[str, object] = {}
        self._R = None
        self._chans = [self._plan(ch) for ch in model.channels]
        if self._zparams:
            self.notes.append("zmodel: parametric pdfs translated from RooFit classes " + ", ".join(sorted(self.classes)))
        if {"RooGenericPdf", "RooLandauCB"} & self.classes:
            from stat_backends.zmodel.roofit import QUAD_NODES, QUAD_PANELS

            self.notes.append(f"zmodel: RooGenericPdf/RooLandauCB are normalised by {QUAD_PANELS}x{QUAD_NODES}-point "
                              "Gauss-Legendre quadrature (RooFit integrates them numerically too)")
        if any(c.kind == "unbinned" for c in self._chans):
            self.notes.append("zmodel: expected yields of unbinned channels are bin-centre densities x width scaled "
                              "to the channel total (Combine's Asimov convention)")
        if check:
            self._check_translations()
        t0 = time.time()
        native = self.prepare(observed_dataset(model))
        self.xla = xla
        self._compile()
        try:
            self.nll_main(self.nominal_values(), native)  # trace now: errors appear at create()
        except tf.errors.OpError as exc:
            if not xla:
                raise
            self.notes.append(f"zmodel: XLA compilation failed ({type(exc).__name__}); using a plain tf.function")
            self.xla = False
            self._compile()
            self.nll_main(self.nominal_values(), native)
        self.trace_seconds = time.time() - t0

    # ----- building -----------------------------------------------------------------------
    def _root(self):
        if self._R is None:
            from modelspec import rootinput

            self._R = rootinput.root()
        return self._R

    def _make_param(self, name):
        import zfit

        if name not in self._zparams:
            if name not in self.model.parameters:
                raise UnsupportedByBackend(f"zmodel: '{name}' is not a model parameter")
            self._zparams[name] = zfit.Parameter(f"zm{self._uid}_{len(self._zparams)}_{name}",
                                                 float(self.model.parameters[name].value))
        return self._zparams[name]

    def _translator(self, ch: I.Channel):
        from stat_backends.zmodel.roofit import Translator

        key = ch.name
        if not hasattr(self, "_translators"):
            self._translators = {}
        if key not in self._translators:
            self._translators[key] = Translator(self._root(), ch.observable.name, ch.observable.lo, ch.observable.hi,
                                                self._zparams, list(self.model.parameters), self._make_param)
        return self._translators[key]

    def _plan(self, ch: I.Channel) -> _Chan:
        kind = ch.data.kind
        edges = np.asarray(ch.observable.edges, dtype=float)
        procs = []
        for proc in ch.processes:
            procs.append(self._plan_proc(ch, proc, kind))
        return _Chan(name=ch.name, kind=kind, lo=ch.observable.lo, hi=ch.observable.hi, edges=edges,
                     integration=ch.bin_integration, procs=procs)

    def _plan_proc(self, ch, proc: I.Process, data_kind) -> _Proc:
        shape = proc.shape
        gmn = any(t.kind == "gmN" for t in proc.norm_terms)
        pp = _Proc(name=proc.name, is_signal=proc.is_signal, rate=1.0 if gmn else float(proc.rate), kind="")
        pp.terms = [self._term(ch, proc, t) for t in proc.norm_terms]
        if shape.kind == "counting":
            pp.kind = "count"
        elif shape.kind == "template":
            pp.kind = "template"
            nom = np.asarray(shape.contents, dtype=float)
            if nom.sum() <= 0:
                pp.empty = True
                pp.nom = np.zeros_like(nom)
                return pp
            pp.nom = nom / nom.sum()
            dhi, dlo, scales = [], [], []
            vsmooth = 1.0
            for s in shape.systs:
                if s.kind != "shape":
                    raise UnsupportedByBackend(f"zmodel: systematic kind '{s.kind}' is not supported")
                up = np.asarray(s.up, dtype=float)
                down = np.asarray(s.down, dtype=float)
                dhi.append(up / up.sum() - pp.nom)
                dlo.append(down / down.sum() - pp.nom)
                pp.syst_idx.append(self.index[s.param])
                scales.append(float(s.scale))
                vsmooth = min(vsmooth, float(s.scale))
                k = S.template_norm_kappas(nom, up, down, s.scale)
                if k is not None:
                    pp.kappas.append((self.index[s.param], k[0], k[1]))
            pp.dhi = np.array(dhi).reshape(len(dhi), len(nom))
            pp.dlo = np.array(dlo).reshape(len(dlo), len(nom))
            pp.scales = np.array(scales)
            pp.vsmooth = vsmooth
        elif shape.kind == "parametric":
            if shape.contents and data_kind != "unbinned":
                pp.kind = "hist"
                pp.contents = np.asarray(shape.contents, dtype=float)
            else:
                pp.kind = "pdf"
                self._translate(ch, pp, shape)
        else:
            raise UnsupportedByBackend(f"zmodel: shape kind '{shape.kind}' is not supported")
        return pp

    def _translate(self, ch, pp, shape):
        from modelspec import rootinput as R

        tr = self._translator(ch)
        ws = R.get_workspace(shape.ref.file, shape.ref.workspace)
        roo = ws.pdf(shape.ref.name)
        if not roo:
            raise KeyError(f"zmodel: pdf '{shape.ref.name}' not found in {shape.ref.file}:{shape.ref.workspace}")
        pp.pdf = tr.pdf(roo)
        self.classes |= tr.classes
        pp.roo = roo
        pp.roo_obs = roo.getVariables().find(ch.observable.name)
        if not pp.roo_obs:
            raise UnsupportedByBackend(f"zmodel: pdf '{shape.ref.name}' does not depend on observable {ch.observable.name}")
        pp.obs_label = tr.space.obs[0]

    def _term(self, ch, proc, t: I.NormTerm) -> Callable:
        import zfit.z.numpy as znp

        from stat_backends.zmodel import formula as F

        if t.kind in ("lnN", "lnU"):
            i, lk = self.index[t.param], math.log(t.kappa_hi)
            return lambda v: znp.exp(v[i] * lk)
        if t.kind == "asym_lnN":
            i, lo, hi = self.index[t.param], t.kappa_lo, t.kappa_hi
            return lambda v: _asym_pow(v[i], lo, hi)
        if t.kind == "gmN":
            i, a = self.index[t.param], t.alpha
            return lambda v: a * v[i]
        if t.kind == "rate_param":
            i = self.index[t.param]
            return lambda v: v[i]
        if t.kind == "formula":
            try:
                node = F.parse(t.formula, t.args)
            except F.FormulaError as exc:
                raise UnsupportedByBackend(f"zmodel: rateParam formula '{t.param}': {exc}") from exc
            fn = F.compile_formula(node, F.TFOps())
            idx = [self.index[a] for a in t.args]
            return lambda v: fn([v[i] for i in idx])
        if t.kind == "ws_norm":
            from modelspec import rootinput as R

            ws = R.get_workspace(t.ref.file, t.ref.workspace)
            roo = ws.arg(t.ref.name)
            sym = self._translator(ch).sym(roo)
            if sym.depends_on_x:
                raise UnsupportedByBackend(f"zmodel: normalisation '{t.ref.name}' depends on the observable")
            idx = {n: self.index[n] for n in sym.leaves}
            fn = sym.fn
            return lambda v: fn(None, {n: v[i] for n, i in idx.items()})
        raise UnsupportedByBackend(f"zmodel: norm term '{t.kind}' is not supported")

    # ----- graph pieces ------------------------------------------------------------------
    def _yield(self, pp: _Proc, v):
        tf = self.tf
        y = tf.constant(pp.rate, tf.float64)
        if pp.is_signal:
            y = y * v[self.poi_index]
        for term in pp.terms:
            y = y * term(v)
        return y

    def _template(self, pp: _Proc, v):
        """(normalisation factor, unit-normalised morphed template)."""
        import zfit.z.numpy as znp

        tf = self.tf
        if pp.empty:
            return tf.constant(0.0, tf.float64), znp.asarray(pp.nom)
        t = znp.asarray(pp.nom)
        norm = tf.constant(1.0, tf.float64)
        if pp.syst_idx:
            theta = tf.gather(v, pp.syst_idx)
            x = theta * pp.scales
            sm = _smooth_step(x, pp.vsmooth)
            dhi, dlo = znp.asarray(pp.dhi), znp.asarray(pp.dlo)
            t = t + znp.sum(0.5 * x[:, None] * ((dhi - dlo) + (dhi + dlo) * sm[:, None]), axis=0)
            t = znp.where(t <= 0, 1e-9, t)
            t = t / znp.sum(t)
        for i, klo, khi in pp.kappas:
            norm = norm * _asym_pow(v[i], klo, khi)
        return norm, t

    def _bin_values(self, ch: _Chan, pp: _Proc, v):
        """(expected counts in the bins, total pdf yield) of a process in a binned/count channel.

        The total is the sum of the bins except for bin-centre shapes (fixed histograms and
        pdfs with bin_integration "center"), where it is the full pdf yield (see nll_main)."""
        import zfit.z.numpy as znp

        y = self._yield(pp, v)
        if pp.kind == "count":
            return znp.reshape(y, [1]), y
        if pp.kind == "template":
            norm, t = self._template(pp, v)
            vals = y * norm * t
            return vals, znp.sum(vals)
        if pp.kind == "hist":
            return y * znp.asarray(pp.contents), y
        if ch.integration == "integral":
            from stat_backends.zmodel.roofit import bin_integrals

            vals = y * bin_integrals(pp.pdf, pp.obs_label, ch.edges)
            return vals, znp.sum(vals)
        return y * self._center_fractions(ch, pp), y

    def _center_fractions(self, ch, pp):
        import zfit.z.numpy as znp

        xc = 0.5 * (ch.edges[:-1] + ch.edges[1:])
        return znp.reshape(pp.pdf.pdf(xc[:, None]), [-1]) * znp.asarray(np.diff(ch.edges))

    def _unbinned_terms(self, ch: _Chan, pp: _Proc, v, x):
        """(total yield, normalised density at x) of a process in an unbinned channel."""
        import zfit.z.numpy as znp

        tf = self.tf
        y = self._yield(pp, v)
        if pp.kind == "template":
            norm, t = self._template(pp, v)
            nb = len(ch.edges) - 1
            idx = tf.clip_by_value(tf.searchsorted(znp.asarray(ch.edges), x, side="right") - 1, 0, nb - 1)
            return y * norm, tf.gather(t / znp.asarray(np.diff(ch.edges)), idx)
        if pp.kind == "pdf":
            return y, znp.reshape(pp.pdf.pdf(x[:, None]), [-1])
        raise UnsupportedByBackend(f"zmodel: process {pp.name} of kind {pp.kind} in unbinned channel {ch.name}")

    def _assign(self, v):
        for name, p in self._zparams.items():
            p.assign(v[self.index[name]])

    def _expected_graph(self, v):
        import zfit.z.numpy as znp

        self._assign(v)
        out = []
        for ch in self._chans:
            if ch.kind != "unbinned":
                out.extend(self._bin_values(ch, pp, v)[0] for pp in ch.procs)
                continue
            raw, total = [], 0.0
            for pp in ch.procs:
                y = self._yield(pp, v)
                if pp.kind == "template":
                    norm, t = self._template(pp, v)
                    y = y * norm
                    raw.append(y * t)
                else:
                    raw.append(y * self._center_fractions(ch, pp))
                total = total + y
            factor = total / znp.sum(znp.stack(raw))
            out.extend(r * factor for r in raw)
        return out

    def _nll_graph(self, v, *data):
        import zfit.z.numpy as znp

        tf = self.tf
        self._assign(v)
        total = tf.constant(0.0, tf.float64)
        bad = tf.constant(False)
        k = 0
        for ch in self._chans:
            if ch.kind != "unbinned":
                n = data[k]
                k += 1
                parts = [self._bin_values(ch, pp, v) for pp in ch.procs]
                mu = tf.add_n([b for b, _ in parts])
                nu_tot = tf.add_n([t for _, t in parts])
                bad = bad | tf.reduce_any((mu <= 0) & (n > 0))
                mu = znp.where(mu <= 0, 1e-300, mu)
                total = total + nu_tot - znp.sum(n * znp.log(mu))
                continue
            x, w = data[k], data[k + 1]
            k += 2
            nu, dens = 0.0, 0.0
            for pp in ch.procs:
                y, f = self._unbinned_terms(ch, pp, v, x)
                nu = nu + y
                dens = dens + y * f
            bad = bad | tf.reduce_any((dens <= 0) & (w != 0))
            dens = znp.where(dens <= 0, 1e-300, dens)
            total = total + nu - znp.sum(w * znp.log(dens))
        return znp.where(bad, np.inf, total)

    def _density_graph(self, ci, v, x):
        self._assign(v)
        ch = self._chans[ci]
        dens = 0.0
        for pp in ch.procs:
            y, f = self._unbinned_terms(ch, pp, v, x)
            dens = dens + y * f
        return dens

    def _compile(self):
        tf = self.tf
        vspec = tf.TensorSpec([len(self.names)], tf.float64)
        specs = [vspec]
        for ch in self._chans:
            if ch.kind == "unbinned":
                specs += [tf.TensorSpec([None], tf.float64), tf.TensorSpec([None], tf.float64)]
            else:
                specs.append(tf.TensorSpec([len(ch.edges) - 1], tf.float64))
        self._nll_fn = tf.function(self._nll_graph, input_signature=specs, jit_compile=self.xla)
        self._exp_fn = tf.function(self._expected_graph, input_signature=[vspec])
        self._dens_fn = {}
        for ci, ch in enumerate(self._chans):
            if ch.kind == "unbinned":
                self._dens_fn[ch.name] = tf.function(lambda v, x, ci=ci: self._density_graph(ci, v, x),
                                                     input_signature=[vspec, tf.TensorSpec([None], tf.float64)])

    # ----- Likelihood interface --------------------------------------------------------------
    def prepare(self, data):
        tf = self.tf
        out = []
        for ch in self._chans:
            d = data.main[ch.name]
            if ch.kind == "unbinned":
                if d.kind != "unbinned":
                    raise ValueError(f"zmodel: channel {ch.name} expects unbinned data, got {d.kind}")
                x = np.asarray(d.values, dtype=float)
                w = np.ones_like(x) if d.weights is None else np.asarray(d.weights, dtype=float)
                if x.size and (x.min() < ch.lo or x.max() > ch.hi):
                    raise ValueError(f"zmodel: channel {ch.name} has events outside [{ch.lo}, {ch.hi}]")
                out += [tf.constant(x, tf.float64), tf.constant(w, tf.float64)]
            else:
                n = np.asarray(d.counts, dtype=float)
                if n.shape != (len(ch.edges) - 1,):
                    raise ValueError(f"zmodel: channel {ch.name} expects {len(ch.edges) - 1} bins, got {n.shape}")
                out.append(tf.constant(n, tf.float64))
        return tuple(out)

    def nll_main(self, values, native) -> float:
        return float(self._nll_fn(self.tf.constant(np.asarray(values, dtype=float)), *native))

    def expected_by_process(self, values):
        arrs = self._exp_fn(self.tf.constant(np.asarray(values, dtype=float)))
        out, k = {}, 0
        for ch in self._chans:
            out[ch.name] = {}
            for pp in ch.procs:
                out[ch.name][pp.name] = np.asarray(arrs[k], dtype=float)
                k += 1
        return out

    def density(self, values, channel, x):
        """Total intensity sum_p nu_p f_p(x) of an unbinned channel."""
        return np.asarray(self._dens_fn[channel](self.tf.constant(np.asarray(values, dtype=float)),
                                                 self.tf.constant(np.asarray(x, dtype=float))))

    def sample_unbinned(self, values, channel, n, rng):
        """Exact accept-reject from the channel's total intensity with the numpy ``rng``.

        The envelope is piecewise constant on SAMPLE_SEGMENTS segments (1.25 x the maximum of
        the intensity on 5 points per segment, floored at 1e-3 of the mean); a proposal above
        the envelope doubles it there and restarts the whole sample, so the result is exact."""
        ch = next(c for c in self._chans if c.name == channel)
        if n <= 0:
            return np.zeros(0)
        K = SAMPLE_SEGMENTS
        grid = np.linspace(ch.lo, ch.hi, 4 * K + 1)
        f = self.density(values, channel, grid)
        if not np.all(np.isfinite(f)) or np.any(f < 0):
            raise RuntimeError(f"zmodel: intensity of channel {channel} is not finite and non-negative")
        env = 1.25 * np.max(np.stack([f[0:-1:4], f[1::4], f[2::4], f[3::4], f[4::4]]), axis=0)
        env = np.maximum(env, 1e-3 * env.mean())
        seg_lo = grid[0:-1:4]
        width = (ch.hi - ch.lo) / K
        while True:
            accepted, restart = [], False
            while sum(len(a) for a in accepted) < n:
                mass = env * width
                p = mass / mass.sum()
                eff = min(1.0, max(1e-3, float(np.sum(f[:-1]) * (ch.hi - ch.lo) / (4 * K) / mass.sum())))
                need = n - sum(len(a) for a in accepted)
                batch = int(min(max(1000, 2.0 * need / eff), 2_000_000))
                k = rng.choice(K, size=batch, p=p)
                x = seg_lo[k] + rng.random(batch) * width
                u = rng.random(batch) * env[k]
                fx = self.density(values, channel, x)
                over = fx > env[k]
                if np.any(over):
                    np.maximum.at(env, k[over], 2.0 * fx[over])
                    restart = True
                    break
                accepted.append(x[u < fx])
            if not restart:
                return np.concatenate(accepted)[:n]

    # ----- validation against RooFit ---------------------------------------------------------
    def _check_translations(self):
        """Compare every translated pdf with RooFit at the IR nominal values: normalised
        densities at 50 points and the bin fractions of the channel's binning, against RooFit
        with precise numerical integrals (``roofit.roofit_reference``).  Where RooFit's
        default numerical normalisation (what Combine uses) differs from the precise one, a
        note gives the size."""
        from stat_backends.zmodel.roofit import bin_integrals, roofit_reference

        v = self.nominal_values()
        self._assign_eager(v)
        for ch in self._chans:
            for pp in ch.procs:
                if pp.kind != "pdf":
                    continue
                saved = self._set_roofit(pp.roo, v)
                method = "integral" if ch.kind != "unbinned" and ch.integration == "integral" else "center"
                try:
                    xs = np.linspace(ch.lo, ch.hi, 52)[1:-1]
                    d_r, b_r, d_def = roofit_reference(self._root(), pp.roo, pp.roo_obs.GetName(), xs,
                                                       ch.edges, method)
                finally:
                    for var, old in saved:
                        var.setVal(old)
                d_z = np.asarray(pp.pdf.pdf(xs[:, None])).ravel()
                if method == "integral":
                    b_z = np.asarray(bin_integrals(pp.pdf, pp.obs_label, ch.edges)).ravel()
                else:
                    b_z = np.asarray(self._center_fractions(ch, pp)).ravel()
                dev_d = float(np.max(np.abs(d_z - d_r)) / np.max(np.abs(d_r)))
                dev_b = float(np.max(np.abs(b_z - b_r)) / np.max(np.abs(b_r)))
                dev_default = float(np.max(np.abs(d_def - d_r)) / np.max(np.abs(d_r)))
                rec = {"channel": ch.name, "process": pp.name, "pdf": pp.roo.GetName(), "class": pp.roo.ClassName(),
                       "density_max_rel_dev": dev_d, "bin_fraction_max_rel_dev": dev_b, "bin_method": method,
                       "roofit_default_normalisation_dev": dev_default}
                self.translation_checks.append(rec)
                self.notes.append(f"zmodel: {ch.name}/{pp.name} ({pp.roo.ClassName()} {pp.roo.GetName()}) agrees with "
                                  f"RooFit: densities {dev_d:.1e}, bin fractions {dev_b:.1e} (max rel. dev.)")
                if dev_default > 1e-9:
                    self.notes.append(f"zmodel: RooFit's default numerical normalisation of {pp.roo.GetName()} (used by "
                                      f"Combine) differs from a precise integral by {dev_default:.1e} (relative); "
                                      "zmodel uses the precise normalisation")
                if not (dev_d <= CHECK_TOLERANCE and dev_b <= CHECK_TOLERANCE):
                    raise RuntimeError(f"zmodel: translated pdf {pp.roo.GetName()} ({pp.roo.ClassName()}) disagrees "
                                       f"with RooFit: density dev {dev_d:.3g}, bin fraction dev {dev_b:.3g}")

    def _assign_eager(self, v):
        for name, p in self._zparams.items():
            p.assign(float(v[self.index[name]]))

    def _set_roofit(self, roo, v):
        saved = []
        for var in roo.getVariables():
            name = var.GetName()
            if name in self.index and var.InheritsFrom("RooRealVar"):
                saved.append((var, var.getVal()))
                var.setVal(float(v[self.index[name]]))
        return saved

    def set_values(self, values):
        """Set the zfit parameters to ``values`` (for direct use of the zfit pdfs)."""
        self._assign_eager(np.asarray(values, dtype=float))

    def pdf(self, channel, process):
        """The zfit pdf of a parametric process (None for other shapes)."""
        ch = next(c for c in self._chans if c.name == channel)
        return next(p for p in ch.procs if p.name == process).pdf
