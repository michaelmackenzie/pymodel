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
    kind: str                      # "count", "template", "hist", "pdf", "hmorph", "vmorph"
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
    cmshist: bool = False          # autoMCStats channel: CMSHistFunc template (floor 1e-9, no renormalisation)
    integral: float = 0.0          # template integral (autoMCStats)
    # fixed histogram
    contents: Optional[np.ndarray] = None
    # shape systematics on a fixed pdf (Shape.pdf_systs): "hmorph" uses nom/dhi/dlo/syst_idx/
    # scales/vsmooth like a template; "vmorph" (VerticalInterpPdf) the arrays below
    widths: Optional[np.ndarray] = None
    raw: Optional[np.ndarray] = None      # (1 + 2K, nbins): I_j * contents_j (nominal, up_k, down_k)
    raw_int: Optional[np.ndarray] = None  # (1 + 2K,): I_j
    # "vmorphf": VerticalInterpPdf of pdfs with floating parameters (roofit.RawPdf per pdf)
    raws: List[object] = field(default_factory=list)       # nominal, up_1, down_1, ...
    raw_roos: List[object] = field(default_factory=list)   # the RooFit pdfs of ``raws``
    raw_idx: Dict[str, int] = field(default_factory=dict)  # IR parameter -> index in the value vector
    logm: bool = False             # shapeN: log-vertical morph (FastVerticalInterpHistPdf2, smoothAlgo < 0)
    lognom: Optional[np.ndarray] = None  # shapeN: log of the unit-normalised nominal (-999 where it is 0)
    # parametric
    pdf: object = None
    obs_label: str = ""
    roo: object = None
    roo_obs: object = None
    # envelope (RooMultiPdf): one "pdf" _Proc per component, selected by the category value
    states: List["_Proc"] = field(default_factory=list)
    cat_index: int = -1
    corrections: List[float] = field(default_factory=list)  # RooMultiPdf::getCorrection() per state
    state_deps: List[set] = field(default_factory=list)      # floating IR parameters of each state


@dataclass
class _Chan:
    name: str
    kind: str                      # "count", "binned", "unbinned"
    lo: float
    hi: float
    edges: np.ndarray
    integration: str
    procs: List[_Proc]
    mcstats: Optional[dict] = None  # autoMCStats index arrays (see _plan_mcstats)


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
        self._plan_envelopes()
        if self._zparams:
            self.notes.append("zmodel: parametric pdfs translated from RooFit classes " + ", ".join(sorted(self.classes)))
        if {"RooGenericPdf", "RooLandauCB"} & self.classes:
            from stat_backends.zmodel.roofit import ROO_INT_STAGES

            self.notes.append("zmodel: RooGenericPdf/RooLandauCB are normalised by an emulation of RooFit's default "
                              f"RooIntegrator1D (Romberg; as Combine; up to {ROO_INT_STAGES} of RooFit's 20 stages)")
        if any(c.kind == "unbinned" for c in self._chans):
            self.notes.append("zmodel: expected yields of unbinned channels are bin-centre densities x width scaled "
                              "to the channel total (Combine's Asimov convention)")
        if check:
            self._check_translations()
            self._check_raw_translations()
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
        out = _Chan(name=ch.name, kind=kind, lo=ch.observable.lo, hi=ch.observable.hi, edges=edges,
                    integration=ch.bin_integration, procs=procs)
        if ch.mcstats is not None:
            out.mcstats = self._plan_mcstats(ch)
        return out

    def _plan_mcstats(self, ch: I.Channel) -> dict:
        """Index arrays of the Barlow-Beeston-lite parameters (semantics.bb_lite_expected)."""
        if ch.data.kind == "unbinned":
            raise UnsupportedByBackend(f"zmodel: autoMCStats on the unbinned channel {ch.name}")
        row = {p.name: i for i, p in enumerate(ch.processes)}
        mc = {"err": np.array([np.sqrt(np.asarray(p.shape.sumw2, dtype=float)) for p in ch.processes]),
              "tot_j": [], "tot_i": [], "pois": [], "pois_i": [], "pois_n": [], "gau": [], "gau_i": []}
        for bp in ch.mcstats.params:
            i = self.index[bp.param]
            if bp.kind == I.MCSTATS_TOTAL:
                mc["tot_j"].append(bp.bin)
                mc["tot_i"].append(i)
            elif bp.kind == I.MCSTATS_POISSON:
                mc["pois"].append([row[bp.process], bp.bin])
                mc["pois_i"].append(i)
                mc["pois_n"].append(bp.n_eff)
            elif bp.kind == I.MCSTATS_GAUSS:
                mc["gau"].append([row[bp.process], bp.bin])
                mc["gau_i"].append(i)
            else:
                raise UnsupportedByBackend(f"zmodel: autoMCStats parameter kind '{bp.kind}'")
        nproc = len(ch.processes)
        # scatter indices of the "total" shifts: (process, bin) for every process of every such bin
        mc["tot_idx"] = [[p, j] for j in mc["tot_j"] for p in range(nproc)]
        mc["pois_n"] = np.asarray(mc["pois_n"], dtype=float)
        return mc

    def _mcstats_values(self, ch: _Chan, v):
        """(per-process yields [nproc, nbins], floored bin totals) of an autoMCStats channel."""
        import zfit.z.numpy as znp

        tf = self.tf
        mc = ch.mcstats
        vals, coefs = [], []
        for pp in ch.procs:
            y = self._yield(pp, v)
            norm, t = self._template(pp, v)
            vals.append(y * norm * t)
            coefs.append(y * norm / pp.integral if pp.integral > 0 else tf.constant(0.0, tf.float64))
        Y = tf.stack(vals)
        base = Y
        ce = tf.stack(coefs)[:, None] * znp.asarray(mc["err"])
        if mc["tot_j"]:
            w = tf.gather(ce, mc["tot_j"], axis=1) ** 2
            s2 = tf.reduce_sum(w, axis=0)
            ok = s2 > 0
            shift = znp.where(ok, tf.gather(v, mc["tot_i"]) / znp.sqrt(znp.where(ok, s2, 1.0)), 0.0)
            Y = tf.tensor_scatter_nd_add(Y, mc["tot_idx"], tf.reshape(tf.transpose(w * shift[None, :]), [-1]))
        if mc["pois"]:
            upd = (tf.gather(v, mc["pois_i"]) / mc["pois_n"] - 1.0) * tf.gather_nd(base, mc["pois"])
            Y = tf.tensor_scatter_nd_add(Y, mc["pois"], upd)
        if mc["gau"]:
            Y = tf.tensor_scatter_nd_add(Y, mc["gau"], tf.gather(v, mc["gau_i"]) * tf.gather_nd(ce, mc["gau"]))
        return Y, znp.maximum(tf.reduce_sum(Y, axis=0), S.CMSHIST_FLOOR)

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
            kinds = {s.kind for s in shape.systs}
            if kinds and (len(kinds) != 1 or not kinds <= {"shape", "shapeN"}):
                raise UnsupportedByBackend(f"zmodel: template {ch.name}/{proc.name} mixes morphing algorithms "
                                           f"{sorted(kinds)} (Combine allows one per shape)")
            pp.logm = kinds == {"shapeN"}
            if pp.logm and ch.mcstats is not None:
                raise UnsupportedByBackend(f"zmodel: shapeN template {ch.name}/{proc.name} in an autoMCStats channel")
            if pp.logm:
                pp.lognom = np.where(pp.nom > 0, np.log(np.where(pp.nom > 0, pp.nom, 1.0)), -999.0)

            def logratio(a):  # 0 where either template is 0 (FastVerticalInterpHistPdf2 smoothAlgo < 0)
                return np.where((a > 0) & (pp.nom > 0),
                                np.log(np.where(a > 0, a, 1.0) / np.where(pp.nom > 0, pp.nom, 1.0)), 0.0)
            for s in shape.systs:
                up = np.asarray(s.up, dtype=float)
                down = np.asarray(s.down, dtype=float)
                if not (up.sum() > 0 and down.sum() > 0):
                    raise UnsupportedByBackend(f"zmodel: template systematic {s.param} of {ch.name}/{proc.name} has a "
                                               "non-positive integral (Combine refuses it)")
                if pp.logm:
                    dhi.append(logratio(up / up.sum()))
                    dlo.append(logratio(down / down.sum()))
                else:
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
            if ch.mcstats is not None:
                # CMSHistFunc (hist-mode 1): up/down rescaled to the nominal integral, i.e. the same
                # deltas of the normalised templates; floored at 1e-9 absolute, not renormalised
                pp.cmshist = True
                pp.integral = float(nom.sum())
        elif shape.kind == "parametric" and shape.pdf_systs:
            self._plan_pdf_morph(ch, pp, shape, data_kind)
        elif shape.kind == "parametric":
            if shape.contents and data_kind != "unbinned":
                pp.kind = "hist"
                pp.contents = np.asarray(shape.contents, dtype=float)
            else:
                pp.kind = "pdf"
                self._translate(ch, pp, shape)
        elif shape.kind == "envelope":
            self._plan_envelope(ch, pp, shape)
        else:
            raise UnsupportedByBackend(f"zmodel: shape kind '{shape.kind}' is not supported")
        return pp

    def _plan_pdf_morph(self, ch, pp, shape, data_kind):
        """Shape systematics on a pdf.  Fixed pdfs on binned data: from the IR histograms
        (semantics.hist_pdf_fractions and semantics.vertical_pdf_fractions); RooHistPdfs on
        unbinned data: the same fractions as a step density; any other pdf with floating
        parameters or on unbinned data: VerticalInterpPdf of the un-normalised RooFit pdfs
        (``vmorphf``, see _vmorph_float)."""
        if shape.pdf_morph != I.PDF_MORPH_HIST and (data_kind == "unbinned" or not shape.contents):
            self._plan_vmorph_float(ch, pp, shape)
            return
        if not shape.contents:
            raise UnsupportedByBackend(f"zmodel: RooHistPdf shape systematics on {ch.name}/{pp.name} without a fixed "
                                       "histogram")
        kinds = {s.kind for s in shape.pdf_systs}
        if len(kinds) != 1 or not kinds <= {"shape", "shapeN"}:
            raise UnsupportedByBackend(f"zmodel: pdf {ch.name}/{pp.name} mixes morphing algorithms {sorted(kinds)}")
        if kinds == {"shapeN"} and shape.pdf_morph != I.PDF_MORPH_HIST:
            raise UnsupportedByBackend(f"zmodel: shapeN on the pdf {ch.name}/{pp.name} (VerticalInterpPdf with "
                                       "smoothAlgo -1, normalised by a numerical integral) is not supported")
        pp.syst_idx = [self.index[s.param] for s in shape.pdf_systs]
        pp.scales = np.array([float(s.scale) for s in shape.pdf_systs])
        pp.vsmooth = min([1.0] + [float(s.scale) for s in shape.pdf_systs])
        pp.widths = np.diff(np.asarray(ch.observable.edges, dtype=float))
        if shape.pdf_morph == I.PDF_MORPH_HIST:
            pp.kind = "hmorph"
            pp.logm = kinds == {"shapeN"}

            def unit(a):  # Combine samples the RooHistPdfs into TH1F
                a = np.asarray(a, dtype=np.float32).astype(float)
                if not a.sum() > 0:
                    raise UnsupportedByBackend(f"zmodel: RooHistPdf histogram of {ch.name}/{pp.name} has no content")
                return a / a.sum()

            def logratio(a, ref):
                return np.where((a > 0) & (ref > 0), np.log(np.where(a > 0, a, 1.0) / np.where(ref > 0, ref, 1.0)), 0.0)

            pp.nom = unit(shape.contents)
            ups = [unit(s.up_contents) for s in shape.pdf_systs]
            downs = [unit(s.down_contents) for s in shape.pdf_systs]
            if pp.logm:
                pp.lognom = np.where(pp.nom > 0, np.log(np.where(pp.nom > 0, pp.nom, 1.0)), -999.0)
                pp.dhi = np.array([logratio(u, pp.nom) for u in ups])
                pp.dlo = np.array([logratio(d, pp.nom) for d in downs])
            else:
                pp.dhi = np.array([u - pp.nom for u in ups])
                pp.dlo = np.array([d - pp.nom for d in downs])
            return
        pp.kind = "vmorph"
        rows, ints = [np.asarray(shape.contents, dtype=float) * shape.raw_integral], [shape.raw_integral]
        for s in shape.pdf_systs:
            rows += [np.asarray(s.up_contents, dtype=float) * s.up_integral,
                     np.asarray(s.down_contents, dtype=float) * s.down_integral]
            ints += [s.up_integral, s.down_integral]
        pp.raw = np.array(rows)
        pp.raw_int = np.array(ints)

    def _plan_vmorph_float(self, ch, pp, shape):
        from modelspec import rootinput as R

        if any(s.kind != "shape" for s in shape.pdf_systs):
            raise UnsupportedByBackend(f"zmodel: shapeN on the pdf {ch.name}/{pp.name} (VerticalInterpPdf with "
                                       "smoothAlgo -1, normalised by a numerical integral) is not supported")
        tr = self._translator(ch)
        refs = [shape.ref] + [r for s in shape.pdf_systs for r in (s.up, s.down)]
        for ref in refs:
            roo = R.get_workspace(ref.file, ref.workspace).pdf(ref.name)
            if not roo:
                raise KeyError(f"zmodel: pdf '{ref.name}' not found in {ref.file}:{ref.workspace}")
            pp.raws.append(tr.raw(roo))
            pp.raw_roos.append(roo)
        self.classes |= tr.classes
        pp.kind = "vmorphf"
        pp.syst_idx = [self.index[s.param] for s in shape.pdf_systs]
        pp.scales = np.array([float(s.scale) for s in shape.pdf_systs])
        pp.vsmooth = min([1.0] + [float(s.scale) for s in shape.pdf_systs])
        pp.widths = np.diff(np.asarray(ch.observable.edges, dtype=float))
        pp.raw_idx = {n: self.index[n] for r in pp.raws for n in r.leaves}
        pp.roo = pp.raw_roos[0]

    def _vcoef(self, pp: _Proc, v):
        """VerticalInterpPdf weights (a_0, a_up_1, a_dn_1, ...) of (nominal, up_k, down_k)
        (semantics.vertical_pdf_coefficients)."""
        import zfit.z.numpy as znp

        x = self.tf.gather(v, pp.syst_idx) * pp.scales
        q = pp.vsmooth
        inside = znp.abs(x) < q
        c_cen = znp.where(inside, -x * x / q, znp.where(x > 0, -x, x))
        c_up = znp.where(inside, x * (q + x) / (2.0 * q), znp.where(x > 0, x, znp.zeros_like(x)))
        c_dn = znp.where(inside, -x * (q - x) / (2.0 * q), znp.where(x > 0, znp.zeros_like(x), -x))
        return znp.concatenate([znp.reshape(1.0 + znp.sum(c_cen), [1]),
                                znp.reshape(znp.stack([c_up, c_dn], axis=1), [-1])])

    def _vmorph_float(self, ch: _Chan, pp: _Proc, v, what, x=None):
        """VerticalInterpPdf of pdfs with floating parameters (docs/statistics.md): F = sum_j a_j
        f_j of the un-normalised values, N = sum_j a_j I_j of their RooFit integrals, F <= 0 ->
        1e-15, N <= 0 -> 1e-10.  ``what``: "density" (F/N at ``x``), "center" (F/N at the bin
        centres times the width) or "integral" (sum_j a_j B_ij / N, B_ij the raw bin integrals;
        N <= 0 -> 0, result <= 0 -> 1e-10: the range version of analyticalIntegralWN)."""
        import zfit.z.numpy as znp

        P = {n: v[i] for n, i in pp.raw_idx.items()}
        a = self._vcoef(pp, v)
        den = znp.sum(a * self.tf.stack([r.integral(ch.lo, ch.hi, P) for r in pp.raws]))
        if what == "integral":
            e = ch.edges
            num = znp.sum(a[:, None] * self.tf.stack([r.integral(e[:-1], e[1:], P) for r in pp.raws]), axis=0)
            ratio = znp.where(den > 0, num / znp.where(den > 0, den, 1.0), znp.zeros_like(num))
            return znp.where(ratio > 0, ratio, S.PDF_INTEGRAL_FLOOR)
        pts = znp.asarray(0.5 * (ch.edges[:-1] + ch.edges[1:])) if what == "center" else x
        num = znp.sum(a[:, None] * self.tf.stack([r.value(pts, P) for r in pp.raws]), axis=0)
        dens = znp.where(num > 0, num, S.PDF_FLOOR) / znp.where(den > 0, den, S.PDF_INTEGRAL_FLOOR)
        return dens * znp.asarray(pp.widths) if what == "center" else dens

    def _pdf_morph_fractions(self, ch: _Chan, pp: _Proc, v):
        """Bin fractions of a pdf with shape systematics (see _plan_pdf_morph)."""
        import zfit.z.numpy as znp

        tf = self.tf
        if pp.kind == "vmorphf":
            return self._vmorph_float(ch, pp, v, "integral" if ch.integration == "integral" else "center")
        x = tf.gather(v, pp.syst_idx) * pp.scales
        w = znp.asarray(pp.widths)
        if pp.kind == "hmorph":
            sm = _smooth_step(x, pp.vsmooth)
            dhi, dlo = znp.asarray(pp.dhi), znp.asarray(pp.dlo)
            delta = znp.sum(0.5 * x[:, None] * ((dhi - dlo) + (dhi + dlo) * sm[:, None]), axis=0)
            if pp.logm:  # shapeN: log space, no crop
                t = znp.exp(znp.asarray(pp.lognom) + delta)
            else:
                t = znp.asarray(pp.nom) + delta
                t = znp.where(t / w < 1e-9, 1e-9 * w, t)
            return t / znp.sum(t)
        coef = self._vcoef(pp, v)
        num = znp.sum(coef[:, None] * znp.asarray(pp.raw), axis=0)
        den = znp.sum(coef * znp.asarray(pp.raw_int))
        if ch.integration == "integral":
            ratio = znp.where(den > 0, num / den, znp.zeros_like(num))
            return znp.where(ratio > 0, ratio, S.PDF_INTEGRAL_FLOOR)
        value = num / w
        value = znp.where(value > 0, value, S.PDF_FLOOR)
        return value / znp.where(den > 0, den, S.PDF_INTEGRAL_FLOOR) * w

    def _translate(self, ch, pp, shape):
        from modelspec import rootinput as R

        ws = R.get_workspace(shape.ref.file, shape.ref.workspace)
        roo = ws.pdf(shape.ref.name)
        if not roo:
            raise KeyError(f"zmodel: pdf '{shape.ref.name}' not found in {shape.ref.file}:{shape.ref.workspace}")
        self._translate_roo(ch, pp, roo)

    def _translate_roo(self, ch, pp, roo):
        tr = self._translator(ch)
        pp.pdf = tr.pdf(roo)
        self.classes |= tr.classes
        pp.roo = roo
        pp.roo_obs = roo.getVariables().find(ch.observable.name)
        if not pp.roo_obs:
            raise UnsupportedByBackend(f"zmodel: pdf '{roo.GetName()}' does not depend on observable {ch.observable.name}")
        pp.obs_label = tr.space.obs[0]

    def _plan_envelope(self, ch, pp, shape):
        """RooMultiPdf: every component translated into its own zfit pdf ("pdf" states); the
        graph evaluates all of them and selects the one of the category value.  The penalty per
        state is RooMultiPdf::getCorrection() (see discrete_penalty)."""
        from modelspec import rootinput as R

        R_ = self._root()
        if not hasattr(R_, "RooMultiPdf"):
            raise UnsupportedByBackend("zmodel: envelopes need RooMultiPdf from the Combine library "
                                       "(set PYMODEL_ROOT_LIBS=libHiggsAnalysisCombinedLimit.so)")
        ws = R.get_workspace(shape.ref.file, shape.ref.workspace)
        multi = ws.pdf(shape.ref.name)
        if not multi or not multi.InheritsFrom("RooMultiPdf"):
            raise UnsupportedByBackend(f"zmodel: envelope '{shape.ref.name}' is not a RooMultiPdf in "
                                       f"{shape.ref.file}:{shape.ref.workspace}")
        cat = ws.cat(shape.category)
        if not cat:
            raise UnsupportedByBackend(f"zmodel: no category '{shape.category}' for {ch.name}/{pp.name}")
        par = self.model.parameters.get(shape.category)
        if par is None or par.role != I.ROLE_DISCRETE:
            raise UnsupportedByBackend(f"zmodel: category '{shape.category}' is not a discrete model parameter")
        n = int(multi.getNumPdfs())
        if cat.numTypes() != n or par.n_states != n:
            raise UnsupportedByBackend(f"zmodel: envelope {shape.ref.name} has {n} pdfs, its category "
                                       f"{cat.numTypes()} states and the IR {par.n_states}")
        pp.kind = "envelope"
        pp.cat_index = self.index[shape.category]
        floating = {p.name for p in self.model.parameters.values() if p.floating}
        old = cat.getCurrentIndex()
        try:
            for i in range(n):
                cat.setIndex(i)
                pp.corrections.append(float(multi.getCorrection()))
                st = _Proc(name=f"{pp.name}[{i}]", is_signal=pp.is_signal, rate=pp.rate, kind="pdf")
                self._translate_roo(ch, st, multi.getPdf(i))
                pp.states.append(st)
                pp.state_deps.append({v.GetName() for v in st.roo.getVariables()} & floating)
        finally:
            cat.setIndex(old)
        self.classes.add("RooMultiPdf")

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
        if pp.cmshist:
            floor = S.CMSHIST_FLOOR / pp.integral
            if pp.syst_idx:
                theta = tf.gather(v, pp.syst_idx)
                x = theta * pp.scales
                sm = _smooth_step(x, pp.vsmooth)
                dhi, dlo = znp.asarray(pp.dhi), znp.asarray(pp.dlo)
                t = t + znp.sum(0.5 * x[:, None] * ((dhi - dlo) + (dhi + dlo) * sm[:, None]), axis=0)
            t = znp.where(t < floor, floor, t)
            for i, klo, khi in pp.kappas:
                norm = norm * _asym_pow(v[i], klo, khi)
            return norm, t
        if pp.syst_idx:
            theta = tf.gather(v, pp.syst_idx)
            x = theta * pp.scales
            sm = _smooth_step(x, pp.vsmooth)
            dhi, dlo = znp.asarray(pp.dhi), znp.asarray(pp.dlo)
            delta = znp.sum(0.5 * x[:, None] * ((dhi - dlo) + (dhi + dlo) * sm[:, None]), axis=0)
            if pp.logm:  # shapeN: the same morph of the log-ratios, exponentiated; no floor
                t = znp.exp(znp.asarray(pp.lognom) + delta)
            else:
                t = znp.where(t + delta <= 0, 1e-9, t + delta)
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
        if pp.kind in ("hmorph", "vmorph"):
            return y * self._pdf_morph_fractions(ch, pp, v), y
        if pp.kind == "vmorphf":
            vals = y * self._pdf_morph_fractions(ch, pp, v)
            return vals, (znp.sum(vals) if ch.integration == "integral" else y)
        if ch.integration == "integral":
            vals = y * self._integral_fractions(ch, pp, v)
            return vals, znp.sum(vals)
        return y * self._center_fractions(ch, pp, v), y

    def _select(self, pp: _Proc, v, per_state):
        """The entry of ``per_state`` (one tensor per envelope state) selected by the category
        value (validated to be an integer state in nll_main / expected_by_process)."""
        tf = self.tf
        k = tf.clip_by_value(tf.cast(tf.round(v[pp.cat_index]), tf.int32), 0, len(per_state) - 1)
        return tf.gather(tf.stack(per_state), k)

    def _integral_fractions(self, ch, pp, v):
        from stat_backends.zmodel.roofit import bin_integrals

        if pp.kind == "envelope":
            return self._select(pp, v, [bin_integrals(st.pdf, st.obs_label, ch.edges) for st in pp.states])
        return bin_integrals(pp.pdf, pp.obs_label, ch.edges)

    def _center_fractions(self, ch, pp, v=None):
        import zfit.z.numpy as znp

        if pp.kind == "envelope":
            return self._select(pp, v, [self._center_fractions(ch, st) for st in pp.states])
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
        if pp.kind == "envelope":
            return y, self._select(pp, v, [znp.reshape(st.pdf.pdf(x[:, None]), [-1]) for st in pp.states])
        if pp.kind == "vmorphf":
            return y, self._vmorph_float(ch, pp, v, "density", x)
        if pp.kind == "hmorph":  # step density of the morphed bin fractions
            nb = len(ch.edges) - 1
            idx = tf.clip_by_value(tf.searchsorted(znp.asarray(ch.edges), x, side="right") - 1, 0, nb - 1)
            return y, tf.gather(self._pdf_morph_fractions(ch, pp, v) / znp.asarray(np.diff(ch.edges)), idx)
        raise UnsupportedByBackend(f"zmodel: process {pp.name} of kind {pp.kind} in unbinned channel {ch.name}")

    def _assign(self, v):
        for name, p in self._zparams.items():
            p.assign(v[self.index[name]])

    def _expected_graph(self, v):
        import zfit.z.numpy as znp

        self._assign(v)
        out = []
        for ch in self._chans:
            if ch.mcstats is not None:
                out.extend(self.tf.unstack(self._mcstats_values(ch, v)[0], num=len(ch.procs), axis=0))
                continue
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
                elif pp.kind == "hmorph":
                    raw.append(y * self._pdf_morph_fractions(ch, pp, v))
                elif pp.kind == "vmorphf":
                    raw.append(y * self._vmorph_float(ch, pp, v, "center"))
                else:
                    raw.append(y * self._center_fractions(ch, pp, v))
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
                if ch.mcstats is not None:
                    mu = self._mcstats_values(ch, v)[1]  # floored at 1e-9: > 0
                    total = total + znp.sum(mu) - znp.sum(n * znp.log(mu))
                    continue
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

    def _check_states(self, values):
        for i, n in self._discrete:
            k = int(round(values[i]))
            if k < 0 or k >= n or abs(values[i] - k) > 1e-9:
                raise ValueError(f"zmodel: invalid state {values[i]} for category {self.names[i]}")

    def nll_main(self, values, native) -> float:
        values = np.asarray(values, dtype=float)
        self._check_states(values)
        out = float(self._nll_fn(self.tf.constant(values), *native))
        if math.isnan(out):
            raise RuntimeError("zmodel: the NLL is NaN at " + str(self.values_dict(values)) + "; a RooGenericPdf/"
                               "RooLandauCB integral needs more RooIntegrator1D stages than roofit.ROO_INT_STAGES, "
                               "or a pdf is undefined there")
        return out

    def expected_by_process(self, values):
        values = np.asarray(values, dtype=float)
        self._check_states(values)
        arrs = self._exp_fn(self.tf.constant(values))
        out, k = {}, 0
        for ch in self._chans:
            out[ch.name] = {}
            for pp in ch.procs:
                out[ch.name][pp.name] = np.asarray(arrs[k], dtype=float)
                k += 1
        return out

    def density(self, values, channel, x):
        """Total intensity sum_p nu_p f_p(x) of an unbinned channel."""
        self._check_states(np.asarray(values, dtype=float))
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

    # ----- envelopes ------------------------------------------------------------------------
    def _plan_envelopes(self):
        """Envelope bookkeeping: (category index, number of states) of every discrete parameter,
        the envelopes, and the floating parameters used outside the envelope components."""
        self._discrete = [(self.index[n], p.n_states) for n, p in self.model.parameters.items()
                          if p.role == I.ROLE_DISCRETE]
        self._envelopes = [pp for ch in self._chans for pp in ch.procs if pp.kind == "envelope"]
        static = {self.poi}
        for ch in self.model.channels:
            if ch.mcstats is not None:
                static |= {bp.param for bp in ch.mcstats.params}
            for proc in ch.processes:
                for t in proc.norm_terms:
                    static |= set(t.args) | ({t.param} if t.param in self.index else set())
                static |= {s.param for s in proc.shape.systs} | {s.param for s in proc.shape.pdf_systs}
                if proc.shape.kind == "parametric":
                    static |= set(proc.shape.params)
        self._static_deps = static
        self._envelope_params = set().union(*[d for pp in self._envelopes for d in pp.state_deps]) \
            if self._envelopes else set()
        if self._envelopes:
            self.notes.append("zmodel: envelope penalty = RooMultiPdf::getCorrection() of the selected pdf, as added "
                              "by Combine (0.5 per non-constant variable of that pdf at construction, observable "
                              "included); all component pdfs are evaluated and the selected one is used")

    def discrete_penalty(self, values):
        return float(sum(pp.corrections[int(round(values[pp.cat_index]))] for pp in self._envelopes))

    def inactive_parameters(self, values):
        if not self._envelopes:
            return []
        active = set(self._static_deps)
        for pp in self._envelopes:
            active |= pp.state_deps[int(round(values[pp.cat_index]))]
        return sorted(self._envelope_params - active)

    # ----- validation against RooFit ---------------------------------------------------------
    def _check_translations(self):
        """Compare every translated pdf with RooFit at the IR nominal values: normalised
        densities at 50 points and the bin fractions of the channel's binning, against RooFit
        with its default integrator configuration (Combine's; ``roofit.roofit_reference``).
        Where that differs from a precise integral, a note gives the size."""
        from stat_backends.zmodel.roofit import bin_integrals, roofit_reference

        v = self.nominal_values()
        self._assign_eager(v)
        for ch in self._chans:
            for pp in [st for q in ch.procs for st in ([q] if q.kind == "pdf" else q.states)]:
                if pp.kind != "pdf":
                    continue
                saved = self._set_roofit(pp.roo, v)
                method = "integral" if ch.kind != "unbinned" and ch.integration == "integral" else "center"
                try:
                    xs = np.linspace(ch.lo, ch.hi, 52)[1:-1]
                    d_r, b_r, d_prec, _ = roofit_reference(self._root(), pp.roo, pp.roo_obs.GetName(), xs,
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
                dev_precise = float(np.max(np.abs(d_prec - d_r)) / np.max(np.abs(d_r)))
                rec = {"channel": ch.name, "process": pp.name, "pdf": pp.roo.GetName(), "class": pp.roo.ClassName(),
                       "density_max_rel_dev": dev_d, "bin_fraction_max_rel_dev": dev_b, "bin_method": method,
                       "roofit_default_vs_precise_dev": dev_precise}
                self.translation_checks.append(rec)
                self.notes.append(f"zmodel: {ch.name}/{pp.name} ({pp.roo.ClassName()} {pp.roo.GetName()}) agrees with "
                                  f"RooFit: densities {dev_d:.1e}, bin fractions {dev_b:.1e} (max rel. dev.)")
                if dev_precise > 1e-9:
                    self.notes.append(f"zmodel: RooFit's default numerical normalisation of {pp.roo.GetName()} (used by "
                                      f"Combine, reproduced by zmodel) differs from a precise integral by "
                                      f"{dev_precise:.1e} (relative) at the nominal values")
                if not (dev_d <= CHECK_TOLERANCE and dev_b <= CHECK_TOLERANCE):
                    raise RuntimeError(f"zmodel: translated pdf {pp.roo.GetName()} ({pp.roo.ClassName()}) disagrees "
                                       f"with RooFit: density dev {dev_d:.3g}, bin fraction dev {dev_b:.3g}")

    def _check_raw_translations(self):
        """Un-normalised values (50 points) and integrals over the observable range of every pdf
        of a ``vmorphf`` morph against RooFit's getVal() / createIntegral() at the nominal values."""
        from stat_backends.zmodel.roofit import roofit_raw_values

        R_ = self._root()
        v = self.nominal_values()
        for ch in self._chans:
            for pp in ch.procs:
                if pp.kind != "vmorphf":
                    continue
                P = {n: self.tf.constant(v[i], self.tf.float64) for n, i in pp.raw_idx.items()}
                xs = np.linspace(ch.lo, ch.hi, 52)[1:-1]
                for raw, roo in zip(pp.raws, pp.raw_roos):
                    saved = self._set_roofit(roo, v)
                    try:
                        obs = roo.getVariables().find(self.model.channel(ch.name).observable.name)
                        if not obs:
                            raise UnsupportedByBackend(f"zmodel: pdf '{roo.GetName()}' does not depend on the "
                                                       f"observable of channel {ch.name}")
                        f_r = roofit_raw_values(R_, roo, obs, xs)
                        i_r = float(roo.createIntegral(R_.RooArgSet(obs)).getVal())
                    finally:
                        for var, old in saved:
                            var.setVal(old)
                    f_z = np.asarray(raw.value(self.tf.constant(xs), P), dtype=float).ravel()
                    i_z = float(raw.integral(ch.lo, ch.hi, P))
                    dev_f = float(np.max(np.abs(f_z - f_r)) / np.max(np.abs(f_r)))
                    dev_i = abs(i_z - i_r) / abs(i_r)
                    self.translation_checks.append({"channel": ch.name, "process": pp.name, "pdf": roo.GetName(),
                                                    "class": roo.ClassName(), "raw_value_max_rel_dev": dev_f,
                                                    "raw_integral_rel_dev": dev_i})
                    self.notes.append(f"zmodel: {ch.name}/{pp.name} morph input {roo.ClassName()} {roo.GetName()}: "
                                      f"un-normalised values {dev_f:.1e}, integral {dev_i:.1e} vs RooFit")
                    if not (dev_f <= CHECK_TOLERANCE and dev_i <= CHECK_TOLERANCE):
                        raise RuntimeError(f"zmodel: un-normalised pdf {roo.GetName()} ({roo.ClassName()}) disagrees "
                                           f"with RooFit: values {dev_f:.3g}, integral {dev_i:.3g}")

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
