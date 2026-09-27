"""RooFit pdf trees -> zfit pdfs (explicit class map, nothing approximated).

``Translator.pdf(roo_pdf)`` walks a RooFit pdf and returns an equivalent zfit pdf over the
channel observable.  IR parameters become shared ``zfit.Parameter`` objects (looked up by
name, exactly as Combine identifies nuisances with workspace variables); constant
``RooRealVar``/``RooConstVar`` leaves become python constants; functions of parameters
(``RooFormulaVar``, ``RooProduct``) become ``zfit.ComposedParameter``.  Any class not in
``PDF_CLASSES``/``FUNCTION_CLASSES`` raises ``UnsupportedByBackend`` naming it.

Class map (RooFit 6.32 definitions):

  RooGaussian     -> zfit.pdf.Gauss(mean, |sigma|)
  RooExponential  -> zfit.pdf.Exponential(lam = c, or -c with negateCoefficient)
  RooCBShape      -> zfit.pdf.CrystalBall(m0, sigma, alpha, n)
  RooCrystalBall  -> single-sided: zfit.pdf.CrystalBall(x0, |sigma|, alphaL, nL) (alphaL < 0: right tail);
                     double-sided: zfit.pdf.GeneralizedCB(x0, |sigmaL|, |alphaL|, nL, |sigmaR|, |alphaR|, nR)
  RooChebychev    -> zfit.pdf.Chebyshev(coeffs), T0 coefficient 1, x mapped to [-1, 1] over the range
  RooBernstein    -> zfit.pdf.Bernstein(coeffs)
  RooPolynomial   -> PolynomialPDF: [1 +] x^lowestOrder * sum c_i x^i (raw x), analytic integral
  RooUniform      -> zfit.pdf.Uniform(lo, hi)
  RooAddPdf       -> zfit.pdf.SumPDF with explicit fractions: N coefficients -> c_i / sum c;
                     N-1 -> c_i and 1 - sum c; recursive -> c_0, (1-c_0) c_1, ..., prod(1-c_i)
  RooHistPdf      -> HistStepPDF: interpolation order 0, density = bin content / bin width,
                     zero outside the histogram, analytic integral
  RooGenericPdf   -> FormulaPDF: the TFormula expression (formula.py), integral by
                     composite Gauss-Legendre quadrature (QUAD_PANELS x QUAD_NODES points)
  RooLandauCB     -> FormulaPDF with Combine's closed form (src/RooLandauCB.cc; Combine uses
                     vdt::fast_exp/fast_pow, which agree with exp/pow to ~1e-15), quadrature
                     integral as RooFit also integrates it numerically
  functions: RooRealVar, RooConstVar, RooFormulaVar, RooProduct (real components only),
             RooRecursiveFraction (l0 * prod_{i>0} (1 - l_i), what RooAddPdf builds for recursive fractions)
"""

import itertools
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

import numpy as np

from stat_backends.base import UnsupportedByBackend
from stat_backends.zmodel import formula as F

QUAD_PANELS = 256
QUAD_NODES = 16

PDF_CLASSES = ("RooGaussian", "RooExponential", "RooCBShape", "RooCrystalBall", "RooChebychev", "RooBernstein",
               "RooPolynomial", "RooUniform", "RooAddPdf", "RooHistPdf", "RooGenericPdf", "RooLandauCB")
FUNCTION_CLASSES = ("RooRealVar", "RooConstVar", "RooFormulaVar", "RooProduct", "RooRecursiveFraction")

_CPP_DONE = False
_UID = itertools.count()


def _cpp(R):
    global _CPP_DONE
    if _CPP_DONE:
        return
    ok = R.gInterpreter.Declare(r'''
#include "RooAbsArg.h"
#include "RooArgProxy.h"
#include "RooAbsCollection.h"
#include "RooAddPdf.h"
namespace pymodel_zmodel {
std::string proxy_name(RooAbsArg& a, int i) { auto p = a.getProxy(i); return p ? std::string(p->name()) : std::string(""); }
RooAbsArg* proxy_arg(RooAbsArg& a, int i) { auto p = dynamic_cast<RooArgProxy*>(a.getProxy(i)); return p ? p->absArg() : nullptr; }
RooAbsCollection* proxy_list(RooAbsArg& a, int i) { return dynamic_cast<RooAbsCollection*>(a.getProxy(i)); }
struct AddPdfAccess : RooAddPdf { static bool recursive(const RooAddPdf& p) { return static_cast<const AddPdfAccess&>(p)._recursive; } };
bool addpdf_recursive(RooAddPdf& p) { return AddPdfAccess::recursive(p); }
}''')
    if not ok:
        raise RuntimeError("zmodel: could not declare the RooFit introspection helpers")
    _CPP_DONE = True


def proxies(R, arg) -> Dict[str, object]:
    """Proxy name -> RooAbsArg (single proxies) or list of RooAbsArg (list/set proxies)."""
    _cpp(R)
    out = {}
    for i in range(arg.numProxies()):
        name = str(R.pymodel_zmodel.proxy_name(arg, i))
        a = R.pymodel_zmodel.proxy_arg(arg, i)
        if a:
            out[name] = a
            continue
        lst = R.pymodel_zmodel.proxy_list(arg, i)
        if lst:
            out[name] = [e for e in lst]
    return out


# ----------------------------------------------------------------------------------------
# quadrature and custom zfit pdfs
# ----------------------------------------------------------------------------------------

_GL_T, _GL_W = np.polynomial.legendre.leggauss(QUAD_NODES)
_Q_T = ((np.arange(QUAD_PANELS)[:, None] + 0.5 * (_GL_T[None, :] + 1.0)) / QUAD_PANELS).ravel()  # in [0, 1]
_Q_W = np.tile(0.5 * _GL_W / QUAD_PANELS, QUAD_PANELS)


def quad(func, lower, upper):
    """Composite Gauss-Legendre integral of the vectorised ``func`` over [lower, upper]."""
    import zfit.z.numpy as znp

    xs = lower + (upper - lower) * znp.asarray(_Q_T)
    return (upper - lower) * znp.sum(func(xs) * znp.asarray(_Q_W))


def _limits(limits):
    import zfit.z.numpy as znp

    lower, upper = limits._rect_limits_tf
    return znp.reshape(lower, [-1])[0], znp.reshape(upper, [-1])[0]


_PDF_TYPES = {}


def _pdf_types():
    """Custom zfit pdf classes (created lazily so that importing this module needs no zfit)."""
    if _PDF_TYPES:
        return _PDF_TYPES
    import zfit
    import zfit.z.numpy as znp

    ANY = zfit.Space(axes=0, limits=(zfit.Space.ANY_LOWER, zfit.Space.ANY_UPPER))

    class FormulaPDF(zfit.pdf.BasePDF):
        """Unnormalised density ``func(x, params)``; normalised by quadrature."""

        _N_OBS = 1

        def __init__(self, obs, params, func, name="FormulaPDF"):
            self._func = func
            super().__init__(obs=obs, params=params, name=name)

        @zfit.supports()
        def _unnormalized_pdf(self, x, params):
            return self._func(x.unstack_x(), params)

    def _formula_integral(limits, params, model):
        lower, upper = _limits(limits)
        return quad(lambda xs: model._func(xs, params), lower, upper)

    FormulaPDF.register_analytic_integral(func=_formula_integral, limits=ANY)

    class PolynomialPDF(zfit.pdf.BasePDF):
        """RooPolynomial: ``[1 +] x^k * sum_i c_i x^i`` with k = lowestOrder (the 1 only if k > 0)."""

        _N_OBS = 1

        def __init__(self, obs, coeffs, lowest_order, name="PolynomialPDF"):
            self._n = len(coeffs)
            self._k = int(lowest_order)
            super().__init__(obs=obs, params={f"c{i}": c for i, c in enumerate(coeffs)}, name=name)

        def _coeffs(self, params):
            return [params[f"c{i}"] for i in range(self._n)]

        @zfit.supports()
        def _unnormalized_pdf(self, x, params):
            x = x.unstack_x()
            val = znp.zeros_like(x)
            for i, c in enumerate(self._coeffs(params)):
                val = val + c * x ** (self._k + i)
            return val + (1.0 if self._k > 0 else 0.0)

    def _poly_integral(limits, params, model):
        lo, hi = _limits(limits)
        total = (hi - lo) if model._k > 0 else 0.0 * lo
        for i, c in enumerate(model._coeffs(params)):
            p = model._k + i + 1
            total = total + c * (hi ** p - lo ** p) / p
        return total

    PolynomialPDF.register_analytic_integral(func=_poly_integral, limits=ANY)

    class HistStepPDF(zfit.pdf.BasePDF):
        """Histogram pdf with interpolation order 0 (RooHistPdf): density content/width in each
        bin, zero outside the histogram."""

        _N_OBS = 1

        def __init__(self, obs, edges, contents, name="HistStepPDF"):
            self._edges = np.asarray(edges, dtype=float)
            self._dens = np.asarray(contents, dtype=float) / np.diff(self._edges)
            super().__init__(obs=obs, params={}, name=name)

        @zfit.supports()
        def _unnormalized_pdf(self, x, params):
            import tensorflow as tf

            x = x.unstack_x()
            e = znp.asarray(self._edges)
            idx = tf.searchsorted(e, x, side="right") - 1
            inside = (x >= self._edges[0]) & (x < self._edges[-1])
            idx = tf.clip_by_value(idx, 0, len(self._dens) - 1)
            return znp.where(inside, tf.gather(znp.asarray(self._dens), idx), znp.zeros_like(x))

    def _hist_integral(limits, params, model):
        lo, hi = _limits(limits)
        e = znp.asarray(model._edges)
        overlap = znp.maximum(znp.minimum(e[1:], hi) - znp.maximum(e[:-1], lo), 0.0)
        return znp.sum(overlap * znp.asarray(model._dens))

    HistStepPDF.register_analytic_integral(func=_hist_integral, limits=ANY)

    _PDF_TYPES.update(FormulaPDF=FormulaPDF, PolynomialPDF=PolynomialPDF, HistStepPDF=HistStepPDF)
    return _PDF_TYPES


# ----------------------------------------------------------------------------------------
# symbolic real-valued functions
# ----------------------------------------------------------------------------------------

@dataclass
class Sym:
    """A RooAbsReal as ``fn(x, P)`` where x is the observable tensor and P maps leaf parameter
    names to values; ``leaves`` are the IR parameters it depends on."""

    fn: Callable
    leaves: Set[str] = field(default_factory=set)
    depends_on_x: bool = False
    constant: Optional[float] = None  # set when it has no leaves and does not depend on x
    leaf: Optional[str] = None        # set when it is exactly one IR parameter


def landau_cb(x, mean, a, b, alpha1, n1, alpha2, n2):
    """Combine RooLandauCB::evaluate (src/RooLandauCB.cc), vectorised with znp."""
    import zfit.z.numpy as znp

    t = x - mean
    core = znp.exp(a * (b * t - znp.exp(b * t)))
    B1 = -alpha1 + n1 / (a * b * (1.0 - znp.exp(-b * alpha1)))
    A1 = znp.exp(a * (-b * alpha1 - znp.exp(-b * alpha1))) * znp.power(B1 + alpha1, n1)
    B2 = -alpha2 - n2 / (a * b * (1.0 - znp.exp(b * alpha2)))
    A2 = znp.exp(a * (b * alpha2 - znp.exp(b * alpha2))) * znp.power(B2 + alpha2, n2)
    left_t = znp.minimum(t, -alpha1)   # keep the unused branches finite
    right_t = znp.maximum(t, alpha2)
    left = A1 * znp.power(B1 - left_t, -n1)
    right = A2 * znp.power(B2 + right_t, -n2)
    return znp.where(t <= -alpha1, left, znp.where(t >= alpha2, right, core))


class Translator:
    """Translate RooFit objects of one channel into zfit objects.

    ``zparams``: IR parameter name -> zfit.Parameter (shared by all channels).
    ``ir_names``: all IR parameter names.
    """

    def __init__(self, R, obs_name: str, lo: float, hi: float, zparams, ir_names, make_param):
        import zfit

        self.R = R
        _cpp(R)
        self.obs_name = obs_name
        self.lo, self.hi = float(lo), float(hi)
        self.space = zfit.Space(f"zm_{obs_name}", limits=(self.lo, self.hi))
        self.zparams = zparams
        self.ir_names = set(ir_names)
        self.make_param = make_param  # name -> zfit.Parameter (creates it on first use)
        self.classes: Set[str] = set()
        self._pdfs = {}
        self._uid = next(_UID)

    def _name(self, base):
        return f"zm{self._uid}_{base}_{next(_UID)}"

    # -- functions ----------------------------------------------------------------------
    def sym(self, a) -> Sym:
        cls = a.ClassName()
        name = a.GetName()
        if a.InheritsFrom("RooRealVar") and cls == "RooRealVar":
            if name == self.obs_name:
                return Sym(fn=lambda x, P: x, depends_on_x=True)
            if name in self.ir_names:
                return Sym(fn=lambda x, P, n=name: P[n], leaves={name}, leaf=name)
            if not a.isConstant():
                raise UnsupportedByBackend(f"zmodel: floating RooRealVar '{name}' is not a model parameter")
            v = float(a.getVal())
            return Sym(fn=lambda x, P, v=v: v, constant=v)
        if cls == "RooConstVar":
            v = float(a.getVal())
            return Sym(fn=lambda x, P, v=v: v, constant=v)
        if cls == "RooFormulaVar":
            self.classes.add(cls)
            deps = [d for d in a.dependents()]
            return self._formula(str(a.expression()), deps, f"RooFormulaVar '{name}'")
        if cls == "RooProduct":
            self.classes.add(cls)
            if a.categorialComponents().getSize():
                raise UnsupportedByBackend(f"zmodel: RooProduct '{name}' has category components")
            parts = [self.sym(c) for c in a.realComponents()]

            def fn(x, P, parts=parts):
                val = parts[0].fn(x, P)
                for p in parts[1:]:
                    val = val * p.fn(x, P)
                return val
            return self._combine(fn, parts)
        if cls == "RooRecursiveFraction":
            self.classes.add(cls)
            lists = [v for v in proxies(self.R, a).values() if isinstance(v, list)]
            if len(lists) != 1 or not lists[0]:
                raise UnsupportedByBackend(f"zmodel: cannot read the fraction list of {cls} '{name}'")
            parts = [self.sym(c) for c in lists[0]]

            def fn(x, P, parts=parts):
                val = parts[0].fn(x, P)
                for p in parts[1:]:
                    val = val * (1.0 - p.fn(x, P))
                return val
            return self._combine(fn, parts)
        raise UnsupportedByBackend(f"zmodel: RooFit function class {cls} ('{name}') is not supported "
                                   f"(supported: {', '.join(FUNCTION_CLASSES)})")

    def _combine(self, fn, parts: List[Sym]) -> Sym:
        leaves = set().union(*[p.leaves for p in parts]) if parts else set()
        depx = any(p.depends_on_x for p in parts)
        s = Sym(fn=fn, leaves=leaves, depends_on_x=depx)
        if not leaves and not depx:
            s.constant = float(self._eval_const(fn))
        return s

    @staticmethod
    def _eval_const(fn):
        return np.asarray(fn(None, {}), dtype=float)

    def _formula(self, expr: str, deps, what: str) -> Sym:
        names = [d.GetName() for d in deps]
        try:
            node = F.parse(expr, names)
        except F.FormulaError as exc:
            raise UnsupportedByBackend(f"zmodel: {what}: {exc}") from exc
        used = F.used_refs(node)
        parts = [self.sym(d) if i in used else Sym(fn=lambda x, P: 0.0, constant=0.0) for i, d in enumerate(deps)]
        tf_eval = F.compile_formula(node, F.TFOps())

        def fn(x, P, parts=parts):
            return tf_eval([p.fn(x, P) for p in parts])
        return self._combine(fn, [parts[i] for i in used])

    def real(self, a, what: str, absolute: bool = False):
        """A pdf argument as a zfit parameter or constant (must not depend on the observable)."""
        s = self.sym(a)
        if s.depends_on_x:
            raise UnsupportedByBackend(f"zmodel: {what} ('{a.GetName()}') depends on the observable; only "
                                       "RooGenericPdf may use functions of the observable")
        if s.constant is not None:
            return abs(s.constant) if absolute else s.constant
        return self.composed(s, a.GetName(), absolute)

    def composed(self, s: Sym, base: str, absolute: bool = False):
        import zfit
        import zfit.z.numpy as znp

        if s.leaf is not None and not absolute:
            return self.make_param(s.leaf)
        params = {n: self.make_param(n) for n in sorted(s.leaves)}
        fn = s.fn

        def func(params, fn=fn, absolute=absolute):
            val = fn(None, params)
            return znp.abs(val) if absolute else val
        return zfit.ComposedParameter(self._name(base), func, params=params)

    def _obs_arg(self, a, what):
        if a.GetName() != self.obs_name:
            raise UnsupportedByBackend(f"zmodel: {what}: the observable slot holds '{a.GetName()}', "
                                       f"not the channel observable '{self.obs_name}'")

    # -- pdfs ---------------------------------------------------------------------------
    def pdf(self, p):
        name = p.GetName()
        if name not in self._pdfs:
            self._pdfs[name] = self._pdf(p)
        return self._pdfs[name]

    def _pdf(self, p):
        import zfit

        R = self.R
        cls = p.ClassName()
        name = p.GetName()
        if cls not in PDF_CLASSES:
            raise UnsupportedByBackend(f"zmodel: RooFit pdf class {cls} ('{name}') is not supported "
                                       f"(supported: {', '.join(PDF_CLASSES)})")
        self.classes.add(cls)
        obs = self.space
        what = f"{cls} '{name}'"
        if cls == "RooGaussian":
            x, mean = p.getX(), p.getMean()
            if x.GetName() != self.obs_name and mean.GetName() == self.obs_name:
                x, mean = mean, x  # symmetric in x and mean
            self._obs_arg(x, what)
            return zfit.pdf.Gauss(mu=self.real(mean, f"{what} mean"),
                                  sigma=self.real(p.getSigma(), f"{what} sigma", absolute=True), obs=obs,
                                  name=self._name("gauss"))
        if cls == "RooExponential":
            self._obs_arg(p.variable(), what)
            c = self.sym(p.coefficient())
            if c.depends_on_x:
                raise UnsupportedByBackend(f"zmodel: {what} coefficient depends on the observable")
            if p.negateCoefficient():
                fn = c.fn
                c = Sym(fn=lambda x, P, fn=fn: -fn(x, P), leaves=c.leaves,
                        constant=None if c.constant is None else -c.constant)
            lam = c.constant if c.constant is not None else self.composed(c, p.coefficient().GetName())
            return zfit.pdf.Exponential(lam=lam, obs=obs, name=self._name("exp"))
        if cls == "RooCBShape":
            px = proxies(R, p)
            self._obs_arg(px["m"], what)
            return zfit.pdf.CrystalBall(mu=self.real(px["m0"], f"{what} m0"), sigma=self.real(px["sigma"], f"{what} sigma"),
                                        alpha=self.real(px["alpha"], f"{what} alpha"),
                                        n=self.real(px["n"], f"{what} n"), obs=obs, name=self._name("cbshape"))
        if cls == "RooCrystalBall":
            px = proxies(R, p)
            self._obs_arg(px["x"], what)
            x0 = self.real(px["x0"], f"{what} x0")
            if "alphaR" not in px:
                if px["sigmaL"].GetName() != px["sigmaR"].GetName():
                    raise UnsupportedByBackend(f"zmodel: single-sided {what} with different sigmaL/sigmaR")
                return zfit.pdf.CrystalBall(mu=x0, sigma=self.real(px["sigmaL"], f"{what} sigma", absolute=True),
                                            alpha=self.real(px["alphaL"], f"{what} alpha"),
                                            n=self.real(px["nL"], f"{what} n"), obs=obs, name=self._name("cb"))
            return zfit.pdf.GeneralizedCB(
                mu=x0, sigmal=self.real(px["sigmaL"], f"{what} sigmaL", absolute=True),
                alphal=self.real(px["alphaL"], f"{what} alphaL", absolute=True), nl=self.real(px["nL"], f"{what} nL"),
                sigmar=self.real(px["sigmaR"], f"{what} sigmaR", absolute=True),
                alphar=self.real(px["alphaR"], f"{what} alphaR", absolute=True), nr=self.real(px["nR"], f"{what} nR"),
                obs=obs, name=self._name("dcb"))
        if cls in ("RooChebychev", "RooBernstein"):
            px = proxies(R, p)
            self._obs_arg(px["x"], what)
            lists = [v for v in px.values() if isinstance(v, list)]  # proxy name differs between ctor and copy
            if len(lists) != 1:
                raise UnsupportedByBackend(f"zmodel: {what}: cannot identify the coefficient list")
            coeffs = [self.real(c, f"{what} coefficient") for c in lists[0]]
            if cls == "RooChebychev":
                if not coeffs:
                    return zfit.pdf.Uniform(low=self.lo, high=self.hi, obs=obs, name=self._name("uniform"))
                return zfit.pdf.Chebyshev(obs=obs, coeffs=coeffs, name=self._name("cheb"))
            return zfit.pdf.Bernstein(obs=obs, coeffs=coeffs, name=self._name("bern"))
        if cls == "RooPolynomial":
            self._obs_arg(p.x(), what)
            coeffs = [self.real(c, f"{what} coefficient") for c in p.coefList()]
            return _pdf_types()["PolynomialPDF"](obs, coeffs, p.lowestOrder(), name=self._name("poly"))
        if cls == "RooUniform":
            px = proxies(R, p)
            xs = px["x"]
            if len(xs) != 1:
                raise UnsupportedByBackend(f"zmodel: {what} has {len(xs)} observables")
            self._obs_arg(xs[0], what)
            return zfit.pdf.Uniform(low=self.lo, high=self.hi, obs=obs, name=self._name("uniform"))
        if cls == "RooAddPdf":
            return self._addpdf(p, what)
        if cls == "RooHistPdf":
            return self._histpdf(p, what)
        if cls == "RooGenericPdf":
            s = self._formula(str(p.expression()), [d for d in p.dependents()], what)
            return self._formula_pdf(s, "generic")
        if cls == "RooLandauCB":
            px = proxies(R, p)
            self._obs_arg(px["x"], what)
            parts = [self.sym(px[k]) for k in ("mean", "a", "b", "alpha1", "n1", "alpha2", "n2")]
            if any(q.depends_on_x for q in parts):
                raise UnsupportedByBackend(f"zmodel: {what} parameters depend on the observable")

            def fn(x, P, parts=parts):
                return landau_cb(x, *[q.fn(x, P) for q in parts])
            s = Sym(fn=fn, leaves=set().union(*[q.leaves for q in parts]), depends_on_x=True)
            return self._formula_pdf(s, "landaucb")
        raise AssertionError(cls)

    def _formula_pdf(self, s: Sym, base):
        import zfit.z.numpy as znp

        params = {n: self.make_param(n) for n in sorted(s.leaves)}
        fn = s.fn

        def func(x, P, fn=fn):
            return znp.asarray(fn(x, P), dtype=np.float64) * znp.ones_like(x)
        return _pdf_types()["FormulaPDF"](self.space, params, func, name=self._name(base))

    def _addpdf(self, p, what):
        import zfit

        if str(p.getCoefRange()):
            raise UnsupportedByBackend(f"zmodel: {what} has a fixed coefficient range '{p.getCoefRange()}'")
        pdfs = [self.pdf(q) for q in p.pdfList()]
        coefs = [self.sym(c) for c in p.coefList()]
        if any(c.depends_on_x for c in coefs):
            raise UnsupportedByBackend(f"zmodel: {what} coefficients depend on the observable")
        n, m = len(pdfs), len(coefs)
        recursive = bool(self.R.pymodel_zmodel.addpdf_recursive(p))
        if m == 0:
            raise UnsupportedByBackend(f"zmodel: {what} without coefficients (extended components) is not supported")
        if m not in (n, n - 1):
            raise UnsupportedByBackend(f"zmodel: {what} has {n} pdfs and {m} coefficients")
        fns = [c.fn for c in coefs]
        leaves = set().union(*[c.leaves for c in coefs])

        def frac(i):
            if m == n:  # yields (or ROOT's RooRecursiveFraction list, summing to 1) -> c_i / sum c
                return lambda x, P: fns[i](x, P) / sum(f(x, P) for f in fns)
            if recursive:
                def f(x, P):
                    val = 1.0
                    for k in range(min(i, m)):
                        val = val * (1.0 - fns[k](x, P))
                    return val * fns[i](x, P) if i < m else val
                return f
            if i < m:
                return fns[i]
            return lambda x, P: 1.0 - sum(f(x, P) for f in fns)

        fracs = []
        for i in range(n):
            s = Sym(fn=frac(i), leaves=leaves)
            if not leaves:
                fracs.append(float(self._eval_const(s.fn)))
            else:
                fracs.append(self.composed(s, f"frac{i}"))
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return zfit.pdf.SumPDF(pdfs, fracs=fracs, obs=self.space, name=self._name("sum"))

    def _histpdf(self, p, what):
        if p.getInterpolationOrder() != 0:
            raise UnsupportedByBackend(f"zmodel: {what} has interpolation order {p.getInterpolationOrder()}; "
                                       "only order 0 is supported")
        if p.haveUnitNorm():
            raise UnsupportedByBackend(f"zmodel: {what} uses unit normalisation")
        px = proxies(self.R, p)
        pobs = px.get("pdfObs", [])
        if len(pobs) != 1:
            raise UnsupportedByBackend(f"zmodel: {what} must have one observable")
        self._obs_arg(pobs[0], what)
        dh = p.dataHist()
        if dh.get().getSize() != 1:
            raise UnsupportedByBackend(f"zmodel: {what} histogram is not 1D")
        hvar = dh.get().first()
        binning = dh.getBinnings()[0]
        nb = binning.numBins()
        edges = [binning.binLow(i) for i in range(nb)] + [binning.binHigh(nb - 1)]
        contents = np.zeros(nb)
        for i in range(dh.numEntries()):
            row = dh.get(i)
            x = row.getRealValue(hvar.GetName())
            k = int(np.searchsorted(edges, x, side="right") - 1)
            contents[k] += dh.weight()
        if np.any(contents < 0):
            raise UnsupportedByBackend(f"zmodel: {what} has negative bin contents")
        return _pdf_types()["HistStepPDF"](self.space, edges, contents, name=self._name("hist"))


def bin_integrals(pdf, obs_name, edges):
    """Normalised integrals of ``pdf`` over the bins ``edges``: ``pdf.integrate`` of each bin,
    vectorised with tf.vectorized_map.

    (zfit's own BinnedFromUnbinnedPDF.rel_counts was found to give wrong bin contents, up to
    6e-4 for a step pdf and 1e-8 for crystal balls, so it is not used.)"""
    import tensorflow as tf
    import zfit

    lims = tf.constant(np.stack([np.asarray(edges[:-1], float), np.asarray(edges[1:], float)], axis=1))

    def one(lim):
        return tf.reshape(pdf.integrate(zfit.Space(obs_name, limits=(lim[0], lim[1]))), [-1])[0]
    return tf.vectorized_map(one, lims)


# ----------------------------------------------------------------------------------------
# numerical check against RooFit
# ----------------------------------------------------------------------------------------

def roofit_densities(R, pdf, obs, xs):
    """RooFit normalised densities of ``pdf`` at ``xs`` (current parameter values)."""
    nset = R.RooArgSet(obs)
    old = obs.getVal()
    out = []
    try:
        for x in xs:
            obs.setVal(float(x))
            out.append(pdf.getVal(nset))
    finally:
        obs.setVal(old)
    return np.array(out)


def roofit_reference(R, pdf, obs_name, xs, edges, method="center"):
    """RooFit reference values at the current parameter values, with every numerical
    integral done precisely (adaptive Gauss-Kronrod, eps 1e-12) on a fresh clone of the
    tree, so that RooFit's default integrator precision (~1e-7, worse for peaked pdfs) does
    not enter the comparison.  Returns (densities, bin fractions, densities with RooFit's
    default normalisation as used by Combine)."""
    from modelspec import rootinput

    obs = pdf.getVariables().find(obs_name)
    d_default = roofit_densities(R, pdf, obs, xs)
    cfg = R.RooAbsReal.defaultIntegratorConfig()
    saved = (cfg.epsAbs(), cfg.epsRel(), str(cfg.method1D().getCurrentLabel()))
    clone = pdf.cloneTree()
    try:
        cfg.setEpsAbs(1e-13)
        cfg.setEpsRel(1e-12)
        cfg.method1D().setLabel("RooAdaptiveGaussKronrodIntegrator1D")
        cobs = clone.getVariables().find(obs_name)
        dens = roofit_densities(R, clone, cobs, xs)
        fracs = np.asarray(rootinput.binned_pdf_contents(clone, cobs, list(edges), method))
    finally:
        cfg.setEpsAbs(saved[0])
        cfg.setEpsRel(saved[1])
        cfg.method1D().setLabel(saved[2])
    return dens, fracs, d_default
