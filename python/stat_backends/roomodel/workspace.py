"""ModelIR -> RooWorkspace.

One RooWorkspace holds the whole model.  Every IR parameter is one RooRealVar (or the
RooCategory of an envelope) with the IR name; workspace objects referenced by the IR are
imported with ``RecycleConflictNodes`` so that their own variables *are* the IR parameters.
Per channel c and process p:

* yield ``n_exp_<c>_<p>`` = RooProduct(rate, r if signal, norm terms), with
    lnN / lnU       exp(theta * log kappa)                         (= kappa**theta)
    asym_lnN        exp(theta * logKappaForX(theta))               (Combine asymPow)
    gmN             alpha * n   (the rate is replaced by 1)
    rate_param      the parameter
    formula         RooFormulaVar of the Combine expression (@0..@n = args)
    ws_norm         the imported <pdf>_norm function
    template shape  asymPow(theta, kappa_lo**scale, kappa_hi**scale) from the template
                    integrals (dropped when both |kappa - 1| < 1e-3, as in Combine)
* bin fractions f_pi:
    counting        1
    template        unit-normalised nominal, vertically morphed with coefficient
                    x = scale*theta and Combine's smooth step (region min(1, scales)):
                    ``shape``:  t_i = nom_i + sum_k x_k/2 (diff_ik + sum_ik S(x_k)), t_i <= 0 -> 1e-9
                    ``shapeN``: t_i = exp(log nom_i + sum_k x_k/2 (ldiff_ik + lsum_ik S(x_k)))
                    (log-ratios of the normalised templates, FastVerticalInterpHistPdf2
                    with smoothAlgo -1), then f_i = t_i / sum_j t_j
    parametric      pdf(bin centre) * width ("center") or the normalised bin integral
                    ("integral"); constants when the pdf depends on no IR parameter
    envelope        one state per RooMultiPdf component, selected by the category
    pdf shape systs Combine getPdf: RooHistPdf nominal -> per-bin morph of the TH1F-rounded
                    unit-normalised histograms (FastVerticalInterpHistPdf2, cropped at a
                    density of 1e-9, renormalised, no normalisation term); other pdfs ->
                    VerticalInterpPdf of the imported nominal/Up/Down pdfs (Combine library)
                    or the equivalent RooRealSumPdf, evaluated like any parametric pdf

Multi-dimensional channels: one RooRealVar per axis (with its binning); bins are flattened
row-major (ir.Observable); parametric pdfs are evaluated at the N-D bin centres times the bin
volume ("center") or integrated over the bin boxes ("integral", unbinned data).

autoMCStats channels (Combine CMSHistFunc + CMSHistErrorPropagator, semantics.py):
    template fractions are h_i / integral with h the un-normalised CMSHistFunc morph, floored
    at 1e-9 and NOT renormalised; per-(process, bin) RooFormulaVars add the Barlow-Beeston-lite
    shifts (x * C_p e_pi**2 C_p / sigma_i for "total" bins, (gamma/n - 1) * yield for
    "poisson", x * C_p e_pi for "gauss", with C_p = n_exp / integral); the evaluator floors
    the bin totals at 1e-9.

The same objects are used to build RooFit pdfs for export and sampling (see export.py).
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from modelspec import ir as I
from modelspec import rootinput as R
from modelspec import semantics as S
from stat_backends.base import UnsupportedByBackend

TEMPLATE_FLOOR = 1e-9  # FastVerticalInterpHistPdf2 CropUnderflows minimum
CMSHIST_FLOOR = 1e-9   # CMSHistFunc / CMSHistErrorPropagator CropUnderflows minimum (absolute)
HIST_PDF_FLOOR = 1e-9  # FastVerticalInterpHistPdf2 of RooHistPdfs: CropUnderflows minimum of the density


def _num(x: float) -> str:
    """Exact decimal representation of a float for formula strings."""
    return repr(float(x))


def smooth_step_expr(arg: str, vsmooth: float) -> str:
    """Combine's smoothStepFunc of ``arg`` as a formula (see semantics.smooth_step)."""
    xn = f"({arg}/{_num(vsmooth)})"
    inner = f"0.125*{xn}*({xn}*{xn}*(3.0*{xn}*{xn}-10.0)+15.0)"
    return f"((abs({arg})>={_num(vsmooth)}) ? (({arg})>0 ? 1.0 : -1.0) : ({inner}))"


def asym_pow_expr(arg: str, kappa_lo: float, kappa_hi: float) -> str:
    """Combine's asymPow(theta, kappa_lo, kappa_hi) = exp(theta * logKappaForX(theta))."""
    lkhi = math.log(kappa_hi)
    lklo = -math.log(kappa_lo)
    avg = 0.5 * (lkhi + lklo)
    half = 0.5 * (lkhi - lklo)
    tx = f"(2.0*{arg})"
    alpha = f"0.125*{tx}*({tx}*{tx}*(3.0*{tx}*{tx}-10.0)+15.0)"
    inner = f"({_num(avg)}+{alpha}*{_num(half)})"
    outer = f"((({arg})>=0) ? {_num(lkhi)} : {_num(lklo)})"
    return f"exp(({arg})*((abs({arg})>=0.5) ? {outer} : {inner}))"


@dataclass
class StateBuild:
    """One shape state of a process (see evaluator.State)."""

    cfrac: Optional[np.ndarray] = None
    ffrac: List[object] = field(default_factory=list)
    pdf: object = None
    deps: set = field(default_factory=set)       # IR parameters the fractions depend on


@dataclass
class ProcBuild:
    name: str
    kind: str
    yield_func: object
    states: List[StateBuild]
    category: object = None                      # RooCategory for envelopes
    multipdf: object = None
    corrections: List[float] = field(default_factory=list)  # RooMultiPdf::getCorrection per state
    yield_deps: set = field(default_factory=set)
    template_fracs: List[object] = field(default_factory=list)  # per-bin fraction functions (templates)


@dataclass
class ChannelBuild:
    name: str
    obs: object
    edges: np.ndarray
    unbinned: bool
    procs: List[ProcBuild]
    bb_terms: List[tuple] = field(default_factory=list)  # (process index, bin, RooAbsReal) additive terms
    floor: Optional[float] = None                         # per-bin floor of the total (autoMCStats)
    obs_list: List[object] = field(default_factory=list)       # one RooRealVar per axis (1D: [obs])
    axis_edges: List[np.ndarray] = field(default_factory=list)  # binning per axis (1D: [edges])
    geometry: object = None                                      # the ir.Observable (nbins, volumes, centres)


class ModelWorkspace:
    """Builds the RooWorkspace for a ModelIR.  ``export`` keeps IR parameter ranges on the
    variables (for a file other tools read); otherwise ranges are removed so that setting any
    value the fitter asks for is never clipped by RooFit."""

    def __init__(self, model: I.ModelIR, name: str = "w", export: bool = False):
        self.model = model
        self.export = export
        self.ROOT = R.root()
        self.ws = self.ROOT.RooWorkspace(name, name)
        self.notes: List[str] = []
        self._sources: Dict[str, tuple] = {}  # node name -> source it was imported from
        self._src_ws: Dict[tuple, object] = {}
        self.vars: Dict[str, object] = {}
        self._keep: List[object] = []
        self._make_observables()
        self._make_parameters()
        self.channels = [self._channel(ch) for ch in model.channels]
        self._finish_parameters()

    # ----- small helpers ---------------------------------------------------------------
    def _imp(self, obj):
        R_ = self.ROOT
        if getattr(self.ws, "import")(obj, R_.RooFit.RecycleConflictNodes(), R_.RooFit.Silence()):
            raise RuntimeError(f"roomodel: could not import {obj.GetName()} into the workspace")
        out = self.ws.arg(obj.GetName())
        if not out:
            raise RuntimeError(f"roomodel: {obj.GetName()} missing after import")
        return out

    def _const(self, name, value):
        return self._imp(self.ROOT.RooConstVar(name, name, float(value)))

    def _formula(self, name, expr, args):
        lst = self.ROOT.RooArgList()
        for a in args:
            lst.add(a)
        f = self.ROOT.RooFormulaVar(name, name, expr, lst)
        return self._imp(f)

    def _source_ws(self, ref: I.RooRef):
        key = (ref.file, ref.workspace)
        if key not in self._src_ws:
            self._src_ws[key] = R.get_workspace(ref.file, ref.workspace)
        return self._src_ws[key]

    def ir_deps(self, obj) -> set:
        """Names of IR parameters that ``obj`` depends on."""
        names = set()
        for v in obj.getVariables():
            if v.GetName() in self.model.parameters:
                names.add(v.GetName())
        return names

    # ----- observables and parameters --------------------------------------------------
    def _make_observables(self):
        """One RooRealVar per observable axis; ``observables[ch]`` is the variable of a 1D
        channel (None for N-D channels), ``obs_lists[ch]`` the variables of all axes."""
        R_ = self.ROOT
        self.observables = {}
        self.obs_lists = {}
        self.observables_edges = {}
        for ch in self.model.channels:
            variables = []
            for o in ch.observable.axis_list():
                edges = np.asarray(o.edges, dtype=float)
                existing = self.ws.var(o.name)
                if existing:
                    old = self.observables_edges[o.name]
                    if len(old) != len(edges) or not np.allclose(old, edges, rtol=0, atol=1e-12):
                        raise UnsupportedByBackend(f"roomodel: channels share the observable '{o.name}' with different "
                                                   "binnings")
                    variables.append(existing)
                    continue
                var = R_.RooRealVar(o.name, o.name, 0.5 * (o.lo + o.hi), o.lo, o.hi)
                arr = R_.std.vector("double")(edges.tolist())
                var.setBinning(R_.RooBinning(len(edges) - 1, arr.data()))
                var = self._imp(var)
                self._sources[o.name] = ("observable",)
                variables.append(var)
                self.observables_edges[o.name] = edges
            if len({v.GetName() for v in variables}) != len(variables):
                raise UnsupportedByBackend(f"roomodel: channel {ch.name} lists an observable axis twice")
            self.obs_lists[ch.name] = variables
            self.observables[ch.name] = variables[0] if ch.observable.ndim == 1 else None

    def _make_parameters(self):
        R_ = self.ROOT
        for name, p in self.model.parameters.items():
            if p.role == I.ROLE_DISCRETE:
                continue  # the RooCategory comes with its RooMultiPdf
            if self.ws.arg(name):
                raise UnsupportedByBackend(f"roomodel: parameter '{name}' has the name of a channel observable")
            v = R_.RooRealVar(name, name, p.value)
            self._imp(v)
            self._sources[name] = ("parameter",)
            self.vars[name] = self.ws.var(name)

    def _finish_parameters(self):
        R_ = self.ROOT
        inf = R_.RooNumber.infinity()
        for name, p in self.model.parameters.items():
            if p.role == I.ROLE_DISCRETE:
                cat = self.ws.cat(name)
                if not cat:
                    raise UnsupportedByBackend(f"roomodel: discrete parameter '{name}' is not the category of an "
                                               "envelope in the model")
                if cat.numTypes() != p.n_states:
                    raise UnsupportedByBackend(f"roomodel: category '{name}' has {cat.numTypes()} states, the IR "
                                               f"says {p.n_states}")
                cat.setIndex(int(round(p.value)))
                cat.setConstant(True)  # never floated by RooFit; the shared fitter profiles it
                self.vars[name] = cat
                continue
            var = self.ws.var(name)
            if self.vars.get(name) is None or var is None:
                raise RuntimeError(f"roomodel: parameter {name} lost during the build")
            self.vars[name] = var
            if self.export:
                lo = p.lo if math.isfinite(p.lo) else -inf
                hi = p.hi if math.isfinite(p.hi) else inf
                var.setRange(lo, hi)
            else:
                var.removeRange()
            var.setVal(p.value)
            var.setConstant(not p.floating)
            var.removeError()

    # ----- importing referenced workspace objects --------------------------------------
    def _import_ref(self, ref: I.RooRef, new_name: str, what: str):
        """Import (a renamed shallow clone of) a workspace object, sharing every node with
        the same name already in the model (Combine's RecycleConflictNodes), after checking
        that such nodes really are the same thing."""
        src_ws = self._source_ws(ref)
        obj = src_ws.arg(ref.name)
        if not obj:
            raise KeyError(f"roomodel: no object '{ref.name}' in {ref.file}:{ref.workspace}")
        source = (ref.file, ref.workspace)
        nodes = self.ROOT.RooArgSet()
        obj.treeNodeServerList(nodes)
        for node in nodes:
            nname = node.GetName()
            if nname == ref.name:
                continue
            mine = self.ws.arg(nname)
            if not mine:
                continue
            prev = self._sources.get(nname)
            if prev == source or prev == ("parameter",):
                continue
            if prev == ("observable",):
                if not node.InheritsFrom("RooRealVar") or abs(node.getMin() - mine.getMin()) > 1e-12 or \
                        abs(node.getMax() - mine.getMax()) > 1e-12:
                    raise UnsupportedByBackend(f"roomodel: {what} uses '{nname}' with range "
                                               f"[{node.getMin()}, {node.getMax()}], but it is the observable "
                                               f"of a channel with range [{mine.getMin()}, {mine.getMax()}]")
                continue
            same = node.ClassName() == mine.ClassName()
            if same and node.InheritsFrom("RooAbsReal"):
                a, b = node.getVal(), mine.getVal()
                same = abs(a - b) <= 1e-12 * max(1.0, abs(a), abs(b))
                if node.InheritsFrom("RooRealVar"):
                    same = same and node.isConstant() == mine.isConstant()
            if not same:
                raise UnsupportedByBackend(
                    f"roomodel: {what} contains '{nname}', which is also defined differently in {prev}; Combine "
                    "would silently share the first definition, so this model is refused")
        clone = obj.Clone(new_name)
        self._keep.append(clone)
        out = self._imp(clone)
        for node in nodes:
            self._sources.setdefault(node.GetName(), source)
        return out

    # ----- channels --------------------------------------------------------------------
    def _channel(self, ch: I.Channel) -> ChannelBuild:
        obs = self.observables[ch.name]
        edges = np.asarray(ch.observable.edges, dtype=float)
        unbinned = ch.data.kind == "unbinned"
        if ch.bin_integration not in ("center", "integral"):
            raise UnsupportedByBackend(f"roomodel: unknown bin integration '{ch.bin_integration}'")
        procs = [self._process(ch, proc, obs, edges, unbinned) for proc in ch.processes]
        cb = ChannelBuild(name=ch.name, obs=obs, edges=edges, unbinned=unbinned, procs=procs,
                          obs_list=self.obs_lists[ch.name],
                          axis_edges=[np.asarray(a.edges, dtype=float) for a in ch.observable.axis_list()],
                          geometry=ch.observable)
        if ch.mcstats is not None:
            self._mcstats(ch, cb)
        return cb

    def _mcstats(self, ch: I.Channel, cb: ChannelBuild):
        """Barlow-Beeston-lite terms of an autoMCStats channel (semantics.bb_lite_expected)."""
        if cb.unbinned:
            raise UnsupportedByBackend(f"roomodel: autoMCStats on the unbinned channel {ch.name}")
        cb.floor = CMSHIST_FLOOR
        pidx = {p.name: i for i, p in enumerate(ch.processes)}
        integ = {p.name: float(np.sum(p.shape.contents)) for p in ch.processes}
        err = {p.name: np.sqrt(np.asarray(p.shape.sumw2, dtype=float)) for p in ch.processes}
        live = [p.name for p in ch.processes if p.rate != 0.0 and integ[p.name] > 0.0]
        for bp in ch.mcstats.params:
            j = bp.bin
            x = self.vars[bp.param]
            name = f"bb_{ch.name}_{bp.param}"
            if bp.kind == I.MCSTATS_TOTAL:
                used = [p for p in live if err[p][j] > 0.0]
                ylds = [cb.procs[pidx[p]].yield_func for p in used]
                cs = [err[p][j] / integ[p] for p in used]  # C_p e_pj = n_exp_p * cs
                s2 = "+".join(f"(@{k + 1}*{_num(c)})*(@{k + 1}*{_num(c)})" for k, c in enumerate(cs))
                for k, p in enumerate(used):
                    ce = f"(@{k + 1}*{_num(cs[k])})"
                    expr = f"(({s2})>0 ? @0*{ce}*{ce}/sqrt({s2}) : 0.0)"
                    cb.bb_terms.append((pidx[p], j, self._formula(f"{name}_{p}", expr, [x] + ylds)))
            elif bp.kind == I.MCSTATS_POISSON:
                pb = cb.procs[pidx[bp.process]]
                st = pb.states[0]
                if st.cfrac is not None:
                    expr = f"(@0/{_num(bp.n_eff)}-1.0)*@1*{_num(st.cfrac[j])}"
                    args = [x, pb.yield_func]
                else:
                    expr = f"(@0/{_num(bp.n_eff)}-1.0)*@1*@2"
                    args = [x, pb.yield_func, st.ffrac[j]]
                cb.bb_terms.append((pidx[bp.process], j, self._formula(name, expr, args)))
            elif bp.kind == I.MCSTATS_GAUSS:
                pb = cb.procs[pidx[bp.process]]
                c = err[bp.process][j] / integ[bp.process]
                cb.bb_terms.append((pidx[bp.process], j,
                                    self._formula(name, f"@0*@1*{_num(c)}", [x, pb.yield_func])))
            else:
                raise UnsupportedByBackend(f"roomodel: autoMCStats parameter kind '{bp.kind}'")

    def _process(self, ch, proc: I.Process, obs, edges, unbinned) -> ProcBuild:
        pfx = f"{ch.name}_{proc.name}"
        kind = proc.shape.kind
        factors = []
        category = None
        multipdf = None
        corrections = []
        template_fracs = []
        if kind == "counting":
            if len(edges) != 2:
                raise UnsupportedByBackend(f"roomodel: counting process {pfx} in a channel with {len(edges) - 1} bins")
            states = [StateBuild(cfrac=np.array([1.0]))]
        elif kind == "template":
            state, norm_factors, template_fracs = self._template(pfx, proc.shape, ch.observable.nbins,
                                                                 cmshist=ch.mcstats is not None)
            states = [state]
            factors += norm_factors
        elif kind == "parametric":
            pdf = self._import_ref(proc.shape.ref, f"shape_{pfx}", f"the shape of {pfx}")
            states = [self._pdf_state(pfx, pdf, obs, edges, ch, unbinned)]
            if proc.shape.contents and not unbinned:
                ref = np.asarray(proc.shape.contents, dtype=float)
                mine = states[0].cfrac if states[0].cfrac is not None else None
                if mine is None or not np.allclose(mine, ref, rtol=1e-9, atol=1e-14):
                    raise RuntimeError(f"roomodel: the fixed histogram of {pfx} in the IR differs from the pdf "
                                       f"{proc.shape.ref.name} evaluated with bin_integration={ch.bin_integration}")
            if proc.shape.pdf_systs and ch.observable.ndim > 1:
                raise UnsupportedByBackend(f"roomodel: shape systematics on the pdf {pfx} of a multi-dimensional channel")
            if proc.shape.pdf_systs and proc.shape.pdf_morph == I.PDF_MORPH_HIST:
                states = [self._hist_pdf_morph(pfx, proc.shape, ch.observable.bin_volumes())]
                kind = "template"  # piecewise-constant fractions: exported and sampled like a template
            elif proc.shape.pdf_systs:
                integral = unbinned or ch.bin_integration == "integral"
                morph, pdfs, coefs, q = self._vertical_pdf_morph(pfx, proc.shape, pdf, integral)
                if integral:
                    # VerticalInterpPdf ignores integration ranges: build the bin integrals here
                    states = [StateBuild(pdf=morph, deps=self.ir_deps(morph),
                                         ffrac=self._vertical_bin_integrals(pfx, pdfs, coefs, q, obs, edges))]
                else:
                    states = [self._pdf_state(pfx, morph, obs, edges, ch, unbinned)]
        elif kind == "envelope":
            if not hasattr(self.ROOT, "RooMultiPdf"):
                raise UnsupportedByBackend("roomodel: envelopes need RooMultiPdf from the Combine library "
                                           "(set PYMODEL_ROOT_LIBS=libHiggsAnalysisCombinedLimit.so)")
            multipdf = self._import_ref(proc.shape.ref, f"shape_{pfx}", f"the envelope of {pfx}")
            if not multipdf.InheritsFrom("RooMultiPdf"):
                raise UnsupportedByBackend(f"roomodel: envelope {proc.shape.ref.name} is a {multipdf.ClassName()}")
            category = self.ws.cat(proc.shape.category)
            if not category:
                raise UnsupportedByBackend(f"roomodel: no category '{proc.shape.category}' for {pfx}")
            states = []
            old = category.getCurrentIndex()
            for i in range(multipdf.getNumPdfs()):
                category.setIndex(i)
                corrections.append(float(multipdf.getCorrection()))
                states.append(self._pdf_state(f"{pfx}_s{i}", multipdf.getPdf(i), obs, edges, ch, unbinned))
            category.setIndex(old)
        else:
            raise UnsupportedByBackend(f"roomodel: shape kind '{kind}' ({pfx})")
        factors = self._norm_factors(pfx, proc) + factors
        lst = self.ROOT.RooArgList()
        for f in factors:
            lst.add(f)
        yld = self._imp(self.ROOT.RooProduct(f"n_exp_{pfx}", f"n_exp_{pfx}", lst))
        return ProcBuild(name=proc.name, kind=kind, yield_func=yld, states=states, category=category,
                         multipdf=multipdf, corrections=corrections, yield_deps=self.ir_deps(yld),
                         template_fracs=template_fracs)

    # ----- yields ----------------------------------------------------------------------
    def _norm_factors(self, pfx, proc: I.Process):
        v = self.vars
        has_gmn = any(t.kind == "gmN" for t in proc.norm_terms)
        out = [self._const(f"rate_{pfx}", 1.0 if has_gmn else proc.rate)]
        if proc.is_signal:
            out.append(v[self.model.poi])
        for k, t in enumerate(proc.norm_terms):
            name = f"norm_{pfx}_{k}_{t.kind}"
            if t.kind in ("lnN", "lnU"):
                out.append(self._formula(name, f"exp(@0*{_num(math.log(t.kappa_hi))})", [v[t.param]]))
            elif t.kind == "asym_lnN":
                out.append(self._formula(name, asym_pow_expr("@0", t.kappa_lo, t.kappa_hi), [v[t.param]]))
            elif t.kind == "gmN":
                out.append(self._formula(name, f"{_num(t.alpha)}*@0", [v[t.param]]))
            elif t.kind == "rate_param":
                out.append(v[t.param])
            elif t.kind == "formula":
                out.append(self._formula(name, t.formula, [v[a] for a in t.args]))
            elif t.kind == "ws_norm":
                out.append(self._import_ref(t.ref, f"{name}_{t.ref.name}", f"the normalisation of {pfx}"))
            else:
                raise UnsupportedByBackend(f"roomodel: norm term '{t.kind}' ({pfx})")
        return out

    # ----- templates -------------------------------------------------------------------
    def _template(self, pfx, shape: I.Shape, nbins, cmshist=False):
        """``cmshist``: CMSHistFunc semantics (autoMCStats channels): fractions h_i / integral with
        h floored at 1e-9 in absolute units and not renormalised."""
        nom = np.asarray(shape.contents, dtype=float)
        if len(nom) != nbins:
            raise UnsupportedByBackend(f"roomodel: template {pfx} has {len(nom)} bins, channel has {nbins}")
        total = nom.sum()
        if total <= 0:
            return StateBuild(cfrac=np.zeros(nbins)), [], []
        nomn = nom / total
        floor = CMSHIST_FLOOR / total if cmshist else TEMPLATE_FLOOR
        if not shape.systs:
            return StateBuild(cfrac=np.maximum(nomn, floor) if cmshist else nomn), [], []
        if cmshist and any(s.kind != "shape" for s in shape.systs):
            raise UnsupportedByBackend(f"roomodel: shapeN template {pfx} in an autoMCStats channel")
        kinds = {s.kind for s in shape.systs}
        if len(kinds) != 1 or not kinds <= {"shape", "shapeN"}:
            raise UnsupportedByBackend(f"roomodel: template {pfx} mixes morphing algorithms {sorted(kinds)} "
                                       "(Combine allows one per shape)")
        log_morph = kinds == {"shapeN"}
        vsmooth = min([1.0] + [s.scale for s in shape.systs])
        norm_factors = []
        args = []   # x_k, S_k pairs
        diffs, sums = [], []
        for k, s in enumerate(shape.systs):
            up = np.asarray(s.up, dtype=float)
            down = np.asarray(s.down, dtype=float)
            if not (up.sum() > 0 and down.sum() > 0):
                raise UnsupportedByBackend(f"roomodel: template systematic {s.param} of {pfx} has a non-positive "
                                           "integral (Combine refuses it)")
            theta = self.vars[s.param]
            x = self._formula(f"morphx_{pfx}_{s.param}", f"{_num(s.scale)}*@0", [theta])
            step = self._formula(f"morphs_{pfx}_{s.param}", smooth_step_expr("@0", vsmooth), [x])
            args += [x, step]
            upn, downn = up / up.sum(), down / down.sum()
            if log_morph:
                hi = np.where((upn > 0) & (nomn > 0), np.log(np.where(upn > 0, upn, 1.0) / np.where(nomn > 0, nomn, 1.0)), 0.0)
                lo = np.where((downn > 0) & (nomn > 0), np.log(np.where(downn > 0, downn, 1.0) / np.where(nomn > 0, nomn, 1.0)), 0.0)
            else:
                hi, lo = upn - nomn, downn - nomn
            diffs.append(hi - lo)
            sums.append(hi + lo)
            kappas = S.template_norm_kappas(nom, up, down, s.scale)
            if kappas is not None:
                norm_factors.append(self._formula(f"systeff_{pfx}_{s.param}", asym_pow_expr("@0", *kappas), [theta]))
        base = np.where(nomn > 0, np.log(np.where(nomn > 0, nomn, 1.0)), -999.0) if log_morph else nomn
        tvals = []
        for i in range(nbins):
            used = [k for k in range(len(shape.systs)) if diffs[k][i] != 0 or sums[k][i] != 0]
            # pass only the arguments the formula uses (RooFormula prunes unused ones, and
            # RooFit's server redirection then crashes on them)
            terms = [f"0.5*@{2 * j}*({_num(diffs[k][i])}+{_num(sums[k][i])}*@{2 * j + 1})"
                     for j, k in enumerate(used)]
            bin_args = [a for k in used for a in args[2 * k:2 * k + 2]]
            if not terms:
                if cmshist:
                    const = max(base[i], floor)
                else:
                    const = math.exp(base[i]) if log_morph else (base[i] if base[i] > 0 else TEMPLATE_FLOOR)
                tvals.append(self._const(f"morpht_{pfx}_{i}", const))
                continue
            expr = f"{_num(base[i])}+" + "+".join(terms)
            if log_morph:
                expr = f"exp({expr})"
            elif cmshist:
                expr = f"((({expr})>{_num(floor)}) ? ({expr}) : {_num(floor)})"
            else:
                expr = f"((({expr})>0) ? ({expr}) : {_num(TEMPLATE_FLOOR)})"
            tvals.append(self._formula(f"morpht_{pfx}_{i}", expr, bin_args))
        deps = {s.param for s in shape.systs}
        if cmshist:
            return StateBuild(ffrac=tvals, deps=deps), norm_factors, tvals
        lst = self.ROOT.RooArgList()
        for t in tvals:
            lst.add(t)
        tsum = self._imp(self.ROOT.RooAddition(f"morphsum_{pfx}", "", lst))
        fracs = [self._formula(f"morphf_{pfx}_{i}", "@0/@1", [tvals[i], tsum]) for i in range(nbins)]
        return StateBuild(ffrac=fracs, deps=deps), norm_factors, fracs

    # ----- parametric pdfs -------------------------------------------------------------
    def _bin_integrals(self, pfx, pdf, obs, edges):
        R_ = self.ROOT
        nset = R_.RooArgSet(obs)
        out = []
        for i in range(len(edges) - 1):
            rname = f"pymodel_{obs.GetName()}_bin{i}"
            if not obs.hasRange(rname):
                obs.setRange(rname, float(edges[i]), float(edges[i + 1]))
            integ = pdf.createIntegral(nset, R_.RooFit.NormSet(nset), R_.RooFit.Range(rname))
            self._keep.append(integ)
            out.append(integ)
        return out

    def _bin_integrals_nd(self, ch, pdf):
        """Normalised integrals of an N-D pdf over the flattened bin boxes of channel ``ch``."""
        R_ = self.ROOT
        variables = self.obs_lists[ch.name]
        nset = R_.RooArgSet()
        for v in variables:
            nset.add(v)
        axes = ch.observable.axes
        out = []
        for k in range(ch.observable.nbins):
            idx = np.unravel_index(k, ch.observable.shape)
            rname = f"pymodel_{ch.name}_nd_bin{k}"
            for v, a, i in zip(variables, axes, idx):
                if not v.hasRange(rname):
                    v.setRange(rname, float(a.edges[i]), float(a.edges[i + 1]))
            integ = pdf.createIntegral(nset, R_.RooFit.NormSet(nset), R_.RooFit.Range(rname))
            self._keep.append(integ)
            out.append(integ)
        return out

    def _centre_values_nd(self, ch, pdf):
        """pdf(N-D bin centre) * bin volume for every flattened bin of channel ``ch``."""
        variables = self.obs_lists[ch.name]
        nset = self.ROOT.RooArgSet()
        for v in variables:
            nset.add(v)
        old = [v.getVal() for v in variables]
        vals = []
        for c, vol in zip(ch.observable.bin_centers(), ch.observable.bin_volumes()):
            for v, x in zip(variables, c):
                v.setVal(float(x))
            vals.append(pdf.getVal(nset) * float(vol))
        for v, x in zip(variables, old):
            v.setVal(x)
        return np.array(vals)

    def _pdf_state(self, pfx, pdf, obs, edges, ch, unbinned) -> StateBuild:
        deps = self.ir_deps(pdf)
        integral = unbinned or ch.bin_integration == "integral"
        state = StateBuild(pdf=pdf, deps=deps)
        if ch.observable.ndim > 1:
            if integral:
                funcs = self._bin_integrals_nd(ch, pdf)
                if deps:
                    state.ffrac = funcs
                else:
                    state.cfrac = np.array([f.getVal() for f in funcs])
            elif not deps:
                state.cfrac = self._centre_values_nd(ch, pdf)
            return state  # binned with deps: evaluated at the N-D bin centres in C++
        if integral:
            funcs = self._bin_integrals(pfx, pdf, obs, edges)
            if deps:
                state.ffrac = funcs
            else:
                state.cfrac = np.array([f.getVal() for f in funcs])
        elif not deps:
            nset = self.ROOT.RooArgSet(obs)
            old = obs.getVal()
            vals = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                obs.setVal(0.5 * (lo + hi))
                vals.append(pdf.getVal(nset) * (hi - lo))
            obs.setVal(old)
            state.cfrac = np.array(vals)
        if not unbinned and state.cfrac is None and not state.ffrac:
            state.pdf = pdf  # evaluated at the bin centres in C++
        return state

    # ----- shape systematics on pdfs (Combine ShapeTools.getPdf) -----------------------
    def _vertical_pdf_morph(self, pfx, shape: I.Shape, nominal, integral: bool):
        """VerticalInterpPdf(nominal, up_k, down_k; x_k = scale_k * theta_k, q = min(1, scales),
        algorithm 0 for ``shape``, -1 for ``shapeN``) exactly as text2workspace builds it.  With
        Combine's library the class itself is used; without it ``shape`` is the equivalent
        RooRealSumPdf of the un-normalised pdfs with the coefficients of
        semantics.vertical_pdf_coefficients (a non-positive morph is then 0 instead of the
        1e-15 / 1e-10 floors of VerticalInterpPdf; noted)."""
        R_ = self.ROOT
        kinds = {s.kind for s in shape.pdf_systs}
        if len(kinds) != 1 or not kinds <= {"shape", "shapeN"}:
            raise UnsupportedByBackend(f"roomodel: pdf {pfx} mixes morphing algorithms {sorted(kinds)}")
        if kinds == {"shapeN"} and integral:
            raise UnsupportedByBackend(f"roomodel: shapeN on the pdf {pfx} with unbinned data / bin integrals")
        algo = -1 if kinds == {"shapeN"} else 0
        q = min([1.0] + [s.scale for s in shape.pdf_systs])
        pdfs, coefs = [nominal], []
        for s in shape.pdf_systs:
            pdfs.append(self._import_ref(s.up, f"shape_{pfx}_{s.param}Up", f"the {s.param}Up shape of {pfx}"))
            pdfs.append(self._import_ref(s.down, f"shape_{pfx}_{s.param}Down", f"the {s.param}Down shape of {pfx}"))
            theta = self.vars[s.param]
            coefs.append(theta if s.scale == 1 else
                         self._formula(f"{s.param}_scaled_{pfx}", f"{_num(s.scale)}*@0", [theta]))
        if hasattr(R_, "VerticalInterpPdf"):
            plist, clist = R_.RooArgList(), R_.RooArgList()
            for a in pdfs:
                plist.add(a)
            for c in coefs:
                clist.add(c)
            return self._imp(R_.VerticalInterpPdf(f"shape_{pfx}_morph", "", plist, clist, q, algo)), pdfs, coefs, q
        if algo != 0:
            raise UnsupportedByBackend(f"roomodel: shapeN on the pdf {pfx} needs Combine's VerticalInterpPdf "
                                       "(set PYMODEL_ROOT_LIBS=libHiggsAnalysisCombinedLimit.so)")
        funcs, cvals = R_.RooArgList(), R_.RooArgList()
        for f, c in zip(pdfs, self._vertical_coefficients(pfx, coefs, q)):
            funcs.add(f)
            cvals.add(c)
        self.notes.append(f"{pfx}: pdf shape systematics as a RooRealSumPdf (no Combine library); identical to "
                          "VerticalInterpPdf except where the morphed pdf is not positive: there it is 0 (NLL = inf "
                          "if the bin/event has data) instead of Combine's floor 1e-15 (1e-10 for the integral)")
        sumpdf = R_.RooRealSumPdf(f"shape_{pfx}_morph", "", funcs, cvals)
        sumpdf.setFloor(True)  # max(0, value), without RooFit's negative-value error messages
        return self._imp(sumpdf), pdfs, coefs, q

    def _vertical_coefficients(self, pfx, coefs, q):
        """[a_0, a_up_1, a_dn_1, ...]: the weights of (nominal, up_k, down_k) in VerticalInterpPdf's
        morph (semantics.vertical_pdf_coefficients; a_0 = 1 + sum_k c_cen,k)."""
        qs = _num(q)
        inside = {"cen": f"(-@0*@0/{qs})", "up": f"(@0*({qs}+@0)/(2.0*{qs}))", "dn": f"(-@0*({qs}-@0)/(2.0*{qs}))"}
        outside = {"cen": "((@0>0) ? -@0 : @0)", "up": "((@0>0) ? @0 : 0.0)", "dn": "((@0>0) ? 0.0 : -@0)"}
        expr = {k: f"((abs(@0)>={qs}) ? {outside[k]} : {inside[k]})" for k in inside}
        cen_terms = "+".join(expr["cen"].replace("@0", f"@{k}") for k in range(len(coefs)))
        out = [self._formula(f"vmorphc_{pfx}_nominal", f"1.0+{cen_terms}", coefs)]
        for k, c in enumerate(coefs):
            out.append(self._formula(f"vmorphc_{pfx}_{k}_up", expr["up"], [c]))
            out.append(self._formula(f"vmorphc_{pfx}_{k}_dn", expr["dn"], [c]))
        return out

    def _vertical_bin_integrals(self, pfx, pdfs, coefs, q, obs, edges):
        """Normalised bin integrals of the VerticalInterpPdf morph: sum_j a_j B_ij / sum_j a_j I_j
        with B_ij / I_j the bin / full integrals of the un-normalised pdfs; a non-positive
        denominator gives 0 and a non-positive result 1e-10 (VerticalInterpPdf::analyticalIntegralWN;
        semantics.vertical_pdf_fractions with method "integral")."""
        R_ = self.ROOT
        iset = R_.RooArgSet(obs)
        a = self._vertical_coefficients(pfx, coefs, q)
        full = [f.createIntegral(iset) for f in pdfs]
        self._keep += full
        nj = len(pdfs)
        den = "+".join(f"@{j}*@{2 * nj + j}" for j in range(nj))
        out = []
        for i in range(len(edges) - 1):
            rname = f"pymodel_{obs.GetName()}_bin{i}"
            if not obs.hasRange(rname):
                obs.setRange(rname, float(edges[i]), float(edges[i + 1]))
            bins = [f.createIntegral(iset, R_.RooFit.Range(rname)) for f in pdfs]
            self._keep += bins
            num = "+".join(f"@{j}*@{nj + j}" for j in range(nj))
            ratio = f"((({den})>0) ? ({num})/({den}) : 0.0)"
            expr = f"(({ratio})>0 ? ({ratio}) : {_num(S.PDF_INTEGRAL_FLOOR)})"
            out.append(self._formula(f"vmorphb_{pfx}_{i}", expr, a + bins + full))
        return out

    def _hist_pdf_morph(self, pfx, shape: I.Shape, widths) -> StateBuild:
        """FastVerticalInterpHistPdf2 of fixed RooHistPdfs (semantics.hist_pdf_fractions): the
        TH1F-rounded, unit-normalised histograms morphed per bin with the smooth step (``shape``:
        linear, cropped at a density of 1e-9; ``shapeN``: in log space), then renormalised.
        ``widths``: the bin widths."""
        widths = np.asarray(widths, dtype=float)
        nbins = len(widths)
        kinds = {s.kind for s in shape.pdf_systs}
        if len(kinds) != 1 or not kinds <= {"shape", "shapeN"}:
            raise UnsupportedByBackend(f"roomodel: pdf {pfx} mixes morphing algorithms {sorted(kinds)}")
        log_morph = kinds == {"shapeN"}

        def unit(a):
            a = np.asarray(a, dtype=np.float32).astype(float)
            if len(a) != nbins or not a.sum() > 0:
                raise UnsupportedByBackend(f"roomodel: RooHistPdf histogram of {pfx} has {len(a)} bins or no content")
            return a / a.sum()

        def logratio(a, ref):
            return np.where((a > 0) & (ref > 0), np.log(np.where(a > 0, a, 1.0) / np.where(ref > 0, ref, 1.0)), 0.0)

        nom = unit(shape.contents)
        q = min([1.0] + [s.scale for s in shape.pdf_systs])
        args, diffs, sums = [], [], []
        for s in shape.pdf_systs:
            up, down = unit(s.up_contents), unit(s.down_contents)
            hi, lo = (logratio(up, nom), logratio(down, nom)) if log_morph else (up - nom, down - nom)
            diffs.append(hi - lo)
            sums.append(hi + lo)
            x = self._formula(f"hmorphx_{pfx}_{s.param}", f"{_num(s.scale)}*@0", [self.vars[s.param]])
            args += [x, self._formula(f"hmorphs_{pfx}_{s.param}", smooth_step_expr("@0", q), [x])]
        base = np.where(nom > 0, np.log(np.where(nom > 0, nom, 1.0)), -999.0) if log_morph else nom
        crop = HIST_PDF_FLOOR * widths
        tvals = []
        for i in range(nbins):
            used = [k for k in range(len(shape.pdf_systs)) if diffs[k][i] != 0 or sums[k][i] != 0]
            if not used:
                const = math.exp(base[i]) if log_morph else (base[i] if base[i] >= crop[i] else crop[i])
                tvals.append(self._const(f"hmorpht_{pfx}_{i}", const))
                continue
            terms = [f"0.5*@{2 * j}*({_num(diffs[k][i])}+@{2 * j + 1}*{_num(sums[k][i])})" for j, k in enumerate(used)]
            expr = f"{_num(base[i])}+" + "+".join(terms)
            if log_morph:
                expr = f"exp({expr})"
            else:
                expr = f"((({expr})<{_num(crop[i])}) ? {_num(crop[i])} : ({expr}))"
            tvals.append(self._formula(f"hmorpht_{pfx}_{i}", expr, [a for k in used for a in args[2 * k:2 * k + 2]]))
        lst = self.ROOT.RooArgList()
        for t in tvals:
            lst.add(t)
        tsum = self._imp(self.ROOT.RooAddition(f"hmorphsum_{pfx}", "", lst))
        fracs = [self._formula(f"hmorphf_{pfx}_{i}", "@0/@1", [tvals[i], tsum]) for i in range(nbins)]
        return StateBuild(ffrac=fracs, deps={s.param for s in shape.pdf_systs})
