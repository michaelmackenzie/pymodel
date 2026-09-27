"""Combine datacard -> ModelIR.

Parsing is done by Combine's own DatacardParser (vendored in third_party/combine), so the
full datacard grammar, ``nuisance edit`` lines and groups behave exactly as in Combine.
This module then resolves every (channel, process) shape with Combine's precedence rules
and converts the result into a ``ModelIR``.  Anything that cannot be represented raises
``UnsupportedFeature``; nothing is dropped silently.
"""

import math
import optparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from modelspec import ir as I
from third_party.combine.DatacardParser import addDatacardParserOptions, parseCard

DEFAULT_POI_RANGE = (0.0, 20.0)  # Combine default physics model: r in [0, 20]


class UnsupportedFeature(RuntimeError):
    """The datacard uses something pymodel cannot represent (yet)."""


def parse_datacard(path: str, mass: str = "120"):
    """Parse a datacard with Combine's parser, applying nuisance edits."""
    parser = optparse.OptionParser()
    addDatacardParserOptions(parser)
    options, _ = parser.parse_args([])
    options.mass = mass
    options.evaluateEdits = True
    with open(path, "r", encoding="utf-8") as handle:
        return parseCard(handle, options)


# ----------------------------------------------------------------------------------------
# Shape resolution (ShapeTools.getShape precedence)
# ----------------------------------------------------------------------------------------

def _shape_entry(dc, channel: str, process: str) -> Optional[List[str]]:
    for ch_key, proc_key in ((channel, process), (channel, "*"), ("*", process), ("*", "*")):
        entry = dc.shapeMap.get(ch_key, {}).get(proc_key)
        if entry is not None:
            return entry
    return None


def _substitute(template: str, channel: str, process: str, mass: str, systematic: str = "") -> str:
    out = template.replace("$CHANNEL", channel).replace("$PROCESS", process).replace("$MASS", mass)
    return out.replace("$SYSTEMATIC", systematic)


class _ShapeResolver:
    def __init__(self, dc, card_dir: str, mass: str):
        self.dc = dc
        self.card_dir = card_dir
        self.mass = mass

    def entry(self, channel, process):
        e = _shape_entry(self.dc, channel, process)
        if e is None or e[0] == "FAKE":
            return None
        return e

    def file(self, entry) -> str:
        # Combine's FileCache: the path as given (relative to the working directory) first,
        # then relative to the datacard directory.
        path = entry[0]
        if os.path.isabs(path) or os.path.exists(path):
            return os.path.abspath(path)
        return os.path.normpath(os.path.join(self.card_dir, path))

    def nominal(self, channel, process):
        from modelspec import rootinput as R

        e = self.entry(channel, process)
        if e is None:
            return None
        if len(e) < 2:
            raise UnsupportedFeature(f"shapes line for {channel}/{process} has no object name")
        spec = _substitute(e[1], channel, process, self.mass)
        path = self.file(e)
        obj, ws = R.get_object(path, spec)
        return obj, path, ws, spec

    def systematic(self, channel, process, syst, direction):
        from modelspec import rootinput as R

        e = self.entry(channel, process)
        if e is None or len(e) < 3:
            raise UnsupportedFeature(
                f"shape systematic '{syst}' on {channel}/{process} but the shapes line has no systematics pattern")
        spec = _substitute(e[2], channel, process, self.mass, syst + direction)
        path = self.file(e)
        obj, ws = R.get_object(path, spec)
        return obj, path, ws, spec

    def systematic_exists(self, channel, process, syst) -> bool:
        from modelspec import rootinput as R

        e = self.entry(channel, process)
        if e is None or len(e) < 3:
            return False
        path = self.file(e)
        return all(R.object_exists(path, _substitute(e[2], channel, process, self.mass, syst + d))
                   for d in ("Up", "Down"))


# ----------------------------------------------------------------------------------------
# IR construction
# ----------------------------------------------------------------------------------------

def _gauss_nuisance(name, lo, hi, origin):
    return I.Parameter(name=name, value=0.0, lo=lo, hi=hi, role=I.ROLE_NUISANCE,
                       constraint=I.Constraint(kind=I.CONSTRAINT_GAUSS, center=0.0), origin=origin)


def _parse_range(text: str) -> Tuple[float, float]:
    lo, hi = text.strip("[]").split(",")
    return float(lo), float(hi)


class _Builder:
    def __init__(self, card_path: str, mass: str, poi_range, bin_integration: str):
        self.card_path = os.path.abspath(card_path)
        self.dc = parse_datacard(card_path, mass)
        self.mass = mass
        self.shapes = _ShapeResolver(self.dc, os.path.dirname(self.card_path), mass)
        self.params: Dict[str, I.Parameter] = {}
        self.notes: List[str] = []
        self.poi_range = poi_range
        self.bin_integration = bin_integration
        self.shape_systs = {}  # name -> pdf type for shape* lines
        self.ws_params: Dict[str, object] = {}  # workspace parameter name -> RooRealVar (first seen)

    # -- parameters -------------------------------------------------------------------
    def add_param(self, par: I.Parameter):
        existing = self.params.get(par.name)
        if existing is None:
            self.params[par.name] = par
            return par
        return existing

    def add_workspace_param(self, var, ref_file, ws_name):
        name = var.GetName()
        if name in self.params:
            return
        if var.InheritsFrom("RooCategory"):
            self.params[name] = I.Parameter(name=name, value=float(var.getIndex()), lo=0.0,
                                            hi=float(var.numTypes() - 1), role=I.ROLE_DISCRETE,
                                            origin="workspace", n_states=int(var.numTypes()))
            return
        lo = var.getMin() if var.hasMin() else -math.inf
        hi = var.getMax() if var.hasMax() else math.inf
        self.params[name] = I.Parameter(name=name, value=var.getVal(), lo=lo, hi=hi,
                                        role=I.ROLE_FREE, origin="workspace")
        self.ws_params[name] = var

    # -- channels ---------------------------------------------------------------------
    def build(self) -> I.ModelIR:
        dc = self.dc
        if dc.binParFlags:
            raise UnsupportedFeature("autoMCStats is not supported yet")
        if dc.extArgs:
            for name, fields in dc.extArgs.items():
                self._ext_arg(name, fields)
        lo, hi = self.poi_range
        self.params["r"] = I.Parameter(name="r", value=1.0, lo=lo, hi=hi, role=I.ROLE_POI, origin="poi")
        for syst in dc.systs:
            name, _nofloat, pdf, args, errline = syst
            if pdf.startswith("shape"):
                self.shape_systs[name] = pdf
            elif pdf == "param":
                # Combine makes a constant workspace variable floating when a param line names it
                var = self._find_constant_ws_var(name)
                if var is not None and var.isConstant():
                    var.setConstant(False)
        channels = [self._channel(b) for b in dc.bins]
        self._systematics(channels)
        self._rate_params(channels)
        self._discretes()
        for name in dc.frozenNuisances:
            if name not in self.params:
                raise UnsupportedFeature(f"'nuisance edit freeze' of unknown parameter '{name}'")
            self.params[name].role = I.ROLE_CONSTANT
        groups = {g: sorted(members) for g, members in dc.groups.items()}
        self._check_unused_params(channels)
        return I.ModelIR(channels=channels, parameters=self.params, poi="r", groups=groups,
                         source=self.card_path, notes=self.notes)

    def _ext_arg(self, name, fields):
        # "name extArg value [range]" or "name extArg file.root:ws"
        if len(fields) >= 3 and ".root" in fields[2]:
            raise UnsupportedFeature(f"extArg '{name}' from a ROOT file is not supported yet")
        value = float(fields[2])
        if len(fields) >= 4:
            lo, hi = _parse_range(fields[3])
            self.params[name] = I.Parameter(name=name, value=value, lo=lo, hi=hi, role=I.ROLE_FREE, origin="extArg")
        else:
            self.params[name] = I.Parameter(name=name, value=value, lo=value, hi=value, role=I.ROLE_CONSTANT,
                                            origin="extArg")

    def _channel(self, b: str) -> I.Channel:
        dc = self.dc
        procs = [p for (bb, p, _s) in dc.keyline if bb == b]
        data_nominal = self.shapes.nominal(b, "data_obs") if self.shapes.entry(b, "data_obs") else None
        shaped = [p for p in procs if self.shapes.entry(b, p) is not None]
        if shaped and len(shaped) != len(procs):
            raise UnsupportedFeature(f"channel '{b}' mixes shape and counting processes")
        if not shaped:
            n_obs = dc.obs.get(b)
            if n_obs is None or n_obs < 0:
                raise UnsupportedFeature(f"counting channel '{b}' needs an observation value")
            observable = I.Observable(name=f"count_{b}", lo=0.0, hi=1.0, edges=[0.0, 1.0])
            data = I.ChannelData(kind="count", counts=[float(n_obs)])
            processes = [I.Process(name=p, is_signal=bool(dc.isSignal[p]), rate=float(dc.exp[b][p]),
                                   shape=I.Shape(kind="counting")) for p in procs]
            return I.Channel(name=b, observable=observable, data=data, processes=processes)
        if data_nominal is None:
            raise UnsupportedFeature(f"channel '{b}' has shapes but no data_obs shapes entry")
        observable, data, obs_var, obs_ref = self._data(b, data_nominal)
        if dc.obs.get(b, -1) >= 0 and abs(dc.obs[b] - data.total) > 1e-6 * max(1.0, data.total):
            self.notes.append(f"channel {b}: observation line {dc.obs[b]} differs from data_obs total "
                              f"{data.total}; data_obs is used (as in Combine)")
        processes = [self._process(b, p, observable, obs_var) for p in procs]
        return I.Channel(name=b, observable=observable, data=data, processes=processes,
                         bin_integration=self.bin_integration, obs_ref=obs_ref)

    def _data(self, b, nominal):
        from modelspec import rootinput as R

        obj, path, ws, spec = nominal
        if obj.InheritsFrom("TH1"):
            edges = R.th1_edges(obj)
            counts, _ = R.th1_contents(obj)
            obs = I.Observable(name=f"CMS_th1x_{b}", lo=edges[0], hi=edges[-1], edges=edges)
            return obs, I.ChannelData(kind="binned", counts=counts), None, None
        if obj.InheritsFrom("RooAbsData"):
            if obj.get().getSize() != 1:
                names = [v.GetName() for v in obj.get()]
                raise UnsupportedFeature(f"data_obs of channel '{b}' has {len(names)} observables {names}; "
                                         "only 1D channels are supported")
            var = obj.get().first()
            ws_obj = R.get_workspace(path, ws)
            var = ws_obj.var(var.GetName())
            edges = R.var_edges(var)
            obs = I.Observable(name=var.GetName(), lo=var.getMin(), hi=var.getMax(), edges=edges)
            obs_ref = I.RooRef(file=path, workspace=ws, name=var.GetName(), class_name=var.ClassName())
            if obj.InheritsFrom("RooDataHist"):
                counts, _ = R.datahist_contents(obj, var, edges)
                return obs, I.ChannelData(kind="binned", counts=counts), var, obs_ref
            values, weights = R.dataset_values(obj, var.GetName())
            return obs, I.ChannelData(kind="unbinned", values=values, weights=weights), var, obs_ref
        raise UnsupportedFeature(f"data_obs '{spec}' of channel '{b}' is a {obj.ClassName()}")

    def _process(self, b, p, observable: I.Observable, obs_var) -> I.Process:
        from modelspec import rootinput as R

        dc = self.dc
        rate = float(dc.exp[b][p])
        obj, path, ws, spec = self.shapes.nominal(b, p)
        is_signal = bool(dc.isSignal[p])
        norm_terms: List[I.NormTerm] = []
        if obj.InheritsFrom("TH1") or obj.InheritsFrom("RooDataHist"):
            if obj.InheritsFrom("TH1"):
                if not np.allclose(R.th1_edges(obj), observable.edges):
                    raise UnsupportedFeature(f"template {spec} binning differs from data_obs in channel {b}")
                contents, sumw2 = R.th1_contents(obj)
            else:
                contents, sumw2 = R.datahist_contents(obj, obs_var, observable.edges)
            if rate == -1:
                rate = float(sum(contents))
            shape = I.Shape(kind="template", contents=contents, sumw2=sumw2)
            shape.systs = self._template_systs(b, p, obj, observable, obs_var)
            return I.Process(name=p, is_signal=is_signal, rate=rate, shape=shape, norm_terms=norm_terms)
        if not obj.InheritsFrom("RooAbsPdf"):
            raise UnsupportedFeature(f"shape '{spec}' for {b}/{p} is a {obj.ClassName()}")
        if obs_var is None:
            raise UnsupportedFeature(f"parametric shape '{spec}' in channel {b} needs RooFit data_obs")
        if rate == -1:
            raise UnsupportedFeature(f"rate -1 is not allowed for the parametric shape {b}/{p}")
        ws_obj = R.get_workspace(path, ws)
        ref = I.RooRef(file=path, workspace=ws, name=obj.GetName(), class_name=obj.ClassName())
        fparams = R.floating_params(obj, obs_var)
        for v in fparams:
            self.add_workspace_param(v, path, ws)
        if obj.InheritsFrom("RooMultiPdf"):
            cats = [v for v in fparams if v.InheritsFrom("RooCategory")]
            if len(cats) != 1:
                raise UnsupportedFeature(f"RooMultiPdf '{spec}' must depend on exactly one floating category, "
                                         f"found {[c.GetName() for c in cats]}")
            shape = I.Shape(kind="envelope", ref=ref, params=[v.GetName() for v in fparams],
                            category=cats[0].GetName())
        else:
            shape = I.Shape(kind="parametric", ref=ref, params=[v.GetName() for v in fparams])
            if not fparams:
                shape.contents = R.binned_pdf_contents(obj, obs_var, observable.edges, self.bin_integration)
        norm = ws_obj.arg(obj.GetName() + "_norm")
        if norm:
            if norm.InheritsFrom("RooRealVar"):
                if norm.isConstant():
                    rate *= norm.getVal()
                else:
                    self.add_workspace_param(norm, path, ws)
                    rate *= 1.0
                    norm_terms.append(I.NormTerm(kind="rate_param", param=norm.GetName()))
            else:
                nparams = R.floating_params(norm, obs_var)
                for v in nparams:
                    self.add_workspace_param(v, path, ws)
                if nparams:
                    norm_terms.append(I.NormTerm(kind="ws_norm", args=[v.GetName() for v in nparams],
                                                 ref=I.RooRef(file=path, workspace=ws, name=norm.GetName(),
                                                              class_name=norm.ClassName())))
                else:
                    rate *= norm.getVal()
        for syst, pdf in self.shape_systs.items():
            if self._entry(syst, b, p) == 0:
                continue
            if pdf.endswith("?") and not self.shapes.systematic_exists(b, p, syst):
                continue
            up = self.shapes.systematic(b, p, syst, "Up")
            down = self.shapes.systematic(b, p, syst, "Down")
            shape.syst_refs[syst] = [I.RooRef(file=up[1], workspace=up[2], name=up[0].GetName(), class_name=up[0].ClassName()),
                                     I.RooRef(file=down[1], workspace=down[2], name=down[0].GetName(),
                                              class_name=down[0].ClassName())]
        return I.Process(name=p, is_signal=is_signal, rate=rate, shape=shape, norm_terms=norm_terms)

    def _entry(self, syst_name, b, p):
        for name, _nofloat, _pdf, _args, errline in self.dc.systs:
            if name == syst_name:
                return errline.get(b, {}).get(p, 0.0) if errline else 0.0
        return 0.0

    def _template_systs(self, b, p, nominal, observable, obs_var) -> List[I.TemplateSyst]:
        from modelspec import rootinput as R

        out = []
        for syst, pdf in self.shape_systs.items():
            scale = self._entry(syst, b, p)
            if scale == 0:
                continue
            if pdf.endswith("?") and not self.shapes.systematic_exists(b, p, syst):
                continue
            kind = "shapeN" if "shapeN" in pdf else "shape"
            if pdf.rstrip("?") not in ("shape", "shapeN"):
                raise UnsupportedFeature(f"systematic type '{pdf}' is not supported")
            arrays = []
            for direction in ("Up", "Down"):
                obj = self.shapes.systematic(b, p, syst, direction)[0]
                if obj.ClassName() != nominal.ClassName():
                    raise UnsupportedFeature(f"{syst}{direction} for {b}/{p} is a {obj.ClassName()}, "
                                             f"nominal is a {nominal.ClassName()}")
                if obj.InheritsFrom("TH1"):
                    arrays.append(R.th1_contents(obj)[0])
                else:
                    arrays.append(R.datahist_contents(obj, obs_var, observable.edges)[0])
            out.append(I.TemplateSyst(param=syst, up=arrays[0], down=arrays[1], scale=float(scale), kind=kind))
        return out

    # -- systematics ------------------------------------------------------------------
    def _proc(self, channels, b, p) -> I.Process:
        for ch in channels:
            if ch.name == b:
                for proc in ch.processes:
                    if proc.name == p:
                        return proc
        raise KeyError((b, p))

    def _systematics(self, channels):
        dc = self.dc
        for name, _nofloat, pdf, args, errline in dc.systs:
            if pdf in ("lnN", "lnU") or (pdf.startswith("shape") and pdf.endswith("?")):
                self._lognormal(channels, name, pdf, errline)
            elif pdf.startswith("shape"):
                par = self.add_param(_gauss_nuisance(name, -4.0, 4.0, "shape"))
                used = any(s.param == name for ch in channels for proc in ch.processes for s in proc.shape.systs)
                used |= any(name in proc.shape.syst_refs for ch in channels for proc in ch.processes)
                if not used:
                    self.notes.append(f"shape systematic '{name}' has no effect on any process")
                par.origin = pdf
            elif pdf == "gmN":
                self._gamma(channels, name, args, errline)
            elif pdf == "param":
                self._param(name, args)
            elif pdf in ("flatParam", "discrete", "rateParam", "extArg"):
                continue
            else:
                raise UnsupportedFeature(f"systematic '{name}' of type '{pdf}' is not supported")
        for name in dc.flatParamNuisances:
            if name in self.params:
                self.params[name].role = I.ROLE_FREE
            else:
                self.notes.append(f"flatParam '{name}' does not match any model parameter")

    def _lognormal(self, channels, name, pdf, errline):
        kind = "lnU" if pdf == "lnU" else "lnN"
        if kind == "lnU":
            par = I.Parameter(name=name, value=0.0, lo=-1.0, hi=1.0, role=I.ROLE_NUISANCE,
                              constraint=I.Constraint(kind=I.CONSTRAINT_FLAT), origin="lnU")
            self.add_param(par)
        else:
            self.add_param(_gauss_nuisance(name, -7.0, 7.0, pdf))
        for b, procs in errline.items():
            for p, val in procs.items():
                if pdf.endswith("?"):
                    if self.shapes.systematic_exists(b, p, name):
                        continue  # a real shape systematic, handled with the templates
                if isinstance(val, list):
                    lo, hi = val
                    self._proc(channels, b, p).norm_terms.append(
                        I.NormTerm(kind="asym_lnN", param=name, kappa_lo=lo, kappa_hi=hi))
                    continue
                if val == 0.0 or (kind == "lnN" and val == 1.0):
                    continue
                if val < 0:
                    raise UnsupportedFeature(f"negative lnN value for {name} on {b}/{p}")
                self._proc(channels, b, p).norm_terms.append(
                    I.NormTerm(kind=kind, param=name, kappa_lo=1.0 / val, kappa_hi=val))

    def _gamma(self, channels, name, args, errline):
        n_obs = float(args[0])
        hi = max(10.0, n_obs + 10.0 * math.sqrt(n_obs + 1.0))
        par = I.Parameter(name=name, value=n_obs, lo=0.0, hi=hi, role=I.ROLE_NUISANCE,
                          constraint=I.Constraint(kind=I.CONSTRAINT_POISSON, center=n_obs), origin="gmN")
        self.add_param(par)
        for b, procs in errline.items():
            for p, alpha in procs.items():
                if alpha == 0.0:
                    continue
                proc = self._proc(channels, b, p)
                if any(t.kind == "gmN" for t in proc.norm_terms):
                    raise UnsupportedFeature(f"more than one gmN on {b}/{p}")
                expected = alpha * n_obs
                if abs(proc.rate - expected) > 1e-3 * max(1.0, expected) and proc.rate != 1e-6:
                    self.notes.append(f"gmN {name} on {b}/{p}: rate {proc.rate} != N*alpha = {expected}; "
                                      "the yield is alpha*n (Combine semantics)")
                proc.norm_terms.append(I.NormTerm(kind="gmN", param=name, alpha=float(alpha)))

    def _param(self, name, args):
        mean = float(args[0])
        if "/" in args[1]:
            lo_s, hi_s = args[1].split("/")
            if not (lo_s.startswith("-") and hi_s.startswith("+")):
                raise UnsupportedFeature(f"asymmetric param '{name}' must be written -x/+y")
            sig_lo, sig_hi = float(lo_s[1:]), float(hi_s[1:])
            kind = I.CONSTRAINT_BIFURGAUSS
        else:
            sig_lo = sig_hi = float(args[1])
            kind = I.CONSTRAINT_GAUSS
        if len(args) >= 3:
            lo, hi = _parse_range(args[2])
        else:
            lo, hi = mean - 4.0 * sig_lo, mean + 4.0 * sig_hi
        constraint = I.Constraint(kind=kind, center=mean, sigma_lo=sig_lo, sigma_hi=sig_hi)
        existing = self.params.get(name)
        if existing is not None and existing.role not in (I.ROLE_FREE, I.ROLE_CONSTANT):
            raise UnsupportedFeature(f"param '{name}' is already a {existing.origin} nuisance")
        value = existing.value if existing is not None else mean
        self.params[name] = I.Parameter(name=name, value=value, lo=lo, hi=hi, role=I.ROLE_NUISANCE,
                                        constraint=constraint, origin="param")

    def _find_constant_ws_var(self, name):
        """The workspace variable a param line refers to (Combine makes it floating), if any."""
        from modelspec import rootinput as R

        seen = set()
        for b in self.dc.bins:
            for (bb, p, _s) in self.dc.keyline:
                if bb != b:
                    continue
                e = self.shapes.entry(b, p)
                if e is None or len(e) < 2 or ":" not in e[1]:
                    continue
                key = (self.shapes.file(e), e[1].split(":", 1)[0])
                if key in seen:
                    continue
                seen.add(key)
                var = R.get_workspace(*key).var(name)
                if var:
                    return var
        return None

    def _rate_params(self, channels):
        for key, entries in self.dc.rateParams.items():
            b, p = key.split("AND", 1)
            proc = self._proc(channels, b, p)
            for spec, rng in entries:
                if spec[-1] == 0:
                    name, value = spec[0], float(spec[1])
                    if rng:
                        lo, hi = _parse_range(rng)
                    else:
                        lo, hi = -math.inf, math.inf
                    par = self.params.get(name)
                    if par is None:
                        par = I.Parameter(name=name, value=value, lo=lo, hi=hi, role=I.ROLE_FREE, origin="rateParam")
                        self.params[name] = par
                    proc.norm_terms.append(I.NormTerm(kind="rate_param", param=name))
                elif spec[-1] == 1:
                    name, formula, args = spec[0], spec[1], spec[2].split(",")
                    missing = [a for a in args if a not in self.params]
                    if missing:
                        raise UnsupportedFeature(f"rateParam formula '{name}' uses unknown parameters {missing}")
                    proc.norm_terms.append(I.NormTerm(kind="formula", param=name, formula=formula, args=args))
                else:
                    raise UnsupportedFeature(f"rateParam '{spec[0]}' read from a ROOT file is not supported yet")

    def _discretes(self):
        for name in self.dc.discretes:
            par = self.params.get(name)
            if par is None or par.role != I.ROLE_DISCRETE:
                raise UnsupportedFeature(f"discrete '{name}' is not the category of a RooMultiPdf in the model")

    def _check_unused_params(self, channels):
        used = {"r"}
        for ch in channels:
            for proc in ch.processes:
                used.update(t.param for t in proc.norm_terms if t.param)
                for t in proc.norm_terms:
                    used.update(t.args)
                used.update(s.param for s in proc.shape.systs)
                used.update(proc.shape.params)
                used.update(proc.shape.syst_refs)
                if proc.shape.category:
                    used.add(proc.shape.category)
        for name, par in self.params.items():
            if name not in used:
                self.notes.append(f"parameter '{name}' ({par.origin}) does not affect any process")


def build_ir(card_path: str, *, mass: str = "120", poi_range=DEFAULT_POI_RANGE,
             bin_integration: str = "center") -> I.ModelIR:
    """Parse a Combine datacard and its shape inputs into a ModelIR."""
    return _Builder(card_path, mass, poi_range, bin_integration).build()
