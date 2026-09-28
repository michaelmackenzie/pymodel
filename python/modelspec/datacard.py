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

def _require_th1(obj, spec, b, p):
    if obj.GetDimension() != 1:
        raise UnsupportedFeature(
            f"{spec} ({b}/{p}) is a {obj.ClassName()}: Combine's ShapeTools accepts only TH1 histograms in shapes lines "
            "(text2workspace fails with 'This method currently supports only TH1s, RooDataHists and RooAbsPdfs'). "
            "Unroll it into a TH1, or put a RooDataHist/RooHistPdf over the N observables in a RooWorkspace")


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
        self.th1_procs = set()  # (channel, process) whose nominal shape is a TH1
        self._mcstats_group: List[str] = []  # autoMCStats parameters (Combine's group_autoMCStats)

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
        self._mcstats(channels)
        self._discretes()
        for name in dc.frozenNuisances:
            if name not in self.params:
                raise UnsupportedFeature(f"'nuisance edit freeze' of unknown parameter '{name}'")
            self.params[name].role = I.ROLE_CONSTANT
        groups = {g: sorted(members) for g, members in dc.groups.items()}
        if self._mcstats_group:
            groups["autoMCStats"] = list(self._mcstats_group)  # Combine's group_autoMCStats
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
            _require_th1(obj, spec, b, "data_obs")
            edges = R.th1_edges(obj)
            counts, _ = R.th1_contents(obj)
            obs = I.Observable(name=f"CMS_th1x_{b}", lo=edges[0], hi=edges[-1], edges=edges)
            return obs, I.ChannelData(kind="binned", counts=counts), None, None
        if obj.InheritsFrom("RooAbsData"):
            if obj.get().getSize() != 1:
                return self._data_nd(b, obj, path, ws, spec)
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

    def _data_nd(self, b, obj, path, ws, spec):
        """data_obs over N >= 2 variables: the channel observable is their product binning
        (flattened row-major in the order of the data_obs variables, ir.Observable)."""
        from modelspec import rootinput as R

        names = [v.GetName() for v in obj.get()]
        if not all(v.InheritsFrom("RooRealVar") for v in obj.get()):
            raise UnsupportedFeature(f"data_obs '{spec}' of channel '{b}' has non-real observables {names}")
        if len(names) > 3:
            raise UnsupportedFeature(f"data_obs of channel '{b}' has {len(names)} observables {names}; Combine's "
                                     "toy and Asimov generation (SinglePdfGenInfo::generateWithHisto) supports at most 3")
        ws_obj = R.get_workspace(path, ws)
        variables = [ws_obj.var(n) for n in names]
        axes = [I.Axis(name=v.GetName(), lo=v.getMin(), hi=v.getMax(), edges=R.var_edges(v)) for v in variables]
        obs = I.Observable.multi(axes)
        self.notes.append(f"channel {b}: {len(names)}-dimensional observable ({', '.join(names)}) with "
                          f"{' x '.join(str(k) for k in obs.shape)} = {obs.nbins} bins")
        if obj.InheritsFrom("RooDataHist"):
            try:
                counts, _ = R.datahist_contents_nd(obj, variables, obs)
            except ValueError as err:
                raise UnsupportedFeature(f"data_obs of channel '{b}': {err}") from err
            return obs, I.ChannelData(kind="binned", counts=counts), variables, None
        values, weights = R.dataset_values_nd(obj, names)
        return obs, I.ChannelData(kind="unbinned", values=values, weights=weights), variables, None

    def _check_pdf_observables(self, b, p, spec, obj, observable: I.Observable, obs_var):
        """A RooHistPdf depends only on its observables: all of them must be axes of the channel
        (with a 1D data_obs Combine would treat the other ones as floating parameters).  In an
        N-D channel every pdf must depend on every axis (RooFit does not normalise a pdf over an
        axis it does not depend on, so its density would carry the axis range as a factor)."""
        axes = [a.name for a in observable.axis_list()]
        if obj.InheritsFrom("RooHistPdf"):
            extra = sorted(v.GetName() for v in obj.getVariables() if v.GetName() not in axes)
            if extra:
                raise UnsupportedFeature(
                    f"the RooHistPdf '{spec}' of {b}/{p} is defined over {extra + axes} but data_obs of channel '{b}' "
                    f"only over {axes}: Combine would treat {extra} as floating parameters of the pdf. Use a data_obs "
                    f"over all of {extra + axes}")
        if observable.ndim > 1:
            missing = [v.GetName() for v in obs_var if not obj.dependsOn(v)]
            if missing:
                raise UnsupportedFeature(f"the pdf '{spec}' of {b}/{p} does not depend on the observable(s) {missing} "
                                         f"of the {observable.ndim}-dimensional channel '{b}'")

    def _process(self, b, p, observable: I.Observable, obs_var) -> I.Process:
        from modelspec import rootinput as R

        dc = self.dc
        rate = float(dc.exp[b][p])
        obj, path, ws, spec = self.shapes.nominal(b, p)
        is_signal = bool(dc.isSignal[p])
        norm_terms: List[I.NormTerm] = []
        if obj.InheritsFrom("TH1") or obj.InheritsFrom("RooDataHist"):
            if obj.InheritsFrom("TH1"):
                _require_th1(obj, spec, b, p)
                if observable.ndim > 1:
                    raise UnsupportedFeature(f"TH1 template {spec} of {b}/{p} in the {observable.ndim}-dimensional "
                                             f"channel '{b}' (Combine would add a separate CMS_th1x observable)")
                self.th1_procs.add((b, p))
                if not np.allclose(R.th1_edges(obj), observable.edges):
                    raise UnsupportedFeature(f"template {spec} binning differs from data_obs in channel {b}")
                contents, sumw2 = R.th1_contents(obj)
            elif observable.ndim > 1:
                try:
                    contents, sumw2 = R.datahist_contents_nd(obj, obs_var, observable)
                except ValueError as err:
                    raise UnsupportedFeature(f"template {spec} of {b}/{p}: {err}") from err
                if self._has_shape_systs(b, p):
                    raise UnsupportedFeature(f"shape systematics on the {observable.ndim}-dimensional RooDataHist "
                                             f"template {b}/{p} are not supported")
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
        self._check_pdf_observables(b, p, spec, obj, observable, obs_var)
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
            if observable.ndim > 1 and self._has_shape_systs(b, p):
                raise UnsupportedFeature(
                    f"shape systematics on the pdf {b}/{p} of the {observable.ndim}-dimensional channel '{b}': Combine's "
                    "text2workspace fails on 2D RooHistPdf variations (ShapeTools.getPdf calls RooArgSet.second(), "
                    "which PyROOT does not provide), and VerticalInterpPdf morphing of N-D pdfs is not supported yet")
            variations = self._pdf_systs(b, p, obj, shape, obs_var)
            if not shape.params and observable.ndim > 1:
                shape.contents = R.binned_pdf_contents_nd(obj, obs_var, observable, self.bin_integration)
            elif not shape.params:
                shape.contents = R.binned_pdf_contents(obj, obs_var, observable.edges, self.bin_integration)
                self._fixed_pdf_systs(shape, obj, variations, obs_var, observable)
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
        if shape.kind == "envelope" and self._has_shape_systs(b, p):
            raise UnsupportedFeature(f"shape systematics on the RooMultiPdf {b}/{p} are not supported")
        return I.Process(name=p, is_signal=is_signal, rate=rate, shape=shape, norm_terms=norm_terms)

    def _has_shape_systs(self, b, p) -> bool:
        return any(self._entry(syst, b, p) != 0 and not (pdf.endswith("?") and
                                                         not self.shapes.systematic_exists(b, p, syst))
                   for syst, pdf in self.shape_systs.items())

    def _pdf_systs(self, b, p, nominal, shape: I.Shape, obs_var):
        """Fill ``shape.pdf_systs`` / ``pdf_morph`` of a RooAbsPdf shape (ShapeTools.getPdf):
        a RooHistPdf nominal is morphed by FastVerticalInterpHistPdf2, any other pdf by
        VerticalInterpPdf.  Returns [(PdfSyst, up pdf, down pdf)].  Combine reads no ``_norm``
        of the Up/Down pdfs (getShape only looks for the nominal one) and adds no normalisation
        term for RooAbsPdf shapes (getExtraNorm)."""
        from modelspec import rootinput as R

        out, algos, ignored = [], set(), []
        for syst, pdf in self.shape_systs.items():
            scale = self._entry(syst, b, p)
            if scale == 0:
                continue
            if pdf.endswith("?") and not self.shapes.systematic_exists(b, p, syst):
                continue
            algo = pdf.rstrip("?")
            if algo not in ("shape", "shapeN"):
                raise UnsupportedFeature(f"systematic type '{pdf}' is not supported on the pdf {b}/{p}")
            algos.add(algo)
            shape_name = self.dc.systematicsShapeMap.get((syst, b, p), syst)
            objs, refs = [], []
            for direction in ("Up", "Down"):
                obj, path, ws, spec = self.shapes.systematic(b, p, shape_name, direction)
                if obj.ClassName() != nominal.ClassName():
                    raise UnsupportedFeature(f"{spec} for {b}/{p} is a {obj.ClassName()}, the nominal pdf is a "
                                             f"{nominal.ClassName()} (Combine: mismatched shape types)")
                if R.get_workspace(path, ws).arg(obj.GetName() + "_norm"):
                    ignored.append(obj.GetName() + "_norm")
                for v in R.floating_params(obj, obs_var):
                    self.add_workspace_param(v, path, ws)
                    if v.GetName() not in shape.params:
                        shape.params.append(v.GetName())
                objs.append(obj)
                refs.append(I.RooRef(file=path, workspace=ws, name=obj.GetName(), class_name=obj.ClassName()))
            out.append((I.PdfSyst(param=syst, up=refs[0], down=refs[1], scale=float(scale), kind=algo), *objs))
        if ignored:
            self.notes.append(f"{b}/{p}: {', '.join(ignored)} ignored (Combine uses only the nominal _norm: a shape "
                              "systematic on a pdf has no normalisation effect)")
        if not out:
            return out
        if len(algos) > 1:
            raise UnsupportedFeature(f"{b}/{p} mixes the morphing algorithms {sorted(algos)} (Combine allows one "
                                     "per shape)")
        if nominal.InheritsFrom("RooParametricHist") or nominal.InheritsFrom("RooMultiPdf"):
            raise UnsupportedFeature(f"shape systematics on the {nominal.ClassName()} {b}/{p} are not supported")
        if nominal.InheritsFrom("RooHistPdf"):
            if nominal.dataHist().get().getSize() != 1:
                raise UnsupportedFeature(f"shape systematics on the multi-dimensional RooHistPdf {b}/{p}")
            if shape.params:
                raise UnsupportedFeature(f"RooHistPdf shape systematics of {b}/{p} depend on floating parameters "
                                         f"{shape.params} (Combine's FastVerticalInterpHistPdf2 refuses them)")
            shape.pdf_morph = I.PDF_MORPH_HIST
        else:
            shape.pdf_morph = I.PDF_MORPH_VERTICAL
        shape.pdf_systs = [o[0] for o in out]
        return out

    def _fixed_pdf_systs(self, shape: I.Shape, nominal, variations, obs_var, observable):
        """Binned contents (and un-normalised integrals) of fixed pdfs and their variations."""
        from modelspec import rootinput as R

        if not variations:
            return
        nset = R.root().RooArgSet(obs_var)
        shape.raw_integral = float(nominal.createIntegral(nset).getVal())
        for syst, up, down in variations:
            syst.up_contents = R.binned_pdf_contents(up, obs_var, observable.edges, self.bin_integration)
            syst.down_contents = R.binned_pdf_contents(down, obs_var, observable.edges, self.bin_integration)
            syst.up_integral = float(up.createIntegral(nset).getVal())
            syst.down_integral = float(down.createIntegral(nset).getVal())

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
                used |= any(s.param == name for ch in channels for proc in ch.processes for s in proc.shape.pdf_systs)
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

    # -- autoMCStats -------------------------------------------------------------------
    def _mcstats(self, channels):
        """Channel.mcstats from ``<channel> autoMCStats threshold [include-signal] [hist-mode]``
        (Combine ShapeTools.doIndividualModels + CMSHistErrorPropagator::setupBinPars)."""
        from modelspec import semantics as S

        for ch in channels:
            flags = self.dc.binParFlags.get(ch.name)
            if flags is None:
                continue
            threshold, include_signal, hist_mode = float(flags[0]), bool(flags[1]), int(flags[2])
            if hist_mode != 1:
                raise UnsupportedFeature(f"autoMCStats hist-mode {hist_mode} in channel '{ch.name}' is not supported "
                                         "(only the default hist-mode 1)")
            bad = [p.name for p in ch.processes if (ch.name, p.name) not in self.th1_procs]
            if bad:
                raise UnsupportedFeature(
                    f"autoMCStats in channel '{ch.name}' with non-TH1 processes {bad}: Combine's "
                    "CMSHistErrorPropagator needs a CMSHistFunc (TH1 template) for every process")
            procs = []
            for proc in ch.processes:
                shape = proc.shape
                if any(s.kind != "shape" for s in shape.systs):
                    raise UnsupportedFeature(f"autoMCStats channel '{ch.name}': {proc.name} uses shapeN "
                                             "(CMSHistFunc LogQuadLinear morphing is not supported)")
                if proc.rate == 0.0:
                    continue  # Combine drops processes with rate 0
                integral = float(np.sum(shape.contents))
                if integral <= 0.0:
                    if any(w > 0 for w in shape.sumw2):
                        raise UnsupportedFeature(f"autoMCStats channel '{ch.name}': template {proc.name} has zero "
                                                 "integral but non-zero bin errors")
                    self.notes.append(f"autoMCStats channel {ch.name}: {proc.name} has an empty template (yield 0)")
                    proc.rate = 0.0
                    continue
                if abs(proc.rate - integral) > 1e-9 * max(1.0, integral):
                    self.notes.append(f"autoMCStats channel {ch.name}: {proc.name} rate {proc.rate:g} replaced by the "
                                      f"template integral {integral:g} (Combine normalises CMSHistFunc templates by "
                                      "their own integral)")
                proc.rate = integral
                for t in proc.norm_terms:
                    if t.kind == "gmN":
                        n_obs = self.params[t.param].constraint.center
                        # Combine: coefficient n / N (n for N = 0) times the raw template
                        t.alpha = integral / n_obs if n_obs > 0 else integral
                h0, snorm = S.cmshist_template(shape.contents, [(s.up, s.down, s.scale) for s in shape.systs],
                                               [self.params[s.param].value for s in shape.systs])
                c0 = self._nominal_norm(proc) * snorm / integral
                procs.append((proc.name, proc.is_signal, c0, h0, np.sqrt(np.asarray(shape.sumw2, dtype=float))))
            cfg = I.MCStats(threshold=threshold, include_signal=include_signal, hist_mode=hist_mode)
            if threshold >= 0.0:
                for j, kind, pname, n_eff, name in S.bb_lite_classify(ch.name, procs, threshold, include_signal):
                    if name in self.params:
                        raise UnsupportedFeature(f"autoMCStats parameter '{name}' clashes with an existing parameter")
                    if kind == "poisson":
                        lo, hi = S.poisson_bb_range(n_eff)
                        par = I.Parameter(name=name, value=n_eff, lo=lo, hi=hi, role=I.ROLE_NUISANCE,
                                          constraint=I.Constraint(kind=I.CONSTRAINT_POISSON, center=n_eff),
                                          origin="autoMCStats")
                    else:
                        par = _gauss_nuisance(name, -S.BB_SIGMA_RANGE, S.BB_SIGMA_RANGE, "autoMCStats")
                    self.params[name] = par
                    cfg.params.append(I.MCStatsParam(bin=j, kind=kind, param=name, process=pname, n_eff=n_eff))
            ch.mcstats = cfg
            names = [bp.param for bp in cfg.params]
            if names:
                if "autoMCStats" in self.dc.groups:
                    raise UnsupportedFeature("a nuisance group named 'autoMCStats' clashes with Combine's own group")
                self._mcstats_group += names

    def _nominal_norm(self, proc: I.Process) -> float:
        """rate * r^[signal] * prod(norm terms) at the nominal parameter values."""
        from modelspec import semantics as S

        v = {n: p.value for n, p in self.params.items()}
        f = 1.0 if any(t.kind == "gmN" for t in proc.norm_terms) else proc.rate
        if proc.is_signal:
            f *= v["r"]
        for t in proc.norm_terms:
            if t.kind in ("lnN", "lnU"):
                f *= t.kappa_hi ** v[t.param]
            elif t.kind == "asym_lnN":
                f *= float(S.asym_pow(v[t.param], t.kappa_lo, t.kappa_hi))
            elif t.kind == "gmN":
                f *= t.alpha * v[t.param]
            elif t.kind == "rate_param":
                f *= v[t.param]
            elif t.kind == "formula":
                from modelspec import rootinput as R

                ROOT = R.root()
                args = ROOT.RooArgList()
                keep = [ROOT.RooRealVar(a, a, v[a]) for a in t.args]
                for a in keep:
                    args.add(a)
                f *= ROOT.RooFormulaVar(f"nominal_{t.param}", t.formula, args).getVal()
            else:
                raise UnsupportedFeature(f"autoMCStats: norm term '{t.kind}' on {proc.name}")
        return f

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
                used.update(s.param for s in proc.shape.pdf_systs)
                if proc.shape.category:
                    used.add(proc.shape.category)
            if ch.mcstats is not None:
                used.update(bp.param for bp in ch.mcstats.params)
        for name, par in self.params.items():
            if name not in used:
                self.notes.append(f"parameter '{name}' ({par.origin}) does not affect any process")


def build_ir(card_path: str, *, mass: str = "120", poi_range=DEFAULT_POI_RANGE,
             bin_integration: str = "center") -> I.ModelIR:
    """Parse a Combine datacard and its shape inputs into a ModelIR."""
    return _Builder(card_path, mass, poi_range, bin_integration).build()
