"""Backend-neutral model intermediate representation (IR).

A ``ModelIR`` is what a Combine datacard (plus its shape inputs) means, written out
explicitly.  Every backend builds its likelihood from a ``ModelIR``; no backend parses
datacards or opens shape files on its own (parametric RooFit objects are the one
exception: they are carried as ``RooRef`` references that the backend resolves).

Semantics follow CMS Combine (text2workspace.py with the default physics model):

* expected yield of process p in channel c::

      nu_cp = rate_cp * r^[p is signal] * prod(norm terms) * shape integral

  where ``rate_cp`` is the datacard ``rate`` (with ``-1`` already resolved to the
  template integral) multiplied by the value of ``<pdf>_norm`` for parametric shapes.
* ``lnN`` (symmetric kappa):          kappa ** theta
* ``lnN`` (asymmetric down/up):       asymPow(theta, kappa_down, kappa_up) (see semantics.py)
* ``lnU``:                            kappa ** theta, theta flat in [-1, 1]
* ``gmN N alpha``:                    alpha * n, with n Poisson-constrained by N
* ``rateParam``:                      a free multiplicative parameter (or formula)
* template ``shape`` systematics:     vertical morphing of the normalised templates with
                                      Combine's smooth step, plus an asymPow normalisation
                                      term with kappa = integral(up|down) / integral(nominal)
* ``shape`` systematics on RooAbsPdfs (``PdfSyst``, ``Shape.pdf_morph``): RooHistPdf
  nominal -> the template morph of the normalised histograms (FastVerticalInterpHistPdf2);
  any other pdf -> VerticalInterpPdf, the morph of the *un-normalised* pdf values divided by
  the same morph of their integrals.  No normalisation effect (the Up/Down ``_norm`` objects
  are ignored, as in Combine).  See ``semantics.vertical_pdf_fractions``.
* constraint terms are listed once per parameter (``Constraint``); every constraint has a
  global observable whose nominal value is ``Constraint.center``.
* ``autoMCStats`` (``Channel.mcstats``): Barlow-Beeston-lite bin parameters of
  CMSHistErrorPropagator, see ``MCStats`` and ``semantics.bb_lite_expected``.
* multi-dimensional channels (``Observable.axes``, data_obs a RooDataHist/RooDataSet over
  several variables): the binned likelihood runs over the flattened product bins (row-major),
  pdfs are evaluated at the N-D bin centres times the bin volume; unbinned data are N-D
  points.  Everything else (yields, templates, constraints) is unchanged.

All arrays are plain python lists of floats so the IR serialises to JSON directly.
"""

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

IR_VERSION = 1

# Parameter roles
ROLE_POI = "poi"
ROLE_NUISANCE = "nuisance"  # has a constraint term
ROLE_FREE = "free"          # floating, unconstrained (rateParam, floating pdf parameters, flatParam)
ROLE_CONSTANT = "constant"  # fixed value, kept for bookkeeping / --set-parameters
ROLE_DISCRETE = "discrete"  # RooMultiPdf category index

# Constraint kinds
CONSTRAINT_GAUSS = "gauss"            # N(g | theta, sigma), sigma_lo == sigma_hi
CONSTRAINT_BIFURGAUSS = "bifurgauss"  # asymmetric Gaussian (param with -x/+y)
CONSTRAINT_POISSON = "poisson"        # Poisson(g | n) for gmN
CONSTRAINT_FLAT = "flat"              # lnU: uniform on the parameter range, no global observable


@dataclass
class Constraint:
    kind: str
    center: float = 0.0     # nominal value of the global observable
    sigma_lo: float = 1.0
    sigma_hi: float = 1.0

    @property
    def has_global_observable(self) -> bool:
        return self.kind != CONSTRAINT_FLAT


@dataclass
class Parameter:
    name: str
    value: float
    lo: float
    hi: float
    role: str
    constraint: Optional[Constraint] = None
    origin: str = ""                   # e.g. "lnN", "shape", "param", "rateParam", "workspace"
    n_states: int = 0                  # for ROLE_DISCRETE: number of category states

    @property
    def floating(self) -> bool:
        return self.role in (ROLE_POI, ROLE_NUISANCE, ROLE_FREE)


@dataclass
class RooRef:
    """Reference to an object inside a RooWorkspace in a ROOT file."""

    file: str
    workspace: str
    name: str
    class_name: str = ""


@dataclass
class NormTerm:
    """One multiplicative factor on a process yield.

    kind:
      "lnN"        -> kappa_hi ** theta (symmetric: kappa_lo == 1/kappa_hi, flag symmetric)
      "asym_lnN"   -> asymPow(theta, kappa_lo, kappa_hi)
      "lnU"        -> kappa_hi ** theta
      "gmN"        -> alpha * n   (the datacard rate is ignored for this process)
      "rate_param" -> value of parameter ``param``
      "formula"    -> value of ``formula`` (Combine/TFormula syntax, @0..@n refer to ``args``)
      "ws_norm"    -> value of the RooAbsReal ``ref`` (a non-trivial <pdf>_norm function)
    """

    kind: str
    param: str = ""
    kappa_lo: float = 1.0
    kappa_hi: float = 1.0
    alpha: float = 1.0
    formula: str = ""
    args: List[str] = field(default_factory=list)
    ref: Optional[RooRef] = None


@dataclass
class TemplateSyst:
    """A ``shape``/``shapeN`` systematic on a histogram template.

    ``up``/``down`` are the raw (un-normalised) varied templates, same binning as nominal.
    ``scale`` is the datacard entry (Combine: theta is effectively divided by it).
    """

    param: str
    up: List[float]
    down: List[float]
    scale: float = 1.0
    kind: str = "shape"  # "shape" (vertical morph) or "shapeN" (log-vertical morph)


# Shape.pdf_morph: how Combine (ShapeTools.getPdf) morphs a RooAbsPdf shape
PDF_MORPH_VERTICAL = "vertical"  # VerticalInterpPdf (any RooAbsPdf that is not a RooHistPdf)
PDF_MORPH_HIST = "hist"          # FastVerticalInterpHistPdf2 (fixed RooHistPdf nominal and variations)


@dataclass
class PdfSyst:
    """A ``shape``/``shapeN`` systematic on a RooAbsPdf shape (datacard ``shapes`` pattern
    with ``$SYSTEMATIC``).

    ``up``/``down`` reference the varied pdfs, ``scale`` is the datacard entry (the morphing
    coefficient is scale * theta; the smooth/quadratic region is min(1, scales of the process)).
    For fixed shapes (``Shape.contents`` set) ``up_contents``/``down_contents`` hold the varied
    pdfs on the channel binning (same convention as ``Shape.contents``) and
    ``up_integral``/``down_integral`` their un-normalised integrals over the observable range
    (the VerticalInterpPdf normalisation).  Up/Down ``_norm`` objects play no role.
    """

    param: str
    up: RooRef
    down: RooRef
    scale: float = 1.0
    kind: str = "shape"  # "shape" or "shapeN"
    up_contents: List[float] = field(default_factory=list)
    down_contents: List[float] = field(default_factory=list)
    up_integral: float = 0.0
    down_integral: float = 0.0


@dataclass
class Shape:
    """Process shape.

    kind:
      "counting"   -> no shape (single-bin counting channel)
      "template"   -> histogram: ``contents`` (+ ``sumw2``) on the channel binning
      "parametric" -> RooAbsPdf referenced by ``ref``; ``params`` lists the names of the IR
                      parameters it depends on.  If neither the pdf nor its ``pdf_systs``
                      variations depend on floating parameters, ``contents`` holds its
                      histogram on the channel binning (see ``Channel.bin_integration``).
                      ``pdf_systs`` are its shape systematics, morphed as ``pdf_morph`` says;
                      ``raw_integral`` is the un-normalised integral of the nominal pdf.
      "envelope"   -> RooMultiPdf referenced by ``ref`` with category ``category``
    """

    kind: str
    contents: List[float] = field(default_factory=list)
    sumw2: List[float] = field(default_factory=list)
    systs: List[TemplateSyst] = field(default_factory=list)
    ref: Optional[RooRef] = None
    params: List[str] = field(default_factory=list)
    category: str = ""
    pdf_systs: List[PdfSyst] = field(default_factory=list)
    pdf_morph: str = ""        # PDF_MORPH_* when pdf_systs is not empty
    raw_integral: float = 0.0  # fixed shapes with pdf_systs: integral of the un-normalised nominal pdf


@dataclass
class Process:
    name: str
    is_signal: bool
    rate: float
    shape: Shape
    norm_terms: List[NormTerm] = field(default_factory=list)


@dataclass
class Axis:
    """One axis of a multi-dimensional observable (a RooRealVar with its binning)."""

    name: str
    lo: float
    hi: float
    edges: List[float]


@dataclass
class Observable:
    """The observable of a channel.

    1D (``axes`` empty): ``name``/``lo``/``hi``/``edges`` (always set; counting: [0, 1]).

    N-D (``axes`` holds the N >= 2 axes in the order of the data_obs variables): the bins are
    the cells of the product of the axis binnings, flattened in row-major order (the last
    axis varies fastest, like numpy's C order and RooDataHist's internal index): cell
    (i_0, ..., i_{N-1}) is bin ``np.ravel_multi_index((i_0, ...), shape)``.  Every per-bin
    array of the IR (counts, template contents, pdf fractions) uses this order.  ``name`` is
    the comma-separated axis names, ``edges`` is empty, ``lo``/``hi`` are 0 and the number
    of bins (the range of the flattened bin index).  Use ``nbins``, ``bin_volumes()`` and
    ``bin_centers()``, which work for both.
    """

    name: str
    lo: float
    hi: float
    edges: List[float] = field(default_factory=list)
    axes: List[Axis] = field(default_factory=list)

    @staticmethod
    def multi(axes: List[Axis]) -> "Observable":
        if len(axes) < 2:
            raise ValueError("a multi-dimensional observable needs at least two axes")
        nbins = 1
        for a in axes:
            nbins *= len(a.edges) - 1
        return Observable(name=",".join(a.name for a in axes), lo=0.0, hi=float(nbins), edges=[], axes=list(axes))

    @property
    def ndim(self) -> int:
        return len(self.axes) if self.axes else 1

    def axis_list(self) -> List[Axis]:
        """The axes (1D: one axis made of name/lo/hi/edges)."""
        return list(self.axes) if self.axes else [Axis(self.name, self.lo, self.hi, list(self.edges))]

    @property
    def shape(self) -> tuple:
        return tuple(len(a.edges) - 1 for a in self.axis_list())

    @property
    def nbins(self) -> int:
        n = 1
        for k in self.shape:
            n *= k
        return n

    def bin_volumes(self):
        """Width (1D) or volume (N-D: product of the axis widths) of every flattened bin."""
        import numpy as np

        vol = np.ones(1)
        for a in self.axis_list():
            vol = np.multiply.outer(vol, np.diff(np.asarray(a.edges, dtype=float)))
        return vol.reshape(-1)

    def bin_centers(self):
        """Array (nbins, ndim) of the bin centres, rows in the flattened bin order."""
        import numpy as np

        cs = [0.5 * (np.asarray(a.edges[:-1], dtype=float) + np.asarray(a.edges[1:], dtype=float))
              for a in self.axis_list()]
        grid = np.meshgrid(*cs, indexing="ij")
        return np.column_stack([g.reshape(-1) for g in grid])


@dataclass
class ChannelData:
    """Observed data of one channel.

    kind "count":    ``counts`` has one entry
    kind "binned":   ``counts`` per bin of the observable (N-D: flattened, see Observable)
    kind "unbinned": ``values`` (and ``weights`` if weighted); N-D: one [x_0, ..., x_{N-1}]
                     list per event
    """

    kind: str
    counts: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    weights: List[float] = field(default_factory=list)

    @property
    def total(self) -> float:
        if self.kind == "unbinned":
            return float(sum(self.weights)) if self.weights else float(len(self.values))
        return float(sum(self.counts))


# autoMCStats bin-parameter kinds (Combine CMSHistErrorPropagator::setupBinPars bintypes 1, 2, 3)
MCSTATS_TOTAL = "total"      # one Gaussian x per bin: nu_i += x * sqrt(sum_p (C_p e_pi)^2)
MCSTATS_POISSON = "poisson"  # per process: nu_pi *= gamma / n_eff, gamma ~ Poisson(n_eff | gamma)
MCSTATS_GAUSS = "gauss"      # per process: nu_pi += x * C_p e_pi


@dataclass
class MCStatsParam:
    """One autoMCStats parameter of bin ``bin`` (index into the channel bins).

    ``process`` is empty for MCSTATS_TOTAL; ``n_eff`` is the rounded effective MC event count
    of the process (the gamma divisor and Poisson global observable) for MCSTATS_POISSON."""

    bin: int
    kind: str
    param: str
    process: str = ""
    n_eff: float = 0.0


@dataclass
class MCStats:
    """``<channel> autoMCStats threshold [include-signal] [hist-mode]`` of one channel.

    The classification of the bins (``params``) is made once, at the nominal parameter
    values, exactly as Combine's setupBinPars; e_pi = sqrt(Shape.sumw2) of the templates.
    In such a channel every process is a TH1 template evaluated like CMSHistFunc (see
    ``semantics.cmshist_template``): the yield is the template integral (the datacard rate is
    not used), up/down templates are rescaled to the nominal integral (hist-mode 1), and
    morphed bins are floored at 1e-9 without renormalisation."""

    threshold: float
    include_signal: bool = False
    hist_mode: int = 1
    params: List[MCStatsParam] = field(default_factory=list)


@dataclass
class Channel:
    name: str
    observable: Observable
    data: ChannelData
    processes: List[Process]
    bin_integration: str = "center"  # parametric pdfs on binned data: "center" (Combine/RooFit) or "integral"
    obs_ref: Optional[RooRef] = None  # RooRealVar of the observable when shapes come from a workspace
    mcstats: Optional[MCStats] = None  # autoMCStats configuration (None: not used)

    @property
    def is_counting(self) -> bool:
        return all(p.shape.kind == "counting" for p in self.processes)


@dataclass
class ModelIR:
    channels: List[Channel]
    parameters: Dict[str, Parameter]
    poi: str = "r"
    groups: Dict[str, List[str]] = field(default_factory=dict)
    source: str = ""        # datacard path the IR was made from
    notes: List[str] = field(default_factory=list)  # informational messages from the conversion
    ir_version: int = IR_VERSION

    # ----- helpers -------------------------------------------------------------------
    def channel(self, name: str) -> Channel:
        for ch in self.channels:
            if ch.name == name:
                return ch
        raise KeyError(f"No channel named '{name}'")

    def floating_parameters(self) -> List[Parameter]:
        return [p for p in self.parameters.values() if p.floating]

    def constrained_parameters(self) -> List[Parameter]:
        return [p for p in self.parameters.values() if p.constraint is not None]

    def features(self) -> set:
        """The set of IR features used by this model; backends compare this with what they support."""
        feats = set()
        for ch in self.channels:
            feats.add(f"data:{ch.data.kind}")
            if ch.observable.ndim > 1:
                feats.add("obs:multidim")  # N-D observable (flattened bins / N-D events)
            if ch.mcstats is not None:
                feats.add("mcstats:bb-lite")
            if ch.data.kind == "unbinned" and ch.data.weights:
                feats.add("data:weighted")
            for proc in ch.processes:
                feats.add(f"shape:{proc.shape.kind}")
                if proc.shape.kind == "parametric" and proc.shape.contents:
                    feats.add("shape:parametric-histogram")
                for syst in proc.shape.systs:
                    feats.add(f"syst:{syst.kind}")
                for syst in proc.shape.pdf_systs:
                    # syst:pdf-morph (VerticalInterpPdf) / syst:histpdf-morph (RooHistPdf), N: shapeN
                    prefix = "histpdf" if proc.shape.pdf_morph == PDF_MORPH_HIST else "pdf"
                    feats.add(f"syst:{prefix}-morph" + ("N" if syst.kind == "shapeN" else ""))
                for term in proc.norm_terms:
                    feats.add(f"norm:{term.kind}")
        for par in self.parameters.values():
            if par.constraint is not None:
                feats.add(f"constraint:{par.constraint.kind}")
            if par.role == ROLE_DISCRETE:
                feats.add("discrete")
        return feats

    def to_dict(self) -> dict:
        return asdict(self)


def _constraint_from_dict(d):
    return None if d is None else Constraint(**d)


def _ref_from_dict(d):
    return None if d is None else RooRef(**d)


def _mcstats_from_dict(d):
    if d is None:
        return None
    d = dict(d)
    d["params"] = [MCStatsParam(**p) for p in d.get("params", [])]
    return MCStats(**d)


def ir_from_dict(d: dict) -> ModelIR:
    if d.get("ir_version") != IR_VERSION:
        raise ValueError(f"Unsupported model IR version {d.get('ir_version')} (expected {IR_VERSION})")
    channels = []
    for ch in d["channels"]:
        procs = []
        for p in ch["processes"]:
            s = dict(p["shape"])
            s["systs"] = [TemplateSyst(**t) for t in s.get("systs", [])]
            s["ref"] = _ref_from_dict(s.get("ref"))
            s["pdf_systs"] = [PdfSyst(**{**t, "up": RooRef(**t["up"]), "down": RooRef(**t["down"])})
                              for t in s.get("pdf_systs", [])]
            terms = []
            for t in p["norm_terms"]:
                t = dict(t)
                t["ref"] = _ref_from_dict(t.get("ref"))
                terms.append(NormTerm(**t))
            procs.append(Process(name=p["name"], is_signal=p["is_signal"], rate=p["rate"],
                                 shape=Shape(**s), norm_terms=terms))
        obs = dict(ch["observable"])
        obs["axes"] = [Axis(**a) for a in obs.get("axes", [])]
        channels.append(Channel(
            name=ch["name"],
            observable=Observable(**obs),
            data=ChannelData(**ch["data"]),
            processes=procs,
            bin_integration=ch.get("bin_integration", "center"),
            obs_ref=_ref_from_dict(ch.get("obs_ref")),
            mcstats=_mcstats_from_dict(ch.get("mcstats")),
        ))
    params = {}
    for name, p in d["parameters"].items():
        p = dict(p)
        p["constraint"] = _constraint_from_dict(p.get("constraint"))
        params[name] = Parameter(**p)
    return ModelIR(channels=channels, parameters=params, poi=d["poi"], groups=d.get("groups", {}),
                   source=d.get("source", ""), notes=d.get("notes", []), ir_version=d["ir_version"])
