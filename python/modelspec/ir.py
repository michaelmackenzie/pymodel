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
* constraint terms are listed once per parameter (``Constraint``); every constraint has a
  global observable whose nominal value is ``Constraint.center``.

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


@dataclass
class Shape:
    """Process shape.

    kind:
      "counting"   -> no shape (single-bin counting channel)
      "template"   -> histogram: ``contents`` (+ ``sumw2``) on the channel binning
      "parametric" -> RooAbsPdf referenced by ``ref``; ``params`` lists the names of the IR
                      parameters it depends on.  If the pdf is a RooHistPdf with no floating
                      parameters, ``contents`` holds its histogram on the channel binning.
      "envelope"   -> RooMultiPdf referenced by ``ref`` with category ``category``
    """

    kind: str
    contents: List[float] = field(default_factory=list)
    sumw2: List[float] = field(default_factory=list)
    systs: List[TemplateSyst] = field(default_factory=list)
    ref: Optional[RooRef] = None
    params: List[str] = field(default_factory=list)
    category: str = ""
    syst_refs: Dict[str, List[RooRef]] = field(default_factory=dict)  # param -> [up, down] pdfs


@dataclass
class Process:
    name: str
    is_signal: bool
    rate: float
    shape: Shape
    norm_terms: List[NormTerm] = field(default_factory=list)


@dataclass
class Observable:
    name: str
    lo: float
    hi: float
    edges: List[float] = field(default_factory=list)  # binning (always set; counting: [0, 1])


@dataclass
class ChannelData:
    """Observed data of one channel.

    kind "count":    ``counts`` has one entry
    kind "binned":   ``counts`` per bin of Observable.edges
    kind "unbinned": ``values`` (and ``weights`` if weighted)
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


@dataclass
class Channel:
    name: str
    observable: Observable
    data: ChannelData
    processes: List[Process]
    bin_integration: str = "center"  # parametric pdfs on binned data: "center" (Combine/RooFit) or "integral"
    obs_ref: Optional[RooRef] = None  # RooRealVar of the observable when shapes come from a workspace

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
            if ch.data.kind == "unbinned" and ch.data.weights:
                feats.add("data:weighted")
            for proc in ch.processes:
                feats.add(f"shape:{proc.shape.kind}")
                if proc.shape.kind == "parametric" and proc.shape.contents:
                    feats.add("shape:parametric-histogram")
                for syst in proc.shape.systs:
                    feats.add(f"syst:{syst.kind}")
                if proc.shape.syst_refs:
                    feats.add("syst:pdf-morph")
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
            s["syst_refs"] = {k: [RooRef(**r) for r in v] for k, v in s.get("syst_refs", {}).items()}
            terms = []
            for t in p["norm_terms"]:
                t = dict(t)
                t["ref"] = _ref_from_dict(t.get("ref"))
                terms.append(NormTerm(**t))
            procs.append(Process(name=p["name"], is_signal=p["is_signal"], rate=p["rate"],
                                 shape=Shape(**s), norm_terms=terms))
        channels.append(Channel(
            name=ch["name"],
            observable=Observable(**ch["observable"]),
            data=ChannelData(**ch["data"]),
            processes=procs,
            bin_integration=ch.get("bin_integration", "center"),
            obs_ref=_ref_from_dict(ch.get("obs_ref")),
        ))
    params = {}
    for name, p in d["parameters"].items():
        p = dict(p)
        p["constraint"] = _constraint_from_dict(p.get("constraint"))
        params[name] = Parameter(**p)
    return ModelIR(channels=channels, parameters=params, poi=d["poi"], groups=d.get("groups", {}),
                   source=d.get("source", ""), notes=d.get("notes", []), ir_version=d["ir_version"])
