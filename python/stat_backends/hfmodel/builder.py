"""ModelIR -> pyhf model specification.

Mapping (Combine semantics on the left, pyhf on the right):

* process yield ``rate * shape``               -> sample ``data`` (per bin)
    - counting:                  [rate]
    - template:                  rate * contents / sum(contents)
    - parametric histogram:      rate * contents   (contents are pdf fractions, not renormalised)
    - process with a gmN term:   alpha * N (the datacard rate is ignored, as in Combine)
* POI ``r`` on signal processes                -> normfactor ``r``
* rateParam                                    -> normfactor (same name)
* lnN kappa (symmetric)                        -> normsys (hi=kappa, lo=1/kappa), code1: exact kappa**theta
* lnN kappa_down/kappa_up (asymmetric)         -> normsys (hi=kappa_up, lo=kappa_down), code1:
                                                  equal to asymPow for |theta| >= 0.5 only
* gmN N alpha                                  -> shapesys on the (single-bin) sample with
                                                  sigma = alpha*sqrt(N), i.e. tau = N and gamma = n/N
* template ``shape`` systematic, scale s       -> histosys on the unit-normalised shapes scaled to
                                                  the nominal yield, hi/lo = nom + s*(up_n - nom_n),
                                                  code4p (Combine's smoothStep with smooth region s)
                                                  + normsys (kd**s, ku**s), code1
* Gaussian constraints N(0, 1)                 -> pyhf normal constraints of normsys/histosys
* Poisson constraint of gmN                    -> pyhf Poisson constraint of shapesys (aux = N)
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from modelspec import ir as I
from modelspec import semantics as S
from stat_backends.base import UnsupportedByBackend

NORMSYS_CODE = "code1"
HISTOSYS_CODE = "code4p"
MODIFIER_SETTINGS = {"normsys": {"interpcode": NORMSYS_CODE}, "histosys": {"interpcode": HISTOSYS_CODE}}
WIDE_BOUND = 1.0e4      # replaces infinite normfactor bounds in the pyhf measurement config
MORPH_FLOOR = 1e-9      # Combine's floor for non-positive morphed template bins


@dataclass
class ParamLink:
    """pyhf parameter ``pyhf_name`` = IR value of ``ir_name`` * ``factor`` (all bins of it)."""

    pyhf_name: str
    ir_name: str
    kind: str          # normfactor, normsys, histosys, shapesys
    factor: float = 1.0


@dataclass
class HFSpec:
    channels: List[dict]
    observations: List[dict]
    parameters: List[dict]                  # pyhf measurement config "parameters"
    links: Dict[str, ParamLink]
    template_samples: List[Tuple[str, str]] = field(default_factory=list)  # (channel, process) with renorm
    notes: List[str] = field(default_factory=list)

    @property
    def model_spec(self) -> dict:
        return {"channels": self.channels, "parameters": self.parameters}

    def workspace(self, poi: str) -> dict:
        return {"channels": self.channels, "observations": self.observations,
                "measurements": [{"name": "pymodel", "config": {"poi": poi, "parameters": self.parameters}}],
                "version": "1.0.0"}


def _require_unit_gauss(par: I.Parameter, use: str):
    c = par.constraint
    if c is None or c.kind != I.CONSTRAINT_GAUSS:
        kind = "none" if c is None else c.kind
        raise UnsupportedByBackend(f"hfmodel: parameter '{par.name}' is used as a pyhf {use} but its constraint is "
                                   f"'{kind}'; pyhf {use} parameters always have a unit Gaussian constraint")
    if c.center != 0.0 or c.sigma_lo != 1.0 or c.sigma_hi != 1.0:
        raise UnsupportedByBackend(f"hfmodel: parameter '{par.name}' ({use}) has a Gaussian constraint with centre "
                                   f"{c.center:g} and sigma {c.sigma_hi:g}; pyhf only has N(0, 1) constraints")


class _Builder:
    def __init__(self, model: I.ModelIR):
        self.model = model
        self.links: Dict[str, ParamLink] = {}
        self.notes: List[str] = []
        self.template_samples: List[Tuple[str, str]] = []
        self.gmn_users: Dict[str, List[Tuple[str, str]]] = {}
        self.asym_lnn = False
        self.inexact_morph: List[str] = []

    def link(self, name: str, kind: str, factor: float = 1.0):
        if name not in self.model.parameters:
            raise UnsupportedByBackend(f"hfmodel: modifier refers to unknown parameter '{name}'")
        old = self.links.get(name)
        group = {"normsys": "normal", "histosys": "normal", "normfactor": "free", "shapesys": "poisson"}
        if old is not None:
            if group[old.kind] != group[kind]:
                raise UnsupportedByBackend(f"hfmodel: parameter '{name}' is used both as pyhf {old.kind} and {kind}")
            if old.kind == "histosys" or kind == "histosys":
                kind = "histosys"
            if kind == "shapesys" and old.factor != factor:
                raise UnsupportedByBackend(f"hfmodel: gmN parameter '{name}' has inconsistent N")
        self.links[name] = ParamLink(pyhf_name=name, ir_name=name, kind=kind, factor=factor)
        par = self.model.parameters[name]
        if kind in ("normsys", "histosys"):
            _require_unit_gauss(par, kind)
        elif kind == "normfactor" and par.constraint is not None:
            raise UnsupportedByBackend(f"hfmodel: rate parameter '{name}' has a '{par.constraint.kind}' constraint; "
                                       "pyhf normfactors are unconstrained")

    # ----- samples ---------------------------------------------------------------------
    def sample(self, ch: I.Channel, proc: I.Process) -> dict:
        nbins = len(ch.observable.edges) - 1
        shape = proc.shape
        mods: List[dict] = []
        normsys: Dict[str, List[float]] = {}  # name -> [hi, lo]; several factors of one name multiply (code1 exact)

        def add_normsys(name, hi, lo):
            if name in normsys:
                normsys[name][0] *= hi
                normsys[name][1] *= lo
            else:
                normsys[name] = [hi, lo]

        gmn = [t for t in proc.norm_terms if t.kind == "gmN"]
        rate = proc.rate
        if gmn:
            term = gmn[0]
            par = self.model.parameters[term.param]
            n_obs = par.constraint.center if par.constraint is not None else float("nan")
            if nbins != 1:
                raise UnsupportedByBackend(
                    f"hfmodel: gmN '{term.param}' on {ch.name}/{proc.name} in a channel with {nbins} bins: a pyhf "
                    "shapesys has one gamma per bin, so it is exact only for single-bin channels")
            if not n_obs > 0:
                raise UnsupportedByBackend(
                    f"hfmodel: gmN '{term.param}' has N = {n_obs:g}; a pyhf shapesys needs tau = N > 0")
            if term.alpha <= 0:
                raise UnsupportedByBackend(f"hfmodel: gmN '{term.param}' on {ch.name}/{proc.name} has alpha "
                                           f"{term.alpha:g}; a pyhf shapesys needs a positive nominal yield")
            rate = term.alpha * n_obs
            self.gmn_users.setdefault(term.param, []).append((ch.name, proc.name))
            mods.append({"name": term.param, "type": "shapesys", "data": [term.alpha * math.sqrt(n_obs)]})
            self.link(term.param, "shapesys", 1.0 / n_obs)

        if shape.kind == "counting":
            data = np.array([rate], dtype=float)
        elif shape.kind == "template":
            nom = np.asarray(shape.contents, dtype=float)
            nom_int = float(nom.sum())
            if nom_int <= 0:
                data = np.zeros(nbins)
                if shape.systs:
                    self.notes.append(f"{ch.name}/{proc.name}: empty nominal template, its shape systematics "
                                      "have no effect (as in the reference semantics)")
            else:
                nom_n = nom / nom_int
                data = rate * nom_n
                self.template_samples.append((ch.name, proc.name))
                scales = [s.scale for s in shape.systs]
                vsmooth = min([1.0] + scales)
                for syst in shape.systs:
                    if syst.kind != "shape":
                        raise UnsupportedByBackend(f"hfmodel: systematic kind '{syst.kind}' is not supported")
                    up = np.asarray(syst.up, dtype=float)
                    down = np.asarray(syst.down, dtype=float)
                    s = syst.scale
                    hi = rate * (nom_n + s * (up / up.sum() - nom_n))
                    lo = rate * (nom_n + s * (down / down.sum() - nom_n))
                    mods.append({"name": syst.param, "type": "histosys",
                                 "data": {"hi_data": hi.tolist(), "lo_data": lo.tolist()}})
                    self.link(syst.param, "histosys")
                    if abs(s - vsmooth) > 1e-12:
                        self.inexact_morph.append(f"{ch.name}/{proc.name}:{syst.param} (scale {s:g}, "
                                                  f"Combine smooth region {vsmooth:g})")
                    kappas = S.template_norm_kappas(nom, up, down, s)
                    if kappas is not None:
                        add_normsys(syst.param, kappas[1], kappas[0])
                        self.asym_lnn |= abs(kappas[0] * kappas[1] - 1.0) > 1e-12
        elif shape.kind == "parametric" and shape.contents and not shape.params:
            data = rate * np.asarray(shape.contents, dtype=float)
        else:
            raise UnsupportedByBackend(f"hfmodel: shape kind '{shape.kind}' of {ch.name}/{proc.name} "
                                       "(parametric shapes with floating parameters and envelopes are not supported)")
        if len(data) != nbins:
            raise UnsupportedByBackend(f"hfmodel: {ch.name}/{proc.name} has {len(data)} bins, channel has {nbins}")

        if proc.is_signal:
            mods.append({"name": self.model.poi, "type": "normfactor", "data": None})
            self.link(self.model.poi, "normfactor")
        for term in proc.norm_terms:
            if term.kind == "gmN":
                continue
            if term.kind == "lnN":
                add_normsys(term.param, term.kappa_hi, term.kappa_lo)
            elif term.kind == "asym_lnN":
                add_normsys(term.param, term.kappa_hi, term.kappa_lo)
                self.asym_lnn = True
            elif term.kind == "rate_param":
                mods.append({"name": term.param, "type": "normfactor", "data": None})
                self.link(term.param, "normfactor")
            else:
                raise UnsupportedByBackend(f"hfmodel: norm term '{term.kind}' on {ch.name}/{proc.name} is not supported")
        for name, (hi, lo) in normsys.items():
            if hi <= 0 or lo <= 0:
                raise UnsupportedByBackend(f"hfmodel: normsys '{name}' on {ch.name}/{proc.name} has a non-positive "
                                           f"factor (hi {hi:g}, lo {lo:g})")
            mods.append({"name": name, "type": "normsys", "data": {"hi": float(hi), "lo": float(lo)}})
            self.link(name, "normsys")
        return {"name": proc.name, "data": [float(x) for x in data], "modifiers": mods}

    # ----- parameters ------------------------------------------------------------------
    def measurement_parameters(self) -> List[dict]:
        out = []
        wide = []
        for name, link in self.links.items():
            par = self.model.parameters[name]
            lo, hi = par.lo, par.hi
            if link.kind == "normfactor":
                if not math.isfinite(lo):
                    lo = -WIDE_BOUND * max(1.0, abs(par.value))
                    wide.append(name)
                if not math.isfinite(hi):
                    hi = WIDE_BOUND * max(1.0, abs(par.value))
                    wide.append(name)
            elif link.kind == "shapesys":
                lo, hi = max(lo * link.factor, 1e-10), hi * link.factor
            entry = {"name": name, "bounds": [[float(lo), float(hi)]], "inits": [float(par.value * link.factor)]}
            if par.role == I.ROLE_CONSTANT:
                entry["fixed"] = True
            out.append(entry)
        if wide:
            self.notes.append(f"infinite IR bounds of {', '.join(sorted(set(wide)))} are written as "
                              f"+-{WIDE_BOUND:g}*max(1,|value|) in the pyhf measurement config (fits use the IR bounds)")
        return out

    def build(self) -> HFSpec:
        channels, observations = [], []
        for ch in self.model.channels:
            if ch.data.kind not in ("count", "binned"):
                raise UnsupportedByBackend(f"hfmodel: channel '{ch.name}' has {ch.data.kind} data")
            channels.append({"name": ch.name, "samples": [self.sample(ch, p) for p in ch.processes]})
            observations.append({"name": ch.name, "data": [float(x) for x in ch.data.counts]})
        for name, users in self.gmn_users.items():
            if len(users) != 1:
                raise UnsupportedByBackend(
                    f"hfmodel: gmN '{name}' acts on {len(users)} processes {users}; a pyhf shapesys maps onto a gmN "
                    "exactly only for one process in one channel")
        parameters = self.measurement_parameters()
        if self.asym_lnn:
            self.notes.append("asymmetric lnN (including template normalisation terms) uses pyhf normsys code1: equal "
                              "to Combine's asymPow for |theta| >= 0.5; inside, code1 is piecewise exponential "
                              "(kink at 0) while asymPow is smooth. The relative yield difference is "
                              "|theta*log(ku*kd)/2*(1-h(2 theta))| <= 0.035*|log(ku*kd)| (at |theta| ~ 0.15; "
                              "0.27% for kappa 0.9/1.2)")
        if self.template_samples:
            self.notes.append("template bins are floored at 1e-9 (unit-normalised) and renormalised as in Combine "
                              "after the pyhf evaluation; the exported pyhf workspace does not do this")
        if self.inexact_morph:
            self.notes.append("vertical morphing uses pyhf histosys code4p (Combine's smoothStep with the smooth "
                              "region |theta| < scale); Combine uses |scale*theta| < min(1, scales of the process), "
                              "so these differ for |theta| < 1 and |theta| < scale: " + "; ".join(self.inexact_morph))
        unused = [p.name for p in self.model.parameters.values()
                  if p.name not in self.links and p.constraint is not None and p.role != I.ROLE_CONSTANT]
        if unused:
            self.notes.append(f"constrained parameters without any pyhf modifier ({', '.join(unused)}) are "
                              "constrained by the shared layer only; they are absent from the exported workspace")
        return HFSpec(channels=channels, observations=observations, parameters=parameters, links=self.links,
                      template_samples=self.template_samples, notes=self.notes)


def build_spec(model: I.ModelIR) -> HFSpec:
    return _Builder(model).build()
