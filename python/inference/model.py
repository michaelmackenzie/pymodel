"""The backend-neutral likelihood interface used by all statistical methods.

A backend turns a ``ModelIR`` into a ``Likelihood`` that can evaluate the *main-measurement*
negative log-likelihood and the expected yields.  Everything else (constraint terms,
global observables, toy generation, fitting, test statistics, limits) is implemented once
in this package, so all backends are treated identically.

Conventions
-----------
* Parameters are ordered as in ``Likelihood.names``; all methods take a full value vector.
* ``nll_main`` is the extended NLL of the main measurement *without* data-only constants,
  summed over channels:

      binned/counting:  nu_tot - sum_i n_i log nu_i
      unbinned:         nu_tot - sum_j w_j log(nu_tot f(x_j))

  where nu_i are the expected yields per bin (``expected_by_process``) and nu_tot is the
  channel's total expected yield, i.e. the sum over processes of the full pdf normalisation.
  For counting channels, templates and ``bin_integration == "integral"`` nu_tot = sum_i nu_i.
  For parametric pdfs evaluated at bin centres (Combine/RooFit's binned likelihood), nu_i =
  nu_p * pdf(centre) * width does not sum to nu_p, and the extended term uses nu_p, exactly as
  Combine's CachingAddNLL / RooFit's extended binned NLL do.  With this convention every
  backend returns the same absolute value for the same model and data.
* Constraint terms use the IR ``Constraint`` of each parameter and the global observables
  stored in the ``Dataset`` (see ``constraint_nll``).
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.special import gammaln

from modelspec import ir as I


@dataclass
class MainData:
    """Observed/generated data of one channel (same meaning as ``ir.ChannelData``)."""

    kind: str                           # "count", "binned", "unbinned"
    counts: Optional[np.ndarray] = None
    values: Optional[np.ndarray] = None
    weights: Optional[np.ndarray] = None

    @property
    def total(self) -> float:
        if self.kind == "unbinned":
            return float(self.weights.sum()) if self.weights is not None else float(len(self.values))
        return float(self.counts.sum())

    def to_dict(self) -> dict:
        out = {"kind": self.kind}
        for key in ("counts", "values", "weights"):
            arr = getattr(self, key)
            if arr is not None:
                out[key] = np.asarray(arr).tolist()
        return out

    @staticmethod
    def from_dict(d) -> "MainData":
        return MainData(kind=d["kind"], **{k: np.asarray(d[k], dtype=float) for k in ("counts", "values", "weights")
                                          if k in d})


@dataclass
class Dataset:
    """Main data for every channel plus the values of the global observables."""

    main: Dict[str, MainData]
    global_obs: Dict[str, float]
    label: str = "data"
    truth: Optional[Dict[str, float]] = None  # parameter values used to generate a toy
    _native: dict = field(default_factory=dict, repr=False, compare=False)  # id(likelihood) -> prepared data

    def to_dict(self) -> dict:
        return {"label": self.label, "main": {k: v.to_dict() for k, v in self.main.items()},
                "global_obs": dict(self.global_obs), "truth": self.truth}

    @staticmethod
    def from_dict(d) -> "Dataset":
        return Dataset(main={k: MainData.from_dict(v) for k, v in d["main"].items()},
                       global_obs=dict(d["global_obs"]), label=d.get("label", "toy"), truth=d.get("truth"))


def observed_dataset(model: I.ModelIR) -> Dataset:
    main = {}
    for ch in model.channels:
        d = ch.data
        if d.kind == "unbinned":
            main[ch.name] = MainData(kind="unbinned", values=np.asarray(d.values, dtype=float),
                                     weights=np.asarray(d.weights, dtype=float) if d.weights else None)
        else:
            main[ch.name] = MainData(kind=d.kind, counts=np.asarray(d.counts, dtype=float))
    gobs = {p.name: p.constraint.center for p in model.parameters.values()
            if p.constraint is not None and p.constraint.has_global_observable}
    return Dataset(main=main, global_obs=gobs, label="data")


class Likelihood(ABC):
    """Backend likelihood built from a ModelIR."""

    backend_name = ""

    def __init__(self, model: I.ModelIR):
        self.model = model
        self.names: List[str] = list(model.parameters)
        self.index = {n: i for i, n in enumerate(self.names)}
        self.poi = model.poi
        self.poi_index = self.index[self.poi]

    # ----- parameters ----------------------------------------------------------------
    @property
    def parameters(self) -> List[I.Parameter]:
        return [self.model.parameters[n] for n in self.names]

    def nominal_values(self) -> np.ndarray:
        return np.array([p.value for p in self.parameters], dtype=float)

    def values_dict(self, values) -> Dict[str, float]:
        return {n: float(v) for n, v in zip(self.names, values)}

    # ----- to implement --------------------------------------------------------------
    def prepare(self, data: Dataset):
        """Convert a Dataset to whatever ``nll_main`` needs (cached on the dataset)."""
        return data.main

    @abstractmethod
    def nll_main(self, values: np.ndarray, native) -> float:
        """Main-measurement NLL (see module docstring) at ``values`` for prepared data."""

    @abstractmethod
    def expected_by_process(self, values: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """Expected yields per channel and process in the bins of each channel's
        observable edges (counting channels: one bin)."""

    def sample_unbinned(self, values: np.ndarray, channel: str, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` observable values of an unbinned channel from the total pdf."""
        raise NotImplementedError(f"{self.backend_name} cannot sample unbinned channel '{channel}'")

    def discrete_penalty(self, values: np.ndarray) -> float:
        """Envelope penalty (NLL units) for the current discrete-parameter state."""
        return 0.0

    def inactive_parameters(self, values: np.ndarray) -> List[str]:
        """Floating parameters that have no effect for the current discrete state."""
        return []

    # ----- shared --------------------------------------------------------------------
    def expected_counts(self, values: np.ndarray) -> Dict[str, np.ndarray]:
        return {ch: np.sum(list(procs.values()), axis=0) for ch, procs in self.expected_by_process(values).items()}

    def native(self, data: Dataset):
        key = id(self)
        if key not in data._native:
            data._native[key] = self.prepare(data)
        return data._native[key]

    def nll(self, values: np.ndarray, data: Dataset) -> float:
        """Total NLL: main measurement + constraints + discrete penalty."""
        val = self.nll_main(values, self.native(data))
        val += constraint_nll(self.model, self.index, values, data.global_obs)
        return val + self.discrete_penalty(values)


def fixed_shape_total_factors(model: I.ModelIR) -> Dict[str, Dict[str, float]]:
    """nu_p(total) / sum_i nu_ip for every process whose binned shape is fixed.

    For parametric pdfs with a fixed histogram (``Shape.contents`` = bin-centre fractions) this
    is 1 / sum(contents); it is 1 for counting and template processes.  Backends that build
    fixed binned shapes use it to form the extended term nu_tot of the NLL convention.
    """
    out = {}
    for ch in model.channels:
        facs = {}
        for proc in ch.processes:
            if proc.shape.kind == "parametric" and proc.shape.contents:
                total = float(np.sum(proc.shape.contents))
                facs[proc.name] = 1.0 / total if total > 0 else 1.0
            else:
                facs[proc.name] = 1.0
        out[ch.name] = facs
    return out


def constraint_nll(model: I.ModelIR, index: Dict[str, int], values, global_obs: Dict[str, float]) -> float:
    total = 0.0
    for p in model.parameters.values():
        c = p.constraint
        if c is None or p.role == I.ROLE_CONSTANT:
            continue
        theta = values[index[p.name]]
        if c.kind == I.CONSTRAINT_GAUSS:
            g = global_obs[p.name]
            total += 0.5 * ((theta - g) / c.sigma_hi) ** 2
        elif c.kind == I.CONSTRAINT_BIFURGAUSS:
            g = global_obs[p.name]
            sigma = c.sigma_lo if theta < g else c.sigma_hi
            total += 0.5 * ((theta - g) / sigma) ** 2
        elif c.kind == I.CONSTRAINT_POISSON:
            g = global_obs[p.name]
            if theta <= 0:
                return math.inf
            total += theta - g * math.log(theta) + gammaln(g + 1.0)
        elif c.kind == I.CONSTRAINT_FLAT:
            continue
        else:
            raise ValueError(f"Unknown constraint kind {c.kind}")
    return total


def global_obs_matching(model: I.ModelIR, index, values) -> Dict[str, float]:
    """Global observables that make each constraint centred on the given parameter values
    (used for post-fit Asimov datasets, like Combine's asimovDatasetWithFit)."""
    return {p.name: float(values[index[p.name]]) for p in model.parameters.values()
            if p.constraint is not None and p.constraint.has_global_observable}
