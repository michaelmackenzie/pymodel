"""roomodel Likelihood: the ModelWorkspace evaluated through the C++ loop of evaluator.py.

* ``nll_main`` follows inference/model.py exactly (no constants, no constraint terms; the
  binned extended term is the total yield sum_p nu_p, as in Combine): the
  sums are done in C++ from RooFit evaluations of the yields, morphing formulas, pdfs and
  bin integrals; RooFit's own NLL classes (which add constants/offsets) are not used.
* Parameter values are pushed into the RooRealVars/RooCategories only where they changed.
* Envelopes: ``discrete_penalty`` is the sum of RooMultiPdf::getCorrection() of the selected
  states (Combine's CachingAddNLL adds it to the NLL; with the default cFactor 0.5 it is 0.5
  per non-constant variable of the selected pdf at construction, the observable included);
  ``inactive_parameters`` are the floating parameters only non-selected states depend on.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from inference.model import Likelihood
from modelspec import ir as I
from modelspec import rootinput as R
from stat_backends.roomodel.evaluator import cpp
from stat_backends.roomodel.workspace import ModelWorkspace


@dataclass
class ChannelData:
    kind: str                       # "binned" (count/binned) or "unbinned"
    counts: np.ndarray = None
    values: np.ndarray = None
    weights: np.ndarray = None


@dataclass
class RooData:
    """Prepared data: contiguous arrays for the C++ loop, plus RooDataHist/RooDataSet
    per channel on request (``roofit(channel)``)."""

    channels: Dict[str, ChannelData]
    lik: object = field(repr=False, default=None)
    _roo: Dict[str, object] = field(default_factory=dict, repr=False)

    def roofit(self, channel: str):
        if channel not in self._roo:
            self._roo[channel] = self.lik.to_roofit_data(channel, self.channels[channel])
        return self._roo[channel]


class RooLikelihood(Likelihood):
    backend_name = "roomodel"

    def __init__(self, model: I.ModelIR):
        super().__init__(model)
        self.notes: List[str] = []
        self.mw = ModelWorkspace(model)
        self.notes += self.mw.notes
        self.ROOT = R.root()
        C = cpp()
        self._handles = [self.mw.vars[n] for n in self.names]
        self._is_cat = [self.model.parameters[n].role == I.ROLE_DISCRETE for n in self.names]
        self._last = np.full(len(self.names), np.nan)
        vec_d = self.ROOT.std.vector("double")
        vec_f = self.ROOT.std.vector("RooAbsReal*")
        self._cpp = {}
        self._chan = {c.name: c for c in self.mw.channels}
        floating = {p.name for p in model.parameters.values() if p.floating}
        static_deps = set()
        self._envelopes = []   # (param index of the category, corrections, [deps per state])
        for cb in self.mw.channels:
            ch = C.Channel(cb.obs, vec_d(cb.edges.tolist()))
            for pb in cb.procs:
                ip = ch.add_proc(pb.yield_func, pb.category if pb.category else self.ROOT.nullptr)
                for st in pb.states:
                    cf = vec_d(st.cfrac.tolist()) if st.cfrac is not None else vec_d()
                    ff = vec_f()
                    for f in st.ffrac:
                        ff.push_back(f)
                    total = 0.0 if (pb.kind == "template" and st.cfrac is not None and not st.cfrac.any()) else 1.0
                    ch.add_state(ip, cf, ff, st.pdf if st.pdf is not None else self.ROOT.nullptr, total)
                static_deps |= pb.yield_deps
                if pb.category is not None:
                    self._envelopes.append((self.index[pb.category.GetName()], pb.corrections,
                                            [st.deps & floating for st in pb.states]))
                else:
                    for st in pb.states:
                        static_deps |= st.deps
            self._cpp[cb.name] = ch
        self._static_deps = static_deps & floating
        if self._envelopes:
            self.notes.append("envelope penalty = RooMultiPdf::getCorrection() of the selected pdf, as added by "
                              "Combine (0.5 per non-constant variable of that pdf at construction, observable "
                              "included)")
        self._envelope_params = set().union(*[set().union(*d) for _, _, d in self._envelopes]) \
            if self._envelopes else set()

    # ----- parameters ------------------------------------------------------------------
    def set_values(self, values):
        v = np.asarray(values, dtype=float)
        changed = np.nonzero(v != self._last)[0]
        for i in changed:
            h = self._handles[i]
            if self._is_cat[i]:
                idx = int(round(v[i]))
                if idx < 0 or idx >= h.numTypes() or abs(v[i] - idx) > 1e-9:
                    raise ValueError(f"roomodel: invalid state {v[i]} for category {self.names[i]}")
                h.setIndex(idx)
            else:
                h.setVal(float(v[i]))
        self._last = v.copy()

    # ----- data --------------------------------------------------------------------------
    def prepare(self, data):
        out = {}
        for ch in self.model.channels:
            md = data.main[ch.name]
            nb = len(ch.observable.edges) - 1
            if ch.data.kind == "unbinned":
                if md.kind != "unbinned":
                    raise ValueError(f"roomodel: channel {ch.name} is unbinned but the dataset has {md.kind} data")
                vals = np.ascontiguousarray(md.values, dtype=float)
                w = None if md.weights is None else np.ascontiguousarray(md.weights, dtype=float)
                if w is not None and len(w) != len(vals):
                    raise ValueError(f"roomodel: channel {ch.name}: {len(vals)} values but {len(w)} weights")
                out[ch.name] = ChannelData(kind="unbinned", values=vals, weights=w)
            else:
                if md.kind == "unbinned":
                    raise ValueError(f"roomodel: channel {ch.name} is binned but the dataset is unbinned")
                counts = np.ascontiguousarray(md.counts, dtype=float)
                if len(counts) != nb:
                    raise ValueError(f"roomodel: channel {ch.name} has {nb} bins, data has {len(counts)}")
                out[ch.name] = ChannelData(kind="binned", counts=counts)
        return RooData(channels=out, lik=self)

    def to_roofit_data(self, channel: str, d: ChannelData):
        """RooDataHist (binned) or RooDataSet (unbinned, weighted if weights) of one channel."""
        R_ = self.ROOT
        cb = self._chan[channel]
        obs = cb.obs
        old = obs.getVal()
        if d.kind == "binned":
            dh = R_.RooDataHist(f"data_{channel}", "", R_.RooArgSet(obs))
            for i, n in enumerate(d.counts):
                obs.setVal(0.5 * (cb.edges[i] + cb.edges[i + 1]))
                dh.set(R_.RooArgSet(obs), float(n))
            obs.setVal(old)
            return dh
        wvar = R_.RooRealVar("weight", "weight", 1.0)
        args = R_.RooArgSet(obs, wvar)
        ds = R_.RooDataSet(f"data_{channel}", "", args, R_.RooFit.WeightVar(wvar))
        for j, x in enumerate(d.values):
            obs.setVal(float(x))
            ds.add(R_.RooArgSet(obs), 1.0 if d.weights is None else float(d.weights[j]))
        obs.setVal(old)
        return ds

    # ----- likelihood ----------------------------------------------------------------------
    def nll_main(self, values, native) -> float:
        self.set_values(values)
        total = 0.0
        for name, ch in self._cpp.items():
            d = native.channels[name]
            if d.kind == "binned":
                total += ch.binned_nll(d.counts)
            else:
                w = d.weights if d.weights is not None else self.ROOT.nullptr
                total += ch.unbinned_nll(d.values, w, len(d.values))
            if not math.isfinite(total):
                return math.inf
        return total

    def expected_by_process(self, values):
        self.set_values(values)
        out = {}
        for cb in self.mw.channels:
            ch = self._cpp[cb.name]
            procs = {}
            for ip, pb in enumerate(cb.procs):
                arr = np.empty(len(cb.edges) - 1)
                ch.expected(ip, arr)
                procs[pb.name] = arr
            out[cb.name] = procs
        return out

    def sample_unbinned(self, values, channel, n, rng):
        """n events of an unbinned channel: process counts from a multinomial with the
        yields (numpy rng), then each pdf generated by RooFit (RooRandom seeded from rng);
        template processes are drawn bin-wise and uniformly inside the bin."""
        R_ = self.ROOT
        self.set_values(values)
        cb = self._chan[channel]
        ch = self._cpp[channel]
        ylds = np.array([ch.proc_yield(ip) for ip in range(len(cb.procs))])
        if n == 0:
            return np.zeros(0)
        if np.any(ylds < 0) or ylds.sum() <= 0:
            raise ValueError(f"roomodel: cannot sample channel {channel} with yields {ylds}")
        counts = rng.multinomial(n, ylds / ylds.sum())
        parts = []
        for ip, (pb, k) in enumerate(zip(cb.procs, counts)):
            if k == 0:
                continue
            state = pb.states[pb.category.getCurrentIndex() if pb.category is not None else 0]
            if state.pdf is not None:
                R_.RooRandom.randomGenerator().SetSeed(int(rng.integers(1, 2 ** 31 - 1)))
                ds = state.pdf.generate(R_.RooArgSet(cb.obs), int(k))
                parts.append(np.asarray(ds.to_numpy()[cb.obs.GetName()], dtype=float))
            else:
                frac = np.empty(len(cb.edges) - 1)
                ch.fractions(ip, frac)
                b = rng.choice(len(frac), size=int(k), p=frac / frac.sum())
                parts.append(cb.edges[b] + rng.random(int(k)) * (cb.edges[b + 1] - cb.edges[b]))
        return rng.permutation(np.concatenate(parts))

    # ----- envelopes ------------------------------------------------------------------------
    def discrete_penalty(self, values):
        return float(sum(corr[int(round(values[i]))] for i, corr, _ in self._envelopes))

    def inactive_parameters(self, values):
        if not self._envelopes:
            return []
        active = set(self._static_deps)
        for i, _, deps in self._envelopes:
            active |= deps[int(round(values[i]))]
        return sorted(self._envelope_params - active)
