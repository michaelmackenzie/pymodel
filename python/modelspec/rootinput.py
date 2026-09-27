"""Read Combine shape inputs (TH1 histograms and RooWorkspace objects) with PyROOT.

Only this module (and the roomodel backend) imports ROOT.  Everything returned is plain
python data, except where a RooFit object is intentionally carried as a ``RooRef``.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

_ROOT = None
_OPEN_FILES: Dict[str, object] = {}


def root():
    """Import ROOT lazily (batch mode, quiet RooFit) and load any user libraries listed in
    the colon-separated environment variable PYMODEL_ROOT_LIBS (e.g. custom pdf classes)."""
    global _ROOT
    if _ROOT is None:
        import ROOT  # noqa: N811

        ROOT.gROOT.SetBatch(True)
        ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)
        for lib in filter(None, os.environ.get("PYMODEL_ROOT_LIBS", "").split(":")):
            if ROOT.gSystem.Load(lib) < 0:
                raise RuntimeError(f"Could not load ROOT library '{lib}' from PYMODEL_ROOT_LIBS")
        _ROOT = ROOT
    return _ROOT


def open_file(path: str):
    path = os.path.abspath(path)
    if path not in _OPEN_FILES:
        R = root()
        f = R.TFile.Open(path)
        if not f or f.IsZombie():
            raise FileNotFoundError(f"Cannot open ROOT file '{path}'")
        _OPEN_FILES[path] = f
    return _OPEN_FILES[path]


_WORKSPACES: Dict[Tuple[str, str], object] = {}


def get_workspace(path: str, ws_name: str):
    """The workspace ``ws_name`` of ``path``, read once and cached: TFile::Get returns a new
    copy of a RooWorkspace on every call, so objects from separate calls would not share
    their variables (e.g. an observable and the pdfs that depend on it)."""
    key = (os.path.abspath(path), ws_name)
    if key not in _WORKSPACES:
        ws = open_file(path).Get(ws_name)
        if not ws or not ws.InheritsFrom("RooWorkspace"):
            raise KeyError(f"No RooWorkspace '{ws_name}' in '{path}'")
        _WORKSPACES[key] = ws
    return _WORKSPACES[key]


def get_object(path: str, spec: str):
    """Resolve a Combine shapes object spec: ``ws:name`` (workspace object) or a TH1 path.

    Returns (object, workspace_name_or_None).  Raises KeyError if the object is missing.
    """
    if ":" in spec:
        ws_name, obj_name = spec.split(":", 1)
        ws = get_workspace(path, ws_name)
        obj = ws.pdf(obj_name) or ws.data(obj_name) or ws.function(obj_name) or ws.arg(obj_name)
        if not obj:
            raise KeyError(f"No object '{obj_name}' in workspace '{ws_name}' of '{path}'")
        return obj, ws_name
    obj = open_file(path).Get(spec)
    if not obj:
        raise KeyError(f"No object '{spec}' in '{path}'")
    return obj, None


def object_exists(path: str, spec: str) -> bool:
    try:
        get_object(path, spec)
        return True
    except KeyError:
        return False


# ----------------------------------------------------------------------------------------
# Histograms
# ----------------------------------------------------------------------------------------

def th1_edges(h) -> List[float]:
    ax = h.GetXaxis()
    return [ax.GetBinLowEdge(i) for i in range(1, h.GetNbinsX() + 1)] + [ax.GetBinUpEdge(h.GetNbinsX())]


def th1_contents(h) -> Tuple[List[float], List[float]]:
    if h.GetDimension() != 1:
        raise ValueError(f"Histogram '{h.GetName()}' is {h.GetDimension()}D; only 1D templates are supported")
    n = h.GetNbinsX()
    return ([h.GetBinContent(i) for i in range(1, n + 1)],
            [h.GetBinError(i) ** 2 for i in range(1, n + 1)])


def var_edges(var) -> List[float]:
    b = var.getBinning()
    return [b.binLow(i) for i in range(b.numBins())] + [b.binHigh(b.numBins() - 1)]


def datahist_contents(dh, var, edges: List[float]) -> Tuple[List[float], List[float]]:
    """Weights of a 1D RooDataHist, in the order of ``edges`` (checked to match)."""
    if dh.get().getSize() != 1:
        raise ValueError(f"RooDataHist '{dh.GetName()}' has {dh.get().getSize()} observables; only 1D is supported")
    counts = np.zeros(len(edges) - 1)
    sumw2 = np.zeros(len(edges) - 1)
    xvar = dh.get().first()
    for i in range(dh.numEntries()):
        row = dh.get(i)
        x = row.getRealValue(xvar.GetName())
        k = np.searchsorted(edges, x, side="right") - 1
        if k < 0 or k >= len(counts):
            continue
        counts[k] += dh.weight()
        sumw2[k] += dh.weightSquared() if hasattr(dh, "weightSquared") else dh.weight() ** 2
    return counts.tolist(), sumw2.tolist()


# ----------------------------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------------------------

def dataset_values(ds, obs_name: str) -> Tuple[List[float], List[float]]:
    """Values (and weights, empty if unweighted) of an unbinned RooDataSet."""
    values, weights = [], []
    weighted = ds.isWeighted()
    for i in range(ds.numEntries()):
        row = ds.get(i)
        values.append(row.getRealValue(obs_name))
        if weighted:
            weights.append(ds.weight())
    return values, weights


# ----------------------------------------------------------------------------------------
# Parametric pdfs
# ----------------------------------------------------------------------------------------

def floating_params(obj, obs) -> List[object]:
    """Floating RooRealVars (and categories) a RooAbsArg depends on, excluding the observable."""
    R = root()
    params = obj.getParameters(R.RooArgSet(obs))
    out = []
    for a in params:
        if a.InheritsFrom("RooRealVar") and not a.isConstant():
            out.append(a)
        elif a.InheritsFrom("RooCategory") and not a.isConstant():
            out.append(a)
    return out


def binned_pdf_contents(pdf, obs, edges: List[float], method: str = "center") -> List[float]:
    """Unit-normalised expected fractions of a 1D pdf in the bins ``edges``.

    method "center":   pdf(x_center) * width, normalised over the observable range, which is
                       what RooFit/Combine do for a pdf fit to a RooDataHist.
    method "integral": exact integral over each bin, divided by the integral over the range.
    """
    R = root()
    nset = R.RooArgSet(obs)
    old = obs.getVal()
    out = []
    try:
        if method == "center":
            for lo, hi in zip(edges[:-1], edges[1:]):
                obs.setVal(0.5 * (lo + hi))
                out.append(pdf.getVal(nset) * (hi - lo))
        elif method == "integral":
            total = pdf.createIntegral(nset, R.RooFit.NormSet(nset)).getVal()
            for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
                rname = f"pymodel_bin_{i}"
                obs.setRange(rname, lo, hi)
                integ = pdf.createIntegral(nset, R.RooFit.NormSet(nset), R.RooFit.Range(rname))
                out.append(integ.getVal() / total if total > 0 else 0.0)
        else:
            raise ValueError(f"Unknown bin integration method '{method}'")
    finally:
        obs.setVal(old)
    return out
