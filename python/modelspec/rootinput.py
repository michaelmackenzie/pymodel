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

def arg_set(obs):
    """RooArgSet of one RooRealVar or of a list of them (the axes of an N-D observable)."""
    R = root()
    if isinstance(obs, (list, tuple)):
        out = R.RooArgSet()
        for v in obs:
            out.add(v)
        return out
    return R.RooArgSet(obs)


def floating_params(obj, obs) -> List[object]:
    """Floating RooRealVars (and categories) a RooAbsArg depends on, excluding the observable
    (one RooRealVar, or a list of them for an N-D observable)."""
    params = obj.getParameters(arg_set(obs))
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


# ----------------------------------------------------------------------------------------
# Multi-dimensional channels (ir.Observable with axes; bins flattened in row-major order)
# ----------------------------------------------------------------------------------------

def _axis_bins(values: np.ndarray, edges: List[float]) -> np.ndarray:
    """Bin index of every value along one axis (-1 outside [lo, hi]; hi belongs to the last bin)."""
    e = np.asarray(edges, dtype=float)
    k = np.searchsorted(e, values, side="right") - 1
    k = np.where(values == e[-1], len(e) - 2, k)
    return np.where((values < e[0]) | (values > e[-1]), -1, k)


def flat_bins(points: np.ndarray, axes) -> np.ndarray:
    """Flattened (row-major) bin index of N-D points (n, ndim); -1 for points outside."""
    points = np.asarray(points, dtype=float).reshape(-1, len(axes))
    idx = np.zeros(len(points), dtype=int)
    inside = np.ones(len(points), dtype=bool)
    for d, a in enumerate(axes):
        k = _axis_bins(points[:, d], a.edges)
        inside &= k >= 0
        idx = idx * (len(a.edges) - 1) + np.maximum(k, 0)
    return np.where(inside, idx, -1)


def datahist_contents_nd(dh, variables, observable) -> Tuple[List[float], List[float]]:
    """Weights (and squared weights) of an N-D RooDataHist in the flattened bins of
    ``observable`` (whose axes are ``variables``, same order).  Every entry must sit at the
    centre of a bin of the axis binnings: RooFit/Combine evaluate the pdfs at the RooDataHist
    coordinates, so a different binning would change the likelihood."""
    names = [v.GetName() for v in variables]
    axes = observable.axes
    stored = sorted(v.GetName() for v in dh.get())
    if stored != sorted(names):
        raise ValueError(f"RooDataHist '{dh.GetName()}' has the variables {stored}, expected {sorted(names)}")
    counts = np.zeros(observable.nbins)
    sumw2 = np.zeros(observable.nbins)
    centres = observable.bin_centers()
    widths = [np.diff(np.asarray(a.edges, dtype=float)) for a in axes]
    for i in range(dh.numEntries()):
        row = dh.get(i)
        x = np.array([row.getRealValue(n) for n in names])
        k = int(flat_bins(x[None, :], axes)[0])
        if k < 0:
            raise ValueError(f"RooDataHist '{dh.GetName()}' entry {i} at {x.tolist()} is outside the observable range")
        per_axis = np.unravel_index(k, observable.shape)
        for d in range(len(axes)):
            if abs(x[d] - centres[k, d]) > 1e-6 * widths[d][per_axis[d]]:
                raise ValueError(f"RooDataHist '{dh.GetName()}' entry {i}: {names[d]} = {x[d]:.10g} is not the centre "
                                 f"of a bin of the {names[d]} binning (a RooDataHist with a different binning)")
        w = dh.weight()
        counts[k] += w
        sumw2[k] += dh.weightSquared() if hasattr(dh, "weightSquared") else w * w
    return counts.tolist(), sumw2.tolist()


def dataset_values_nd(ds, names: List[str]) -> Tuple[List[List[float]], List[float]]:
    """Events [x_0, ..., x_{N-1}] (and weights, empty if unweighted) of an N-D RooDataSet."""
    values, weights = [], []
    weighted = ds.isWeighted()
    for i in range(ds.numEntries()):
        row = ds.get(i)
        values.append([row.getRealValue(n) for n in names])
        if weighted:
            weights.append(ds.weight())
    return values, weights


def binned_pdf_contents_nd(pdf, variables, observable, method: str = "center") -> List[float]:
    """Unit-normalised expected fractions of an N-D pdf in the flattened bins of ``observable``
    (axes = ``variables``): "center": pdf(bin centre) * bin volume with the pdf normalised over
    all axes (RooFit/Combine for a RooDataHist); "integral": the integral over each bin box
    divided by the integral over the full range."""
    R = root()
    nset = arg_set(list(variables))
    old = [v.getVal() for v in variables]
    out = []
    try:
        if method == "center":
            for c, vol in zip(observable.bin_centers(), observable.bin_volumes()):
                for v, x in zip(variables, c):
                    v.setVal(float(x))
                out.append(pdf.getVal(nset) * float(vol))
        elif method == "integral":
            total = pdf.createIntegral(nset, R.RooFit.NormSet(nset)).getVal()
            shape = observable.shape
            for k in range(observable.nbins):
                idx = np.unravel_index(k, shape)
                rname = f"pymodel_nd_bin_{k}"
                for v, a, i in zip(variables, observable.axes, idx):
                    v.setRange(rname, float(a.edges[i]), float(a.edges[i + 1]))
                integ = pdf.createIntegral(nset, R.RooFit.NormSet(nset), R.RooFit.Range(rname))
                out.append(integ.getVal() / total if total > 0 else 0.0)
        else:
            raise ValueError(f"Unknown bin integration method '{method}'")
    finally:
        for v, x in zip(variables, old):
            v.setVal(x)
    return out
