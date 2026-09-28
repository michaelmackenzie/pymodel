#!/usr/bin/env python3
"""pymodel test runner (plain script; rootana has no pytest).

    python3 tests/run_all.py [--fast | --slow | --all] [--backend NAME] [--only TEXT] [--workdir DIR]

--fast (default)  datacard grammar, bundle round trip, semantic oracle vs the independent
                  reference (tests/reference/reference_stats.py), every backend vs the oracle,
                  every backend vs the Combine fixtures (tests/fixtures/*.json), toy sanity.
--slow            toy CLs vs Combine HybridNew and vs exact Poisson CLs, Feldman-Cousins vs
                  the FC98 table / exact enumeration / Combine LHC-FC points, r pulls, FC coverage.
--all             both tiers.

"Backends" are the registered stat_backends plus ``semantic`` (the numpy oracle,
inference.semantic_likelihood), so the shared inference code is tested even when no real
backend is installed.  A backend that cannot be imported, or that refuses a model
(UnsupportedByBackend), is reported as SKIP with the reason.  Every test prints
PASS / FAIL / SKIP; the exit code is non-zero if any test failed.

Examples are copied to a work directory (default: a new temporary directory; never the
repository) and their make_inputs.py is run there.  The envelope and real mumep cards need
Combine's library (RooMultiPdf): set PYMODEL_ROOT_LIBS=libHiggsAnalysisCombinedLimit.so (the
Combine environment does), otherwise those tests are skipped.
"""

import argparse
import contextlib
import copy
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, os.path.join(HERE, "reference"))

import reference_stats as REF  # noqa: E402  (independent of pymodel)
from inference.asymptotic import AsymptoticLimits  # noqa: E402
from inference.fitting import Fitter  # noqa: E402
from inference.model import observed_dataset  # noqa: E402
from inference.semantic_likelihood import SemanticLikelihood  # noqa: E402
from modelspec import ir as I  # noqa: E402
from modelspec.datacard import UnsupportedFeature, build_ir  # noqa: E402
from modelspec.semantics import template_norm_kappas  # noqa: E402
from stat_backends import BACKEND_NAMES, get_backend  # noqa: E402
from stat_backends.base import Backend, UnsupportedByBackend  # noqa: E402

MUMEP_CARDS = "/exp/mu2e/app/users/mmackenz/mumep/mumep_ana/analysis/combine"
FIXTURES = os.path.join(HERE, "fixtures")


class Skip(Exception):
    pass


# ----------------------------------------------------------------------------------------
# registry and helpers
# ----------------------------------------------------------------------------------------

TESTS = []


def test(tier, name):
    def deco(fn):
        TESTS.append((tier, name, fn))
        return fn
    return deco


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-12)


from semantic_backend import SemanticBackend  # noqa: E402  (importable for parallel workers)


def backend_names(selected):
    names = ["semantic"] + list(BACKEND_NAMES)
    return [n for n in names if selected is None or n == selected]


_BACKENDS = {}


def backend(name):
    if name not in _BACKENDS:
        if name == "semantic":
            _BACKENDS[name] = SemanticBackend()
        else:
            try:
                _BACKENDS[name] = get_backend(name)
            except (ImportError, AttributeError) as exc:
                _BACKENDS[name] = Skip(f"backend {name} not importable: {type(exc).__name__}: {exc}")
    b = _BACKENDS[name]
    if isinstance(b, Skip):
        raise Skip(str(b))
    return b


def options_for(b, card):
    import pymodel_core

    return pymodel_core.build_parser(b).parse_args(["nll", card])


def likelihood(name, model, card="card.txt"):
    b = backend(name)
    try:
        return b.build_likelihood(model, options_for(b, card))
    except (UnsupportedByBackend, ImportError) as exc:
        raise Skip(f"{name}: {type(exc).__name__}: {exc}") from exc


def per_backend(fn, names, *args):
    """Run fn(name, *args) for every backend; SKIPs are collected, failures raised."""
    done, skipped, failed = [], [], []
    for name in names:
        try:
            out = fn(name, *args)
            done.append(f"{name} ({out})" if out else name)
        except Skip as exc:
            skipped.append(f"{name}: {exc}")
        except AssertionError as exc:
            failed.append(f"[{name}] {exc}")
    if failed:
        raise AssertionError(" || ".join(failed) + (f" (passed: {', '.join(done)})" if done else ""))
    if not done:
        raise Skip("; ".join(skipped))
    return f"ran on {', '.join(done)}" + (f"; skipped {'; '.join(skipped)}" if skipped else "")


@contextlib.contextmanager
def chdir(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


class Work:
    """Scratch copies of the examples (inputs generated there) and of the mumep cards."""

    def __init__(self, root):
        self.root = root
        self._done = {}

    def has_combine_lib(self):
        return "libHiggsAnalysisCombinedLimit" in os.environ.get("PYMODEL_ROOT_LIBS", "")

    def example(self, name):
        if name not in self._done:
            dest = os.path.join(self.root, "examples", name)
            if os.path.exists(dest):
                shutil.rmtree(dest)
            shutil.copytree(os.path.join(REPO, "examples", name), dest,
                            ignore=shutil.ignore_patterns("*.root", "higgsCombine*"))
            if os.path.exists(os.path.join(dest, "make_inputs.py")):
                if name == "envelope" and not self.has_combine_lib():
                    raise Skip("envelope needs Combine's library (set PYMODEL_ROOT_LIBS)")
                proc = subprocess.run([sys.executable, "make_inputs.py"], cwd=dest, capture_output=True, text=True)
                if proc.returncode != 0:
                    raise RuntimeError(f"make_inputs.py of {name} failed:\n{proc.stderr[-2000:]}")
            self._done[name] = dest
        return self._done[name]

    def mumep(self, card, workspace):
        key = ("mumep", card)
        if key not in self._done:
            if not os.path.exists(os.path.join(MUMEP_CARDS, "datacards", card)):
                raise Skip(f"{card} not available")
            dest = os.path.join(self.root, "mumep", os.path.splitext(card)[0])
            os.makedirs(os.path.join(dest, "datacards"), exist_ok=True)
            os.makedirs(os.path.join(dest, "workspaces"), exist_ok=True)
            shutil.copy(os.path.join(MUMEP_CARDS, "datacards", card), os.path.join(dest, "datacards"))
            shutil.copy(os.path.join(MUMEP_CARDS, "workspaces", workspace), os.path.join(dest, "workspaces"))
            self._done[key] = dest
        return self._done[key]

    def tmp(self, name):
        d = os.path.join(self.root, "tmp", name)
        os.makedirs(d, exist_ok=True)
        return d


WORK: Work = None
ARGS = None

# fixture name -> how to build the model: (kind, example dir or card, card, workspace)
FIXTURE_MODELS = {
    "counting": ("example", "counting", "card.txt"),
    "counting_multibin": ("example", "counting", "card_multibin.txt"),
    "templates": ("example", "templates", "card.txt"),
    "parametric_binned": ("example", "parametric_binned", "card.txt"),
    "parametric_unbinned": ("example", "parametric_unbinned", "card.txt"),
    "low_background_n0": ("example", "low_background", "card_n0.txt"),
    "low_background_n1": ("example", "low_background", "card_n1.txt"),
    "envelope": ("example", "envelope", "card.txt"),
    "mumem_75_hists": ("mumep", "combine_mumem_75_evt_r0104_hists.txt", "workspace_mumem_75_evt_r0104_hists.root"),
    "mumem_75_funcs": ("mumep", "combine_mumem_75_evt_r0104_funcs.txt", "workspace_mumem_75_evt_r0104_funcs.root"),
    "mumep_40_env": ("mumep", "combine_mumep_40_evt_r0104_env.txt", "workspace_mumep_40_evt_r0104_env.root"),
    # shape systematics on RooAbsPdfs (syst:pdf-morph / syst:histpdf-morph)
    "pdf_shape_syst": ("example", "pdf_shape_syst", "card.txt"),
    "pdf_shape_syst_histpdf": ("example", "pdf_shape_syst", "card_histpdf.txt"),
    "pdf_shape_syst_floating": ("example", "pdf_shape_syst", "card_floating.txt"),
    "pdf_shape_syst_unbinned": ("example", "pdf_shape_syst", "card_unbinned.txt"),
    # autoMCStats (Barlow-Beeston-lite)
    "mcstats": ("example", "mcstats", "card.txt"),
    # multi-dimensional (p, t0) channels (obs:multidim)
    "two_dim": ("example", "two_dim", "card.txt"),
    "two_dim_param": ("example", "two_dim", "card_param.txt"),
    "two_dim_unbinned": ("example", "two_dim", "card_unbinned.txt"),
}
NEEDS_COMBINE_LIB = {"envelope", "mumep_40_env", "mumem_75_funcs"}  # mumem_75_funcs: RooLandauCB

_MODELS = {}


def fixture_model(name, poi_range=(0.0, 20.0)):
    key = (name, poi_range)
    if key not in _MODELS:
        kind, a, b = FIXTURE_MODELS[name]
        if name in NEEDS_COMBINE_LIB and not WORK.has_combine_lib():
            raise Skip(f"{name} needs Combine's library (set PYMODEL_ROOT_LIBS)")
        if kind == "example":
            d = WORK.example(a)
            card = b
        else:
            d = WORK.mumep(a, b)
            card = os.path.join("datacards", a)
        with chdir(d):
            _MODELS[key] = (build_ir(card, poi_range=poi_range), os.path.join(d, card), d)
    model, card, d = _MODELS[key]
    return copy.deepcopy(model), card, d


def load_fixture(name):
    path = os.path.join(FIXTURES, f"{name}.json")
    if not os.path.exists(path):
        raise Skip(f"no fixture {path}; run tests/make_combine_fixtures.sh")
    with open(path) as handle:
        return json.load(handle)


def random_points(lik, n, rng):
    """Random parameter vectors inside the ranges (Gaussian-ish around the nominal)."""
    pts = []
    for _ in range(n):
        x = lik.nominal_values().copy()
        for i, p in enumerate(lik.parameters):
            if p.role == I.ROLE_POI:
                x[i] = rng.uniform(0.0, 3.0)
            elif p.role == I.ROLE_NUISANCE and p.constraint is not None:
                c = p.constraint
                if c.kind == I.CONSTRAINT_POISSON:
                    x[i] = c.center * rng.uniform(0.8, 1.2)
                elif c.kind == I.CONSTRAINT_FLAT:
                    x[i] = rng.uniform(p.lo, p.hi)
                else:
                    x[i] = c.center + rng.normal(0.0, 1.0) * c.sigma_hi
            elif p.role == I.ROLE_FREE:
                x[i] = p.value * rng.uniform(0.8, 1.2) if p.value != 0 else rng.uniform(-0.1, 0.1)
            x[i] = min(max(x[i], p.lo), p.hi)
        pts.append(x)
    return pts


def write_card(dirname, fname, text, files=()):
    d = WORK.tmp(dirname)
    for src in files:
        shutil.copy(src, d)
    path = os.path.join(d, fname)
    with open(path, "w") as handle:
        handle.write(text)
    return path


# ----------------------------------------------------------------------------------------
# reference models (built by hand from the example definitions, not from pymodel)
# ----------------------------------------------------------------------------------------

def ref_counting():
    N = REF.RefNuisance
    procs = [
        REF.RefProcess("sig", [12.0], signal=True, sym_lnN=[("lumi", 1.025), ("sig_eff", 1.05)]),
        REF.RefProcess("bkg", [50.0], sym_lnN=[("lumi", 1.025), ("bkg_norm", 1.10)], lnN=[("bkg_asym", 0.95, 1.08)]),
        REF.RefProcess("bkg_cr", [20.0], gmN=("cr_stat", 0.5)),
        REF.RefProcess("bkg_free", [10.0], rate_params=["scale_free"]),
    ]
    nuis = [N("lumi"), N("sig_eff"), N("bkg_norm"), N("bkg_asym"),
            N("cr_stat", kind="poisson", center=40.0, lo=0.0, hi=200.0),
            N("scale_free", center=1.0, sigma_lo=0.2, sigma_hi=0.2, lo=0.2, hi=1.8)]
    return REF.RefModel([REF.RefChannel("sr", [80.0], procs)], nuis)


def ref_multibin():
    chans = []
    for b, s, bk, n in (("b1", 4, 10, 12), ("b2", 6, 30, 28), ("b3", 2, 60, 65)):
        bkg = REF.RefProcess("bkg", [bk], sym_lnN=[("lumi", 1.025), ("bkg_norm", 1.10)])
        if b == "b3":
            bkg.lnN.append(("bkg_b3", 0.97, 1.05))
        chans.append(REF.RefChannel(b, [n], [REF.RefProcess("sig", [s], signal=True, sym_lnN=[("lumi", 1.025)]), bkg]))
    N = REF.RefNuisance
    return REF.RefModel(chans, [N("lumi"), N("bkg_norm"), N("bkg_b3")])


def ref_low_background(n):
    N = REF.RefNuisance
    procs = [REF.RefProcess("sig", [1.0], signal=True, sym_lnN=[("sig_eff", 1.10)]),
             REF.RefProcess("bkg", [0.2], sym_lnN=[("bkg_unc", 1.30)])]
    return REF.RefModel([REF.RefChannel("sr", [float(n)], procs)], [N("sig_eff"), N("bkg_unc")])


def ref_templates():
    """Reads templates.root directly with PyROOT (no pymodel code)."""
    import ROOT

    d = WORK.example("templates")
    f = ROOT.TFile.Open(os.path.join(d, "templates.root"))

    def h(path):
        obj = f.Get(path)
        return [obj.GetBinContent(i) for i in range(1, obj.GetNbinsX() + 1)]

    chans = []
    for ch in ("ch1", "ch2"):
        sig = REF.RefProcess("sig", h(f"{ch}/sig"), signal=True, sym_lnN=[("lumi", 1.025)])
        if ch == "ch1":
            sig.shapes.append(("sig_width", h(f"{ch}/sig_sig_widthUp"), h(f"{ch}/sig_sig_widthDown"), 1.0))
        bkg = REF.RefProcess("bkg", h(f"{ch}/bkg"), sym_lnN=[("lumi", 1.025), ("bkg_norm", 1.05)],
                             shapes=[("bkg_shape", h(f"{ch}/bkg_bkg_shapeUp"), h(f"{ch}/bkg_bkg_shapeDown"), 1.0)])
        chans.append(REF.RefChannel(ch, h(f"{ch}/data_obs"), [sig, bkg]))
    f.Close()
    N = REF.RefNuisance
    return REF.RefModel(chans, [N("lumi"), N("bkg_norm"), N("bkg_shape", lo=-4, hi=4), N("sig_width", lo=-4, hi=4)])


REF_MODELS = {"counting": ref_counting, "counting_multibin": ref_multibin, "templates": ref_templates,
              "low_background_n0": lambda: ref_low_background(0), "low_background_n1": lambda: ref_low_background(1)}


def to_ref_vector(ref, lik, x):
    return np.array([x[lik.index[n]] for n in ref.names])


# ----------------------------------------------------------------------------------------
# FAST: datacard grammar
# ----------------------------------------------------------------------------------------

def _grammar_th1_file(d):
    import ROOT

    path = os.path.join(d, "g.root")
    f = ROOT.TFile(path, "RECREATE")
    vals = {"sig": [1, 4, 1], "bkg": [10, 10, 10], "data_obs": [11, 15, 9],
            "bkg_sUp": [12, 10, 9], "bkg_sDown": [8, 10, 11], "sig_125": [2, 8, 2],
            "bkg_125": [10, 10, 10], "data_obs_125": [11, 15, 9], "other_sig": [3, 3, 3]}
    for name, v in vals.items():
        hh = ROOT.TH1D(name, name, 3, 0, 3)
        for i, c in enumerate(v):
            hh.SetBinContent(i + 1, c)
        hh.Write()
    f.Close()
    return path


def _proc(model, ch, p):
    for proc in model.channel(ch).processes:
        if proc.name == p:
            return proc
    raise KeyError((ch, p))


@test("fast", "grammar: shapes lines (4-token, $MASS, precedence, shape?) and multi-bin/swapped process lines")
def t_grammar_shapes():
    d = WORK.tmp("grammar")
    _grammar_th1_file(d)
    card = write_card("grammar", "shapes.txt", """
imax 1
jmax 1
kmax *
shapes * * g.root $PROCESS_$MASS $PROCESS_$SYSTEMATIC
shapes sig a g.root other_sig
shapes bkg a g.root bkg bkg_$SYSTEMATIC
shapes data_obs a g.root data_obs
bin a
observation -1
bin a a
process 1 0
process bkg sig
rate -1 -1
s shape? 1 1
""")
    m = build_ir(card, mass="125")
    sig, bkg = _proc(m, "a", "sig"), _proc(m, "a", "bkg")
    check(sig.is_signal and not bkg.is_signal, "swapped process lines: sig must be the signal (id 0)")
    check(sig.shape.contents == [3, 3, 3], f"(channel, process) shapes line must win over (*, *): {sig.shape.contents}")
    check(abs(sig.rate - 9) < 1e-9 and abs(bkg.rate - 30) < 1e-9, "rate -1 must be the template integral")
    check(m.channel("a").data.counts == [11, 15, 9], "data_obs from its own shapes line")
    check([s.param for s in bkg.shape.systs] == ["s"], "shape? with templates present is a shape systematic")
    check(not sig.shape.systs and not any(t.param == "s" for t in sig.norm_terms),
          "shape? without templates for sig (no systematics pattern) falls back to lnN 1 = no effect")
    # $MASS substitution
    card2 = write_card("grammar", "mass.txt", """
imax 1
jmax 1
kmax 0
shapes * * g.root $PROCESS_$MASS
bin a
observation -1
bin a a
process sig bkg
process 0 1
rate -1 -1
""")
    m2 = build_ir(card2, mass="125")
    check(_proc(m2, "a", "sig").shape.contents == [2, 8, 2], "$MASS must be substituted by --mass")
    # multi-bin pairing: observation order follows the 'bin' line, not the process columns
    card3 = write_card("grammar", "multibin.txt", """
imax 2
jmax 1
kmax 1
bin B A
observation 7 3
bin A A B B
process sig bkg sig bkg
process 0 1 0 1
rate 1 2 3 4
u lnN - 1.1 - 1.2
""")
    m3 = build_ir(card3)
    check(m3.channel("A").data.counts == [3.0] and m3.channel("B").data.counts == [7.0], "bin/observation pairing")
    check(_proc(m3, "A", "bkg").rate == 2 and _proc(m3, "B", "sig").rate == 3, "rates per (bin, process)")
    k = [t for t in _proc(m3, "B", "bkg").norm_terms if t.param == "u"][0]
    check(abs(k.kappa_hi - 1.2) < 1e-12 and k.kind == "lnN", "lnN per bin")


@test("fast", "grammar: shapes * * FAKE gives a counting model")
def t_grammar_fake():
    card = write_card("grammar", "fake.txt", """
imax 1
jmax 1
kmax 0
shapes * * FAKE
bin a
observation 5
bin a a
process s b
process 0 1
rate 2 3
""")
    m = build_ir(card)
    check(m.channel("a").is_counting and m.channel("a").data.counts == [5.0], "FAKE shapes -> counting")


@test("fast", "grammar: lnN sym/asym, lnU, gmN, rateParam (+formula), param asym/range, freeze, group")
def t_grammar_norms():
    card = write_card("grammar", "norms.txt", """
imax 1
jmax 2
kmax *
bin a
observation 50
bin a a a
process s b c
process 0 1 2
rate 5 20 999
k1 lnN 1.1 - -
k2 lnN - 0.8/1.3 -
ku lnU - 1.5 -
g gmN 30 - - 0.5
p1 param 0.5 -0.1/+0.3 [0,2]
p2 param 1 0.2
nb rateParam a b 1 [0,5]
nc rateParam a b (@0*@1) nb,p1
frz lnN - 1.2 -
nuisance edit freeze frz
grp group = k1 k2
""")
    m = build_ir(card)
    P = m.parameters
    s, b, c = _proc(m, "a", "s"), _proc(m, "a", "b"), _proc(m, "a", "c")
    t = {x.param: x for x in s.norm_terms}
    check(t["k1"].kind == "lnN" and abs(t["k1"].kappa_hi - 1.1) < 1e-12, "symmetric lnN")
    tb = {x.param: x for x in b.norm_terms}
    check(tb["k2"].kind == "asym_lnN" and tb["k2"].kappa_lo == 0.8 and tb["k2"].kappa_hi == 1.3,
          "asymmetric lnN is down/up (Combine order)")
    check(tb["ku"].kind == "lnU" and P["ku"].constraint.kind == I.CONSTRAINT_FLAT and (P["ku"].lo, P["ku"].hi) == (-1, 1),
          "lnU: flat on [-1, 1]")
    tc = {x.param: x for x in c.norm_terms}
    check(tc["g"].kind == "gmN" and tc["g"].alpha == 0.5, "gmN alpha")
    check(P["g"].constraint.kind == I.CONSTRAINT_POISSON and P["g"].constraint.center == 30, "gmN N = 30")
    check(any("gmN" in n for n in m.notes), "gmN with rate != N*alpha must leave a note")
    check(P["p1"].constraint.kind == I.CONSTRAINT_BIFURGAUSS and P["p1"].constraint.sigma_lo == 0.1
          and P["p1"].constraint.sigma_hi == 0.3 and (P["p1"].lo, P["p1"].hi) == (0, 2) and P["p1"].constraint.center == 0.5,
          f"asymmetric param with range: {P['p1']}")
    check(P["p2"].constraint.kind == I.CONSTRAINT_GAUSS and abs(P["p2"].lo - 0.2) < 1e-12 and abs(P["p2"].hi - 1.8) < 1e-12,
          "param default range is mean +- 4 sigma")
    check(P["nb"].role == I.ROLE_FREE and (P["nb"].lo, P["nb"].hi) == (0, 5), "rateParam with range")
    check(tb["nb"].kind == "rate_param", "rateParam term on b")
    check(tb["nc"].kind == "formula" and tb["nc"].args == ["nb", "p1"] and tb["nc"].formula == "(@0*@1)", "formula rateParam")
    check(P["frz"].role == I.ROLE_CONSTANT, "nuisance edit freeze")
    check(sorted(m.groups.get("grp", [])) == ["k1", "k2"], f"group: {m.groups}")


@test("fast", "grammar: discrete (RooMultiPdf) in the envelope example")
def t_grammar_discrete():
    model, _, _ = fixture_model("envelope")
    P = model.parameters
    check(P["pdfindex"].role == I.ROLE_DISCRETE and P["pdfindex"].n_states == 3, f"pdfindex: {P.get('pdfindex')}")
    bkg = _proc(model, "sr", "bkg")
    check(bkg.shape.kind == "envelope" and bkg.shape.category == "pdfindex", "envelope shape")
    check(any(t.kind == "rate_param" and t.param == "bkg_norm" for t in bkg.norm_terms), "floating bkg_norm")
    # a discrete line naming something that is not a RooMultiPdf category must fail
    d = WORK.example("envelope")
    bad = write_card("grammar_env", "bad.txt", open(os.path.join(d, "card.txt")).read() + "nosuchcat discrete\n",
                     files=[os.path.join(d, "workspace.root")])
    try:
        build_ir(bad)
    except UnsupportedFeature:
        return
    raise AssertionError("discrete of an unknown category must raise UnsupportedFeature")


@test("fast", "grammar: parametric examples (param makes a constant workspace var floating, <pdf>_norm)")
def t_grammar_parametric():
    m, _, _ = fixture_model("parametric_binned")
    P = m.parameters
    check(P["sig_scale"].role == I.ROLE_NUISANCE and P["sig_scale"].constraint.kind == I.CONSTRAINT_GAUSS,
          "sig_scale param -> Gaussian nuisance")
    check("sig_scale" in _proc(m, "sr", "sig").shape.params, "signal pdf depends on sig_scale")
    check(m.channel("sr").data.kind == "binned" and len(m.channel("sr").observable.edges) == 41, "40 bins")
    u, _, _ = fixture_model("parametric_unbinned")
    check(u.channel("sr").data.kind == "unbinned" and len(u.channel("sr").data.values) == 420, "unbinned data")
    check(u.parameters["slope_bkg"].role == I.ROLE_FREE and u.parameters["bkg_norm"].role == I.ROLE_FREE,
          "floating slope and bkg_norm")


# ----------------------------------------------------------------------------------------
# FAST: bundle round trip
# ----------------------------------------------------------------------------------------

@test("fast", "bundle save/load round trip (identical IR and NLL at random points)")
def t_bundle():
    from modelspec.bundle import load_bundle, save_bundle

    def one(name, fixture):
        model, card, d = fixture_model(fixture)
        path = os.path.join(WORK.tmp("bundle"), f"{fixture}_{name}.json")
        with chdir(d):
            save_bundle(model, path)
        loaded = load_bundle(path)
        check(len(loaded.constrained_parameters()) == len(model.constrained_parameters()), "constraint count")
        check(list(loaded.parameters) == list(model.parameters), "parameter order")
        with chdir(d):
            lik_a = likelihood(name, model, card)
        lik_b = likelihood(name, loaded, path)
        rng = np.random.default_rng(7)
        da, db = observed_dataset(model), observed_dataset(loaded)
        for x in random_points(lik_a, 5, rng):
            a, b = lik_a.nll(x, da), lik_b.nll(x, db)
            check(a == b or abs(a - b) <= 1e-10 * max(1.0, abs(a)), f"{fixture}: NLL {a!r} != {b!r} after reload")

    msgs = []
    for fixture in ("counting", "templates", "parametric_binned", "parametric_unbinned"):
        msgs.append(f"{fixture}: " + per_backend(one, backend_names(ARGS.backend), fixture))
    return " | ".join(msgs)


# ----------------------------------------------------------------------------------------
# FAST: oracle vs independent reference
# ----------------------------------------------------------------------------------------

@test("fast", "oracle vs reference_stats: NLL at random points (counting, multibin, templates, low_background)")
def t_oracle_nll():
    out = []
    for fixture, make in REF_MODELS.items():
        model, _, _ = fixture_model(fixture)
        lik = SemanticLikelihood(model)
        ref = make()
        data = observed_dataset(model)
        worst = 0.0
        for x in random_points(lik, 20, np.random.default_rng(11)):
            a = lik.nll(x, data)
            b = ref.nll(to_ref_vector(ref, lik, x))
            worst = max(worst, abs(a - b))
            check(abs(a - b) < 1e-8 * max(1.0, abs(b)), f"{fixture}: oracle NLL {a:.12g} vs reference {b:.12g} at "
                                                         f"{lik.values_dict(x)}")
        out.append(f"{fixture} max|dNLL|={worst:.1e}")
    return ", ".join(out)


@test("fast", "oracle vs reference_stats: asymptotic CLs observed/expected within 0.5%")
def t_oracle_asymptotic():
    out, bad = [], []
    for fixture, make in REF_MODELS.items():
        model, _, _ = fixture_model(fixture)
        lik = SemanticLikelihood(model)
        res = AsymptoticLimits(lik, Fitter(lik)).run()
        ref = REF.asymptotic_limits(make())
        pairs = [("obs", res.observed, ref.observed)] + [(f"exp{q:g}", res.expected[q], ref.expected[q])
                                                         for q in REF.QUANTILES]
        for label, a, b in pairs:
            if a is None or rel(a, b) > 0.005:
                bad.append(f"{fixture} {label}: pymodel {a} vs reference {b:.5g}")
        out.append(f"{fixture} obs {res.observed:.4f}/{ref.observed:.4f}")
        if res.flags:
            bad.append(f"{fixture}: flags {res.flags}")
    check(not bad, "; ".join(bad))
    return ", ".join(out)


@test("fast", "reference: exact Poisson CLs and FC98 Table IV (b=3, 90%) by Neyman construction")
def t_reference_fc98():
    check(abs(REF.poisson_cls_limit(1.0, 0.0, 0) - 2.9957) < 1e-3, "CLs n=0 limit is -ln(0.05)")
    bad = []
    for n, (lo, hi) in REF.FC98_B3_CL90.items():
        rlo, rhi = REF.fc_interval(n, 3.0, 0.90, mu_max=n + 15.0)
        if n == 0:
            check(abs(rhi - 0.95) < 0.006, f"plain construction n=0 upper {rhi} (expected 0.95 before FC's b-monotone fix)")
            continue
        if abs(rlo - lo) > 0.006 or abs(rhi - hi) > 0.006:
            bad.append(f"n={n}: [{rlo:.3f}, {rhi:.3f}] vs FC98 [{lo}, {hi}]")
    check(not bad, "; ".join(bad))
    return "n=1..10 reproduce FC98 to 0.005; n=0 plain construction 0.95 (FC98 1.08 includes the b-monotone fix, slow tier)"


# ----------------------------------------------------------------------------------------
# FAST: backends vs oracle
# ----------------------------------------------------------------------------------------

@test("fast", "backends vs oracle: NLL and expected yields at random points")
def t_backend_vs_oracle():
    names = [n for n in backend_names(ARGS.backend) if n != "semantic"]
    msgs = []
    for fixture in ("counting", "counting_multibin", "templates", "low_background_n1", "mumem_75_hists",
                    "mumem_75_funcs"):
        try:
            model, card, d = fixture_model(fixture)
        except Skip as exc:
            msgs.append(f"{fixture}: SKIP {exc}")
            continue
        try:
            oracle = SemanticLikelihood(model)
        except NotImplementedError as exc:  # e.g. a regenerated mumep card whose pdfs now float
            msgs.append(f"{fixture}: SKIP oracle: {exc}")
            continue
        data = observed_dataset(model)
        pts = random_points(oracle, 10, np.random.default_rng(3))

        def one(name):
            with chdir(d):
                lik = likelihood(name, model, card)
            data_b = observed_dataset(model)
            notes = getattr(lik, "notes", [])
            # a backend may approximate asymmetric lnN only if it declares it; the documented
            # bound is a relative yield error <= 0.035*|ln(ku*kd)| per term
            approx = {}
            if any("asymmetric lnN" in n for n in notes):
                for ch in model.channels:
                    for proc in ch.processes:
                        kappas = [(t.kappa_lo, t.kappa_hi) for t in proc.norm_terms if t.kind == "asym_lnN"]
                        for s in proc.shape.systs:  # template normalisation terms are asymPow too
                            k = template_norm_kappas(proc.shape.contents, s.up, s.down, s.scale)
                            if k is not None:
                                kappas.append(k)
                        approx[(ch.name, proc.name)] = sum(0.035 * abs(math.log(lo * hi)) for lo, hi in kappas)
            for x in pts:
                a, b = lik.nll(x, data_b), oracle.nll(x, data)
                ea, eb = lik.expected_by_process(x), oracle.expected_by_process(x)
                nll_tol = 1e-6 * max(1.0, abs(b))
                for ch in eb:
                    n = np.asarray(data.main[ch].counts)
                    mu = np.sum(list(eb[ch].values()), axis=0)
                    for p in eb[ch]:
                        rel = approx.get((ch, p), 0.0)
                        check(np.allclose(ea[ch][p], eb[ch][p], rtol=1e-6 + 1.001 * rel, atol=1e-9),
                              f"{fixture}/{name}: expected {ch}/{p} {ea[ch][p]} vs {eb[ch][p]}"
                              + (f" (declared bound {rel:.3g})" if rel else ""))
                        if rel:
                            # first-order NLL bound from the allowed yield differences
                            dnu = 1.001 * rel * np.asarray(eb[ch][p])
                            nll_tol += float(np.sum(dnu * (1.0 + n / np.maximum(mu, 1e-300))))
                check(abs(a - b) < nll_tol, f"{fixture}/{name}: NLL {a:.10g} vs oracle {b:.10g} "
                                            f"(tolerance {nll_tol:.3g}) at {oracle.values_dict(x)}"
                                            + (f" (backend notes: {notes})" if notes else ""))
        try:
            msgs.append(f"{fixture}: " + per_backend(one, names))
        except Skip as exc:
            msgs.append(f"{fixture}: SKIP {exc}")
    if all("SKIP" in m for m in msgs):
        raise Skip(" | ".join(msgs))
    return " | ".join(msgs)


# ----------------------------------------------------------------------------------------
# FAST: backends vs Combine fixtures
# ----------------------------------------------------------------------------------------

def _profile_crossings(lik, fitter, data, free, level):
    """r values where 2*DeltaNLL = level (lower edge 0 if not crossed above the r bound)."""
    from scipy.optimize import brentq

    rhat = free.values[lik.poi_index]
    lo_b, hi_b = fitter.bounds[lik.poi]

    def f(r):
        res = fitter.fit(data, start=free.values, fixed={lik.poi: r})
        if not res.valid:
            raise AssertionError(f"conditional fit at r={r} failed: {res.status}")
        return 2.0 * (res.nll - free.nll) - level

    hi = rhat + 0.5 * max(1.0, abs(rhat))
    while f(hi) < 0:
        hi = rhat + 2.0 * (hi - rhat)
        if hi > 1e5:
            raise AssertionError("no upper crossing")
    upper = brentq(f, rhat, hi, xtol=1e-5)
    lower = lo_b if f(lo_b) < 0 else brentq(f, lo_b, rhat, xtol=1e-5)
    return lower, upper


def _vs_combine(name, fixture):
    fx = load_fixture(fixture)
    model, card, d = fixture_model(fixture)
    with chdir(d):
        lik = likelihood(name, model, card)
    extra = []
    fitter = Fitter(lik)
    data = observed_dataset(model)
    # asymptotic limits
    res = AsymptoticLimits(lik, fitter).run()
    ca = fx["asymptotic"]
    errs = []
    if res.observed is None or rel(res.observed, ca["observed"]) > 0.01:
        errs.append(f"observed {res.observed} vs Combine {ca['observed']:.5g}")
    for q in (0.025, 0.16, 0.5, 0.84, 0.975):
        a, b = res.expected.get(q), ca["expected"][f"{q:g}"]
        if a is None or rel(a, b) > 0.01:
            errs.append(f"expected {q:g}: {a} vs Combine {b:.5g}")
    extra.append(f"obs {res.observed:.4g}/{ca['observed']:.4g}")
    # best fit and 68% interval (MultiDimFit singles, same r range)
    lo, hi = fx["r_range"]
    fitter.set_range(lik.poi, lo, hi)
    free = fitter.fit(data)
    check(free.valid, f"free fit failed: {free.status}")
    s = fx["multidimfit_singles"]
    width = s["upper"] - s["lower"]
    rhat = free.values[lik.poi_index]
    if abs(rhat - s["best_fit"]) > 0.02 * width:
        # On a very flat likelihood the minimiser tolerance (EDM ~1e-4) limits where either
        # program stops; accept only if Combine's r_hat is an equivalent minimum for pymodel.
        at_c = fitter.fit(data, start=free.values, fixed={lik.poi: s["best_fit"]})
        dq = 2.0 * (at_c.nll - free.nll)
        if abs(dq) < 0.005:
            extra.append(f"NOTE best fit r {rhat:.4g} vs Combine {s['best_fit']:.4g} differs by "
                         f"{abs(rhat - s['best_fit']) / width:.1%} of the 68% width, but 2dNLL between them is {dq:.1e} "
                         "(flat likelihood, minimiser precision)")
        else:
            errs.append(f"best fit r {rhat:.5g} vs Combine {s['best_fit']:.5g} (tol 2% of the 68% width {width:.3g}; "
                        f"2dNLL at Combine's r_hat {dq:.3g})")
    l68, u68 = _profile_crossings(lik, fitter, data, free, 1.0)
    for label, a, b in (("lower", l68, s["lower"]), ("upper", u68, s["upper"])):
        if abs(a - b) > 0.02 * width:
            errs.append(f"68% {label} {a:.5g} vs Combine singles {b:.5g}")
    extra.append(f"r {rhat:.4g} [{l68:.4g},{u68:.4g}] vs [{s['lower']:.4g},{s['upper']:.4g}]")
    # MultiDimFit grid: 2 DeltaNLL at the same r values
    worst = 0.0
    for r, q in fx["multidimfit_grid"]["points"][::5]:
        c = fitter.fit(data, start=free.values, fixed={lik.poi: r})
        mine = 2.0 * (c.nll - free.nll)
        diff = abs(mine - q)
        worst = max(worst, diff)
        if diff > 0.02 + 0.01 * q:
            errs.append(f"grid r={r:.4g}: 2dNLL {mine:.4g} vs Combine {q:.4g}")
    extra.append(f"grid max|d(2dNLL)| {worst:.3g}")
    # significance
    from inference.hybrid import significance

    sig_fitter = Fitter(lik)
    sig_fitter.set_range(lik.poi, 0.0, max(hi, fitter.bounds[lik.poi][1]))  # Combine's Significance is not capped at r=20
    sig = significance(lik, sig_fitter)
    z = sig.get("asymptotic", {}).get("Z")
    if z is None or abs(z - fx["significance"]["Z"]) > 0.01 + 0.01 * fx["significance"]["Z"]:
        errs.append(f"significance {z} vs Combine {fx['significance']['Z']:.4g}")
    check(not errs, f"{fixture}/{name}: " + "; ".join(errs))
    return ", ".join(extra)


def _make_combine_test(fixture):
    @test("fast", f"backends vs Combine fixture {fixture} (asymptotic 1%, best fit/68% interval 2%, grid, Z)")
    def t():
        details = []

        def one(name):
            details.append(f"{name}: {_vs_combine(name, fixture)}")
        msg = per_backend(one, backend_names(ARGS.backend))
        return msg + " || " + " | ".join(details)
    return t


for _fx in FIXTURE_MODELS:
    _make_combine_test(_fx)


# ----------------------------------------------------------------------------------------
# FAST: zmodel envelopes and floating pdf morphs vs roomodel (zmodel feature gaps)
# ----------------------------------------------------------------------------------------

@test("fast", "zmodel vs roomodel: nll_main, envelope penalty and inactive parameters per state; floating/unbinned pdf morphs")
def t_zmodel_vs_roomodel():
    if ARGS.backend not in (None, "zmodel"):
        raise Skip("zmodel-specific")
    out = []
    for fixture in ("envelope", "pdf_shape_syst_floating", "pdf_shape_syst_unbinned"):
        model, card, d = fixture_model(fixture)
        with chdir(d):
            z = likelihood("zmodel", model, card)
            r = likelihood("roomodel", model, card)
        data = observed_dataset(model)
        rng = np.random.default_rng(11)
        x0 = z.nominal_values()
        cats = [i for i, p in enumerate(z.parameters) if p.role == I.ROLE_DISCRETE]
        states = [(i, k) for i in cats for k in range(z.parameters[i].n_states)] or [(None, None)]
        worst = 0.0
        for ic, k in states:
            ref = None
            for x in random_points(z, 8, rng):
                if ic is not None:
                    x[ic] = k
                a, b = z.nll(x, data), r.nll(x, data)
                ref = ref if ref is not None else (a, b)
                worst = max(worst, abs((a - ref[0]) - (b - ref[1])))
                check(abs(a - b) < 1e-8 * max(1.0, abs(b)), f"{fixture}: zmodel NLL {a!r} vs roomodel {b!r}")
                if ic is not None:
                    check(z.discrete_penalty(x) == r.discrete_penalty(x), f"{fixture}: penalty differs in state {k}")
                    check(z.inactive_parameters(x) == r.inactive_parameters(x),
                          f"{fixture}: inactive {z.inactive_parameters(x)} vs {r.inactive_parameters(x)}")
        out.append(f"{fixture} max|d(dNLL)|={worst:.1e}")
    return " | ".join(out)


# ----------------------------------------------------------------------------------------
# FAST: toys
# ----------------------------------------------------------------------------------------

@test("fast", "toys: frequentist count variance = mean, same seed -> identical, prior toys inflate the variance")
def t_toys():
    from inference.toys import ToyConfig, generate_toys

    def one(name):
        model, card, d = fixture_model("counting")
        with chdir(d):
            lik = likelihood(name, model, card)
        fitter = Fitter(lik)
        n = 4000
        toys, _ = generate_toys(lik, fitter, ToyConfig(ntoys=n, expect_signal=1.0, frequentist=True, bypass_fit=True),
                                np.random.default_rng(5))
        c = np.array([t.main["sr"].counts[0] for t in toys])
        mean, var = c.mean(), c.var(ddof=1)
        expected = lik.expected_counts(np.array([toys[0].truth[k] for k in lik.names]))["sr"][0]
        check(abs(mean - expected) < 4 * math.sqrt(expected / n), f"frequentist mean {mean:.3f} vs {expected:.3f}")
        # Var of the sample variance of a Poisson: ~ (2 mu^2 + mu) / n
        check(abs(var - mean) < 4 * math.sqrt((2 * expected ** 2 + expected) / n),
              f"frequentist variance {var:.3f} vs mean {mean:.3f}")
        g = np.array([t.global_obs["lumi"] for t in toys])
        check(abs(g.mean()) < 4 / math.sqrt(n) and abs(g.std() - 1) < 0.05, "frequentist global observables ~ N(0, 1)")
        again, _ = generate_toys(lik, fitter, ToyConfig(ntoys=50, expect_signal=1.0, frequentist=True,
                                                        bypass_fit=True), np.random.default_rng(5))
        check(all(np.array_equal(a.main["sr"].counts, b.main["sr"].counts) and a.global_obs == b.global_obs
                  for a, b in zip(toys[:50], again)), "same seed must give identical toys")
        # prior-sampled (Combine default) toys: Var(n) = E[nu] + Var(nu)
        prior, _ = generate_toys(lik, fitter, ToyConfig(ntoys=n, expect_signal=1.0), np.random.default_rng(6))
        cp = np.array([t.main["sr"].counts[0] for t in prior])
        nus = np.array([lik.expected_counts(np.array([t.truth[k] for k in lik.names]))["sr"][0] for t in prior])
        pred = nus.mean() + nus.var()
        check(abs(cp.var(ddof=1) - pred) < 4 * pred * math.sqrt(2.0 / n),
              f"prior toys variance {cp.var(ddof=1):.2f} vs E[nu]+Var[nu] = {pred:.2f}")
        check(cp.var(ddof=1) > mean + 3 * math.sqrt((2 * expected ** 2 + expected) / n), "prior toys must inflate the variance")
        gm = np.array([t.truth["cr_stat"] for t in prior])
        check(abs(gm.mean() - 41) < 4 * math.sqrt(41 / n), f"gmN prior Gamma(N+1): mean {gm.mean():.2f} vs 41")
        check(all(t.global_obs == prior[0].global_obs for t in prior), "prior toys keep the nominal global observables")
    return per_backend(one, backend_names(ARGS.backend))


# ----------------------------------------------------------------------------------------
# FAST: shape systematics on RooAbsPdfs (syst:pdf-morph, syst:histpdf-morph)
# ----------------------------------------------------------------------------------------

def _pdf_morph_points(lik, rng, n):
    """Random points plus fixed ones in the quadratic/smooth region and beyond |x| = 1."""
    special = [dict(sigshift=0.3, sigwidth=0.2, bkgslope=-0.4), dict(sigshift=-1.7, sigwidth=2.3, bkgslope=1.4),
               dict(sigshift=0.9, sigwidth=-0.45, bkgslope=-2.5), dict(sigshift=1.0, sigwidth=0.49, bkgslope=1.0),
               dict(sigshift=-3.0, sigwidth=-3.0, bkgslope=3.2, r=2.0)]
    pts = []
    for d in special:
        x = lik.nominal_values().copy()
        for k, v in d.items():
            x[lik.index[k]] = v
        pts.append(x)
    return pts + random_points(lik, n, rng)


@test("fast", "pdf shape systs: IR of the example cards (morph kind, scales, fixed contents, Up/Down _norm ignored)")
def t_pdf_morph_ir():
    model, _, d = fixture_model("pdf_shape_syst")
    sig = _proc(model, "sr", "sig")
    check(sig.shape.pdf_morph == I.PDF_MORPH_VERTICAL, f"sig morph {sig.shape.pdf_morph}")
    check([(s.param, s.scale, s.kind) for s in sig.shape.pdf_systs] == [("sigshift", 1.0, "shape"),
                                                                       ("sigwidth", 0.5, "shape")], "sig systs")
    check(sig.shape.pdf_systs[0].up.name == "sig_pdf_sigshiftUp", "$SYSTEMATIC pattern resolution")
    check(abs(sig.shape.raw_integral - math.sqrt(2 * math.pi) * 0.5) < 1e-9, "raw integral of the nominal Gaussian")
    check(abs(sig.shape.pdf_systs[1].up_integral - math.sqrt(2 * math.pi) * 0.55) < 1e-9, "raw integral of sigwidthUp")
    check(sig.rate == 30.0 and not sig.norm_terms[1:], "yield: rate * nominal _norm, no Up/Down normalisation term")
    check(any("sig_pdf_sigshiftUp_norm" in n and "ignored" in n for n in model.notes), "note on the ignored Up/Down _norm")
    check({"syst:pdf-morph"} <= model.features() and "syst:histpdf-morph" not in model.features(), "features")
    hist, _, _ = fixture_model("pdf_shape_syst_histpdf")
    check(_proc(hist, "sr", "bkg").shape.pdf_morph == I.PDF_MORPH_HIST and "syst:histpdf-morph" in hist.features(),
          "RooHistPdf nominal -> hist morph")
    flo, _, _ = fixture_model("pdf_shape_syst_floating")
    fs = _proc(flo, "sr", "sig").shape
    check(fs.params == ["sig_scale"] and not fs.contents and not fs.pdf_systs[0].up_contents,
          f"floating pdf: params {fs.params}, no fixed contents")
    # Combine refusals: mismatched classes, mixed algorithms
    base = open(os.path.join(d, "card.txt")).read()
    bad_class = base.replace("shapes * * workspace.root w:$PROCESS_pdf w:$PROCESS_pdf_$SYSTEMATIC",
                             "shapes sig * workspace.root w:sig_pdf w:bkg_pdf_$SYSTEMATIC\n"
                             "shapes bkg * workspace.root w:bkg_pdf w:bkg_pdf_$SYSTEMATIC")
    bad_class = bad_class.replace("bkgslope     shape -     1", "bkgslope     shape 1     1")
    bad_class = bad_class.replace("sigshift     shape 1     -", "sigshift     shape -     -")
    bad_class = bad_class.replace("sigwidth     shape 0.5   -", "sigwidth     shape -     -")
    mixed = base.replace("sigwidth     shape 0.5", "sigwidth     shapeN 0.5")
    for label, text, what in (("mismatched classes", bad_class, "mismatched shape types"),
                              ("mixed algorithms", mixed, "mixes the morphing algorithms")):
        card = write_card("pdf_morph_bad", f"{label.replace(' ', '_')}.txt", text,
                          [os.path.join(d, "workspace.root")])
        try:
            with chdir(os.path.dirname(card)):
                build_ir(card)
        except UnsupportedFeature as exc:
            check(what in str(exc), f"{label}: unexpected message {exc}")
        else:
            raise AssertionError(f"{label} must raise UnsupportedFeature")
    return "vertical/hist kinds, scales, raw integrals, ignored Up/Down _norm, refusals"


@test("fast", "pdf shape systs: oracle fractions vs an independent evaluation of the workspace pdfs")
def t_pdf_morph_reference():
    """VerticalInterpPdf (algorithm 0) re-derived from the raw RooFit values of the pdfs, and
    FastVerticalInterpHistPdf2 from TH1F-rounded bin-centre values, without pymodel code."""
    from modelspec import rootinput as R

    ROOT = R.root()
    model, _, d = fixture_model("pdf_shape_syst")
    hmodel, _, _ = fixture_model("pdf_shape_syst_histpdf")
    oracle, horacle = SemanticLikelihood(model), SemanticLikelihood(hmodel)
    systs = {"sig": [("sigshift", 1.0), ("sigwidth", 0.5)], "bkg": [("bkgslope", 1.0)]}
    worst = [0.0, 0.0]
    for wsname, lik, idx in (("workspace.root", oracle, 0), ("workspace_hist.root", horacle, 1)):
        w = R.get_workspace(os.path.join(d, wsname), "w")
        x = w.var("x")
        nset = ROOT.RooArgSet(x)
        edges = np.asarray(lik.model.channels[0].observable.edges)
        cen, wid = 0.5 * (edges[1:] + edges[:-1]), np.diff(edges)

        def raw(name):
            out = []
            for c in cen:
                x.setVal(c)
                out.append(w.pdf(name).getVal())
            return np.array(out)

        def norm_hist(name):
            out = []
            for c, wi in zip(cen, wid):
                x.setVal(c)
                out.append(np.float32(w.pdf(name).getVal(nset) * wi))
            out = np.array(out, dtype=float)
            return out / out.sum()

        for pt in _pdf_morph_points(lik, np.random.default_rng(21), 6):
            exp = lik.expected_by_process(pt)["sr"]
            for proc, sy in systs.items():
                q = min([1.0] + [sc for _, sc in sy])
                if idx == 0:
                    f0, i0 = raw(f"{proc}_pdf"), w.pdf(f"{proc}_pdf").createIntegral(nset).getVal()
                    num, den = f0.copy(), i0
                    for s, sc in sy:
                        c = sc * pt[lik.index[s]]
                        fu, fd = raw(f"{proc}_pdf_{s}Up"), raw(f"{proc}_pdf_{s}Down")
                        iu = w.pdf(f"{proc}_pdf_{s}Up").createIntegral(nset).getVal()
                        idn = w.pdf(f"{proc}_pdf_{s}Down").createIntegral(nset).getVal()
                        if abs(c) >= q:
                            num += c * (fu - f0) if c > 0 else c * (f0 - fd)
                            den += c * (iu - i0) if c > 0 else c * (i0 - idn)
                        else:
                            cu, cd, cc = c * (q + c) / (2 * q), -c * (q - c) / (2 * q), -c * c / q
                            num += cu * fu + cd * fd + cc * f0
                            den += cu * iu + cd * idn + cc * i0
                    ref = np.where(num > 0, num, 1e-15) / (den if den > 0 else 1e-10) * wid
                else:
                    t = norm_hist(f"{proc}_pdf")
                    nom = t.copy()
                    for s, sc in sy:
                        c = sc * pt[lik.index[s]]
                        dhi, dlo = norm_hist(f"{proc}_pdf_{s}Up") - nom, norm_hist(f"{proc}_pdf_{s}Down") - nom
                        xn = c / q
                        step = math.copysign(1.0, c) if abs(c) >= q else 0.125 * xn * (xn * xn * (3 * xn * xn - 10) + 15)
                        t = t + 0.5 * c * ((dhi - dlo) + (dhi + dlo) * step)
                    t = np.where(t / wid < 1e-9, 1e-9 * wid, t)
                    ref = t / t.sum()
                mine = exp[proc] / np.sum(exp[proc]) * np.sum(ref)
                worst[idx] = max(worst[idx], float(np.max(np.abs(mine - ref) / np.maximum(ref, 1e-300))))
    check(max(worst) < 1e-10, f"relative fraction differences {worst}")
    return f"max rel. difference: VerticalInterpPdf {worst[0]:.1e}, FastVerticalInterpHistPdf2 {worst[1]:.1e}"


@test("fast", "pdf shape systs: backends vs oracle (NLL, yields) incl. |x| < q and |x| > 1; hfmodel only equal scales")
def t_pdf_morph_backends():
    names = [n for n in backend_names(ARGS.backend) if n != "semantic"]
    msgs = []
    model_s1, _, d = fixture_model("pdf_shape_syst_histpdf")
    for ch in model_s1.channels:  # a variant with one common scale (hfmodel code4p is exact only then)
        for proc in ch.processes:
            for s in proc.shape.pdf_systs:
                s.scale = 1.0
    cases = [("card", fixture_model("pdf_shape_syst")[0]), ("histpdf", fixture_model("pdf_shape_syst_histpdf")[0]),
             ("histpdf scale 1", model_s1)]
    for label, model in cases:
        oracle = SemanticLikelihood(model)
        data = observed_dataset(model)
        pts = _pdf_morph_points(oracle, np.random.default_rng(4), 10)

        def one(name):
            with chdir(d):
                lik = likelihood(name, model, os.path.join(d, "card.txt"))
            notes = getattr(lik, "notes", [])
            if any("vertical morphing uses pyhf histosys code4p" in n for n in notes):
                return "declared inexact smooth region (mixed scales), not compared"
            worst = 0.0
            for x in pts:
                a, b = lik.nll(x, data), oracle.nll(x, data)
                worst = max(worst, abs(a - b))
                check(abs(a - b) < 1e-8 * max(1.0, abs(b)), f"{label}/{name}: NLL {a:.12g} vs oracle {b:.12g} at "
                                                          f"{oracle.values_dict(x)}")
                ea, eb = lik.expected_by_process(x), oracle.expected_by_process(x)
                for p in eb["sr"]:
                    check(np.allclose(ea["sr"][p], eb["sr"][p], rtol=1e-9, atol=1e-12),
                          f"{label}/{name}: expected {p} {ea['sr'][p]} vs {eb['sr'][p]}")
            return f"max|dNLL| {worst:.1e}"
        msgs.append(f"{label}: " + per_backend(one, names))
    return " | ".join(msgs)


@test("fast", "pdf shape systs: roomodel on floating/unbinned pdfs vs a RooFit NLL of the same VerticalInterpPdf")
def t_pdf_morph_roomodel_floating():
    """roomodel's NLL differences equal those of an independently built RooFit model (the
    workspace pdfs, VerticalInterpPdf or the equivalent RooRealSumPdf, RooAddPdf, RooNLLVar)."""
    from modelspec import rootinput as R

    ROOT = R.root()
    out = []
    for fixture in ("pdf_shape_syst_floating", "pdf_shape_syst_unbinned"):
        model, card, d = fixture_model(fixture)
        with chdir(d):
            lik = likelihood("roomodel", model, card)
        w = R.get_workspace(os.path.join(d, "workspace.root"), "w")
        x = w.var("x")
        nuis = {}
        for n in lik.names:
            v = w.var(n)
            nuis[n] = v if v else ROOT.RooRealVar(n, n, 0.0)
            nuis[n].setConstant(False)
        keep, pdfs, coefs = [], ROOT.RooArgList(), ROOT.RooArgList()
        for proc in model.channels[0].processes:
            sh = proc.shape
            q = min([1.0] + [s.scale for s in sh.pdf_systs])
            funcs, cs = ROOT.RooArgList(w.pdf(sh.ref.name)), ROOT.RooArgList()
            terms = []
            for s in sh.pdf_systs:
                funcs.add(w.pdf(s.up.name))
                funcs.add(w.pdf(s.down.name))
                c = ROOT.RooFormulaVar(f"c_{proc.name}_{s.param}", f"{s.scale!r}*@0", ROOT.RooArgList(nuis[s.param]))
                keep.append(c)
                terms.append(c)
            if hasattr(ROOT, "VerticalInterpPdf"):
                for c in terms:
                    cs.add(c)
                morph = ROOT.VerticalInterpPdf(f"ref_{proc.name}", "", funcs, cs, q, 0)
            else:
                qs = repr(q)
                cen = "+".join(f"((abs(@{k})>={qs}) ? ((@{k}>0) ? -@{k} : @{k}) : (-@{k}*@{k}/{qs}))"
                               for k in range(len(terms)))
                al = ROOT.RooArgList()
                for c in terms:
                    al.add(c)
                c0 = ROOT.RooFormulaVar(f"c0_{proc.name}", f"1.0+{cen}", al)
                cs.add(c0)
                keep.append(c0)
                for c in terms:
                    for e in (f"((abs(@0)>={qs}) ? ((@0>0) ? @0 : 0.0) : (@0*({qs}+@0)/(2.0*{qs})))",
                              f"((abs(@0)>={qs}) ? ((@0>0) ? 0.0 : -@0) : (-@0*({qs}-@0)/(2.0*{qs})))"):
                        f = ROOT.RooFormulaVar(f"{c.GetName()}_{len(keep)}", e, ROOT.RooArgList(c))
                        keep.append(f)
                        cs.add(f)
                morph = ROOT.RooRealSumPdf(f"ref_{proc.name}", "", funcs, cs)
                morph.setFloor(True)
            keep += [funcs, cs, morph]
            factors = [ROOT.RooConstVar(f"rate_{proc.name}", "", proc.rate)]
            if proc.is_signal:
                factors.append(nuis["r"])
            for t in proc.norm_terms:
                factors.append(ROOT.RooFormulaVar(f"n_{proc.name}_{t.param}", f"exp(@0*{math.log(t.kappa_hi)!r})",
                                                  ROOT.RooArgList(nuis[t.param])))
            fl = ROOT.RooArgList()
            for f in factors:
                fl.add(f)
            yld = ROOT.RooProduct(f"y_{proc.name}", "", fl)
            keep += factors + [fl, yld]
            pdfs.add(morph)
            coefs.add(yld)
        tot = ROOT.RooAddPdf("ref_total", "", pdfs, coefs)
        dname = "data_obs_unbinned" if "unbinned" in fixture else "data_obs"
        nll = tot.createNLL(w.data(dname), ROOT.RooFit.Extended(True), ROOT.RooFit.EvalBackend("legacy"))
        data = observed_dataset(model)
        pts = _pdf_morph_points(lik, np.random.default_rng(8), 8)
        cons = lambda x: sum(0.5 * x[lik.index[p.name]] ** 2 for p in model.constrained_parameters())  # noqa: E731

        def ref(xv):
            for n, v in zip(lik.names, xv):
                nuis[n].setVal(float(v))
            return nll.getVal() + cons(xv)
        x0 = lik.nominal_values()
        r0, m0 = ref(x0), lik.nll(x0, data)
        worst = 0.0
        for xv in pts:
            worst = max(worst, abs((lik.nll(xv, data) - m0) - (ref(xv) - r0)))
        check(worst < 1e-8, f"{fixture}: max |dNLL(roomodel) - dNLL(RooFit)| = {worst:.2e}")
        out.append(f"{fixture} {worst:.1e}")
    return ", ".join(out)


# ----------------------------------------------------------------------------------------
# FAST: parallel toys, job splitting/merging, impacts, 2D scans (inference layer + CLI)
# ----------------------------------------------------------------------------------------

def _no_zmodel(names, toys=False):
    """Tests of the shared layer run on semantic/hfmodel/roomodel.  zmodel only when selected
    explicitly, and never for toy tests (TF call overhead makes toy fits slow)."""
    out = [n for n in names if n != "zmodel" or (ARGS.backend == "zmodel" and not toys)]
    if not out:
        raise Skip("zmodel is not used for the toy tests of the shared inference layer (slow toy fits)")
    return out


def run_cli(bname, argv, cwd):
    """Run a pymodel command in-process with backend ``bname`` (output silenced)."""
    import io
    import pymodel_core as C

    b = backend(bname)
    args = C.build_parser(b).parse_args(argv)
    with chdir(cwd), contextlib.redirect_stdout(io.StringIO()):
        try:
            getattr(C, "cmd_" + args.command)(b, args)
        except UnsupportedByBackend as exc:
            raise Skip(f"{bname}: {exc}") from exc


def _read(path):
    with open(path) as handle:
        return json.load(handle)


@test("fast", "parallel toys: --jobs 1 and --jobs 3 give bit-identical toys (limit, fc, significance, fit -t)")
def t_parallel_determinism():
    def one(name):
        d = WORK.example("counting")
        out = os.path.join(WORK.root, "parallel", name)
        os.makedirs(out, exist_ok=True)
        f = lambda s: os.path.join(out, s)
        common = ["card.txt", "--seed", "7"]
        for j in (1, 3):
            run_cli(name, ["limit"] + common + ["--method", "toys", "--grid", "1.5,2.5", "--toys-per-point", "40",
                                                "--refine", "1", "-j", str(j), "-o", f(f"l{j}.json"),
                                                "--save-toy-results", f(f"lraw{j}.json")], d)
            run_cli(name, ["fc"] + common + ["--grid", "0.5,1.8", "--toys-per-point", "40", "--refine", "0", "-j",
                                             str(j), "-o", f(f"fc{j}.json"), "--save-toy-results", f(f"fraw{j}.json")],
                    d)
            run_cli(name, ["significance"] + common + ["--method", "toys", "--toys-per-point", "40", "-j", str(j),
                                                       "-o", f(f"z{j}.json")], d)
            run_cli(name, ["fit"] + common + ["-t", "12", "--expect-signal", "1", "--toys-frequentist", "-j", str(j),
                                              "-o", f(f"t{j}.json")], d)
        a, b = _read(f("lraw1.json")), _read(f("lraw3.json"))
        check(a["points"] == b["points"], "CLs raw test statistics differ between --jobs 1 and 3")
        check(len(a["points"]) == 3 and all(len(p["q_b"]) + p["failed_b"] == 40 for p in a["points"]),
              "expected 3 points (grid + 1 refinement) with 40 b-only toys each")
        la, lb = _read(f("l1.json"))["result"], _read(f("l3.json"))["result"]
        check(la["observed"] == lb["observed"] and la["expected"] == lb["expected"], "CLs limits differ")
        check(_read(f("fraw1.json"))["points"] == _read(f("fraw3.json"))["points"], "FC raw toys differ")
        check(_read(f("z1.json"))["result"]["toys"] == _read(f("z3.json"))["result"]["toys"], "toy significance differs")
        check(_read(f("t1.json"))["result"]["fits"] == _read(f("t3.json"))["result"]["fits"], "toy fits differ")
        return f"limit {la['observed']:.4g}"
    return per_backend(one, _no_zmodel(backend_names(ARGS.backend), toys=True))


@test("fast", "job splitting: --toy-chunk and --points jobs merged with 'merge' equal the single run exactly")
def t_split_merge():
    def one(name):
        d = WORK.example("counting")
        out = os.path.join(WORK.root, "merge", name)
        os.makedirs(out, exist_ok=True)
        f = lambda s: os.path.join(out, s)
        base = ["card.txt", "--seed", "11", "--toys-per-point", "40", "--grid", "1.5,2.5"]
        run_cli(name, ["limit", "--method", "toys"] + base + ["--refine", "0", "-o", f("full.json"),
                                                               "--save-toy-results", f("full_raw.json")], d)
        for i in range(3):
            run_cli(name, ["limit", "--method", "toys"] + base + ["--toy-chunk", f"{i}/3", "-o", f(f"c{i}.json"),
                                                                   "--save-toy-results", f(f"c{i}_raw.json")], d)
        for r in ("1.5", "2.5"):
            run_cli(name, ["limit", "--method", "toys"] + base + ["--points", r, "-o", f(f"p{r}.json"),
                                                                   "--save-toy-results", f(f"p{r}_raw.json")], d)
        run_cli(name, ["merge", f("c0_raw.json"), f("c1_raw.json"), f("c2_raw.json"), "-o", f("mc.json"),
                       "--save-toy-results", f("mc_raw.json")], d)
        run_cli(name, ["merge", f("p1.5_raw.json"), f("p2.5_raw.json"), "-o", f("mp.json")], d)
        full = _read(f("full.json"))["result"]
        for tag in ("mc", "mp"):
            m = _read(f(f"{tag}.json"))["result"]
            check(m["observed"] == full["observed"] and m["expected"] == full["expected"],
                  f"merged ({tag}) limit {m['observed']} vs single run {full['observed']}")
            for key in ("CLs", "CLb", "CLsplusb", "n_sb", "n_b"):
                check([p[key] for p in m["points"]] == [p[key] for p in full["points"]], f"merged ({tag}) {key} differ")
        fr, mr = _read(f("full_raw.json"))["points"], _read(f("mc_raw.json"))["points"]
        check(all(sorted(a["q_sb"]) == sorted(b["q_sb"]) and sorted(a["q_b"]) == sorted(b["q_b"])
                  for a, b in zip(fr, mr)), "merged raw toys are not the single-run toys")
        # a file merged with itself would double count its toys: refused
        try:
            run_cli(name, ["merge", f("c0_raw.json"), f("c0_raw.json"), "-o", f("bad.json")], d)
        except SystemExit as exc:
            check("more than one file" in str(exc), f"unexpected merge error: {exc}")
        else:
            raise AssertionError("merging a file with itself was not refused")
        # FC: two chunks merged equal the single run
        fcb = ["card.txt", "--seed", "11", "--toys-per-point", "30", "--grid", "0.5,1.8"]
        run_cli(name, ["fc"] + fcb + ["--refine", "0", "-o", f("fcf.json")], d)
        for i in range(2):
            run_cli(name, ["fc"] + fcb + ["--toy-chunk", f"{i}/2", "-o", f(f"fc{i}.json"),
                                          "--save-toy-results", f(f"fc{i}_raw.json")], d)
        run_cli(name, ["merge", f("fc0_raw.json"), f("fc1_raw.json"), "-o", f("fcm.json")], d)
        a, b = _read(f("fcf.json"))["result"], _read(f("fcm.json"))["result"]
        check([p["p"] for p in a["points"]] == [p["p"] for p in b["points"]] and a["upper"] == b["upper"],
              "merged FC p-values differ from the single run")
        return f"limit {full['observed']:.4g}"
    return per_backend(one, _no_zmodel(backend_names(ARGS.backend), toys=True))


@test("fast", "adaptive toys: --cls-acc/--p-acc add toys until the target, caps are recorded and flagged")
def t_adaptive_toys():
    from inference.hybrid import feldman_cousins, toy_cls_limit
    from inference.parallel import ToySeeds

    def one(name):
        model, card, d = fixture_model("counting")
        with chdir(d):
            lik = likelihood(name, model, card)
        res = toy_cls_limit(lik, Fitter(lik), ToySeeds(3), [1.9, 4.0], ntoys=25, cls_acc=0.02, max_toys=200,
                            expected=False)
        p19, p4 = res.points
        cls, err = p19.pvalues()[4:6]
        check(p19.n_sb > 25, f"no toys added at r=1.9 (CLs {cls:.3f} +- {err:.3f})")
        check(err <= 0.02 or p19.n_sb == 200, f"r=1.9 stopped at CLs error {err:.3f} with {p19.n_sb} toys")
        check("sigma" in p4.stop_reason or "accuracy" in p4.stop_reason, f"r=4 stop reason: {p4.stop_reason}")
        capped = toy_cls_limit(lik, Fitter(lik), ToySeeds(3), [1.9], ntoys=25, cls_acc=0.001, max_toys=50,
                               expected=False)
        check(capped.points[0].n_sb == 50 and capped.points[0].stop_reason.startswith("max toys"),
              f"cap not applied: {capped.points[0].n_sb} toys, {capped.points[0].stop_reason}")
        check(any("max toys reached" in fl for fl in capped.flags), "a capped point must be flagged")
        fc = feldman_cousins(lik, Fitter(lik), ToySeeds(3), [1.6], ntoys=25, p_acc=0.03, max_toys=400)
        p, e = fc.points[0].pvalue()
        check(e <= 0.03 or fc.points[0].n == 400 or abs(p - 0.1) > 3 * e, f"FC p {p:.3f} +- {e:.3f}")
        check(fc.points[0].n > 25, "no FC toys added near alpha")
        return f"r=1.9: {p19.n_sb} toys, CLs {cls:.3f}+-{err:.3f}; FC r=1.6: {fc.points[0].n} toys"
    return per_backend(one, _no_zmodel(backend_names(ARGS.backend), toys=True))


def _independent_impact(ref, name, level, r_bounds=(-5.0, 5.0)):
    """Impact of one nuisance with the independent reference model (scipy, no pymodel code)."""
    from scipy.optimize import brentq, minimize

    bounds = list(ref.bounds)
    bounds[0] = r_bounds
    i = ref.idx[name]

    def prof(fix=None, start=None):
        free = [k for k in range(len(bounds)) if k != fix]
        x0 = ref.nominal_point() if start is None else np.array(start, dtype=float)
        x0[0] = 0.0 if start is None else x0[0]

        def f(z):
            x = x0.copy()
            x[free] = z
            val = ref.nll(x)
            return val if math.isfinite(val) else 1e30
        res = minimize(f, x0[free], method="L-BFGS-B", bounds=[bounds[k] for k in free],
                       options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 5000})
        res = minimize(f, res.x, method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": 20000})
        x = x0.copy()
        x[free] = res.x
        return float(res.fun), x

    nll0, x0 = prof()

    def at(v):
        start = x0.copy()
        start[i] = v
        return prof(fix=i, start=start)

    th = x0[i]
    step = 1.0 if name != "cr_stat" else 6.0
    g = lambda v: 2.0 * (at(v)[0] - nll0) - level
    lo = brentq(g, th - 3 * step, th, xtol=1e-6)
    hi = brentq(g, th, th + 3 * step, xtol=1e-6)
    return {"fit": [lo, th, hi], "r": [at(lo)[1][0], x0[0], at(hi)[1][0]]}


@test("fast", "impacts: counting vs an independent scipy calculation; counting/templates vs combineTool Impacts")
def t_impacts():
    from inference.impacts import LEVEL_68, impacts

    def one(name):
        msgs = []
        for fixture, r_range in (("counting", (-5.0, 5.0)), ("templates", (0.0, 20.0))):
            fx = load_fixture(f"{fixture}_impacts")["impacts"]
            model, card, d = fixture_model(fixture, poi_range=r_range)
            try:
                with chdir(d):
                    lik = likelihood(name, model, card)
            except Skip as exc:
                msgs.append(f"{fixture} skipped ({exc})")
                continue
            res = impacts(lik, Fitter(lik), observed_dataset(model))
            mine = {p["name"]: p for p in res["params"]}
            check(all(p["valid"] for p in res["params"]), f"{fixture}: failed impacts {res['flags']}")
            order = [p["impact"] for p in res["params"]]
            check(order == sorted(order, reverse=True), "parameters must be sorted by |impact|")
            (c_lo, _c, c_hi), (m_lo, _m, m_hi) = fx["POIs"][0]["fit"], res["POIs"][0]["fit"]
            check(abs(c_lo - m_lo) < 3e-3 and abs(c_hi - m_hi) < 3e-3,
                  f"{fixture}: r interval [{m_lo:.4f}, {m_hi:.4f}] vs Combine [{c_lo:.4f}, {c_hi:.4f}]")
            worst = 0.0
            # hfmodel: pyhf's template interpolation is not exactly Combine's (docs/backend-hfmodel.md)
            tol_r = 1e-2 if name == "hfmodel" else 3e-3
            for p in fx["params"]:
                m = mine[p["name"]]
                width = p["prefit"][2] - p["prefit"][0] if p["prefit"][2] > p["prefit"][0] else 1.0
                for k in (0, 2):
                    check(abs(m["fit"][k] - p["fit"][k]) < (5 if name == "hfmodel" else 1) * 2e-3 * width,
                          f"{fixture} {p['name']}: theta crossing {m['fit'][k]:.5f} vs Combine {p['fit'][k]:.5f}")
                    check(abs(m["r"][k] - p["r"][k]) < tol_r,
                          f"{fixture} {p['name']}: r {m['r'][k]:.5f} vs Combine {p['r'][k]:.5f}")
                check(abs(m["impact"] - p["impact_r"]) < tol_r,
                      f"{fixture} {p['name']}: impact {m['impact']:.5f} vs Combine {p['impact_r']:.5f}")
                if "pull" in m:
                    check(abs(m["prefit"][0] - p["prefit"][0]) < 1e-3 * width, f"{p['name']}: prefit interval")
                worst = max(worst, abs(m["impact"] - p["impact_r"]))
            msgs.append(f"{fixture}: max |impact - Combine| {worst:.1e}")
            if fixture == "counting":
                ref = ref_counting()
                wi = 0.0
                for pname in ("bkg_norm", "cr_stat"):
                    ind = _independent_impact(ref, pname, LEVEL_68)
                    m = mine[pname]
                    wi = max(wi, abs(m["r"][0] - ind["r"][0]), abs(m["r"][2] - ind["r"][2]))
                    for k in (0, 2):
                        check(abs(m["fit"][k] - ind["fit"][k]) < 1e-3 * (6.0 if pname == "cr_stat" else 1.0),
                              f"{pname}: crossing {m['fit'][k]:.5f} vs independent {ind['fit'][k]:.5f}")
                        # Minuit's EDM goal (tolerance 0.01) limits r to a few 1e-3 here
                        check(abs(m["r"][k] - ind["r"][k]) < 3e-3,
                              f"{pname}: r {m['r'][k]:.5f} vs independent {ind['r'][k]:.5f}")
                msgs.append(f"counting vs independent: max |d r| {wi:.1e}")
        if all("skipped" in m for m in msgs):
            raise Skip("; ".join(msgs))
        return "; ".join(msgs)
    return per_backend(one, _no_zmodel(backend_names(ARGS.backend)))


@test("fast", "2D scan (r, bkg_norm) on templates vs Combine MultiDimFit --algo grid (2DeltaNLL, contours)")
def t_scan_2d():
    from inference.scan import LEVELS_2D, contour_lines, profile_scan_2d

    fx = load_fixture("templates_scan2d")["scan2d"]
    grid = np.array(fx["grid"])
    xs_all, ys_all = np.unique(grid[:, 0]), np.unique(grid[:, 1])
    xs, ys = xs_all[::2], ys_all[::2]   # 10 x 10 of Combine's 20 x 20 points
    cz = {(round(x, 5), round(y, 5)): z for x, y, z in grid}
    zc = np.array([[cz[(round(x, 5), round(y, 5))] for y in ys] for x in xs])

    def one(name):
        model, card, d = fixture_model("templates")
        with chdir(d):
            lik = likelihood(name, model, card)
        fitter = Fitter(lik)
        out = profile_scan_2d(lik, fitter, observed_dataset(model), fx["params"], xs, ys)
        zm = np.array([p["deltaNLL2"] for p in out["points"]], dtype=float).reshape(len(xs), len(ys))
        diff = np.abs(zm - zc)
        # hfmodel: pyhf's interpolation differs slightly from Combine's (docs/backend-hfmodel.md);
        # same tolerance as the 1D grid comparison of the Combine-fixture tests
        tol = (0.02 + 0.01 * zc) if name == "hfmodel" else (2e-3 + 1e-3 * zc)
        check(np.all(diff <= tol), f"2DeltaNLL differs from Combine by up to {diff.max():.4f} "
                                   f"at {np.unravel_index(diff.argmax(), diff.shape)}")
        check(abs(out["best_fit"][0] - fx["best_fit"][0]) < 3e-3 and abs(out["best_fit"][1] - fx["best_fit"][1]) < 5e-3,
              f"best fit {out['best_fit']} vs Combine {fx['best_fit']}")
        for key, level in LEVELS_2D.items():
            ref = np.vstack(contour_lines(xs, ys, zc, level))
            got = np.asarray(out["contours"][key]["y_range"])
            check(np.all(np.abs(got - [ref[:, 1].min(), ref[:, 1].max()]) < 0.02),
                  f"{key}% contour y range {got} vs Combine grid {ref[:, 1].min():.3f}..{ref[:, 1].max():.3f}")
        return f"max |2DeltaNLL - Combine| {diff.max():.1e}"
    return per_backend(one, _no_zmodel(backend_names(ARGS.backend)))


# ----------------------------------------------------------------------------------------
# FAST: autoMCStats (Barlow-Beeston-lite, mcstats:bb-lite)
# ----------------------------------------------------------------------------------------

# Combine's setupBinPars for examples/mcstats (text2workspace.py output, Combine 137dbced):
# name -> (kind, nominal value, range lo, range hi) with ranges rounded as Combine prints them
MCSTATS_COMBINE_PARAMS = {
    **{f"prop_binch1_bin{j}": ("total", 0.0, -7.0, 7.0) for j in range(4)},
    "prop_binch1_bin4_sig": ("gauss", 0.0, -7.0, 7.0), "prop_binch1_bin4_bkg1": ("gauss", 0.0, -7.0, 7.0),
    "prop_binch1_bin4_bkg2": ("poisson", 1.0, 0.00, 30.85), "prop_binch1_bin5_sig": ("gauss", 0.0, -7.0, 7.0),
    "prop_binch1_bin5_bkg1": ("poisson", 6.0, 0.03, 43.60), "prop_binch1_bin5_bkg2": ("poisson", 1.0, 0.00, 30.85),
    "prop_binch1_bin6_sig": ("poisson", 4.0, 0.00, 38.96), "prop_binch1_bin6_bkg1": ("gauss", 0.0, -7.0, 7.0),
    "prop_binch1_bin6_bkg2": ("gauss", 0.0, -7.0, 7.0), "prop_binch1_bin7_bkg1": ("poisson", 4.0, 0.00, 38.96),
    **{f"prop_binch2_bin{j}": ("total", 0.0, -7.0, 7.0) for j in range(4)},
    "prop_binch2_bin4_sig": ("poisson", 2.0, 0.00, 33.79), "prop_binch2_bin4_bkg": ("poisson", 2.0, 0.00, 33.79),
}


def _mcstats_card(tag, stats_lines, extra=""):
    """A copy of the mcstats example card with other autoMCStats lines (same mcstats.root)."""
    src = WORK.example("mcstats")
    with open(os.path.join(src, "card.txt")) as handle:
        text = "".join(l for l in handle if "autoMCStats" not in l)
    return write_card("mcstats", f"{tag}.txt", text + extra + stats_lines + "\n",
                      files=[os.path.join(src, "mcstats.root")])


@test("fast", "autoMCStats grammar/IR: parameters, kinds and ranges = Combine setupBinPars; variants")
def t_mcstats_grammar():
    model, card, d = fixture_model("mcstats")
    got = {bp.param: bp for ch in model.channels for bp in ch.mcstats.params}
    check(list(got) == list(MCSTATS_COMBINE_PARAMS), f"parameter list/order {list(got)}")
    for name, (kind, value, lo, hi) in MCSTATS_COMBINE_PARAMS.items():
        p = model.parameters[name]
        check(got[name].kind == kind, f"{name}: kind {got[name].kind} vs Combine {kind}")
        check(p.value == value and round(p.lo, 2) == lo and round(p.hi, 2) == hi,
              f"{name}: value/range {p.value} [{p.lo:.3f}, {p.hi:.3f}] vs Combine {value} [{lo}, {hi}]")
        want = I.CONSTRAINT_POISSON if kind == "poisson" else I.CONSTRAINT_GAUSS
        check(p.constraint.kind == want and p.constraint.center == value and p.role == I.ROLE_NUISANCE,
              f"{name}: constraint {p.constraint}")
        if kind == "poisson":
            check(got[name].n_eff == value, f"{name}: n_eff {got[name].n_eff}")
    check(model.groups.get("autoMCStats") == list(MCSTATS_COMBINE_PARAMS), "group autoMCStats")
    check("mcstats:bb-lite" in model.features(), "feature string")
    ch1, ch2 = model.channel("ch1").mcstats, model.channel("ch2").mcstats
    check((ch1.threshold, ch1.include_signal, ch1.hist_mode) == (10.0, False, 1), f"ch1 flags {ch1}")
    check((ch2.threshold, ch2.include_signal, ch2.hist_mode) == (5.0, True, 1), f"ch2 flags {ch2}")
    # JSON round trip of the IR
    check(I.ir_from_dict(json.loads(json.dumps(model.to_dict()))) == model, "IR JSON round trip")
    # wildcard channel pattern; include-signal changes the ch1 decision in bin 4 (n_eff 9 -> 11 > 10)
    with chdir(WORK.tmp("mcstats")):
        m = build_ir(_mcstats_card("wild", "* autoMCStats 10"))
        check(all(ch.mcstats is not None and ch.mcstats.threshold == 10.0 for ch in m.channels), "wildcard")
        m = build_ir(_mcstats_card("incsig", "ch1 autoMCStats 10 1"))
        kinds = {bp.bin: bp.kind for bp in m.channel("ch1").mcstats.params if bp.bin == 4}
        check(kinds == {4: "total"} and m.channel("ch2").mcstats is None, f"include-signal: bin 4 {kinds}")
        m = build_ir(_mcstats_card("neg", "ch1 autoMCStats -1"))
        check(m.channel("ch1").mcstats is not None and not m.channel("ch1").mcstats.params,
              "negative threshold: CMSHistFunc semantics, no bin parameters")
        for tag, line in (("hist2", "ch1 autoMCStats 10 0 2"), ("hist0", "ch1 autoMCStats 10 0 0")):
            try:
                build_ir(_mcstats_card(tag, line))
                check(False, f"{line}: not refused")
            except UnsupportedFeature:
                pass
    # counting channels cannot have autoMCStats in Combine (no CMSHistFunc)
    cnt = write_card("mcstats", "counting.txt", """
imax 1
jmax 1
kmax *
bin a
observation 5
bin a a
process s b
process 0 1
rate 1 4
a autoMCStats 0
""")
    try:
        build_ir(cnt)
        check(False, "autoMCStats on a counting channel not refused")
    except UnsupportedFeature:
        pass
    return f"{len(got)} parameters as Combine; variants ok"


@test("fast", "autoMCStats oracle vs an independent evaluation of the TH1 templates (total/poisson/gauss bins)")
def t_mcstats_oracle():
    import ROOT

    model, card, d = fixture_model("mcstats")
    lik = SemanticLikelihood(model)
    f = ROOT.TFile.Open(os.path.join(d, "mcstats.root"))

    def th1(path):
        h = f.Get(path)
        return (np.array([h.GetBinContent(i + 1) for i in range(h.GetNbinsX())]),
                np.array([h.GetBinError(i + 1) for i in range(h.GetNbinsX())]))

    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(10):
        v = {n: p.value for n, p in model.parameters.items()}
        v["r"] = rng.uniform(0.0, 3.0)
        for n in ("lumi", "bkg1_norm", "bkg2_norm", "bkg_shape"):
            v[n] = rng.normal(0.0, 1.0)
        for n, (kind, value, lo, hi) in MCSTATS_COMBINE_PARAMS.items():
            v[n] = value * rng.uniform(0.3, 2.0) if kind == "poisson" else rng.normal(0.0, 1.5)
        x = np.array([v[n] for n in lik.names])
        exp = lik.expected_counts(x)
        # ch1: C_p = lnN factors (x r for sig); yield = C_p * template (+ BB terms)
        c = {"sig": v["r"] * 1.025 ** v["lumi"], "bkg1": 1.025 ** v["lumi"] * 1.10 ** v["bkg1_norm"],
             "bkg2": 1.025 ** v["lumi"] * 1.20 ** v["bkg2_norm"]}
        h = {p: th1(f"ch1/{p}") for p in c}
        nu = sum(c[p] * np.maximum(h[p][0], 1e-9) for p in c)
        for j in range(4):
            nu[j] += v[f"prop_binch1_bin{j}"] * math.sqrt(sum((c[p] * h[p][1][j]) ** 2 for p in c))
        for name, (kind, value, lo, hi) in MCSTATS_COMBINE_PARAMS.items():
            if not name.startswith("prop_binch1_") or kind == "total":
                continue
            j, p = int(name.split("_")[2][3:]), name.split("_")[3]
            nu[j] += (v[name] / value - 1.0) * c[p] * h[p][0][j] if kind == "poisson" else v[name] * c[p] * h[p][1][j]
        worst = max(worst, float(np.max(np.abs(exp["ch1"] - nu) / np.maximum(np.abs(nu), 1e-9))))
        # ch2: bkg shape systematic, CMSHistFunc hist-mode 1 (up/down scaled to the nominal integral,
        # no renormalisation) with asymPow(kd, ku) of the integrals; bins 0-3 total, bin 4 Poisson
        sig, bkg, up, dn = th1("ch2/sig"), th1("ch2/bkg"), th1("ch2/bkg_bkg_shapeUp")[0], th1("ch2/bkg_bkg_shapeDown")[0]
        t = v["bkg_shape"]
        ku, kd = up.sum() / bkg[0].sum(), dn.sum() / bkg[0].sum()
        s = 1.0 if t >= 1 else -1.0 if t <= -1 else 0.125 * t * (t * t * (3 * t * t - 10) + 15)
        hb = bkg[0] + 0.5 * t * ((up / ku - dn / kd) + (up / ku + dn / kd - 2 * bkg[0]) * s)
        hb = np.maximum(hb, 1e-9)
        cb = 1.025 ** v["lumi"] * float(REF_asym_pow(t, kd, ku))
        cs = v["r"] * 1.025 ** v["lumi"]
        nu2 = cs * sig[0] + cb * hb
        for j in range(4):
            nu2[j] += v[f"prop_binch2_bin{j}"] * math.sqrt((cs * sig[1][j]) ** 2 + (cb * bkg[1][j]) ** 2)
        nu2[4] += (v["prop_binch2_bin4_sig"] / 2 - 1) * cs * sig[0][4] + (v["prop_binch2_bin4_bkg"] / 2 - 1) * cb * hb[4]
        worst = max(worst, float(np.max(np.abs(exp["ch2"] - nu2) / np.maximum(np.abs(nu2), 1e-9))))
    f.Close()
    check(worst < 1e-12, f"oracle vs independent yields: max rel diff {worst:.2e}")
    return f"max rel yield diff {worst:.1e}"


def REF_asym_pow(theta, kd, ku):
    """Combine asymPow written out independently (CombineMathFuncs.h logKappaForX)."""
    if abs(theta) >= 0.5:
        return ku ** theta if theta >= 0 else kd ** (-theta)
    lhi, llo = math.log(ku), -math.log(kd)
    x2 = 2 * theta
    alpha = 0.125 * x2 * (x2 * x2 * (3 * x2 * x2 - 10) + 15)
    return math.exp(theta * (0.5 * (lhi + llo) + alpha * 0.5 * (lhi - llo)))


@test("fast", "autoMCStats backends vs oracle: NLL and yields (incl. floored bins); hfmodel exact for symmetric kappas")
def t_mcstats_backends():
    model, card, d = fixture_model("mcstats")
    sym = copy.deepcopy(model)
    for ch in sym.channels:  # template norm terms symmetric: hfmodel's normsys code1 is then exact
        for proc in ch.processes:
            for s in proc.shape.systs:
                ku, kd = sum(s.up) / sum(proc.shape.contents), sum(s.down) / sum(proc.shape.contents)
                s.down = list(np.asarray(s.down) / (ku * kd))
    names = [n for n in backend_names(ARGS.backend) if n != "semantic"]
    msgs = []
    for label, m in (("mcstats", model), ("mcstats-symmetric-kappas", sym)):
        oracle = SemanticLikelihood(m)
        rng = np.random.default_rng(5)
        pts = []
        for k in range(12):
            x = oracle.nominal_values().copy()
            for i, p in enumerate(oracle.parameters):
                if p.role == I.ROLE_POI:
                    x[i] = rng.uniform(0.0, 3.0)
                elif p.constraint is not None and p.constraint.kind == I.CONSTRAINT_POISSON:
                    x[i] = p.value * rng.uniform(0.3, 2.0)
                elif p.floating:
                    x[i] = rng.normal(0.0, 1.5 if k < 9 else 3.5)  # large |x| drives bins to the floor
            pts.append(x)

        def one(name):
            with chdir(d):
                lik = likelihood(name, m, card)
            notes = getattr(lik, "notes", [])
            if name == "hfmodel" and label == "mcstats":
                check(any("asymmetric lnN" in n for n in notes), "hfmodel must declare the asymPow approximation")
                return "approximate (declared asymPow note); exactness tested on the symmetric-kappa model"
            da, db = observed_dataset(m), observed_dataset(m)
            worst_nll = worst_y = 0.0
            for x in pts:
                a, b = lik.nll(x, da), oracle.nll(x, db)
                worst_nll = max(worst_nll, abs(a - b) / max(1.0, abs(b)))
                ea, eb = lik.expected_by_process(x), oracle.expected_by_process(x)
                for ch in eb:
                    for p in eb[ch]:
                        worst_y = max(worst_y, float(np.max(np.abs(ea[ch][p] - eb[ch][p])
                                                            / np.maximum(np.abs(eb[ch][p]), 1e-9))))
            check(worst_nll < 1e-10 and worst_y < 1e-10, f"{label}/{name}: NLL {worst_nll:.2e}, yields {worst_y:.2e}")
            return f"NLL {worst_nll:.0e}, yields {worst_y:.0e}"
        msgs.append(f"{label}: " + per_backend(one, names))
    return " | ".join(msgs)


@test("fast", "autoMCStats toys: prior toys sample the Poisson gammas inside their ranges; Asimov refit at nominal")
def t_mcstats_toys():
    from inference.toys import ToyConfig, generate_toys

    model, card, d = fixture_model("mcstats")
    lik = SemanticLikelihood(model)
    fitter = Fitter(lik)
    toys, _ = generate_toys(lik, fitter, ToyConfig(ntoys=20), np.random.default_rng(3))
    pois = [p for p in model.parameters.values() if p.origin == "autoMCStats" and p.constraint.kind == "poisson"]
    for t in toys:
        for p in pois:
            check(p.lo <= t.truth[p.name] <= p.hi, f"{p.name} = {t.truth[p.name]} outside [{p.lo}, {p.hi}]")
    asimov, _ = generate_toys(lik, fitter, ToyConfig(ntoys=-1, expect_signal=1.0), np.random.default_rng(3))
    res = fitter.fit(asimov[0])
    check(res.valid, f"Asimov fit failed: {res.status}")
    dev = max(abs(res.values[lik.index[n]] - model.parameters[n].value) for n in MCSTATS_COMBINE_PARAMS)
    check(abs(res.values[lik.poi_index] - 1.0) < 1e-3 and dev < 2e-2, f"Asimov fit r {res.values[lik.poi_index]}, "
          f"max |BB parameter - nominal| {dev:.3g}")
    return f"20 prior toys; Asimov r_hat {res.values[lik.poi_index]:.4f}, max BB deviation {dev:.1e}"



# ----------------------------------------------------------------------------------------
# FAST: multi-dimensional channels (obs:multidim, examples/two_dim)
# ----------------------------------------------------------------------------------------

def _two_dim_reference():
    """Independent 2D expectation of examples/two_dim: rate * outer(p_x, p_t) from the
    example's own analytic fractions, and the data histogrammed with numpy from the RooDataSet."""
    import importlib.util

    import ROOT

    d = WORK.example("two_dim")
    spec = importlib.util.spec_from_file_location("two_dim_inputs", os.path.join(d, "make_inputs.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    f = ROOT.TFile.Open(os.path.join(d, "workspace.root"))
    ds = f.Get("w").data("data_obs_unbinned")
    pts = np.array([[ds.get(i).getRealValue("obs_p"), ds.get(i).getRealValue("obs_t")] for i in range(ds.numEntries())])
    f.Close()
    n, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=[mod.P_EDGES, mod.T_EDGES])
    shapes = {p: np.outer(mod.p_fractions(p), mod.t_fractions(p)).reshape(-1) for p in mod.RATES}
    return mod, n.reshape(-1), shapes, pts


@test("fast", "2D grammar/IR: axes, row-major bins = numpy histogram2d, refusals (TH2, 2D shape systs, 1D data)")
def t_two_dim_ir():
    mod, n_ref, shapes, _ = _two_dim_reference()
    model, card, d = fixture_model("two_dim")
    ch = model.channels[0]
    check(ch.observable.ndim == 2 and [a.name for a in ch.observable.axes] == ["obs_p", "obs_t"], "axes")
    check(ch.observable.shape == (20, 9) and ch.observable.nbins == 180, f"shape {ch.observable.shape}")
    check(np.allclose(ch.observable.bin_volumes(), np.multiply.outer(np.diff(mod.P_EDGES), np.diff(mod.T_EDGES)).reshape(-1)),
          "bin volumes")
    check(np.array_equal(np.asarray(ch.data.counts), n_ref), "data_obs counts != numpy histogram2d (row-major)")
    check("obs:multidim" in model.features(), "feature obs:multidim")
    for proc in ch.processes:
        check(np.allclose(proc.shape.contents, shapes[proc.name], rtol=1e-6, atol=1e-12), f"{proc.name} fractions")
    unb, _, _ = fixture_model("two_dim_unbinned")
    check(np.asarray(unb.channels[0].data.values).shape == (int(n_ref.sum()), 2), "unbinned values (n, 2)")
    refused = []
    text_1d = open(os.path.join(d, "card.txt")).read().replace("w:data_obs\n", "w:data_obs_p\n")
    for fname, text in (("card_th2.txt", None), ("card_syst.txt", None), ("card_1d.txt", text_1d)):
        path = os.path.join(d, fname)
        if text is not None:
            with open(path, "w") as handle:
                handle.write(text)
        try:
            with chdir(d):
                build_ir(fname)
            raise AssertionError(f"{fname} must be refused")
        except UnsupportedFeature as exc:
            refused.append(f"{fname}: {exc}")
    check("TH1" in refused[0] and "text2workspace" in refused[1] and "obs_t" in refused[2], f"refusal reasons {refused}")
    refused = [r[:70] for r in refused]
    # bundle round trip
    from modelspec.bundle import load_bundle, save_bundle

    path = save_bundle(model, os.path.join(WORK.tmp("two_dim_bundle"), "b.json"))
    back = load_bundle(path)
    check(back.channels[0].observable == ch.observable and back.channels[0].data == ch.data, "bundle round trip")
    x = SemanticLikelihood(model).nominal_values()
    check(abs(SemanticLikelihood(back).nll(x, observed_dataset(back)) - SemanticLikelihood(model).nll(
        x, observed_dataset(model))) < 1e-12, "bundle NLL")
    return "; ".join(refused)


@test("fast", "2D oracle vs an independent numpy NLL; backends vs oracle (NLL, yields) at random points")
def t_two_dim_oracle_backends():
    mod, n_ref, shapes, _ = _two_dim_reference()
    model, card, d = fixture_model("two_dim")
    oracle = SemanticLikelihood(model)
    data = observed_dataset(model)
    pts = random_points(oracle, 10, np.random.default_rng(11))
    for x in pts:
        v = oracle.values_dict(x)
        nu = (mod.RATES["sig"] * v["r"] * 1.10 ** v["lumi"] * shapes["sig"]
              + mod.RATES["dio"] * 1.10 ** v["lumi"] * 1.05 ** v["dioN"] * shapes["dio"]
              + mod.RATES["cosmic"] * v["csm_scale"] * shapes["cosmic"])
        ref = nu.sum() - np.sum(n_ref * np.log(nu)) + 0.5 * (v["lumi"] ** 2 + v["dioN"] ** 2)
        check(abs(oracle.nll(x, data) - ref) < 1e-9 * max(1, abs(ref)), f"oracle {oracle.nll(x, data)} vs {ref}")

    def one(name):
        with chdir(d):
            lik = likelihood(name, model, card)
        data_b = observed_dataset(model)
        for x in pts:
            a, b = lik.nll(x, data_b), oracle.nll(x, data)
            check(abs(a - b) < 1e-9 * max(1.0, abs(b)), f"NLL {a} vs oracle {b}")
            ea, eb = lik.expected_by_process(x)["sr"], oracle.expected_by_process(x)["sr"]
            for p in eb:
                check(np.allclose(ea[p], eb[p], rtol=1e-9, atol=1e-12), f"yields {p}")
    return per_backend(one, [n for n in backend_names(ARGS.backend) if n != "semantic"])


@test("fast", "2D backends vs Combine's CachingSimNLL: NLL differences at the fixture points (binned, param, unbinned)")
def t_two_dim_combine_nll():
    msgs = []
    for fixture in ("two_dim", "two_dim_param", "two_dim_unbinned"):
        fx = load_fixture(fixture)
        model, card, d = fixture_model(fixture)

        def one(name):
            with chdir(d):
                lik = likelihood(name, model, card)
            data = observed_dataset(model)
            base, worst = None, 0.0
            for pt in fx["nll_points"]:
                x = lik.nominal_values().copy()
                for k, val in pt["params"].items():
                    check(k in lik.index, f"Combine parameter {k} is not in the model")
                    x[lik.index[k]] = val
                val = lik.nll(x, data)
                if base is None:
                    base = (val, pt["nll"])
                    continue
                dd = abs((val - base[0]) - (pt["nll"] - base[1]))
                worst = max(worst, dd)
                check(dd < 1e-7 * max(1.0, abs(pt["nll"] - base[1])), f"{fixture}: dNLL differs by {dd:.3g} at {pt}")
            return f"max {worst:.1e}"
        msgs.append(f"{fixture}: " + per_backend(one, backend_names(ARGS.backend)))
    return " | ".join(msgs)


@test("fast", "2D toys: binned toy means = expected per flattened bin, Asimov = expected; unbinned 2D sampling")
def t_two_dim_toys():
    from inference.toys import ToyConfig, generate_toys
    from scipy.stats import chi2

    def one(name, fixture):
        model, card, d = fixture_model(fixture)
        with chdir(d):
            lik = likelihood(name, model, card)
        fitter = Fitter(lik)
        cfg = ToyConfig(ntoys=150, expect_signal=1.0, frequentist=True, bypass_fit=True)
        toys, _ = generate_toys(lik, fitter, cfg, np.random.default_rng(7))
        x = np.array([toys[0].truth[k] for k in lik.names])
        mu = lik.expected_counts(x)["sr"]
        asimov, _ = generate_toys(lik, fitter, ToyConfig(ntoys=-1, expect_signal=1.0), np.random.default_rng(7))
        ch = model.channels[0]
        if ch.data.kind == "binned":
            c = np.array([t.main["sr"].counts for t in toys])
            check(c.shape == (150, 180), f"toy shape {c.shape}")
            n = c.sum(axis=0)
            check(np.allclose(asimov[0].main["sr"].counts, mu), "Asimov != expected")
        else:
            vals = [np.asarray(t.main["sr"].values) for t in toys]
            check(all(v.ndim == 2 and v.shape[1] == 2 for v in vals), "unbinned toys must be (n, 2)")
            edges = [np.asarray(a.edges) for a in ch.observable.axes]
            n = np.histogramdd(np.concatenate(vals), bins=edges)[0].reshape(-1)
            a = asimov[0].main["sr"]
            check(np.allclose(a.values, ch.observable.bin_centers()) and np.allclose(a.weights, mu), "unbinned Asimov")
        # Pearson chi2 of the summed toys against 150 * expected (bins with enough expectation)
        e = 150 * mu
        keep = e > 5
        stat = float(np.sum((n[keep] - e[keep]) ** 2 / e[keep]))
        p = chi2.sf(stat, keep.sum())
        check(p > 1e-4, f"toys vs expected: chi2 {stat:.1f} / {keep.sum()} bins (p = {p:.2g})")
        return f"{fixture} chi2/ndf {stat:.0f}/{keep.sum()}"

    msgs = []
    for fixture in ("two_dim", "two_dim_unbinned"):
        msgs.append(per_backend(one, backend_names(ARGS.backend), fixture))
    return " | ".join(msgs)


# ----------------------------------------------------------------------------------------
# SLOW
# ----------------------------------------------------------------------------------------

def _pvalue_compare(label, mine, mine_err, ref, ref_err, nsig=3.0):
    tot = math.hypot(mine_err, ref_err)
    ok = abs(mine - ref) <= nsig * max(tot, 1e-3)
    return ok, f"{label}: {mine:.4f}+-{mine_err:.4f} vs {ref:.4f}+-{ref_err:.4f}"


@test("slow", "toy CLs (LHC-limits) vs Combine HybridNew points (counting, low_background n0/n1)")
def t_toy_cls_vs_hybrid():
    from inference.hybrid import toy_cls_limit

    def one(name):
        bad, n_cmp, skipped = [], 0, []
        for fixture in ("counting", "low_background_n0", "low_background_n1"):
            fx = load_fixture(fixture)
            model, card, d = fixture_model(fixture)
            try:
                with chdir(d):
                    lik = likelihood(name, model, card)
            except Skip as exc:
                skipped.append(f"{fixture}: {exc}")
                continue
            pts = fx["hybrid_cls"]["points"]
            res = toy_cls_limit(lik, Fitter(lik), np.random.default_rng(99), [p["r"] for p in pts],
                                ntoys=fx["hybrid_cls"]["toys"], expected=False)
            mine = {p.r: p.to_dict() for p in res.points}
            for p in pts:
                m = mine[p["r"]]
                for key in ("CLs", "CLb", "CLsplusb"):
                    if p[f"{key}_err"] < 0 or p[key] < 0:
                        continue
                    ok, txt = _pvalue_compare(f"{fixture} r={p['r']:g} {key}", m[key], m[f"{key}_err"], p[key],
                                              p[f"{key}_err"])
                    n_cmp += 1
                    if not ok:
                        bad.append(txt)
        # with ~50 comparisons at 3 sigma a single outlier is possible; more is a failure
        if n_cmp == 0:
            raise Skip("; ".join(skipped))
        check(len(bad) <= 1, f"{name}: {len(bad)}/{n_cmp} p-values differ by > 3 sigma: " + "; ".join(bad))
        return f"{n_cmp} p-values within 3 sigma" + (f", {len(bad)} outlier" if bad else "") + \
            (f"; skipped {'; '.join(skipped)}" if skipped else "")
    return per_backend(one, backend_names(ARGS.backend))


def _stat_only_card(dirname, s, b, n):
    return write_card(dirname, f"card_s{s:g}_b{b:g}_n{n}.txt", f"""
imax 1
jmax 1
kmax 0
bin sr
observation {n}
bin sr sr
process sig bkg
process 0 1
rate {s} {b}
""")


@test("slow", "toy CLs vs exact Poisson CLs (stat-only counting, b=0.2 n=0 and b=3 n=2)")
def t_toy_cls_exact():
    from inference.hybrid import toy_cls_limit

    def one(name):
        bad = []
        for b, n, grid in ((0.2, 0, [1.0, 2.0, 3.0, 4.0]), (3.0, 2, [2.0, 4.0, 6.0, 8.0])):
            card = _stat_only_card("exact_cls", 1.0, b, n)
            model = build_ir(card)
            lik = likelihood(name, model, card)
            res = toy_cls_limit(lik, Fitter(lik), np.random.default_rng(1), grid, ntoys=2000, expected=False)
            for p in res.points:
                dct = p.to_dict()
                exact = REF.poisson_cls(p.r, b, n)
                ok, txt = _pvalue_compare(f"b={b} n={n} r={p.r:g} CLs", dct["CLs"], dct["CLs_err"], exact, 0.0)
                if not ok:
                    bad.append(txt)
            exact_lim = REF.poisson_cls_limit(1.0, b, n)
            if res.observed is not None and abs(res.observed - exact_lim) > 3 * (res.observed_err or 0) + 0.05 * exact_lim:
                bad.append(f"b={b} n={n}: toy limit {res.observed:.3f}+-{res.observed_err:.3f} vs exact {exact_lim:.3f}")
        check(not bad, f"{name}: " + "; ".join(bad))
    return per_backend(one, backend_names(ARGS.backend))


@test("slow", "reference FC98 n=0 with the b-monotone compensation reproduces 1.08")
def t_reference_fc98_n0():
    up = REF.fc_upper_b_monotone(0, 3.0, 0.90, b_max=6.0, b_step=0.002, mu_window=(0.9, 1.2), step=0.0025)
    check(abs(up - 1.08) < 0.01, f"compensated upper limit {up:.4f} vs FC98 1.08")
    return f"compensated n=0 upper = {up:.4f}"


def _fc_exact_pvalue(n_obs, b, mu):
    """p(mu) = P(t >= t_obs) for the FC likelihood-ratio ordering (exact enumeration)."""
    from scipy.stats import poisson

    nmax = int(mu + b + 30 + 10 * math.sqrt(mu + b + 1))
    n = np.arange(nmax + 1)
    rank = poisson.pmf(n, mu + b) / poisson.pmf(n, np.maximum(0.0, n - b) + b)
    r_obs = poisson.pmf(n_obs, mu + b) / poisson.pmf(n_obs, max(0.0, n_obs - b) + b)
    return float(np.sum(poisson.pmf(n, mu + b)[rank <= r_obs * (1 + 1e-12)]))


@test("slow", "FC toys vs exact enumeration and the FC98 table (stat-only, b=3, n=0,3,6,10)")
def t_fc_fc98():
    from inference.hybrid import feldman_cousins

    def one(name):
        bad = []
        for n in (0, 3, 6, 10):
            lo98, hi98 = REF.FC98_B3_CL90[n]
            card = _stat_only_card("fc98", 1.0, 3.0, n)
            model = build_ir(card)
            lik = likelihood(name, model, card)
            grid = sorted({max(0.0, lo98 - 0.3), max(0.0, lo98 + 0.3), hi98 - 0.3, hi98 + 0.3, 0.5 * (lo98 + hi98)})
            res = feldman_cousins(lik, Fitter(lik), np.random.default_rng(n), grid, ntoys=1500, cl=0.90)
            for p in res.points:
                d = p.to_dict()
                exact = _fc_exact_pvalue(n, 3.0, p.r)
                ok, txt = _pvalue_compare(f"n={n} r={p.r:g} p", d["p"], d["p_err"], exact, 0.0)
                if not ok:
                    bad.append(txt)
                inside98 = lo98 - 1e-9 <= p.r <= hi98 + 1e-9
                # the FC98 table is slightly conservative (b-monotone fix), so only check clear cases
                if (d["p"] > 0.10 + 3 * d["p_err"]) and not inside98:
                    bad.append(f"n={n} r={p.r:g} accepted (p={d['p']:.3f}) but outside FC98 [{lo98}, {hi98}]")
                if (d["p"] < 0.10 - 3 * d["p_err"]) and inside98 and not (n == 0 and p.r > 0.95):
                    bad.append(f"n={n} r={p.r:g} rejected (p={d['p']:.3f}) but inside FC98 [{lo98}, {hi98}]")
        check(not bad, f"{name}: " + "; ".join(bad))
    return per_backend(one, backend_names(ARGS.backend))


@test("slow", "FC toys vs Combine LHC-feldman-cousins points (counting, low_background n0/n1)")
def t_fc_vs_combine():
    from inference.hybrid import feldman_cousins

    def one(name):
        bad, n_cmp, skipped = [], 0, []
        for fixture in ("counting", "low_background_n0", "low_background_n1"):
            fx = load_fixture(fixture)
            model, card, d = fixture_model(fixture)
            try:
                with chdir(d):
                    lik = likelihood(name, model, card)
            except Skip as exc:
                skipped.append(f"{fixture}: {exc}")
                continue
            pts = fx["hybrid_fc"]["points"]
            res = feldman_cousins(lik, Fitter(lik), np.random.default_rng(42), [p["r"] for p in pts],
                                  ntoys=fx["hybrid_fc"]["toys"], cl=0.90)
            mine = {p.r: p.to_dict() for p in res.points}
            for p in pts:
                m = mine[p["r"]]
                ok, txt = _pvalue_compare(f"{fixture} r={p['r']:g} p", m["p"], m["p_err"], p["CLsplusb"],
                                          p["CLsplusb_err"])
                n_cmp += 1
                if not ok:
                    bad.append(txt)
        if n_cmp == 0:
            raise Skip("; ".join(skipped))
        check(len(bad) <= 1, f"{name}: {len(bad)}/{n_cmp} FC p-values differ by > 3 sigma: " + "; ".join(bad))
        return f"{n_cmp} FC p-values within 3 sigma" + (f", {len(bad)} outlier" if bad else "") + \
            (f"; skipped {'; '.join(skipped)}" if skipped else "")
    return per_backend(one, backend_names(ARGS.backend))


@test("slow", "pull test: r pulls of frequentist toys at r=1 (s=50, b=80, lnN) have mean 0 and width 1")
def t_pulls():
    from inference.toys import ToyConfig, generate_toys

    card = write_card("pulls", "card.txt", """
imax 1
jmax 1
kmax *
bin sr
observation 130
bin sr sr
process sig bkg
process 0 1
rate 50 80
lumi lnN 1.025 1.025
bkg_norm lnN - 1.05
""")

    def one(name):
        model = build_ir(card)
        lik = likelihood(name, model, card)
        fitter = Fitter(lik)
        fitter.set_range("r", -5.0, 20.0)  # no boundary for the pull test
        toys, _ = generate_toys(lik, fitter, ToyConfig(ntoys=400, expect_signal=1.0, frequentist=True),
                                np.random.default_rng(8))
        pulls = []
        for t in toys:
            res = fitter.fit(t, hesse=True)
            if res.valid and "r" in res.errors:
                pulls.append((res.value(lik, "r") - 1.0) / res.errors["r"])
        pulls = np.array(pulls)
        n = len(pulls)
        check(n >= 390, f"only {n} valid toy fits")
        check(abs(pulls.mean()) < 3.5 / math.sqrt(n), f"pull mean {pulls.mean():.3f} (n={n})")
        check(abs(pulls.std(ddof=1) - 1.0) < 3.5 / math.sqrt(2 * n), f"pull width {pulls.std(ddof=1):.3f} (n={n})")
        return f"mean {pulls.mean():.3f} width {pulls.std(ddof=1):.3f}"
    return per_backend(one, backend_names(ARGS.backend))


@test("slow", "FC coverage (stat-only, s=1, b=3): 200 pseudo-experiments at r_true = 0.5 and 2")
def t_fc_coverage():
    from inference.hybrid import feldman_cousins
    from scipy.stats import poisson

    def one(name):
        rng = np.random.default_rng(2026)
        cache = {}
        report = []
        for r_true in (0.5, 2.0):
            ns = rng.poisson(r_true + 3.0, 200)
            for n in sorted(set(ns.tolist())):
                if n in cache:
                    continue
                card = _stat_only_card("fc_cov", 1.0, 3.0, n)
                model = build_ir(card)
                lik = likelihood(name, model, card)
                lo98, hi98 = REF.fc_interval(n, 3.0, 0.90, mu_max=n + 15.0, step=0.01)
                grid = sorted({round(x, 3) for x in (0.0, lo98 - 0.25, lo98 + 0.25, hi98 - 0.25, hi98 + 0.25,
                                                      0.5 * (lo98 + hi98)) if x >= 0})
                res = feldman_cousins(lik, Fitter(lik), np.random.default_rng(100 + n), grid, ntoys=400, cl=0.90,
                                      refine=2)
                cache[n] = (res.lower if res.lower is not None else 0.0,
                            res.upper if res.upper is not None else math.inf)
            covered = np.mean([cache[n][0] <= r_true <= cache[n][1] for n in ns])
            err = math.sqrt(0.9 * 0.1 / len(ns))
            exact = sum(poisson.pmf(n, r_true + 3.0) for n, (lo, hi) in cache.items() if lo <= r_true <= hi)
            report.append(f"r={r_true}: {covered:.3f} +- {err:.3f} (exact over computed n: {exact:.3f})")
            check(covered >= 0.90 - 2.5 * err, f"coverage {covered:.3f} at r_true={r_true} below 0.90 - 2.5 sigma")
        return "; ".join(report)
    return per_backend(one, backend_names(ARGS.backend))


# ----------------------------------------------------------------------------------------

def main():
    global WORK, ARGS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--fast", action="store_true", help="fast tier (default)")
    g.add_argument("--slow", action="store_true", help="slow tier")
    g.add_argument("--all", action="store_true", help="both tiers")
    ap.add_argument("--backend", default=None, help="only this backend (semantic, " + ", ".join(BACKEND_NAMES) + ")")
    ap.add_argument("--only", default=None, help="run tests whose name contains this text")
    ap.add_argument("--workdir", default=None, help="scratch directory (default: a new temporary directory)")
    ARGS = ap.parse_args()
    tiers = {"fast", "slow"} if ARGS.all else ({"slow"} if ARGS.slow else {"fast"})
    root = ARGS.workdir or tempfile.mkdtemp(prefix="pymodel_tests_")
    root = os.path.abspath(root)
    if root == REPO or root.startswith(REPO + os.sep) or root.startswith(os.path.abspath(MUMEP_CARDS)):
        raise SystemExit("the work directory must not be inside the repository or mumep_ana")
    WORK = Work(root)
    print(f"work directory: {root}")
    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    t_all = time.time()
    for tier, name, fn in TESTS:
        if tier not in tiers or (ARGS.only and ARGS.only not in name):
            continue
        t0 = time.time()
        try:
            msg = fn()
            status = "PASS"
        except Skip as exc:
            status, msg = "SKIP", str(exc)
        except AssertionError as exc:
            status, msg = "FAIL", str(exc)
        except Exception as exc:  # an error in a test is a failure, with its traceback
            status, msg = "FAIL", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        counts[status] += 1
        print(f"{status:4s} [{tier}] {name} ({time.time() - t0:.1f} s)" + (f"\n       {msg}" if msg else ""), flush=True)
    print(f"\n{counts['PASS']} passed, {counts['FAIL']} failed, {counts['SKIP']} skipped "
          f"in {time.time() - t_all:.0f} s")
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
