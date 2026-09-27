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


class SemanticBackend(Backend):
    """The numpy oracle wrapped as a backend (not a user-facing backend)."""

    name = "semantic"
    supported_features = frozenset(SemanticLikelihood.supported | {"shape:parametric"})

    def runtime_versions(self):
        return [("numpy", np.__version__)]

    def create(self, model, options):
        try:
            return SemanticLikelihood(model)
        except NotImplementedError as exc:
            raise UnsupportedByBackend(f"semantic oracle: {exc}") from exc


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
}
NEEDS_COMBINE_LIB = {"envelope", "mumep_40_env"}

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
        oracle = SemanticLikelihood(model)
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
