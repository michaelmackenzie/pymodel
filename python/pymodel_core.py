"""pymodel command line: ``pymodel <backend> <command> INPUT [options]``.

INPUT is a Combine datacard or a model bundle written by ``build``.  The commands mirror
Combine's methods:

  build         datacard -> model bundle (JSON, plus a ROOT file for RooFit objects)
  inspect       print the model: channels, processes, parameters, notes
  nll           evaluate the NLL at given parameter values (validation)
  fit           maximum-likelihood fit to data or toys (FitDiagnostics-like)
  scan          1D or 2D profile-likelihood scan (MultiDimFit --algo grid, one or two -P)
  impacts       nuisance-parameter impacts, pulls and constraints (combineTool.py -M Impacts)
  limit         CLs upper limit: --method asymptotic (AsymptoticLimits) or toys (HybridNew LHC-limits)
  fc            Feldman-Cousins interval (HybridNew LHC-feldman-cousins)
  significance  discovery significance (Significance; asymptotic or toys)
  generate      generate and save toy datasets (GenerateOnly)
  merge         merge raw toy results of split limit/fc jobs and compute the result
                (hadd + HybridNew --readHybridResults)
  export        write the model in the backend's native format

Toy commands take --jobs N (worker processes; results are identical for any N).

Every command writes a JSON result file (``--output``); its ``flags`` list names anything
that makes the result suspect.
"""

import argparse
import copy
import json
import math
import os
import sys

import numpy as np

COMMANDS = ("build", "inspect", "nll", "fit", "scan", "impacts", "limit", "fc", "significance", "generate", "merge",
            "export")


# ----------------------------------------------------------------------------------------
# argument parsing
# ----------------------------------------------------------------------------------------

def _kv_list(text, cast=float):
    out = {}
    if not text:
        return out
    for item in text.split(","):
        key, val = item.split("=", 1)
        out[key.strip()] = cast(val)
    return out


def _range_list(text):
    out = {}
    if not text:
        return out
    for item in text.split(","):
        key, val = item.split("=", 1)
        lo, hi = val.split(":")
        out[key.strip()] = (float(lo), float(hi))
    return out


def _grid(text):
    """'lo:hi:n' (n points including both ends) or a comma-separated list."""
    if ":" in text:
        lo, hi, n = text.split(":")
        return list(np.linspace(float(lo), float(hi), int(n)))
    return [float(x) for x in text.split(",")]


def _add_common(p, backend):
    g = p.add_argument_group("model")
    g.add_argument("input", help="Combine datacard, or a model bundle (.json) written by 'build'")
    g.add_argument("--mass", default="120", help="value substituted for $MASS in shapes lines")
    g.add_argument("--rmin", type=float, default=0.0, help="POI lower bound (default 0)")
    g.add_argument("--rmax", type=float, default=20.0, help="POI upper bound (default 20, as in Combine)")
    g.add_argument("--bin-integration", choices=("center", "integral"), default="center",
                   help="how parametric pdfs are evaluated on binned data: bin centre x width (Combine/RooFit, "
                        "default) or exact bin integrals")
    g.add_argument("--set-parameters", default="", help="name=value,... (initial/fixed values)")
    g.add_argument("--freeze-parameters", default="", help="name,... parameters to fix at their values")
    g.add_argument("--freeze-nuisance-groups", default="", help="group,... datacard groups to freeze")
    g.add_argument("--set-parameter-ranges", default="", help="name=lo:hi,...")
    g = p.add_argument_group("fit / output")
    g.add_argument("--seed", type=int, default=123456, help="random seed (default 123456, as in Combine)")
    g.add_argument("--strategy", type=int, default=1, choices=(0, 1, 2), help="Minuit strategy")
    g.add_argument("--tolerance", type=float, default=0.01, help="Minuit tolerance (EDM goal 0.002*tol*0.5)")
    g.add_argument("--output", "-o", default=None, help="result JSON (default pymodel_<command>.json)")
    g.add_argument("--plot", action="store_true", help="write plots for this command")
    g.add_argument("--plot-dir", default="plots")
    g.add_argument("--verbose", "-v", action="count", default=0)
    backend.add_arguments(p)


def _add_toy_options(p, allow_asimov=True):
    g = p.add_argument_group("toys (Combine semantics)")
    g.add_argument("--toys", "-t", type=int, default=0,
                   help="number of toys" + (" (-1: Asimov dataset)" if allow_asimov else ""))
    g.add_argument("--expect-signal", type=float, default=0.0, help="r used to generate toys (default 0)")
    g.add_argument("--toys-frequentist", action="store_true",
                   help="fit nuisances to data and randomise global observables (Combine --toysFrequentist)")
    g.add_argument("--bypass-frequentist-fit", action="store_true",
                   help="frequentist toys around the pre-fit nuisance values")
    g.add_argument("--toys-no-systematics", action="store_true", help="do not randomise nuisances")
    g.add_argument("--toys-file", default=None, help="read datasets saved by 'generate' instead of generating")


def _add_jobs(p):
    p.add_argument("--jobs", "-j", type=int, default=1,
                   help="worker processes for toys / fits (spawned, each rebuilds the model; results do not "
                        "depend on N)")


def _add_split_options(p, what):
    g = p.add_argument_group("job splitting (merge the saved files with the 'merge' command)")
    g.add_argument("--points", default=None,
                   help=f"compute only these r points (comma-separated), no refinement: one job of a {what} grid")
    g.add_argument("--toy-chunk", default=None, metavar="I/N",
                   help="run only chunk I (0-based) of N of the --toys-per-point toys at every point (fixed toys, "
                        "no refinement)")
    g.add_argument("--save-toy-results", default=None, metavar="FILE",
                   help="write the raw per-point test-statistic arrays (JSON) for 'merge'")


def build_parser(backend):
    parser = argparse.ArgumentParser(prog=f"pymodel {backend.name}", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build", help="datacard -> model bundle")
    _add_common(p, backend)
    p.add_argument("--bundle", default="model.json", help="output bundle path (default model.json)")

    p = sub.add_parser("inspect", help="print the model")
    _add_common(p, backend)

    p = sub.add_parser("nll", help="evaluate the NLL")
    _add_common(p, backend)
    p.add_argument("--at", default="", help="name=value,... parameter point (default: nominal)")
    _add_toy_options(p)

    p = sub.add_parser("fit", help="maximum-likelihood fit")
    _add_common(p, backend)
    _add_toy_options(p)
    p.add_argument("--minos", default="", help="comma-separated parameters for MINOS errors ('all' for all)")
    p.add_argument("--fix-r", type=float, default=None, help="fit with r fixed to this value")
    _add_jobs(p)

    p = sub.add_parser("scan", help="1D or 2D profile-likelihood scan")
    _add_common(p, backend)
    p.add_argument("--param", default="r", help="parameter to scan, or 'a,b' for a 2D scan")
    p.add_argument("--points", type=int, default=50,
                   help="grid points (2D: ceil(sqrt(N)) per axis, as Combine)")
    p.add_argument("--range", default=None,
                   help="lo:hi, or lo:hi,lo:hi for a 2D scan (default: the parameter ranges)")
    _add_toy_options(p)

    p = sub.add_parser("impacts", help="nuisance-parameter impacts on r")
    _add_common(p, backend)
    p.add_argument("--errors", choices=("profile", "hesse"), default="profile",
                   help="post-fit +-1 sigma of each nuisance: profile-likelihood crossings (Combine --robustFit 1, "
                        "default) or the Hesse errors of the initial fit")
    p.add_argument("--params", default="", help="comma-separated nuisances (default: every floating nuisance)")
    p.add_argument("--exclude", default="", help="comma-separated nuisances to leave out")
    p.add_argument("--plot-max", type=int, default=30, help="number of parameters in the impact plot")
    _add_jobs(p)
    _add_toy_options(p)

    p = sub.add_parser("limit", help="CLs upper limit")
    _add_common(p, backend)
    p.add_argument("--method", choices=("asymptotic", "toys"), default="asymptotic")
    p.add_argument("--cl", type=float, default=0.95)
    p.add_argument("--run", choices=("both", "observed", "expected", "blind"), default="both",
                   help="blind: expected only, from a pre-fit Asimov dataset")
    p.add_argument("--grid", default=None, help="toys: r grid 'lo:hi:n' or list (default: around the asymptotic limit)")
    p.add_argument("--toys-per-point", type=int, default=500, help="toys: initial s+b and b-only toys per point")
    p.add_argument("--refine", type=int, default=3, help="toys: (max) bisection points added around the crossing")
    p.add_argument("--bypass-frequentist-fit", action="store_true")
    p.add_argument("--toys-file", default=None, help="use a saved dataset as the observed data")
    p.add_argument("--toy-index", type=int, default=0)
    g = p.add_argument_group("adaptive toys (HybridNew --clsAcc / --rAbsAcc / --rRelAcc)")
    g.add_argument("--cls-acc", type=float, default=None,
                   help="add toys at each point until the CLs error is below this (or CLs is > 3 sigma from 1-CL)")
    g.add_argument("--r-abs-acc", type=float, default=None, help="refine until the limit error is below this")
    g.add_argument("--r-rel-acc", type=float, default=None,
                   help="refine until the limit error is below this fraction of the limit")
    g.add_argument("--max-toys-per-point", type=int, default=None,
                   help="cap on toys per point for the adaptive options (default 20 x --toys-per-point)")
    _add_split_options(p, "limit")
    _add_jobs(p)

    p = sub.add_parser("fc", help="Feldman-Cousins interval")
    _add_common(p, backend)
    p.add_argument("--cl", type=float, default=0.90)
    p.add_argument("--grid", default=None, help="r grid 'lo:hi:n' or list (default: from a likelihood scan)")
    p.add_argument("--toys-per-point", type=int, default=500, help="initial toys per point")
    p.add_argument("--refine", type=int, default=3, help="bisection points added around each edge")
    p.add_argument("--bypass-frequentist-fit", action="store_true")
    p.add_argument("--toys-file", default=None)
    p.add_argument("--toy-index", type=int, default=0)
    g = p.add_argument_group("adaptive toys")
    g.add_argument("--p-acc", type=float, default=None,
                   help="add toys at each point until the p-value error is below this (or p is > 3 sigma from 1-CL)")
    g.add_argument("--max-toys-per-point", type=int, default=None,
                   help="cap on toys per point with --p-acc (default 20 x --toys-per-point)")
    _add_split_options(p, "Feldman-Cousins")
    _add_jobs(p)

    p = sub.add_parser("significance", help="discovery significance")
    _add_common(p, backend)
    p.add_argument("--method", choices=("asymptotic", "toys"), default="asymptotic")
    p.add_argument("--toys-per-point", type=int, default=1000)
    p.add_argument("--bypass-frequentist-fit", action="store_true")
    p.add_argument("--toys-file", default=None)
    p.add_argument("--toy-index", type=int, default=0)
    _add_jobs(p)

    p = sub.add_parser("generate", help="generate and save toys")
    _add_common(p, backend)
    _add_toy_options(p)
    p.add_argument("--toys-out", default="toys.json")
    _add_jobs(p)

    p = sub.add_parser("merge", help="merge raw toy results (limit/fc --save-toy-results) and compute the result")
    p.add_argument("files", nargs="+", help="files written with --save-toy-results")
    p.add_argument("--cl", type=float, default=None, help="confidence level (default: the one the files were made with)")
    p.add_argument("--save-toy-results", default=None, metavar="FILE", help="also write the merged raw toy results")
    p.add_argument("--output", "-o", default=None, help="result JSON (default pymodel_merge.json)")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--plot-dir", default="plots")

    p = sub.add_parser("export", help="native export")
    _add_common(p, backend)
    p.add_argument("--native-out", required=True)
    return parser


# ----------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------

class Session:
    def __init__(self, backend, args):
        from inference.fitting import FitSettings, Fitter
        from modelspec import ir as I
        from modelspec.bundle import load_model

        self.backend, self.args = backend, args
        self.flags = []
        model = load_model(args.input, mass=args.mass, poi_range=(args.rmin, args.rmax),
                           bin_integration=args.bin_integration)
        model = copy.deepcopy(model)
        if args.input.endswith(".json"):
            model.parameters[model.poi].lo, model.parameters[model.poi].hi = args.rmin, args.rmax
        for name, val in _kv_list(args.set_parameters).items():
            self._param(model, name).value = val
        for name, (lo, hi) in _range_list(args.set_parameter_ranges).items():
            par = self._param(model, name)
            par.lo, par.hi = lo, hi
        frozen = [n for n in args.freeze_parameters.split(",") if n]
        for grp in [g for g in args.freeze_nuisance_groups.split(",") if g]:
            if grp not in model.groups:
                raise SystemExit(f"unknown nuisance group '{grp}' (groups: {', '.join(model.groups) or 'none'})")
            frozen += model.groups[grp]
        for name in frozen:
            self._param(model, name).role = I.ROLE_CONSTANT
        self.model = model
        self.lik = backend.build_likelihood(model, args)
        for note in list(model.notes) + list(getattr(self.lik, "notes", [])):
            print(f"note: {note}")
        from inference.parallel import ToySeeds

        self.rng = np.random.default_rng(args.seed)
        self.seeds = ToySeeds(args.seed)
        self.fitter = Fitter(self.lik, FitSettings(strategy=args.strategy, tolerance=args.tolerance),
                             rng=np.random.default_rng(args.seed + 1))

    @staticmethod
    def _param(model, name):
        if name not in model.parameters:
            raise SystemExit(f"unknown parameter '{name}'")
        return model.parameters[name]

    def observed(self):
        from inference.model import Dataset, observed_dataset

        path = getattr(self.args, "toys_file", None)
        if path and self.args.command in ("limit", "fc", "significance"):
            with open(path, "r", encoding="utf-8") as handle:
                doc = json.load(handle)
            return Dataset.from_dict(doc["datasets"][self.args.toy_index])
        return observed_dataset(self.model)

    def datasets(self):
        """Datasets for fit/scan/nll/generate: observed data, saved toys, or new toys."""
        from inference.model import Dataset
        from inference.toys import ToyConfig, generate_toys

        a = self.args
        if a.toys_file:
            with open(a.toys_file, "r", encoding="utf-8") as handle:
                doc = json.load(handle)
            return [Dataset.from_dict(d) for d in doc["datasets"]], {"mode": f"read from {a.toys_file}"}
        if a.toys == 0:
            return [self.observed()], {"mode": "observed data"}
        cfg = ToyConfig(ntoys=a.toys, expect_signal=a.expect_signal, frequentist=a.toys_frequentist,
                        bypass_fit=a.bypass_frequentist_fit, no_systematics=a.toys_no_systematics)
        return generate_toys(self.lik, self.fitter, cfg, self.seeds, observed=self.observed())

    def toy_config(self):
        from inference.toys import ToyConfig

        a = self.args
        return ToyConfig(ntoys=a.toys, expect_signal=a.expect_signal, frequentist=a.toys_frequentist,
                         bypass_fit=a.bypass_frequentist_fit, no_systematics=a.toys_no_systematics)

    def executor(self):
        """Serial executor, or a spawn pool whose workers rebuild this session (--jobs)."""
        from inference.parallel import Executor

        jobs = getattr(self.args, "jobs", 1)
        return Executor(self.lik, self.fitter, jobs=jobs,
                        factory=SessionFactory(self.backend, self.args) if jobs > 1 else None)

    def write(self, result, flags=()):
        from inference.results import write_result

        path = self.args.output or f"pymodel_{self.args.command}.json"
        write_result(path, command=self.args.command, backend=self.backend.name,
                     versions=dict(self.backend.runtime_versions()), seed=self.args.seed,
                     input_path=self.args.input,
                     model_notes=list(self.model.notes) + [f"{self.backend.name}: {n}" for n in
                                                           getattr(self.lik, "notes", [])],
                     flags=list(self.flags) + list(flags), result=result)
        allflags = list(dict.fromkeys(list(self.flags) + list(flags)))
        if allflags:
            print("\nFLAGS (check before using the result):")
            for f in allflags:
                print(f"  - {f}")
        print(f"\nWrote {path}")


class SessionFactory:
    """Picklable recipe that rebuilds a Session's likelihood and fitter in a worker process.
    Registered backends are re-imported by name; other backends (e.g. the tests' semantic
    oracle) are pickled, so their class must be importable."""

    def __init__(self, backend, args):
        from stat_backends import BACKEND_NAMES, get_backend

        registered = backend.name in BACKEND_NAMES and type(get_backend(backend.name)) is type(backend)
        self.backend_name = backend.name if registered else None
        self.backend = None if registered else backend
        self.args = copy.copy(args)
        self.args.jobs = 1

    def __call__(self):
        from inference.parallel import WorkContext
        from stat_backends import get_backend

        backend = get_backend(self.backend_name) if self.backend_name else self.backend
        s = Session(backend, self.args)
        return WorkContext(lik=s.lik, fitter=s.fitter)


def _fmt(x, digits=4):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}g}"


# ----------------------------------------------------------------------------------------
# commands
# ----------------------------------------------------------------------------------------

def cmd_build(backend, args):
    from modelspec.bundle import save_bundle

    s = Session(backend, args)
    path = save_bundle(s.model, args.bundle)
    print(f"Model built and checked with {backend.name}; bundle written to {path}")


def cmd_inspect(backend, args):
    s = Session(backend, args)
    m = s.model
    print(f"Model from {m.source}")
    for ch in m.channels:
        obs = ch.observable
        axes = ", ".join(f"{a.name} [{a.lo:g}, {a.hi:g}] ({len(a.edges) - 1} bins)" for a in obs.axis_list())
        print(f"channel {ch.name}: data {ch.data.kind} (total {ch.data.total:g}), observable {axes}"
              + (f": {obs.nbins} flattened bins" if obs.ndim > 1 else ""))
        if ch.mcstats is not None:
            mc = ch.mcstats
            kinds = [bp.kind for bp in mc.params]
            print(f"   autoMCStats threshold {mc.threshold:g}, include-signal {int(mc.include_signal)}, hist-mode "
                  f"{mc.hist_mode}: " + ", ".join(f"{kinds.count(k)} {k}" for k in ("total", "poisson", "gauss")))
        exp = s.lik.expected_by_process(s.lik.nominal_values())[ch.name]
        for proc in ch.processes:
            terms = ", ".join(f"{t.kind}({t.param or t.formula})" for t in proc.norm_terms) or "-"
            print(f"   {'S' if proc.is_signal else 'B'} {proc.name:<16} rate {proc.rate:<10.5g} "
                  f"expected {float(np.sum(exp[proc.name])):<10.5g} shape {proc.shape.kind:<10} norm: {terms}")
    print("parameters:")
    for p in m.parameters.values():
        c = p.constraint
        ctext = "" if c is None else f" constraint {c.kind}(center {c.center:g}, sigma {c.sigma_lo:g}/{c.sigma_hi:g})"
        print(f"   {p.name:<32} {p.role:<9} value {p.value:<10.5g} range [{p.lo:g}, {p.hi:g}]{ctext}")
    if m.groups:
        print("groups:", ", ".join(f"{g}({len(v)})" for g, v in m.groups.items()))
    s.write({"features": sorted(m.features()), "parameters": list(m.parameters)})


def cmd_nll(backend, args):
    from inference.model import constraint_nll

    s = Session(backend, args)
    x = s.lik.nominal_values()
    for name, val in _kv_list(args.at).items():
        x[s.lik.index[s._param(s.model, name).name]] = val
    datasets, info = s.datasets()
    out = []
    for d in datasets:
        main = s.lik.nll_main(x, s.lik.native(d))
        cons = constraint_nll(s.model, s.lik.index, x, d.global_obs)
        out.append({"dataset": d.label, "nll_main": main, "nll_constraints": cons, "nll": main + cons})
        print(f"{d.label}: NLL = {main + cons:.10g} (main {main:.10g}, constraints {cons:.10g})")
    s.write({"point": s.lik.values_dict(x), "datasets": out, "data_mode": info})


def _fit_datasets(s, args, fixed, minos):
    """(dataset, FitResult) pairs.  Generated toys (-t N) are made and fitted through the
    executor (toy i from stream STREAM_GEN, i: identical for any --jobs)."""
    from inference.toys import toy_base

    a = args
    if a.toys_file or a.toys <= 0:
        datasets, info = s.datasets()
        return [(d, s.fitter.fit(d, fixed=fixed, hesse=True, minos=minos)) for d in datasets], info
    cfg = s.toy_config()
    base, info = toy_base(s.lik, s.fitter, cfg, s.observed())
    return _toy_tasks(s, cfg, base, fit=True, fixed=fixed, minos=minos), info


def _toy_tasks(s, cfg, base, fit, fixed=None, minos=()):
    from inference.toys import toy_fit_task

    n = cfg.ntoys
    with s.executor() as ex:
        size = ex.chunk_size(n)
        tasks = [{"seed": s.args.seed, "cfg": cfg, "base": np.asarray(base).tolist(), "start": i,
                  "stop": min(i + size, n), "fit": fit, "fixed": fixed, "minos": list(minos)}
                 for i in range(0, n, size)]
        return [pair for chunk in ex.map(toy_fit_task, tasks) for pair in chunk]


def cmd_fit(backend, args):
    from inference.scan import fit_summary

    s = Session(backend, args)
    minos = [p.name for p in s.lik.parameters if p.floating] if args.minos == "all" else \
        [n for n in args.minos.split(",") if n]
    fixed = {s.lik.poi: args.fix_r} if args.fix_r is not None else None
    fits, info = _fit_datasets(s, args, fixed, minos)
    datasets = [d for d, _ in fits]
    result = {"data_mode": info, "fits": []}
    flags = []
    for d, res in fits:
        summ = fit_summary(s.lik, res, d)
        summ["dataset"] = d.label
        if d.truth is not None:
            summ["truth"] = d.truth
        result["fits"].append(summ)
        if not res.valid:
            flags.append(f"fit to {d.label} is not valid: {res.status}")
        if len(datasets) == 1 and res.at_limit:
            flags.append(f"parameters at a bound in the fit to {d.label}: {', '.join(res.at_limit)}; their Hesse "
                         "errors are not meaningful (use --minos)")
    if len(datasets) == 1:
        d, res = fits[0]
        print(f"Fit to {d.label}: {'valid' if res.valid else 'INVALID (' + res.status + ')'}, NLL {res.nll:.8g}")
        for name, e in result["fits"][0]["parameters"].items():
            if e["role"] in ("poi", "nuisance", "free"):
                extra = f" pull {e['pull']:+.3f}" if "pull" in e else ""
                mn = e.get("minos")
                mtext = f" minos [{mn['lower']:+.4g}, {mn['upper']:+.4g}]" if mn else ""
                print(f"  {name:<32} {e['value']:+.6g} +- {_fmt(e.get('error'))}{mtext}{extra}")
    else:
        rs = np.array([r.value(s.lik, s.lik.poi) for _, r in fits if r.valid])
        errs = np.array([r.errors.get(s.lik.poi, np.nan) for _, r in fits if r.valid])
        truth = args.expect_signal
        pulls = (rs - truth) / errs
        at_bound = int(np.sum(np.isclose(rs, s.fitter.bounds[s.lik.poi][0])))
        stats = {"n_toys": len(fits), "n_valid": int(len(rs)), "r_mean": float(np.mean(rs)), "r_std": float(np.std(rs)),
                 "pull_mean": float(np.nanmean(pulls)), "pull_std": float(np.nanstd(pulls)),
                 "n_at_lower_bound": at_bound}
        result["toy_summary"] = stats
        print(f"{len(fits)} toys ({info['mode']}): {len(rs)} valid fits; r_hat mean {stats['r_mean']:.4g} "
              f"std {stats['r_std']:.4g}; pull mean {stats['pull_mean']:+.3f} std {stats['pull_std']:.3f}")
        if at_bound:
            flags.append(f"{at_bound} toy fits have r_hat at the lower bound; pulls there are not Gaussian")
        if len(rs) < len(fits):
            flags.append(f"{len(fits) - len(rs)} toy fits failed and are excluded from the summary")
    if args.plot:
        from inference import plots

        plots.plot_fit(s, fits, args.plot_dir)
    s.write(result, flags)


def _scan_range(par, text, which):
    if text:
        return tuple(float(x) for x in text.split(":"))
    return par.lo, par.hi


def cmd_scan(backend, args):
    from inference.scan import grid_axes_2d, grid_points, profile_scan, profile_scan_2d

    s = Session(backend, args)
    names = [n for n in args.param.split(",") if n]
    if len(names) not in (1, 2):
        raise SystemExit("--param takes one parameter or two (a,b)")
    pars = [s._param(s.model, n) for n in names]
    texts = args.range.split(",") if args.range else [None] * len(names)
    if len(texts) != len(names):
        raise SystemExit(f"--range needs one lo:hi per scanned parameter ({len(names)})")
    ranges = [_scan_range(p, t, i) for i, (p, t) in enumerate(zip(pars, texts))]
    datasets, info = s.datasets()
    if len(datasets) != 1:
        raise SystemExit("scan works on one dataset; use --toys -1 or a single toy")
    if len(names) == 1:
        out = profile_scan(s.lik, s.fitter, datasets[0], names[0], grid_points(*ranges[0], args.points))
        for key, iv in out["intervals"].items():
            print(f"{names[0]}: best fit {out['best_fit']:.5g}; {key}% interval [{_fmt(iv['lower'])}, "
                  f"{_fmt(iv['upper'])}]")
    else:
        xs, ys = grid_axes_2d(ranges, args.points)
        # the scanned parameters must be free to move over the whole grid
        for n, (lo, hi) in zip(names, ranges):
            blo, bhi = s.fitter.bounds[n]
            s.fitter.set_range(n, min(blo, lo), max(bhi, hi))
        out = profile_scan_2d(s.lik, s.fitter, datasets[0], names, xs, ys)
        bx, by = out["best_fit"]
        print(f"2D scan of ({names[0]}, {names[1]}) on {len(xs)} x {len(ys)} points; best fit ({bx:.5g}, {by:.5g})")
        for key, c in out["contours"].items():
            if "x_range" in c:
                print(f"  {key}% contour (2DeltaNLL = {c['level']:.3f}): {names[0]} in [{c['x_range'][0]:.4g}, "
                      f"{c['x_range'][1]:.4g}], {names[1]} in [{c['y_range'][0]:.4g}, {c['y_range'][1]:.4g}]")
    if args.plot:
        from inference import plots

        (plots.plot_scan if len(names) == 1 else plots.plot_scan_2d)(out, args.plot_dir)
    out["data_mode"] = info
    s.write(out, out["flags"])


def cmd_impacts(backend, args):
    from inference.impacts import impacts

    s = Session(backend, args)
    datasets, info = s.datasets()
    if len(datasets) != 1:
        raise SystemExit("impacts work on one dataset: the data, --toys -1 (Asimov) or a single toy")
    with s.executor() as ex:
        res = impacts(s.lik, s.fitter, datasets[0], seed=args.seed,
                      names=[n for n in args.params.split(",") if n] or None,
                      exclude=[n for n in args.exclude.split(",") if n], errors=args.errors, executor=ex)
    lo, v, hi = res["POIs"][0]["fit"]
    print(f"{res['poi']} = {v:.5g} -{v - lo:.4g}/+{hi - v:.4g}   impacts ({args.errors} errors), sorted by |impact|:")
    print(f"  {'parameter':<28} {'pull':>8} {'constr':>7} {'dr(+1s)':>9} {'dr(-1s)':>9}")
    for p in res["params"]:
        if not p["valid"]:
            print(f"  {p['name']:<28} FAILED: {p['error']}")
            continue
        pull = f"{p['pull']:+.3f}" if "pull" in p else "free"
        constr = f"{p['constraint']:.3f}" if "constraint" in p else "-"
        print(f"  {p['name']:<28} {pull:>8} {constr:>7} {p['impact_hi']:+9.4f} {p['impact_lo']:+9.4f}")
    if args.plot:
        from inference import plots

        plots.plot_impacts(res, args.plot_dir, args.plot_max)
    res["data_mode"] = info
    s.write(res, res["flags"])


def _default_toy_grid(s, cl):
    from inference.asymptotic import AsymptoticLimits

    res = AsymptoticLimits(s.lik, s.fitter, cl=cl).run(data=s.observed())
    vals = [v for v in [res.observed] + list(res.expected.values()) if v]
    if not vals:
        raise SystemExit("could not derive a toy grid from asymptotic limits; pass --grid")
    return list(np.linspace(0.5 * min(vals), 1.5 * max(vals), 8)), res


def _split_plan(args):
    """(grid override, toy_range, refine, notes) from --points / --toy-chunk."""
    notes = []
    grid = _grid(args.points) if args.points else None
    toy_range = None
    refine = args.refine
    if args.toy_chunk:
        try:
            i, n = (int(x) for x in args.toy_chunk.split("/"))
        except ValueError:
            raise SystemExit(f"--toy-chunk takes I/N, got '{args.toy_chunk}'")
        if not 0 <= i < n:
            raise SystemExit(f"--toy-chunk {args.toy_chunk}: need 0 <= I < N")
        N = args.toys_per_point
        toy_range = (i * N // n, (i + 1) * N // n)
        notes.append(f"toy chunk {i}/{n}: toys [{toy_range[0]}, {toy_range[1]}) of {N} at every point")
    if grid is not None or toy_range is not None:
        if grid is None and not args.grid:
            raise SystemExit("--toy-chunk needs an explicit --grid (or --points), identical in every job")
        refine = 0
        notes.append("partial job (--points/--toy-chunk): no refinement; combine the --save-toy-results files "
                     "with 'merge'")
        if not args.save_toy_results:
            raise SystemExit("--points/--toy-chunk make a partial job; add --save-toy-results FILE")
    return grid, toy_range, refine, notes


def _max_toys(args, adaptive):
    if args.max_toys_per_point is None:
        return 20 * args.toys_per_point if adaptive else args.toys_per_point
    if args.max_toys_per_point < args.toys_per_point:
        raise SystemExit("--max-toys-per-point must be >= --toys-per-point")
    return args.max_toys_per_point


def _save_toy_results(s, kind, res, cl):
    from inference.hybrid import toy_results_doc

    meta = {"input": os.path.abspath(s.args.input), "backend": s.backend.name,
            "bypass_frequentist_fit": s.args.bypass_frequentist_fit, "argv": sys.argv}
    doc = toy_results_doc(kind, res.points, cl, s.args.seed, s.fitter.bounds[s.lik.poi][0], meta,
                          res.engine_flags)
    with open(s.args.save_toy_results, "w", encoding="utf-8") as handle:
        json.dump(doc, handle)
    print(f"Wrote raw toy results for {len(res.points)} points to {s.args.save_toy_results}")


def _print_cls_points(res):
    print(f"  {'r':>10} {'CLs':>9} {'+-':>8} {'n_sb':>6} {'n_b':>6}  stop")
    for p in res.points:
        d = p.to_dict()
        print(f"  {p.r:10.4g} {_fmt(d['CLs']):>9} {_fmt(d['CLs_err'], 2):>8} {p.n_sb:6d} {p.n_b:6d}  {p.stop_reason}")


def cmd_limit(backend, args):
    s = Session(backend, args)
    data = s.observed()
    if args.method == "asymptotic":
        from inference.asymptotic import AsymptoticLimits, QUANTILES

        run = AsymptoticLimits(s.lik, s.fitter, cl=args.cl, bypass_frequentist_fit=(args.run == "blind"
                                                                                  or args.bypass_frequentist_fit))
        res = run.run(data=data, observed=args.run in ("both", "observed"), expected=args.run != "observed")
        print(f"Asymptotic CLs limits on r at {args.cl:.0%} CL")
        if args.run in ("both", "observed"):
            print(f"  observed        r < {_fmt(res.observed)}   (r_hat = {_fmt(res.r_hat)})")
        for q in QUANTILES:
            if q in res.expected:
                print(f"  expected {q * 100:5.1f}%  r < {_fmt(res.expected[q])}")
        if args.plot:
            from inference import plots

            plots.plot_asymptotic(res, args.plot_dir)
        s.write(res.to_dict(), res.flags)
        return
    from inference.hybrid import toy_cls_limit

    split_grid, toy_range, refine, notes = _split_plan(args)
    adaptive = bool(args.cls_acc or args.r_abs_acc or args.r_rel_acc)
    if toy_range is not None and adaptive:
        raise SystemExit("--toy-chunk runs a fixed number of toys; it cannot be combined with --cls-acc/--r-*-acc")
    grid = split_grid or (_grid(args.grid) if args.grid else _default_toy_grid(s, args.cl)[0])
    s.fitter.set_range(s.lik.poi, args.rmin, max(args.rmax, 1.2 * max(grid)))
    for n in notes:
        print(f"note: {n}")
    with s.executor() as ex:
        res = toy_cls_limit(s.lik, s.fitter, s.seeds, grid, args.toys_per_point, cl=args.cl,
                            bypass_fit=args.bypass_frequentist_fit, refine=refine, data=data, executor=ex,
                            cls_acc=args.cls_acc, max_toys=_max_toys(args, adaptive),
                            r_abs_acc=None if split_grid else args.r_abs_acc,
                            r_rel_acc=None if split_grid else args.r_rel_acc, toy_range=toy_range)
    print(f"Toy CLs limits on r at {args.cl:.0%} CL ({args.toys_per_point} initial toys per point, "
          f"--jobs {args.jobs})")
    _print_cls_points(res)
    print(f"  observed        r < {_fmt(res.observed)} +- {_fmt(res.observed_err, 2)}   "
          f"[{res.refinement['stop_reason']}]")
    for q, v in res.expected.items():
        print(f"  expected {q * 100:5.1f}%  r < {_fmt(v)}")
    if args.save_toy_results:
        _save_toy_results(s, "cls", res, args.cl)
    if args.plot:
        from inference import plots

        plots.plot_toy_cls(res, args.plot_dir)
    out = res.to_dict()
    out["notes"] = notes
    s.write(out, res.flags)


def cmd_fc(backend, args):
    from inference.hybrid import feldman_cousins
    from inference.scan import grid_points, profile_scan

    s = Session(backend, args)
    data = s.observed()
    split_grid, toy_range, refine, notes = _split_plan(args)
    if toy_range is not None and args.p_acc:
        raise SystemExit("--toy-chunk runs a fixed number of toys; it cannot be combined with --p-acc")
    if split_grid:
        grid = split_grid
    elif args.grid:
        grid = _grid(args.grid)
    else:
        sc = profile_scan(s.lik, s.fitter, data, s.lik.poi, grid_points(args.rmin, args.rmax, 40))
        hi95 = sc["intervals"]["95"]["upper"] or args.rmax
        grid = list(np.linspace(args.rmin, 1.5 * hi95, 12))
    s.fitter.set_range(s.lik.poi, args.rmin, max(args.rmax, 1.2 * max(grid)))
    for n in notes:
        print(f"note: {n}")
    with s.executor() as ex:
        res = feldman_cousins(s.lik, s.fitter, s.seeds, grid, args.toys_per_point, cl=args.cl,
                              bypass_fit=args.bypass_frequentist_fit, refine=refine, data=data, executor=ex,
                              p_acc=args.p_acc, max_toys=_max_toys(args, bool(args.p_acc)), toy_range=toy_range)
    print(f"Feldman-Cousins {args.cl:.0%} CL interval for r: [{_fmt(res.lower)}, {_fmt(res.upper)}]")
    for p in res.points:
        d = p.to_dict()
        print(f"  r = {p.r:<10.4g} p = {_fmt(d['p'])} +- {_fmt(d['p_err'], 2)}  ({p.n} toys; {p.stop_reason})")
    if args.save_toy_results:
        _save_toy_results(s, "fc", res, args.cl)
    if args.plot:
        from inference import plots

        plots.plot_fc(res, args.plot_dir)
    out = res.to_dict()
    out["notes"] = notes
    s.write(out, res.flags)


def cmd_significance(backend, args):
    from inference.hybrid import significance

    s = Session(backend, args)
    n = args.toys_per_point if args.method == "toys" else 0
    with s.executor() as ex:
        res = significance(s.lik, s.fitter, s.seeds, ntoys=n, bypass_fit=args.bypass_frequentist_fit,
                           data=s.observed(), executor=ex)
    if res.get("asymptotic"):
        print(f"Asymptotic significance Z = {res['asymptotic']['Z']:.4g} (p = {res['asymptotic']['p']:.4g})")
    if res.get("toys"):
        t = res["toys"]
        print(f"Toy significance Z = {t['Z']:.4g} (p = {t['p']:.4g} +- {t['p_err']:.2g}, {t['n_toys']} toys)")
    s.write(res, res["flags"])


def cmd_generate(backend, args):
    from inference.toys import toy_base

    s = Session(backend, args)
    if args.toys == 0:
        raise SystemExit("generate needs --toys N (or -1 for Asimov)")
    if args.toys > 0 and not args.toys_file:
        cfg = s.toy_config()
        base, info = toy_base(s.lik, s.fitter, cfg, s.observed())
        datasets = [d for d, _ in _toy_tasks(s, cfg, base, fit=False)]
    else:
        datasets, info = s.datasets()
    with open(args.toys_out, "w", encoding="utf-8") as handle:
        json.dump({"format": "pymodel-toys", "mode": info["mode"], "seed": args.seed,
                   "datasets": [d.to_dict() for d in datasets]}, handle)
    print(f"Wrote {len(datasets)} datasets ({info['mode']}) to {args.toys_out}")
    s.write({"toys_out": args.toys_out, "n": len(datasets), "data_mode": info})


def cmd_merge(backend, args):
    """Merge raw toy results: no model is needed, only the files."""
    from inference.hybrid import cls_readout, fc_readout, merge_toy_results, toy_results_doc
    from inference.results import write_result

    docs = []
    for path in args.files:
        with open(path, "r", encoding="utf-8") as handle:
            docs.append(json.load(handle))
    try:
        kind, points, cl, r_lo, flags, info = merge_toy_results(docs)
    except ValueError as exc:
        raise SystemExit(f"merge failed: {exc}")
    cl = args.cl if args.cl is not None else cl
    if kind == "cls":
        res = cls_readout(points, cl, True, flags)
        res.refinement = {"stop_reason": f"merged {info['n_files']} files"}
        print(f"Merged {info['n_files']} files: {info['n_points']} points; toy CLs limits at {cl:.0%} CL")
        _print_cls_points(res)
        print(f"  observed        r < {_fmt(res.observed)} +- {_fmt(res.observed_err, 2)}")
        for q, v in res.expected.items():
            print(f"  expected {q * 100:5.1f}%  r < {_fmt(v)}")
    else:
        res = fc_readout(points, cl, r_lo, flags)
        print(f"Merged {info['n_files']} files: {info['n_points']} points; "
              f"Feldman-Cousins {cl:.0%} CL interval for r: [{_fmt(res.lower)}, {_fmt(res.upper)}]")
    if args.save_toy_results:
        meta = dict(docs[0]["meta"], merged_from=[os.path.abspath(f) for f in args.files])
        seeds = sorted({d["seed"] for d in docs})
        with open(args.save_toy_results, "w", encoding="utf-8") as handle:
            json.dump(toy_results_doc(kind, points, cl, seeds[0] if len(seeds) == 1 else seeds, r_lo, meta,
                                      flags), handle)
        print(f"Wrote merged raw toy results to {args.save_toy_results}")
    if args.plot:
        from inference import plots

        (plots.plot_toy_cls if kind == "cls" else plots.plot_fc)(res, args.plot_dir)
    out = res.to_dict()
    out.update({"kind": kind, "merged_files": [os.path.abspath(f) for f in args.files]})
    path = args.output or "pymodel_merge.json"
    write_result(path, command="merge", backend=backend.name, versions=dict(backend.runtime_versions()),
                 seed=sorted({str(d["seed"]) for d in docs}), input_path=docs[0]["meta"].get("input"),
                 model_notes=[], flags=res.flags, result=out)
    if res.flags:
        print("\nFLAGS (check before using the result):")
        for f in res.flags:
            print(f"  - {f}")
    print(f"\nWrote {path}")


def cmd_export(backend, args):
    s = Session(backend, args)
    backend.export(s.model, args.native_out, args)
    print(f"Wrote {backend.name} native model to {args.native_out}")


def run(argv=None):
    from stat_backends import BACKEND_NAMES, get_backend

    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in BACKEND_NAMES:
        print(f"usage: pymodel {{{','.join(BACKEND_NAMES)}}} {{{','.join(COMMANDS)}}} INPUT [options]", file=sys.stderr)
        return 2
    backend = get_backend(argv[0])
    args = build_parser(backend).parse_args(argv[1:])
    versions = " | ".join(f"{n} {v}" for n, v in backend.runtime_versions())
    print(f"pymodel {backend.name} {args.command}  [{versions}]", flush=True)
    globals()[f"cmd_{args.command}"](backend, args)
    return 0
