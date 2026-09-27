"""pymodel command line: ``pymodel <backend> <command> INPUT [options]``.

INPUT is a Combine datacard or a model bundle written by ``build``.  The commands mirror
Combine's methods:

  build         datacard -> model bundle (JSON, plus a ROOT file for RooFit objects)
  inspect       print the model: channels, processes, parameters, notes
  nll           evaluate the NLL at given parameter values (validation)
  fit           maximum-likelihood fit to data or toys (FitDiagnostics-like)
  scan          1D profile-likelihood scan of any parameter (MultiDimFit --algo grid)
  limit         CLs upper limit: --method asymptotic (AsymptoticLimits) or toys (HybridNew LHC-limits)
  fc            Feldman-Cousins interval (HybridNew LHC-feldman-cousins)
  significance  discovery significance (Significance; asymptotic or toys)
  generate      generate and save toy datasets (GenerateOnly)
  export        write the model in the backend's native format

Every command writes a JSON result file (``--output``); its ``flags`` list names anything
that makes the result suspect.
"""

import argparse
import copy
import json
import math
import sys

import numpy as np

COMMANDS = ("build", "inspect", "nll", "fit", "scan", "limit", "fc", "significance", "generate", "export")


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

    p = sub.add_parser("scan", help="1D profile-likelihood scan")
    _add_common(p, backend)
    p.add_argument("--param", default="r")
    p.add_argument("--points", type=int, default=50)
    p.add_argument("--range", default=None, help="lo:hi (default: POI range, or the parameter range)")
    _add_toy_options(p)

    p = sub.add_parser("limit", help="CLs upper limit")
    _add_common(p, backend)
    p.add_argument("--method", choices=("asymptotic", "toys"), default="asymptotic")
    p.add_argument("--cl", type=float, default=0.95)
    p.add_argument("--run", choices=("both", "observed", "expected", "blind"), default="both",
                   help="blind: expected only, from a pre-fit Asimov dataset")
    p.add_argument("--grid", default=None, help="toys: r grid 'lo:hi:n' or list (default: around the asymptotic limit)")
    p.add_argument("--toys-per-point", type=int, default=500)
    p.add_argument("--refine", type=int, default=3, help="toys: bisection points added around the crossing")
    p.add_argument("--bypass-frequentist-fit", action="store_true")
    p.add_argument("--toys-file", default=None, help="use a saved dataset as the observed data")
    p.add_argument("--toy-index", type=int, default=0)

    p = sub.add_parser("fc", help="Feldman-Cousins interval")
    _add_common(p, backend)
    p.add_argument("--cl", type=float, default=0.90)
    p.add_argument("--grid", default=None, help="r grid 'lo:hi:n' or list (default: from a likelihood scan)")
    p.add_argument("--toys-per-point", type=int, default=500)
    p.add_argument("--refine", type=int, default=3, help="bisection points added around each edge")
    p.add_argument("--bypass-frequentist-fit", action="store_true")
    p.add_argument("--toys-file", default=None)
    p.add_argument("--toy-index", type=int, default=0)

    p = sub.add_parser("significance", help="discovery significance")
    _add_common(p, backend)
    p.add_argument("--method", choices=("asymptotic", "toys"), default="asymptotic")
    p.add_argument("--toys-per-point", type=int, default=1000)
    p.add_argument("--bypass-frequentist-fit", action="store_true")
    p.add_argument("--toys-file", default=None)
    p.add_argument("--toy-index", type=int, default=0)

    p = sub.add_parser("generate", help="generate and save toys")
    _add_common(p, backend)
    _add_toy_options(p)
    p.add_argument("--toys-out", default="toys.json")

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
        self.rng = np.random.default_rng(args.seed)
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
        return generate_toys(self.lik, self.fitter, cfg, self.rng, observed=self.observed())

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
        print(f"channel {ch.name}: data {ch.data.kind} (total {ch.data.total:g}), observable {ch.observable.name} "
              f"[{ch.observable.lo:g}, {ch.observable.hi:g}] with {len(ch.observable.edges) - 1} bins")
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


def cmd_fit(backend, args):
    from inference.scan import fit_summary

    s = Session(backend, args)
    datasets, info = s.datasets()
    minos = [p.name for p in s.lik.parameters if p.floating] if args.minos == "all" else \
        [n for n in args.minos.split(",") if n]
    fixed = {s.lik.poi: args.fix_r} if args.fix_r is not None else None
    fits = []
    for d in datasets:
        res = s.fitter.fit(d, fixed=fixed, hesse=True, minos=minos)
        fits.append((d, res))
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


def cmd_scan(backend, args):
    from inference.scan import grid_points, profile_scan

    s = Session(backend, args)
    par = s._param(s.model, args.param)
    if args.range:
        lo, hi = (float(x) for x in args.range.split(":"))
    else:
        lo, hi = par.lo, par.hi
    datasets, info = s.datasets()
    if len(datasets) != 1:
        raise SystemExit("scan works on one dataset; use --toys -1 or a single toy")
    out = profile_scan(s.lik, s.fitter, datasets[0], args.param, grid_points(lo, hi, args.points))
    for key, iv in out["intervals"].items():
        print(f"{args.param}: best fit {out['best_fit']:.5g}; {key}% interval [{_fmt(iv['lower'])}, {_fmt(iv['upper'])}]")
    if args.plot:
        from inference import plots

        plots.plot_scan(out, args.plot_dir)
    out["data_mode"] = info
    s.write(out, out["flags"])


def _default_toy_grid(s, cl):
    from inference.asymptotic import AsymptoticLimits

    res = AsymptoticLimits(s.lik, s.fitter, cl=cl).run(data=s.observed())
    vals = [v for v in [res.observed] + list(res.expected.values()) if v]
    if not vals:
        raise SystemExit("could not derive a toy grid from asymptotic limits; pass --grid")
    return list(np.linspace(0.5 * min(vals), 1.5 * max(vals), 8)), res


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

    grid = _grid(args.grid) if args.grid else _default_toy_grid(s, args.cl)[0]
    s.fitter.set_range(s.lik.poi, args.rmin, max(args.rmax, 1.2 * max(grid)))
    res = toy_cls_limit(s.lik, s.fitter, s.rng, grid, args.toys_per_point, cl=args.cl,
                        bypass_fit=args.bypass_frequentist_fit, refine=args.refine, data=data)
    print(f"Toy CLs limits on r at {args.cl:.0%} CL ({args.toys_per_point} toys per point)")
    print(f"  observed        r < {_fmt(res.observed)} +- {_fmt(res.observed_err, 2)}")
    for q, v in res.expected.items():
        print(f"  expected {q * 100:5.1f}%  r < {_fmt(v)}")
    if args.plot:
        from inference import plots

        plots.plot_toy_cls(res, args.plot_dir)
    s.write(res.to_dict(), res.flags)


def cmd_fc(backend, args):
    from inference.hybrid import feldman_cousins
    from inference.scan import grid_points, profile_scan

    s = Session(backend, args)
    data = s.observed()
    if args.grid:
        grid = _grid(args.grid)
    else:
        sc = profile_scan(s.lik, s.fitter, data, s.lik.poi, grid_points(args.rmin, args.rmax, 40))
        hi95 = sc["intervals"]["95"]["upper"] or args.rmax
        grid = list(np.linspace(args.rmin, 1.5 * hi95, 12))
    s.fitter.set_range(s.lik.poi, args.rmin, max(args.rmax, 1.2 * max(grid)))
    res = feldman_cousins(s.lik, s.fitter, s.rng, grid, args.toys_per_point, cl=args.cl,
                          bypass_fit=args.bypass_frequentist_fit, refine=args.refine, data=data)
    print(f"Feldman-Cousins {args.cl:.0%} CL interval for r: [{_fmt(res.lower)}, {_fmt(res.upper)}]")
    if args.plot:
        from inference import plots

        plots.plot_fc(res, args.plot_dir)
    s.write(res.to_dict(), res.flags)


def cmd_significance(backend, args):
    from inference.hybrid import significance

    s = Session(backend, args)
    n = args.toys_per_point if args.method == "toys" else 0
    res = significance(s.lik, s.fitter, s.rng, ntoys=n, bypass_fit=args.bypass_frequentist_fit, data=s.observed())
    if res.get("asymptotic"):
        print(f"Asymptotic significance Z = {res['asymptotic']['Z']:.4g} (p = {res['asymptotic']['p']:.4g})")
    if res.get("toys"):
        t = res["toys"]
        print(f"Toy significance Z = {t['Z']:.4g} (p = {t['p']:.4g} +- {t['p_err']:.2g}, {t['n_toys']} toys)")
    s.write(res, res["flags"])


def cmd_generate(backend, args):
    s = Session(backend, args)
    if args.toys == 0:
        raise SystemExit("generate needs --toys N (or -1 for Asimov)")
    datasets, info = s.datasets()
    with open(args.toys_out, "w", encoding="utf-8") as handle:
        json.dump({"format": "pymodel-toys", "mode": info["mode"], "seed": args.seed,
                   "datasets": [d.to_dict() for d in datasets]}, handle)
    print(f"Wrote {len(datasets)} datasets ({info['mode']}) to {args.toys_out}")
    s.write({"toys_out": args.toys_out, "n": len(datasets), "data_mode": info})


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
