#!/usr/bin/env python3
"""Validate the hfmodel (pyhf) backend against the semantic oracle.

Writes small Combine datacards (counting single/multi-bin with lnN, asymmetric lnN, gmN and
rateParam; TH1 templates with shape systematics in two channels) into a work directory and
checks, at random parameter points, that ``nll_main`` and ``expected_by_process`` of hfmodel
agree with ``inference.semantic_likelihood.SemanticLikelihood``:

* exact where the pyhf mapping is exact (tolerance 1e-9 relative on yields, 1e-8 on the NLL);
* where pyhf approximates Combine (asymmetric lnN with code1 for |theta| < 0.5, vertical
  morphing with a scale != the process smooth region) the maximum differences are printed.

It also checks that pyhf's own full logpdf (with its constraints and auxdata built from the
Dataset global observables) equals -(nll_main + constraints + data constants), which
validates the parameter/auxdata mapping used for the native export.

Usage (after ``source setup_env.sh``):
    python3 tests/backend_hfmodel_check.py [--workdir DIR] [--card extra_card.txt ...]
Extra cards are parsed from the current directory (their shapes paths are resolved as in
Combine), so run from the directory the card expects.  Exit status 1 if an exact check fails.
"""

import argparse
import math
import os
import sys
import tempfile

import numpy as np
from scipy.special import gammaln

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))

from inference.model import Dataset, constraint_nll, observed_dataset  # noqa: E402
from inference.semantic_likelihood import SemanticLikelihood  # noqa: E402
from modelspec import ir as I  # noqa: E402
from modelspec import semantics as S  # noqa: E402
from modelspec.datacard import build_ir  # noqa: E402
from stat_backends import get_backend  # noqa: E402
from stat_backends.base import UnsupportedByBackend  # noqa: E402

THETAS = (-2.0, -1.0, -0.5, 0.0, 0.3, 0.5, 1.0, 2.0)

CARDS = {
    "count_single.txt": """imax 1
jmax 3
kmax *
---
bin a
observation 25
---
bin      a     a     a     a
process  sig   bkg   ctl   side
process  0     1     2     3
rate     4.0   12.0  3.0   6.0
---
lumi     lnN   1.025 1.025 -     -
bkgsym   lnN   -     1.30  -     -
bkgasym  lnN   0.85/1.10 -  1.2/0.9 -
ctlN     gmN 15 -    -     0.2   -
rp  rateParam a side 1.0 [0,5]
""",
    "count_sym.txt": """imax 1
jmax 1
kmax *
---
bin a
observation 20
---
bin      a     a
process  sig   bkg
process  0     1
rate     5.0   14.0
---
lumi     lnN   1.03  1.03
bkgsym   lnN   -     1.25
bkgN     gmN 28 -    0.5
""",
    "count_multi.txt": """imax 3
jmax 2
kmax *
---
bin         b1   b2   b3
observation 3    9    40
---
bin      b1    b1    b1    b2    b2    b2    b3    b3    b3
process  sig   bkg   oth   sig   bkg   oth   sig   bkg   oth
process  0     1     2     0     1     2     0     1     2
rate     1.5   2.0   0.5   3.0   6.0   1.0   2.0   35.0  2.0
---
lumi     lnN   1.02  1.02  1.02  1.02  1.02  1.02  1.02  1.02  1.02
effsig   lnN   0.9/1.15 -  -     0.9/1.15 - -     0.9/1.15 - -
bkgsh    lnN   -     1.2   -     -     0.8/1.1 -  -     1.05  -
othN     gmN 8 -     -     -     -     -     -     -     -     0.25
mu_bkg rateParam * bkg 1.0 [0,10]
""",
}


def _template_cards(workdir):
    """Two-channel TH1 template cards: one exact (scales 1 and 0.5 on separate processes) and
    one with mixed scales on one process (pyhf approximation)."""
    import ROOT

    ROOT.gROOT.SetBatch(True)
    rng = np.random.default_rng(7)
    path = os.path.join(workdir, "templates.root")
    out = ROOT.TFile(path, "RECREATE")
    x = np.linspace(-0.9, 0.9, 10)
    shapes = {
        "sig": np.exp(-0.5 * (x / 0.25) ** 2) * 10 + 0.05,
        "bkg": 20 * np.exp(-x) + 1,
        "bkg2": 5 + 3 * x + 0.1,
    }
    for ch, scale in (("chA", 1.0), ("chB", 0.7)):
        d = out.mkdir(ch)
        d.cd()
        total = np.zeros(10)
        for proc, nom in shapes.items():
            nom = nom * scale
            total += nom
            n0 = nom.sum()
            # jes/res: generic variations (asymmetric normalisation effect); jesS/resS: the same
            # shapes with a symmetric (kappa_down = 1/kappa_up) or no normalisation effect
            jes_up, jes_dn = nom * (1 + 0.15 * x), nom * (1 - 0.1 * x - 0.02)
            res_up, res_dn = nom * (1 + 0.2 * x ** 2), nom * (1 - 0.15 * x ** 2 + 0.03)
            for suffix, arr in (("", nom),
                                ("_jesUp", jes_up), ("_jesDown", jes_dn), ("_resUp", res_up), ("_resDown", res_dn),
                                ("_jesSUp", jes_up * 1.08 * n0 / jes_up.sum()),
                                ("_jesSDown", jes_dn * n0 / (1.08 * jes_dn.sum())),
                                ("_resSUp", res_up * n0 / res_up.sum()), ("_resSDown", res_dn * n0 / res_dn.sum()),
                                # large variation: the morph goes negative for |theta| > ~1.2 (floor + renormalise)
                                ("_bigUp", nom * (1 + 0.9 * x) * n0 / (nom * (1 + 0.9 * x)).sum()),
                                ("_bigDown", nom * (1 - 0.9 * x) * n0 / (nom * (1 - 0.9 * x)).sum())):
                h = ROOT.TH1D(proc + suffix, "", 10, -1.0, 1.0)
                for i, v in enumerate(arr):
                    h.SetBinContent(i + 1, float(v))
                h.Write()
        h = ROOT.TH1D("data_obs", "", 10, -1.0, 1.0)
        for i, v in enumerate(rng.poisson(total)):
            h.SetBinContent(i + 1, float(v))
        h.Write()
    out.Close()
    head = """imax 2
jmax 2
kmax *
---
shapes * * templates.root $CHANNEL/$PROCESS $CHANNEL/$PROCESS_$SYSTEMATIC
---
bin chA chB
observation -1 -1
---
bin      chA   chA   chA   chB   chB   chB
process  sig   bkg   bkg2  sig   bkg   bkg2
process  0     1     2     0     1     2
rate     -1    -1    -1    -1    -1    -1
---
lumi     lnN   1.025 1.025 1.025 1.025 1.025 1.025
bkgn     lnN   -     1.1   -     -     1.1   -
"""
    exact = head + """jesS     shape 1     1     -     1     1     -
resS     shape -     -     0.5   -     -     0.5
"""
    mixed = head + """jes      shape 1     2     0.5   1     -     -
res      shape -     1     -     -     0.5   -
bkg2asym lnN   -     -     0.9/1.2 -   -     0.85/1.1
"""
    clip = head + """big      shape 1     1     -     1     -     -
"""
    return {"tmpl_exact.txt": exact, "tmpl_mixed.txt": mixed, "tmpl_clip.txt": clip}


def random_points(lik, rng, n):
    base = lik.nominal_values()
    pts = []
    for _ in range(n):
        v = base.copy()
        for i, p in enumerate(lik.parameters):
            if not p.floating:
                continue
            if p.role == I.ROLE_POI:
                v[i] = rng.uniform(0.0, 3.0)
            elif p.constraint is not None and p.constraint.kind == I.CONSTRAINT_POISSON:
                v[i] = p.value * rng.uniform(0.5, 1.5)
            elif p.constraint is not None:
                v[i] = rng.uniform(-2.5, 2.5)
            else:
                v[i] = p.value * rng.uniform(0.5, 2.0)
        pts.append(v)
    pts[0][lik.poi_index] = 0.0  # r = 0 (signal factor zero) is where toys and CLs start
    return pts


def random_dataset(model, lik, values, rng):
    """A toy-like dataset: Poisson main data and shifted global observables."""
    exp = lik.expected_counts(values)
    data = observed_dataset(model)
    for ch, mu in exp.items():
        data.main[ch].counts = rng.poisson(mu).astype(float)
    for name in data.global_obs:
        c = model.parameters[name].constraint
        data.global_obs[name] = float(rng.poisson(c.center)) + 1 if c.kind == I.CONSTRAINT_POISSON \
            else float(rng.normal(0.0, 1.0))
    data._native.clear()
    return data


def compare(model, label, exact, npoints=20, seed=1):
    backend = get_backend("hfmodel")
    lik = backend.build_likelihood(model, None)
    ref = SemanticLikelihood(model)
    rng = np.random.default_rng(seed)
    obs = observed_dataset(model)
    max_rel_y, max_dnll, max_dlogpdf, n_clipped = 0.0, 0.0, 0.0, 0
    # parameters with a pyhf modifier (others are constrained by the shared layer only)
    pyhf_part = I.ModelIR(channels=model.channels, poi=model.poi,
                          parameters={n: p for n, p in model.parameters.items() if n in lik.spec.links})
    for v in random_points(lik, rng, npoints):
        e_hf, e_ref = lik.expected_by_process(v), ref.expected_by_process(v)
        for ch in e_ref:
            for p in e_ref[ch]:
                a, b = e_hf[ch][p], e_ref[ch][p]
                rel = np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12))
                max_rel_y = max(max_rel_y, float(rel) if np.isfinite(rel) else math.inf)
        for data in (obs, random_dataset(model, lik, v, rng)):
            d = abs(lik.nll_main(v, lik.native(data)) - ref.nll_main(v, ref.prepare(data)))
            max_dnll = max(max_dnll, d if np.isfinite(d) else math.inf)
            # pyhf's own full logpdf (constraints + constants) vs the shared-layer convention; not
            # comparable where hfmodel floors negative morphed bins (pyhf itself does not)
            raw = lik.pdf.main_model.expected_data(lik.pyhf_pars(v), return_by_sample=True)
            if np.any(raw[lik._template_mask] <= 0):
                n_clipped += 1
                continue
            n = lik.native(data)
            n_gauss = sum(lik.pdf.config.param_set(nm).n_parameters for nm in lik.pdf.config.auxdata_order
                          if lik.pdf.config.param_set(nm).pdf_type == "normal")
            expect = -(lik.nll_main(v, n) + float(np.sum(gammaln(n + 1.0)))
                       + constraint_nll(pyhf_part, lik.index, v, data.global_obs)
                       + 0.5 * math.log(2 * math.pi) * n_gauss)
            dl = abs(lik.pyhf_logpdf(v, data) - expect)
            max_dlogpdf = max(max_dlogpdf, dl if np.isfinite(dl) else math.inf)
    ok = max_rel_y < 1e-9 and max_dnll < 1e-8 if exact else True
    ok &= max_dlogpdf < 1e-7
    status = "OK" if ok else "FAIL"
    kind = "exact" if exact else "approx"
    print(f"[{status}] {label:<22} ({kind}) {npoints} points: max rel yield diff {max_rel_y:.3g}, "
          f"max |dNLL_main| {max_dnll:.3g}, max |d pyhf logpdf| {max_dlogpdf:.3g}"
          + (f" ({n_clipped} evaluations with floored bins skipped in the logpdf check)" if n_clipped else ""))
    for note in lik.notes:
        print(f"      note: {note}")
    return ok, lik


def template_theta_check(model, lik, label):
    """Per template systematic: hfmodel yield vs semantics.template_expected at fixed thetas."""
    worst = 0.0
    for ch in model.channels:
        for proc in ch.processes:
            if proc.shape.kind != "template" or not proc.shape.systs:
                continue
            for syst in proc.shape.systs:
                diffs, sdiffs = [], []
                for th in THETAS:
                    v = lik.nominal_values()
                    v[lik.index[syst.param]] = th
                    thetas = [th if s.param == syst.param else 0.0 for s in proc.shape.systs]
                    ref = S.template_expected(proc.shape.contents, proc.rate,
                                              [(s.up, s.down, s.scale) for s in proc.shape.systs], thetas)
                    got = lik.expected_by_process(v)[ch.name][proc.name]
                    if proc.is_signal:
                        got = got / v[lik.poi_index]
                    diffs.append(float(np.max(np.abs(got / ref - 1.0))))
                    sdiffs.append(float(np.max(np.abs(got / got.sum() / (ref / ref.sum()) - 1.0))))
                worst = max(worst, max(d if np.isfinite(d) else math.inf for d in diffs))
                text = " ".join(f"{t:+.1f}:{d:.2g}/{e:.2g}" for t, d, e in zip(THETAS, diffs, sdiffs))
                print(f"      {label} {ch.name}/{proc.name}/{syst.param} (scale {syst.scale:g}) "
                      f"max rel diff yield/shape-only per theta  {text}")
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--card", action="append", default=[], help="extra datacard(s) to compare (approx mode)")
    ap.add_argument("--points", type=int, default=20)
    args = ap.parse_args()
    workdir = args.workdir or tempfile.mkdtemp(prefix="hfcheck_")
    os.makedirs(workdir, exist_ok=True)
    cards = dict(CARDS)
    cards.update(_template_cards(workdir))
    for name, text in cards.items():
        with open(os.path.join(workdir, name), "w", encoding="utf-8") as handle:
            handle.write(text)
    all_ok = True
    exact = {"count_sym.txt": True, "count_single.txt": False, "count_multi.txt": False,
             "tmpl_exact.txt": True, "tmpl_mixed.txt": False, "tmpl_clip.txt": True}
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        for name in cards:
            model = build_ir(name)
            ok, lik = compare(model, name, exact[name], args.points)
            all_ok &= ok
            if name.startswith("tmpl"):
                worst = template_theta_check(model, lik, name)
                if exact[name] and worst > 1e-9:
                    print(f"[FAIL] {name}: template morph differs by {worst:.3g}")
                    all_ok = False
        # gmN acting on two processes must be refused at create time
        model = build_ir("count_multi.txt")
        bad = model.channel("b2").processes[2]
        bad.norm_terms.append(I.NormTerm(kind="gmN", param="othN", alpha=0.1))
        try:
            get_backend("hfmodel").build_likelihood(model, None)
            print("[FAIL] gmN on two processes was not refused")
            all_ok = False
        except UnsupportedByBackend as err:
            print(f"[OK] gmN on two processes refused: {err}")
    finally:
        os.chdir(cwd)
    for card in args.card:
        model = build_ir(card)
        ok, lik = compare(model, os.path.basename(card), False, args.points)
        all_ok &= ok
    print("ALL OK" if all_ok else "SOME CHECKS FAILED", f"(cards in {workdir})")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
