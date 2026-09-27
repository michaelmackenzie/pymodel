# hfmodel (pyhf)

Code: `python/stat_backends/hfmodel/` (`builder.py` IR → pyhf spec; `likelihood.py`).
It needs pyhf, which rootana 2.5.0 lacks: run `scripts/install_python_deps.sh` once. That
installs pyhf 0.7.6 and jsonpatch into `.pydeps/`, which `setup_env.sh` adds to the path.

## Supported model features
- **Data:** counting and binned.
- **Shapes:** counting, TH1/RooDataHist templates, and parametric pdfs *with a fixed
  histogram* (no floating shape parameters).
- **Systematics:** `shape`, `lnN` (symmetric and asymmetric), `rateParam`, and `gmN` (one
  process in one single-bin channel only, via an exactly equivalent shapesys).
- **Constraints:** Gaussian constraints with centre 0 and width 1.

Everything else (unbinned data, floating parametric pdfs, envelopes, lnU, formulas,
BifurGauss `param`s) is rejected when the likelihood is built.

## Mapping and exactness
- **Main term only.** `nll_main` is computed from pyhf's per-sample expected rates. pyhf's own
  constraint terms are not used: the shared layer adds the IR constraints.
- **Symmetric lnN** → normsys code 1: exactly κ^θ.
- **Asymmetric lnN** → normsys code 1. It equals Combine's asymPow for |θ| ≥ 0.5. Inside that
  region pyhf is piecewise exponential, a relative yield difference of at most
  0.035·|ln(κu·κd)| (0.27% for 0.9/1.2). Code 1 was measured to be closer than code 4. A note
  is added to the model.
- **Template shapes** → histosys code 4p on the unit-normalised shapes, plus a normsys for the
  normalisation. This is algebraically Combine's smooth-step morph, and it is exact (≤ 1e-15)
  when all shape systematics on a process share one scale ≤ 1. Otherwise the smooth region
  differs for |θ| < 1 (≤ 0.8% at scale 2), and a note names the process.
- **Negative morphed bins** are floored and renormalised after pyhf's evaluation, as in
  Combine. The exported pyhf JSON cannot express this.

## Export
`hfmodel export INPUT --native-out model.json` writes a pyhf workspace. It also writes a
`model_settings.json` sidecar holding the interpolation codes, the parameter map and the
notes, which pyhf JSON cannot carry.

## Validation
`tests/backend_hfmodel_check.py` compares against the numpy oracle at random points.

| models | NLL agreement |
|---|---|
| exact models | ≤ 1e-12 |
| approximated models | the differences listed above |

On `combine_mumem_75_evt_r0104_hists.txt` the asymptotic limits are 9.086 (observed) and
14.29 (median). Combine gives 9.087 and 14.31.
