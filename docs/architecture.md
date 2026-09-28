# Architecture

```
Combine datacard + ROOT shapes
        │  modelspec/datacard.py  (Combine's own DatacardParser, vendored in third_party/combine)
        ▼
    ModelIR  ──────────────────────  modelspec/bundle.py  (build: JSON + <stem>_objects.root)
        │  stat_backends/<name>  (hfmodel: pyhf, zmodel: zfit, roomodel: RooFit)
        ▼
    Likelihood  (nll_main, expected_by_process, sample_unbinned)
        │  inference/  (shared by every backend)
        ▼
  fitting · toys · test statistics · asymptotic CLs · toy CLs · Feldman–Cousins · scans
        │  pymodel_core.py (CLI)  →  JSON result with `flags`  (+ plots)
```

## Layers

| Layer | Files | Responsibility |
|---|---|---|
| Datacard parsing | `third_party/combine/` | Combine's parser, unmodified apart from relative imports |
| Model IR | `modelspec/ir.py` | Explicit, JSON-serialisable description of the model (the semantics are in its docstring) |
| IR construction | `modelspec/datacard.py`, `modelspec/rootinput.py` | Shape resolution with Combine's precedence and file lookup; reading TH1 / RooDataHist / RooDataSet / RooAbsPdf / `_norm` / RooMultiPdf; everything unsupported raises `UnsupportedFeature` |
| Reference formulas | `modelspec/semantics.py` | numpy ports of Combine's asymPow, smooth step and vertical morphing |
| Backends | `stat_backends/<name>/` | `ModelIR` → `Likelihood`; each declares `supported_features`, and `build_likelihood` rejects models that use anything else |
| Oracle | `inference/semantic_likelihood.py` | numpy `Likelihood` built from `semantics.py`; backends are tested against it |
| Inference | `inference/*.py` | Fitter (iminuit), constraint terms and global observables, toy modes, q̃/t/q₀, AsymptoticLimits, HybridNew-style toy CLs and FC (adaptive toys, raw-result merging), significance, 1D/2D scans, impacts, per-toy seeds and the spawn process pool (`parallel.py`), result schema, plots |
| CLI | `python/pymodel`, `python/pymodel_core.py`, `bin/*` | `pymodel <backend> <command> INPUT [options]` |

## Design rules
1. **Combine is the reference.** Definitions live in `docs/statistics.md`, and the tests
   compare them against Combine outputs (`tests/fixtures/`).
2. **One implementation of each statistical method.** Backends evaluate likelihoods and never
   implement fits, toys or limits.
3. **Fail loudly.** Unsupported datacard content raises an error. Anything that makes a number
   suspect goes into the result's `flags`. Nothing is dropped or approximated silently.
4. **Constraint terms and global observables belong to the shared layer.** They come from
   the IR, so all backends treat them identically. Backends compute the main measurement only.
5. **One NLL convention** (see `inference/model.py`), so backends can be compared number for
   number.

## Adding things
- **A datacard feature.** Parse it in `modelspec/datacard.py` into IR fields (never a
  free-form string). Give it a feature name in `ModelIR.features()`, add it to the oracle
  and to `docs/statistics.md`, implement it in each backend that can support it exactly, and
  add a card to the grammar tests in `tests/`.
- **A statistical method.** Add it under `inference/`, built on `Fitter`, `TestStat` and
  `toys`. Give it a CLI command in `pymodel_core.py`, a result schema with flags, and a
  Combine fixture comparison.
- **A backend.** Create a `stat_backends/<name>/` package with `BACKEND`, add it to
  `stat_backends.BACKEND_NAMES`, and give it a `tests/backend_<name>_check.py`.
