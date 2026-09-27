# roomodel (RooFit)

Code: `python/stat_backends/roomodel/`:
- `workspace.py`: IR → RooWorkspace
- `evaluator.py`: C++ bin and event loop, compiled once through cling
- `likelihood.py`
- `export.py`

roomodel is the most general backend. It uses the pdfs from the Combine workspaces as they
are.

## Supported model features
Everything in the IR except `syst:pdf-morph` (shape systematics on RooAbsPdfs):
- **Data:** counting, binned, unbinned and weighted.
- **Shapes:** templates (`shape` and `shapeN`), parametric pdfs with floating parameters,
  `<pdf>_norm` variables and functions, and RooMultiPdf envelopes (`discrete`).
- **Norm terms:** `lnN` (symmetric and asymmetric), `lnU`, `gmN`, `rateParam` (plain and
  formula).
- **Constraints:** every constraint type.

Workspaces with Combine classes (RooMultiPdf, RooLandauCB, …) need
`PYMODEL_ROOT_LIBS=libHiggsAnalysisCombinedLimit.so` in the Combine environment.

## Implementation
- **Parameters.** Each IR parameter is one RooRealVar, created before the workspace objects
  are imported with RecycleConflictNodes, so the imported pdfs use the IR parameters
  directly. Nodes with the same name but different definitions from different files are
  refused (Combine would silently share them).
- **Yields.** `n_exp_<ch>_<proc>` = rate × r × norm terms. lnN is `exp(θ ln κ)`, asymmetric
  lnN is Combine's `logKappaForX` formula, and gmN is α·n.
- **Templates.** Per-bin RooFormulaVars implement Combine's smooth-step vertical morph (and
  the shapeN log morph), with clipping and renormalisation. The Combine library is not needed
  for this.
- **Binned parametric pdfs** are evaluated at the bin centre × width (or as bin integrals with
  `--bin-integration integral`). The extended term uses the full yield, as Combine does.
- **Envelopes.** The penalty is `RooMultiPdf::getCorrection()`, exactly what Combine's
  CachingAddNLL adds; RooFit counts the observable as a parameter there. Parameters of the
  non-selected pdfs are frozen.
- **Speed.** The NLL is summed in C++ from RooFit evaluations, at about 7–17 µs per call on the
  mumep_ana cards.

## Export
`roomodel export INPUT --native-out ws.root` writes a RooWorkspace `w` containing:
- `model_s`, the main pdf × the constraint pdfs;
- the `<p>_Pdf` / `<p>_In` constraint pdfs and global observables;
- `data_obs`;
- `ModelConfig` and `ModelConfig_bonly`;
- the IR as JSON.

## Validation (`tests/backend_roomodel_check.py`)
**Against the numpy oracle:** ≤ 2e-15 relative.

**Against Combine**, as ΔNLL between parameter points on the text2workspace model of the same
card:

| card | ΔNLL agreement |
|---|---|
| `combine_mumem_75_evt_r0104_hists` | 3e-14 |
| `combine_mumem_75_evt_r0104_funcs` | 4e-14 |
| `combine_mumep_40_evt_r0104_env` (both envelope indices) | 4e-11 |
| `combine_total_mumem_75_evt_r0101` | 3e-14 |
| TH1 templates | 5e-7, from Combine rounding the template κ to `%f` |

**Asymptotic limits (observed / median) against Combine:**

| card | roomodel | Combine |
|---|---|---|
| funcs | 9.136 / 14.44 | 9.142 / 14.44 |
| env (`--rmax 1000`) | 236.6 / 197.1 | 237.0 / 196.5 |
