# Testing and validation

pymodel is validated against three independent references:
1. **`tests/reference/reference_stats.py`.** A pure numpy/scipy implementation that imports no
   pymodel code. It covers counting and template models, Combine's asymptotic CLs, exact
   Poisson CLs, and Feldman–Cousins by Neyman construction. It reproduces FC98 Table IV.
2. **Combine itself.** `tests/fixtures/*.json` hold Combine results (AsymptoticLimits,
   MultiDimFit grid and singles, FitDiagnostics, Significance, HybridNew LHC-limits and
   LHC-FC) for every example and for three mumep_ana cards. Each file records the exact
   command lines and the Combine commit.
3. **The numpy oracle** (`inference/semantic_likelihood.py`). Each backend's NLL and yields are
   compared with it at random parameter points.

## Running
```bash
source setup_env.sh
python3 tests/run_all.py --fast                   # ~2 min: grammar, bundles, oracle, backends vs Combine, toy sanity
python3 tests/run_all.py --slow                   # toy CLs vs HybridNew and exact Poisson, FC vs FC98/Combine, pulls, coverage
python3 tests/run_all.py --fast --backend hfmodel --only "Combine fixture"
```
Envelope cards (RooMultiPdf) need the Combine environment and
`PYMODEL_ROOT_LIBS=libHiggsAnalysisCombinedLimit.so`. If you use that environment, add
`$PYMODEL_REPO/.pydeps` to `PYTHONPATH` so that pyhf is found. A backend that cannot be used
for a test is reported as SKIP, with the reason.

Per-backend checks, which are more detailed and print the size of every approximation:
```bash
python3 tests/backend_hfmodel_check.py
python3 tests/backend_zmodel_check.py
python3 tests/backend_roomodel_check.py --workdir /tmp/<scratch>
```

## Regenerating the Combine fixtures
```bash
tests/make_combine_fixtures.sh /path/to/scratch/workdir [--only counting,templates] [--jobs 16]
```
Combine is run only inside the work directory, on copies. The script refuses to run inside
the repository or in mumep_ana. The envelope fixtures use `--cminRunAllDiscreteCombinations`,
because Combine's default discrete minimisation is not exhaustive: it gave a 97.5% expected
limit of 390.6 on mumep_40_env, against 383.2 exhaustively.

## Tolerances
| check | tolerance |
|---|---|
| NLL vs oracle | 1e-6 relative |
| asymptotic limits vs Combine | 1% (typical agreement 0.1–0.3%) |
| best fit and 68% interval | 2% |
| grid 2ΔNLL | 1e-3 |
| toy p-values vs HybridNew | 3σ binomial |
| FC coverage | ≥ 1 − α within 2σ |

A backend may exceed the oracle tolerance only for an approximation that it declares in its
notes (for example hfmodel's asymmetric-lnN interpolation). The test then applies the
documented bound instead.

Never loosen a tolerance to make a test pass. Find the cause.
