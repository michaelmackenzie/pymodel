---
name: validate-stats
description: Run pymodel's validation suite (oracle, independent reference, Combine fixtures, toys) and interpret the results. Use after any change to modelspec/, inference/, stat_backends/ or the CLI, and before committing.
---

# Validate pymodel statistics

1. Set up a fresh shell. Use the Combine environment so that the envelope tests run:
   ```bash
   cd /exp/mu2e/app/users/mmackenz/mumep/combine/HiggsAnalysis/CombinedLimit
   source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh && pyenv rootana 2.5.0 && source env_standalone_mu2e.sh
   cd /exp/mu2e/app/users/mmackenz/mumep/pymodel
   export PYTHONPATH=$PYTHONPATH:$PWD/python:$PWD/.pydeps PYMODEL_ROOT_LIBS=libHiggsAnalysisCombinedLimit.so
   ```
   Without Combine, `source setup_env.sh` is enough. The envelope tests then SKIP.
2. Run the fast tier: `python3 tests/run_all.py --fast`, about 2 minutes. It must be all PASS.
   To iterate on one area, use `--only "<test title text>"` and/or `--backend <name>`.
3. If the change touches toys, test statistics, hybrid.py or the fitter, also run
   `python3 tests/run_all.py --slow --backend roomodel`. That takes about 10–20 minutes; the
   zmodel slow tier is much slower.
4. If a backend changed, run its own check, `python3 tests/backend_<name>_check.py`. It prints
   the size of every declared approximation.

## Interpreting failures
- **Backend vs oracle.** The backend's `nll_main` or yields break the convention in
  `inference/model.py`, or the semantics in `modelspec/semantics.py`. Evaluate both at the
  reported parameter point to find the term.
- **Backend vs Combine fixture.** Compare with the numbers in `tests/fixtures/<name>.json`,
  whose `commands` show exactly what Combine ran. Differences of about 0.3% in limits come
  from Combine's 0.5% accuracy target and are fine. Larger differences are real.
- **Toy tests** are statistical. Before calling a failure a fluctuation, rerun it with a
  different seed.
- **Never loosen a tolerance** and never skip a test to get a PASS. Find the cause, and
  report the numbers before and after the fix.
