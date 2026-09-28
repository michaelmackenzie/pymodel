# pymodel: instructions for AI agents and contributors

pymodel runs Combine-style statistical analyses of **CMS Combine datacards**. It has three
interchangeable likelihood backends: `hfmodel` (pyhf), `zmodel` (zfit) and `roomodel`
(RooFit). **Combine is the reference semantics.** The definitions are in
`docs/statistics.md`; read it before touching anything statistical.

## Environment (fresh shell)
```bash
source setup_env.sh                  # rootana 2.5.0 + python/ on PYTHONPATH + .pydeps (pyhf)
scripts/install_python_deps.sh       # once: installs pyhf into .pydeps (rootana lacks it)
```
- **Combine environment.** Cards that use Combine classes (RooMultiPdf, RooLandauCB, …), and
  anything that runs `combine` / `text2workspace.py`, need Combine's environment. See
  `tests/make_combine_fixtures.sh` for the exact setup, and set
  `PYMODEL_ROOT_LIBS=libHiggsAnalysisCombinedLimit.so` so pymodel loads the classes.
- **No pytest.** The tests are plain scripts.

## Map
See `docs/architecture.md`. In short:
- datacard → `modelspec/` (ModelIR)
- ModelIR → `stat_backends/<name>/` (`Likelihood`)
- `inference/`: all fits, toys and limits, shared by every backend
- `pymodel_core.py`: the CLI

## Hard rules
1. **Match Combine.** Any change to a statistical definition must update
   `docs/statistics.md` and pass `tests/run_all.py --fast`. Quote before/after numbers in the
   change description.
2. **No silent fallbacks.**
   - Never write `except Exception: pass`, and never return a default value on failure.
   - Unsupported datacard content raises `UnsupportedFeature`. A backend that lacks a
     feature leaves it out of `supported_features`.
   - Suspect results get an entry in the result `flags`.
3. **Backends evaluate likelihoods only.** Never add fitting, toy, limit or constraint logic
   to a backend. Constraint terms and global observables come from the IR in `inference/`.
4. **Keep the NLL convention** (`inference/model.py`): main measurement only, without
   data-only constants. Backends must agree with `inference/semantic_likelihood.py`.
5. **Never report an unbracketed limit or a failed fit as a result.** They are `None` plus a
   flag.
6. **No legacy formats.** Inputs are Combine datacards with ROOT shapes, or bundles made by
   `build`.
7. **Toys are seeded per toy.** Random numbers in toy code come from
   `inference.parallel.ToySeeds` keys (stream, point, toy index, purpose), never from a shared
   sequential generator. That is what makes `--jobs N` and merged split jobs bit-identical.
   Work that runs in workers is a module-level task function that takes picklable tasks
   (`portable(dataset)`).

## Validation
```bash
python3 tests/run_all.py --fast                   # minutes; must pass before any commit
python3 tests/run_all.py --slow                   # toys: CLs, FC coverage, pulls
python3 tests/run_all.py --fast --backend roomodel
tests/make_combine_fixtures.sh WORKDIR           # regenerate Combine reference numbers (slow)
scripts/compare_combine.py CARD --workdir SCRATCH # pymodel vs Combine on any card (copies inputs)
```
- Each backend also has its own check: `tests/backend_<name>_check.py`.
- Claude Code skills live in `.claude/skills/`: `validate-stats`, `compare-combine` and
  `new-card-feature`.
- After changing CLI options, regenerate `docs/cli-reference.md` with
  `scripts/make_cli_reference.sh`.

## Boundaries
- **The user's analysis area is read-only.** Never write into
  `/exp/mu2e/app/users/mmackenz/mumep/mumep_ana`. To run pymodel or Combine on its cards,
  copy the card and the workspaces it uses into a scratch directory (with the same relative
  layout: `datacards/` + `workspaces/`) and run from there.
- **No `/cvmfs` scans.** Never run `find` or `grep -r` over `/cvmfs`.
- **Keep Combine runs small.** Running `combine` on copies is allowed, but keep the CPU
  budget modest.
- **Vendored code.** `third_party/combine/` is vendored and must not be edited; re-vendor it
  instead (see its `__init__.py`).

## Status / open work
`todo.org` holds the open items. `docs/improvement-plan.md` holds the review findings that
motivated the v2 rewrite.
