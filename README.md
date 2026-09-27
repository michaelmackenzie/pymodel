# pymodel

pymodel runs Combine-style statistical analyses (fits, likelihood scans, CLs limits,
Feldman–Cousins intervals, significances) on **CMS Combine datacards** with one of three
likelihood backends:

| backend | engine | typical use |
|---|---|---|
| `roomodel` | RooFit | anything a Combine workspace contains: parametric pdfs, `_norm` functions, RooMultiPdf envelopes |
| `zmodel` | zfit | counting, templates, and parametric pdfs translated from RooFit (explicit class map) |
| `hfmodel` | pyhf | counting and histogram-template models |

All statistics (fitting, toys, test statistics, limits and intervals) are implemented once
and shared by the backends. They follow Combine's definitions, which are documented in
[docs/statistics.md](docs/statistics.md). Each backend only evaluates the likelihood, so the
same card gives the same numbers on every backend that supports it. The tests check this
against Combine itself.

## Setup
```bash
source setup_env.sh                 # every new shell
scripts/install_python_deps.sh      # once: pyhf (not in rootana) into .pydeps/
```
Cards that use Combine's own classes (e.g. RooMultiPdf) also need Combine's libraries. See
[AGENTS.md](AGENTS.md#environment-fresh-shell).

## Usage
```bash
pymodel <backend> <command> INPUT [options]      # or: roomodel <command> ..., hfmodel ..., zmodel ...
```
`INPUT` is a Combine datacard, or a model bundle written by `build`.

| command | Combine equivalent | example |
|---|---|---|
| `limit` | AsymptoticLimits / HybridNew LHC-limits | `roomodel limit card.txt` · `roomodel limit card.txt --method toys --toys-per-point 1000` |
| `fc` | HybridNew LHC-feldman-cousins | `roomodel fc card.txt --cl 0.9 --grid 0:4:17` |
| `fit` | FitDiagnostics / MultiDimFit singles | `roomodel fit card.txt --minos r` · `roomodel fit card.txt -t 500 --expect-signal 1 --toys-frequentist` |
| `scan` | MultiDimFit --algo grid | `roomodel scan card.txt --param r --points 50 --range 0:5` |
| `significance` | Significance | `roomodel significance card.txt --method toys` |
| `generate` | GenerateOnly | `roomodel generate card.txt -t 100 --toys-out toys.json` |
| `build` / `inspect` / `nll` / `export` | text2workspace | `roomodel build card.txt --bundle model.json` |

Common options include:
- `--rmin/--rmax`
- `--set-parameters a=1,b=2`, `--freeze-parameters`, `--freeze-nuisance-groups`,
  `--set-parameter-ranges`
- `--seed`, `--output result.json`, `--plot`

The toy options follow Combine: `-t N`, `-t -1` (Asimov), `--expect-signal`,
`--toys-frequentist`, `--bypass-frequentist-fit` and `--toys-no-systematics`.
See `pymodel <backend> <command> --help`.

Every command writes a JSON result (format described in `python/inference/results.py`).
**Check its `flags` list before using a number.** Unbracketed limits, failed fits,
parameters at a bound and excluded toys are all reported there, and are also printed at the
end of the run.

## Documentation
- [docs/statistics.md](docs/statistics.md): what is computed, with the matching Combine code
- [docs/architecture.md](docs/architecture.md): code map and design rules
- [docs/cards-and-conversion.md](docs/cards-and-conversion.md): supported datacard features per backend
- [docs/testing-and-regression.md](docs/testing-and-regression.md): validation suite and Combine fixtures
- [examples/](examples/): runnable datacards
- [AGENTS.md](AGENTS.md): rules for AI agents and contributors
