# Getting started

```bash
cd /path/to/pymodel
source setup_env.sh                 # rootana 2.5.0, pymodel on PATH/PYTHONPATH
scripts/install_python_deps.sh      # once: pyhf for hfmodel
```

## A counting experiment
```bash
roomodel inspect examples/counting/card.txt       # how the card was interpreted
roomodel limit   examples/counting/card.txt       # asymptotic CLs (AsymptoticLimits)
roomodel fit     examples/counting/card.txt --minos r
roomodel scan    examples/counting/card.txt --points 40 --range 0:4 --plot
```
Every command writes `pymodel_<command>.json` (use `-o` to choose the name). It prints
**FLAGS** if anything makes the result suspect.

## Shapes
```bash
cd examples/parametric_unbinned && python3 make_inputs.py && cd -
roomodel limit examples/parametric_unbinned/card.txt
zmodel   limit examples/parametric_unbinned/card.txt       # same model, zfit likelihood
roomodel fit   examples/parametric_unbinned/card.txt -t 200 --expect-signal 1 --toys-frequentist --plot
```

## Low backgrounds: use toys
Asymptotic formulas fail for b ≲ 1. Use toy CLs or Feldman–Cousins instead:
```bash
roomodel limit examples/low_background/card_n0.txt --method toys --toys-per-point 2000
roomodel fc    examples/low_background/card_n0.txt --cl 0.9 --grid 0:5:11 --toys-per-point 1000
```
For comparison, Combine gives 2.11 asymptotic and 2.93 with toys for `card_n0`.

## Your own Combine cards
Run from the directory that the card's shape paths are relative to, exactly as for Combine:
```bash
cd mumep_ana/analysis/combine
roomodel limit datacards/combine_mumem_75_evt_r0104_hists.txt
```
Cards that use Combine classes (RooMultiPdf, RooLandauCB, …) need Combine's environment and
`export PYMODEL_ROOT_LIBS=libHiggsAnalysisCombinedLimit.so`.

To reuse a model across many runs, save it once as a self-contained bundle:
`roomodel build card.txt --bundle model.json`, then `hfmodel limit model.json`.
