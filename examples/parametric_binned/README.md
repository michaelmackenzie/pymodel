# parametric_binned

RooWorkspace `w` (in `workspace.root`) with a binned RooDataHist `data_obs` (x in [0, 10],
40 bins), a Gaussian signal whose mean is `5 * (1 + 0.01 * sig_scale)` (a RooFormulaVar), and
an exponential background with a fixed slope.  The datacard line `sig_scale param 0 1` turns
the constant workspace variable into a Gaussian-constrained nuisance (Combine semantics).
The pdfs are evaluated at bin centres times the bin width, as RooFit/Combine do.

```bash
python3 examples/parametric_binned/make_inputs.py
pymodel <backend> limit examples/parametric_binned/card.txt
```

Combine reference: `tests/fixtures/parametric_binned.json`.
