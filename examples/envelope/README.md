# envelope

Discrete profiling: the background `bkg` is a RooMultiPdf of an exponential, a power law and a
2nd-order Bernstein polynomial (each with free shape parameters), selected by the category
`pdfindex` (`pdfindex discrete` in the datacard), with a floating normalisation `bkg_norm`.
Data: binned RooDataHist (x in [1, 11], 50 bins).

RooMultiPdf lives in Combine's library, so both building the inputs and reading them need it:

```bash
source <combine environment>        # or export PYMODEL_ROOT_LIBS=libHiggsAnalysisCombinedLimit.so
python3 examples/envelope/make_inputs.py
pymodel <backend> limit examples/envelope/card.txt
```

Combine reference (run with `--cminDefaultMinimizerStrategy 0 --X-rtd
MINIMIZER_freezeDisassociatedParams`): `tests/fixtures/envelope.json`.
