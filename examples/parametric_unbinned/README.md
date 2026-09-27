# parametric_unbinned

As `parametric_binned` but with an unbinned RooDataSet `data_obs`, a floating background
slope (`slope_bkg`, free) and a floating background normalisation `bkg_norm` (Combine's
`<pdf>_norm` convention; the datacard rate 1 multiplies it).  x has 50 bins: that binning is
used for Asimov datasets.

```bash
python3 examples/parametric_unbinned/make_inputs.py
pymodel <backend> limit examples/parametric_unbinned/card.txt
```

Combine reference: `tests/fixtures/parametric_unbinned.json`.
