# templates

Two channels (`ch1`, `ch2`) of TH1 templates (10 bins on [0, 10]) with

* `bkg_shape`: a background shape systematic in both channels whose Up/Down templates also
  change the normalisation (+5% / -3%), so Combine adds an asymPow normalisation term;
* `sig_width`: a signal width systematic in ch1 only that leaves the normalisation unchanged
  (Combine drops its normalisation part, |kappa - 1| < 1e-3);
* lnN `lumi` and `bkg_norm`.  autoMCStats is not used.

```bash
python3 examples/templates/make_inputs.py          # writes templates.root (fixed seed)
pymodel <backend> limit examples/templates/card.txt
pymodel <backend> fit   examples/templates/card.txt
```

Combine reference: `tests/fixtures/templates.json`.
