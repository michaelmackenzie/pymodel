# counting

Two counting cards (no shape inputs, so there is no `make_inputs.py`):

* `card.txt` - one bin, s = 12, b = 80 (50 + 20 + 10), n = 80.  Demonstrates symmetric lnN
  (`lumi`, `bkg_norm`), asymmetric lnN `0.95/1.08` (`bkg_asym`, Combine's down/up order), a
  gmN control-region term (`cr_stat gmN 40` with alpha 0.5, i.e. 20 events), a rateParam with
  a Gaussian `param` constraint (`scale_free`), and nuisance `group`s.
* `card_multibin.txt` - three counting bins with correlated and bin-specific (asymmetric)
  lnN; tests the multi-bin `bin`/`observation` pairing.

```bash
pymodel <backend> limit examples/counting/card.txt                     # asymptotic CLs
pymodel <backend> limit examples/counting/card.txt --method toys       # toy CLs (LHC-limits)
pymodel <backend> fc    examples/counting/card.txt                     # Feldman-Cousins
pymodel <backend> scan  examples/counting/card.txt --points 50 --range 0:4

# Combine reference (run in a scratch copy, never in the repository)
text2workspace.py card.txt -o card.root && combine -M AsymptoticLimits card.root
```

The Combine numbers are in `tests/fixtures/counting.json` and `counting_multibin.json`.
