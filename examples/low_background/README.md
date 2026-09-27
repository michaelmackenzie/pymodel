# low_background

A Mu2e-like single-channel search: 1.0 expected signal event at r = 1 (10% lnN), background
0.2 events (30% lnN), with the observed count n = 0 (`card_n0.txt`) or n = 1 (`card_n1.txt`).
No shape inputs, so there is no `make_inputs.py`.

With b ~ 0.2 the asymptotic approximation does not hold (compare `limit` with and without
`--method toys`); use toy CLs or Feldman-Cousins here.

```bash
pymodel <backend> limit examples/low_background/card_n0.txt                   # asymptotic (not valid here)
pymodel <backend> limit examples/low_background/card_n0.txt --method toys     # toy CLs
pymodel <backend> fc    examples/low_background/card_n1.txt                   # FC interval
```

Combine references (including HybridNew LHC-limits and LHC-feldman-cousins points):
`tests/fixtures/low_background_n0.json`, `low_background_n1.json`.
