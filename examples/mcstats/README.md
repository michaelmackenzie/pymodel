# mcstats

Barlow-Beeston-lite MC statistical uncertainties (`autoMCStats`) on TH1 templates made from
low-statistics weighted MC (`make_inputs.py` builds every bin from (number of MC events,
weight) pairs, so the TH1 errors are √Σw²).

* `ch1` (8 bins, `ch1 autoMCStats 10`: threshold 10, signal left out of the n_eff decision):
  - bins 0–3: n_eff > 10, one Gaussian parameter per bin (`prop_binch1_bin<i>`) scaling the
    total MC error;
  - bins 4–7: n_eff ≤ 10, per-process parameters: Poisson γ for processes with few MC events
    (`prop_binch1_bin5_bkg1`, n_eff = 6), Gaussian for processes with many (signal in bins 4–5,
    bkg1 in bins 4 and 6), the "Poisson not viable" Gaussian for bkg2 in bin 6 (a +2.0 and a
    −1.7 weight: content 0.3 < error 2.6), and no parameter for the empty bkg2 and signal
    templates in bin 7 (content 0, error 0);
* `ch2` (5 bins, `ch2 autoMCStats 5 1`: signal included): a background shape systematic
  (CMSHistFunc hist-mode-1 morphing), Gaussian bins 0–3 and Poisson parameters for both
  processes in bin 4.

The parameters, kinds and ranges are exactly those Combine's text2workspace prints (checked in
`tests/run_all.py`, "autoMCStats grammar/IR"). Semantics: `docs/statistics.md`, section
"autoMCStats".

```bash
python3 examples/mcstats/make_inputs.py          # writes mcstats.root (fixed seed)
pymodel <backend> inspect examples/mcstats/card.txt
pymodel <backend> limit   examples/mcstats/card.txt
pymodel <backend> fit     examples/mcstats/card.txt
```

Combine reference: `tests/fixtures/mcstats.json`, made on a workspace whose
CMSHistErrorPropagators are integrated with RooBinIntegrator (without that, this Combine
build integrates them numerically and its fits fail; see `docs/statistics.md`).
