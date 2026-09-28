# two_dim

A two-dimensional (momentum, t0) channel in the style of mumep_ana's `build_model.C` with
`do_2d_fit_`: every process is a RooHistPdf over (`obs_p`, `obs_t`) made from the product of
independent momentum and time histograms, and data_obs is a 2D RooDataHist over the same
variables. `obs_t` has a variable binning, so the bin areas differ. `make_inputs.py` writes
`workspace.root` (and `th2.root` for the TH2 refusal check).

| card | content | backends |
|---|---|---|
| `card.txt` | 2D RooHistPdfs, lnN, a floating rateParam, 2D RooDataHist | roomodel, hfmodel, oracle |
| `card_param.txt` | 2D parametric signal (RooProdPdf Gaussian(p) x exponential(t0)) whose momentum depends on a `param` nuisance | roomodel |
| `card_unbinned.txt` | `card.txt` on the same events as a 2D RooDataSet | roomodel |
| `card_syst.txt` | `shape` systematics on the 2D RooHistPdfs | refused (Combine's text2workspace fails) |
| `card_th2.txt` | TH2 histograms in the `shapes` line | refused (Combine accepts only TH1) |

What Combine does, and why the refused cards are refused: docs/statistics.md, "Multi-dimensional
channels". The bins are flattened row-major (`obs_p` slowest); `pymodel <backend> fit --plot`
draws the projections onto each axis.

```bash
python3 examples/two_dim/make_inputs.py
pymodel roomodel limit examples/two_dim/card.txt
```

Combine references: `tests/fixtures/two_dim{,_param,_unbinned}.json` (with Combine's own NLL,
CachingSimNLL, at 12 parameter points: `nll_points`).
