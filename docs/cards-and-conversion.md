# Datacards and model conversion

pymodel reads **Combine datacards**, and nothing else. The card is parsed by Combine's own
`DatacardParser` (vendored in `python/third_party/combine`), so the syntax, wildcard
expansion, `nuisance edit` and `group` lines behave exactly as in Combine. Shape files are
looked up the way Combine does it: first relative to the working directory, then relative
to the card's directory. `$PROCESS`, `$CHANNEL`, `$SYSTEMATIC` and `$MASS` (`--mass`) are
substituted. Shapes are resolved with Combine's precedence: (channel, process) >
(channel, `*`) > (`*`, process) > (`*`, `*`).

The card and its shape inputs become a `ModelIR` (`python/modelspec/`). There is no
card-to-card conversion step: every backend builds its likelihood directly from the IR.
`build` saves the IR as a self-contained bundle (`model.json`, plus `model_objects.root`
when RooFit objects are referenced). `export` writes a backend's native model: a pyhf
workspace JSON, or a RooWorkspace with a ModelConfig.

## Support matrix

✓ supported · — refused with an error (never dropped silently)

| datacard content | IR feature | roomodel | zmodel | hfmodel |
|---|---|---|---|---|
| counting channels (`shapes * * FAKE` or no shapes) | `shape:counting` | ✓ | ✓ | ✓ |
| TH1 / RooDataHist templates | `shape:template` | ✓ | ✓ | ✓ |
| RooAbsPdf, fixed parameters | `shape:parametric(-histogram)` | ✓ | ✓ | ✓ |
| RooAbsPdf, floating parameters | `shape:parametric` | ✓ | ✓ (class map) | — |
| RooMultiPdf + `discrete` | `shape:envelope`, `discrete` | ✓ | ✓ | — |
| unbinned (RooDataSet) data, weighted or not | `data:unbinned`, `data:weighted` | ✓ | ✓ | — |
| `<pdf>_norm` RooRealVar | `norm:rate_param` | ✓ | ✓ | ✓ |
| `<pdf>_norm` function | `norm:ws_norm` | ✓ | ✓ (formula/product) | — |
| `lnN κ` | `norm:lnN` | ✓ | ✓ | ✓ |
| `lnN κd/κu` | `norm:asym_lnN` | ✓ | ✓ | ✓ (pyhf interpolation) |
| `lnU` | `norm:lnU` | ✓ | ✓ | — |
| `gmN N α` | `norm:gmN` | ✓ | ✓ | ✓ (one process only) |
| `rateParam` value / `[range]` | `norm:rate_param` | ✓ | ✓ | ✓ |
| `rateParam` formula | `norm:formula` | ✓ | ✓ | — |
| `shape` (histograms) | `syst:shape` | ✓ | ✓ | ✓ (pyhf interpolation) |
| `shapeN` | `syst:shapeN` | ✓ | ✓ | — |
| `shape` on RooHistPdfs (fixed; FastVerticalInterpHistPdf2) | `syst:histpdf-morph` | ✓ | ✓ | ✓ (pyhf interpolation) |
| `shape` on other RooAbsPdfs (VerticalInterpPdf), fixed parameters | `syst:pdf-morph` | ✓ | ✓ (binned) | — |
| `shape` on RooAbsPdfs with floating parameters / unbinned data | `syst:pdf-morph` | ✓ | ✓ (RooGaussian, RooExponential, RooPolynomial, RooUniform, RooGenericPdf, RooLandauCB) | — |
| `shapeN` on RooAbsPdfs | `syst:pdf-morphN`, `syst:histpdf-morphN` | ✓ (pdf-morphN: Combine library, binned) | histpdf-morphN only | — |
| `param m σ` / `m -σl/+σh` / `[range]` | `constraint:gauss` / `bifurgauss` | ✓ | ✓ | centre 0, σ 1 only |
| `flatParam`, `extArg` (value) | free / constant parameter | ✓ | ✓ | ✓ |
| `group`, `nuisance edit` (incl. `freeze`) | handled by the parser | ✓ | ✓ | ✓ |
| `autoMCStats` (TH1 channels, hist-mode 1) | `mcstats:bb-lite` | ✓ | ✓ | ✓ (BB-lite terms in numpy; no export) |
| multi-dimensional data_obs (2D/3D RooDataHist or RooDataSet; RooHistPdfs, RooAbsPdfs, RooDataHist templates over the same variables) | `obs:multidim` | ✓ | — | ✓ (fixed binned shapes) |
| shape systematics in multi-dimensional channels | refused (Combine's text2workspace fails on 2D RooHistPdf variations) | — | — | — |
| TH2/TH3 histograms in `shapes` lines | refused (Combine accepts only TH1) | — | — | — |

`pymodel <backend> inspect card.txt` shows how a card was interpreted: channels, per-process
expected yields, parameters with their roles and constraints, and notes. Notes point out
suspicious inputs, for example a `param` that affects no process, or an observation line that
disagrees with data_obs.
