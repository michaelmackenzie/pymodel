# pdf_shape_syst

Shape systematics on RooAbsPdfs: the `shapes` line has a `$SYSTEMATIC` pattern whose objects
are pdfs, as in mumep_ana (`workspace:mumem_75_$PROCESS_pdf workspace:mumem_75_$PROCESS_pdf_$SYSTEMATIC`).
`make_inputs.py` writes `workspace.root` (RooGaussian signal with a mean shift `sigshift` and
a width change `sigwidth`, RooExponential background with a slope change `bkgslope`, plus
`_norm` constants for every pdf) and `workspace_hist.root` (the same shapes as fixed
RooHistPdfs).

| card | Combine morph | backends |
|---|---|---|
| `card.txt` | VerticalInterpPdf (scales 1 and 0.5 on the signal) | roomodel, zmodel, oracle |
| `card_histpdf.txt` | FastVerticalInterpHistPdf2 (RooHistPdfs) | all (hfmodel: declared smooth-region approximation, mixed scales) |
| `card_floating.txt` | VerticalInterpPdf, signal mean depends on the `param` nuisance `sig_scale` | roomodel |
| `card_unbinned.txt` | VerticalInterpPdf on a RooDataSet | roomodel |

What Combine does (docs/statistics.md, "Shape systematics on RooAbsPdfs"): no normalisation
effect (the Up/Down `_norm` objects are ignored; `inspect` lists them in the notes); a
RooHistPdf nominal is morphed like a template; any other pdf is morphed through its
**un-normalised** values with VerticalInterpPdf's quadratic/linear interpolation. The
variations here are kept moderate: large ones make the linear extrapolation for |s·θ| > 1
negative where the pdf still has probability, and fits can then exploit Combine's floors.

```bash
python3 examples/pdf_shape_syst/make_inputs.py
pymodel roomodel limit examples/pdf_shape_syst/card.txt
```

Combine references: `tests/fixtures/pdf_shape_syst{,_histpdf,_floating,_unbinned}.json`.
