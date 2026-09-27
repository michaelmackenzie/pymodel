# zmodel (zfit)

Code: `python/stat_backends/zmodel/`:
- `roofit.py`: RooFit → zfit class map
- `formula.py`: TFormula/RooFormula parser
- `likelihood.py`

## Supported model features
- **Data:** counting, binned, unbinned and weighted.
- **Shapes:** counting, TH1/RooDataHist templates (`shape`), and parametric pdfs translated
  from RooFit.
- **Norm terms:** `lnN` (symmetric and asymmetric), `lnU`, `gmN`, `rateParam` (plain and
  formula), and `_norm` functions built from RooFormulaVar / RooProduct / RooRealVar.
- **Constraints:** every constraint type.
- **Not supported:** envelopes, `shapeN`, and pdf-morph shape systematics.

## RooFit → zfit translation
Translation is explicit. An unlisted class raises an error naming it when the likelihood is
built.

| RooFit | zfit |
|---|---|
| RooGaussian, RooExponential (`negateCoefficient` honoured), RooChebychev, RooBernstein, RooUniform | native pdfs |
| RooCBShape; RooCrystalBall single-sided | CrystalBall |
| RooCrystalBall double-sided | GeneralizedCB (both σ) |
| RooPolynomial (lowestOrder honoured), RooHistPdf (interpolation order 0), RooGenericPdf, RooLandauCB | custom pdfs; RooGenericPdf is normalised by 256×16-point Gauss–Legendre integration |
| RooAddPdf | SumPDF; N coefficients, N−1 fractions and recursive fractions as RooFit defines them |
| RooFormulaVar, RooProduct | composed parameters |

**Formula parser.** `formula.py` is a real tokenizer and parser. It follows C++ semantics:
integer division, right-associative `^`, and `?:`. It covers TMath/std functions and `@i`,
`x[i]` and named arguments. It agrees with RooFormulaVar to 1e-16 on 61 test formulas.

**Checks when the likelihood is built.** Every translated pdf is compared with RooFit at the
nominal parameters (densities, bin fractions and integrals). The build fails above 1e-6.
The results are written to the backend notes.

## Numerical notes
- **Speed.** `nll_main` is one XLA-compiled function of the parameter vector, at about
  0.4–0.7 ms per call. `--zmodel-no-xla` disables XLA. CLI start-up takes about 12–15 s
  (TensorFlow).
- **Bin integrals** are computed per bin with `integrate`. zfit's `rel_counts` was found to be
  inaccurate at the 1e-8 to 6e-4 level.
- **Normalisation.** Translated pdfs use precise normalisation. On
  `combine_mumem_75_evt_r0104_funcs.txt`, RooFit's default numerical integrator (which Combine
  uses) mis-normalises two RooGenericPdfs, by 3.4% (rmc_ext) and 0.09% (dio). zmodel does not
  reproduce that error, and says so in the notes. Cards whose pdfs have no floating
  parameters use the IR histograms, which carry the RooFit (Combine) values.

## Validation (`tests/backend_zmodel_check.py`)
- **Numpy oracle:** ≤ 7e-16 relative (counting, templates with scales 0.5, 2 and 0.7 up to
  |θ| = 2.5, parametric histograms).
- **RooFit extended NLL** on parametric cards (unbinned, weighted, binned centre, binned
  integral): ΔNLL agreement 1.5e-10.
- **Counting limits:** identical to roomodel and hfmodel.
