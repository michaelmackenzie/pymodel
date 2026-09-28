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
- **autoMCStats** (`mcstats:bb-lite`): CMSHistFunc templates and the Barlow-Beeston-lite
  terms inside the TF graph (scatter-adds per bin). Exact against the oracle (≤ 1e-15).
- **Templates with `shapeN`** (`syst:shapeN`): FastVerticalInterpHistPdf2's log-vertical morph
  (smoothAlgo < 0: the smooth-step morph of the log-ratios of the unit-normalised templates,
  0 where a template is 0, log 0 → −999, then exp and renormalisation; no floor), plus the
  same asymPow normalisation term as `shape`. One algorithm per process (mixing is refused);
  refused in autoMCStats channels. Equal to roomodel (which matches Combine) to 4e-16.
- **Shape systematics on pdfs** (`syst:pdf-morph`, `syst:histpdf-morph`, `syst:histpdf-morphN`):
  - fixed pdfs on binned data: the reference formulas (`semantics.vertical_pdf_fractions` from
    the IR's bin contents and raw integrals; `hist_pdf_fractions` for RooHistPdfs, also in log
    space for `shapeN`) in the TF graph; ΔNLL vs Combine ≤ 4e-13;
  - fixed RooHistPdfs on unbinned data: the same morphed fractions as a step density;
  - pdfs with floating parameters, or on unbinned data: VerticalInterpPdf of the
    **un-normalised** RooFit pdfs (`roofit.RawPdf`: RooFit's `evaluate()` and its
    `createIntegral`), F = Σ a_j f_j(x) (≤ 0 → 1e-15), N = Σ a_j I_j (≤ 0 → 1e-10); bin
    integrals use Σ a_j B_ij / N. Classes: RooGaussian, RooExponential, RooPolynomial,
    RooUniform (RooFit's analytic integrals, e.g. its erfc-based Gaussian integral),
    RooGenericPdf, RooLandauCB (RooIntegrator1D emulation); other classes are refused.
    Every input pdf is checked against RooFit's `getVal()` / `createIntegral()` at build time.
    `pdf_shape_syst` card_floating / card_unbinned: NLL equal to roomodel to 1e-15.
- **Envelopes** (`shape:envelope`, `discrete`; needs `PYMODEL_ROOT_LIBS=libHiggsAnalysisCombinedLimit.so`):
  each RooMultiPdf component is translated into its own zfit pdf; the graph evaluates all of
  them and selects the one of the category value (integer states are validated in python).
  `discrete_penalty` is RooMultiPdf::getCorrection() of the selected state (as roomodel:
  0.5 per non-constant variable of the pdf, the observable included);
  `inactive_parameters` are the floating parameters only non-selected components use.
  nll_main vs roomodel at 20 points per state: ≤ 3e-12 (envelope example), ≤ 4e-11 on
  `combine_mumep_40_evt_r0104_env.txt` (NLL ≈ −35000). About 0.8–1.0 ms per nll_main call.
- **Not supported:** `shapeN` on non-RooHistPdf pdfs (`syst:pdf-morphN`: VerticalInterpPdf
  smoothAlgo −1 is normalised by a numerical integral of the product morph), and pdf morphs
  of classes outside the RawPdf list (e.g. RooCrystalBall, RooAddPdf, RooChebychev).

## RooFit → zfit translation
Translation is explicit. An unlisted class raises an error naming it when the likelihood is
built.

| RooFit | zfit |
|---|---|
| RooGaussian, RooExponential (`negateCoefficient` honoured), RooChebychev, RooBernstein, RooUniform | native pdfs |
| RooCBShape; RooCrystalBall single-sided | CrystalBall |
| RooCrystalBall double-sided | GeneralizedCB (both σ) |
| RooPolynomial (lowestOrder honoured), RooHistPdf (interpolation order 0), RooGenericPdf, RooLandauCB | custom pdfs; RooGenericPdf / RooLandauCB are normalised like RooFit (RooIntegrator1D emulation, see below) |
| RooMultiPdf | one pdf per component, selected by the category (envelopes) |
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
- **Normalisation.** Pdfs without an analytic RooFit integral (RooGenericPdf, RooLandauCB)
  are normalised with an exact emulation of RooFit's default RooIntegrator1D (trapezoid
  stages + 5-point Romberg extrapolation, stop at the first stage ≥ 5 with error ≤ 1e-7
  relative or absolute; `roofit.roo_integrate`), i.e. with Combine's (imprecise) numbers:
  bit-identical to RooFit in a numpy test and ≤ 1e-15 relative in the likelihood. RooFit's
  default is not a precise integral: next to a kink (`max(0, …)` in the mumep_40 envelope) it
  is off by up to 2.5e-4, which moved the NLL by up to 1.4 before the emulation (the precise
  deviation is reported in the notes). The graph computes 15 of RooFit's 20 stages
  (16385 points); where RooFit would need more the integral is NaN and nll_main raises.
  A pdf with a non-default integrator configuration is refused.

## Validation (`tests/backend_zmodel_check.py`)
- **Numpy oracle:** ≤ 7e-16 relative (counting, templates with scales 0.5, 2 and 0.7 up to
  |θ| = 2.5, parametric histograms).
- **RooFit extended NLL** on parametric cards (unbinned, weighted, binned centre, binned
  integral): ΔNLL agreement 1.5e-10.
- **Counting limits:** identical to roomodel and hfmodel.
