# Statistical conventions

This page is the authoritative definition of what pymodel computes. It follows CMS Combine
(HiggsAnalysis-CombinedLimit, `mu2e_dev` branch, commit `137dbced`). Every statement below
points to the pymodel code that implements it and, where relevant, to the Combine source
it reproduces. A change that alters any of these definitions must update this page and
must be checked with the validation suite (`tests/run_all.py`).

## 1. Likelihood

For channels c, bins or events i, processes p and parameters (r, θ):

    L(r, θ) = Π_c L_c(data_c | r, θ) × Π_k C_k(g_k | θ_k)

- **Main measurement.** Binned and counting channels use `Π_i Pois(n_i | ν_i)`. Unbinned
  channels use the extended likelihood `Pois(N | ν) Π_j f(x_j)^{w_j}`. Backends return
  `nll_main` without data-only constants (`inference/model.py`):
  - binned: `Σ_i ν_i − n_i ln ν_i`
  - unbinned: `ν − Σ_j w_j ln(ν f(x_j))`

  With this convention, every backend gives the same absolute NLL.
- **Yields.** `ν_cp = rate_cp × r^[p is signal] × Π(norm terms)`. The datacard `rate` is
  multiplied by the workspace `<pdf>_norm` when one exists. `rate -1` means "the template
  integral" and is only allowed for histograms. See `modelspec/ir.py`.
- **Parametric pdfs on binned data** are evaluated at the bin centre × bin width and are not
  renormalised. This is what RooFit and Combine do. `--bin-integration integral` uses exact
  bin integrals instead.

### Norm terms (`modelspec/semantics.py`, Combine `CombineMathFuncs.h`)

| datacard | response | constraint on θ | θ range |
|---|---|---|---|
| `lnN κ` | κ^θ | Gauss(g ∣ θ, 1), g₀ = 0 | [−7, 7] |
| `lnN κd/κu` | asymPow(θ, κd, κu): `exp(θ · logKappaForX(θ))`, smooth for \|θ\| < 0.5 | Gauss | [−7, 7] |
| `lnU κ` | κ^θ | flat | [−1, 1] |
| `gmN N α` | α · n (the datacard rate is ignored) | Pois(N ∣ n) | [0, …] |
| `rateParam` | the parameter value (or a formula) | none | as given; unbounded if no range |
| `param m σ` / `m −σl/+σh` | enters the pdfs | Gauss / BifurGauss(g ∣ θ, σ), g₀ = m | given, or m ± 4σ |

### Template shape systematics (`semantics.template_expected`, Combine `ShapeTools.py` and `FastVerticalInterpHistPdf2`)

- **Morphing.** The unit-normalised nominal template is morphed vertically with coefficient
  x = s·θ, where s is the datacard value:

      t = t₀ + Σ x/2 · [(t↑ − t↓) + (t↑ + t↓ − 2 t₀) · S(x)]

  - S is the smooth step: (3u⁵ − 10u³ + 15u)/8 for \|u\| < 1 and ±1 outside, with
    u = x / min(1, s).
  - Bins are clipped at 1e-9 and the result is renormalised.
- **Normalisation.** The template integrals change the normalisation by
  asymPow(θ, κd^s, κu^s), with κ = ∫up / ∫nominal and ∫down / ∫nominal. The term is dropped
  when both κ are within 1e-3 of 1.
- **Nuisance range.** `shape` nuisances have θ ∈ [−4, 4].

### Shape systematics on RooAbsPdfs (`semantics.vertical_pdf_fractions` / `hist_pdf_fractions`)

Datacard: a `shapes` line with a systematics pattern whose objects are RooAbsPdfs, e.g. the
mumep_ana style `workspace:mumem_75_$PROCESS_pdf workspace:mumem_75_$PROCESS_pdf_$SYSTEMATIC`,
and `shape`/`shapeN` lines with a scale s per (channel, process). IR: `Shape.pdf_systs`
(`PdfSyst`), `Shape.pdf_morph`; features `syst:pdf-morph[N]` and `syst:histpdf-morph[N]`.
Combine: `python/ShapeTools.py` `getShape` / `getPdf` / `getExtraNorm`,
`src/VerticalInterpPdf.cc`, `src/VerticalInterpHistPdf.cc` (FastVerticalInterpHistPdf2),
`src/FastTemplate_Old.cc`.

**Normalisation: none.** `getShape` reads `<pdf>_norm` only for the nominal object (not for
`<pdf>_<syst>Up/Down`), and `getExtraNorm` returns only that nominal `_norm` for RooAbsPdf
shapes: there is no asymPow term, and Up/Down `_norm` objects (which mumep_ana's
`build_model.C` writes) are **ignored**. The yield is rate × nominal `_norm` × norm terms;
pymodel adds a conversion note for every ignored Up/Down `_norm`.

**Which morph.** Combine refuses Up/Down objects of a different class than the nominal
(`Mismatched shape types`) and more than one algorithm per process (`shape` with `shapeN`);
pymodel raises `UnsupportedFeature` in both cases. The coefficient of systematic k is
x_k = s_k θ_k (`<syst>_scaled_<ch>_<proc>` = s·θ when s ≠ 1) and the smooth/quadratic region
is q = min(1, s_k) over the systematics of the process. Then, by the class of the nominal:

1. **RooHistPdf nominal → FastVerticalInterpHistPdf2** (Combine asserts that the pdfs have no
   parameters). Each pdf is sampled with `createHistogram` on the observable binning (a TH1F:
   the bin-centre density × width, *rounded to single precision*) and normalised to unit
   integral. The morph is the template one,

       t = t₀ + Σ_k x_k/2 · [(t↑ − t↓) + (t↑ + t↓ − 2t₀) · S(x_k / q)],

   followed by `CropUnderflows`: every bin whose **density** t_i / w_i is below 1e-9 is set to
   1e-9 · w_i (also bins that are positive but small), then the result is renormalised.
   `shapeN`: the same in log space (log-ratios, 0 where a template is 0; log 0 → −999), then
   exp and renormalisation, no crop. No normalisation term (unlike TH1 templates).
2. **Any other pdf → VerticalInterpPdf** (algorithm 0 for `shape`). It morphs the
   **un-normalised** pdf values f_j(x) (RooAbsPdf::getVal() without normalisation set) and
   normalises with the same morph of their integrals I_j over the observable range:

       F(x) = f₀(x) + Σ_k [c↑(x_k) f↑_k(x) + c↓(x_k) f↓_k(x) + c₀(x_k) f₀(x)]
       N    = I₀    + Σ_k [c↑(x_k) I↑_k    + c↓(x_k) I↓_k    + c₀(x_k) I₀]
       density(x) = F̃(x) / Ñ,   F̃ = F if F > 0 else 1e-15,   Ñ = N if N > 0 else 1e-10

   with, for |x| < q, c↑ = x(q + x)/2q, c↓ = −x(q − x)/2q, c₀ = −x²/q (a quadratic that is
   continuous but *not* differentiable at |x| = q; not the smooth step), and for |x| ≥ q the
   linear extrapolation x·(f↑ − f₀) (x > 0) or x·(f₀ − f↓) (x < 0).
   Because the *raw* values are morphed, the result depends on each pdf's un-normalised scale:
   e.g. a Gaussian whose width changes has raw integral σ√(2π), so the morph is not the morph
   of the normalised pdfs. `shapeN` (algorithm −1): F = f₀ Π_k κ_k^{|x_k|}, κ = f↑/f₀ (x > 0)
   or f₀/f↓ (x < 0), normalised by a numerical integral (`_forceNumInt`).
   On binned data F̃/Ñ is evaluated at the bin centres times the width, as for any pdf.
   VerticalInterpPdf ignores integration ranges, so for bin integrals (unbinned channels'
   binned expectations, `bin_integration = integral`) pymodel uses the range version of its
   analytical integral: (Σ a_j B_ij) / (Σ a_j I_j), B_ij the bin integrals of the raw pdfs,
   with ≤ 0 → 1e-10 — what VerticalInterpPdf::analyticalIntegralWN would give with ranges.

**Pitfalls of VerticalInterpPdf (Combine behaviour, reproduced exactly).** For |x| > 1 the
raw values are extrapolated linearly, so wherever f↑ (or f↓) is smaller than f₀ the morph
becomes negative for large enough |x| (a narrower Down width, a shifted mean, a steeper slope
all do this in the tails). There F̃ = 1e-15 while N still subtracts the negative area, so the
pdf is no longer normalised; a fit can exploit this. The floors are **absolute**: a pdf whose
raw values are tiny (the mumep_ana DIO RooGenericPdf has a raw integral of 1.7e-117) gets a
density of 1e-15 / 1.7e-117 ≈ 1e102 wherever the morph is ≤ 0, and fits run away (seen in
Combine and pymodel alike: ΔNLL ≈ −1.2e4). Keep the raw scale of the pdfs O(1) (or use
RooHistPdf shapes) and the variations moderate.

### autoMCStats: Barlow-Beeston-lite (`semantics.bb_lite_classify` / `cmshist_template` / `bb_lite_expected`)

Datacard: `<channel> autoMCStats <threshold> [include-signal = 0] [hist-mode = 1]` (the
channel may be a `fnmatch` pattern; Combine `DatacardParser.py`). Combine builds such a
channel from a `CMSHistErrorPropagator` over `CMSHistFunc`s (`ShapeTools.py`
`doIndividualModels`/`getPdf`/`shape2Pdf`, `src/CMSHistErrorPropagator.cc`,
`src/CMSHistFunc.cc`). pymodel reproduces it as follows (IR: `Channel.mcstats`, feature
`mcstats:bb-lite`).

**Which processes.** Every process of the channel must be a TH1 template: Combine's
propagator needs a CMSHistFunc per process, so RooDataHist/RooAbsPdf/counting processes in an
autoMCStats channel are refused (`UnsupportedFeature`), as are `shapeN` (LogQuadLinear) and
hist-mode ≠ 1. Processes with rate 0 are dropped (as in Combine). All processes, signal
included, enter the error sums and can get per-process parameters; `include-signal` only
decides whether signal processes count in the n_eff *decision* below
(`skipForErrorSum`).

**Templates (CMSHistFunc, hist-mode 1).** In an autoMCStats channel the template is *not*
normalised: the process coefficient is `r^[sig] × Π(norm terms) × asymPow(shape norms)` and
the yield is that coefficient times the raw template, so the rate is the template integral
(the datacard rate is replaced by it; Combine refuses a >1% mismatch). The shape morph
uses the Up/Down templates rescaled to the nominal integral:

    h = h₀ + Σ_k x_k/2 · [(u_k − d_k) + (u_k + d_k − 2h₀) S(x_k)],  x_k = s_k θ_k

then `h = max(h, 1e-9)` per bin, **without renormalisation** (FastVerticalInterpHistPdf2 in
channels without autoMCStats renormalises). The asymPow normalisation terms are the same as
without autoMCStats.

**Bin classification** (`setupBinPars`, done once at the nominal parameter values, with
e_pi = √sumw2 of the TH1, C_p⁰ the nominal coefficients):

1. Over the processes that count (signal only with `include-signal`):
   V = Σ C_p⁰ h_pi, E² = Σ (C_p⁰ e_pi)². If E ≤ 0: no parameter for this bin.
2. n = floor(0.5 + V²/E²). If **n > threshold** (and the error of all processes is > 0): one
   parameter `prop_bin<ch>_bin<i>` ("total"), x ∈ [−7, 7], Gauss(g | x, 1), g₀ = 0.
3. Otherwise, for **every** process p (v = h_pi, e = e_pi, no coefficient):
   - e ≤ 0, or v < 0: no parameter;
   - v > 0 and v ≥ 0.999 e: n_p = floor(0.5 + v²/e²). If n_p ≤ threshold, a Poisson
     parameter `prop_bin<ch>_bin<i>_<p>` ("poisson") γ with value n_p and range
     [½ χ²⁻¹(Φc(7); 2n_p), ½ χ²⁻¹(1 − Φc(7); 2n_p + 2)], constraint Pois(g | γ) with g₀ = n_p
     (RooPoisson without rounding; the same constraint kind as gmN). Otherwise a Gaussian
     parameter (as below);
   - v ≥ 0 and e > v ("Poisson not viable"): a Gaussian parameter
     `prop_bin<ch>_bin<i>_<p>` ("gauss"), x ∈ [−7, 7], Gauss(g | x, 1).
   With a negative threshold no parameters are made (the CMSHistFunc template semantics
   still apply).

Combine puts all of them in the group `autoMCStats` (`--freezeNuisanceGroups autoMCStats`);
pymodel does the same.

**Yields** (`updateCache`), with C_p(θ) the *current* coefficients and h_p(θ) the morphed
templates:

| bin kind | expected yield |
|---|---|
| total | ν_i = Σ_p C_p h_pi + x_i · √(Σ_p (C_p e_pi)²) |
| poisson (process p) | ν_pi = C_p h_pi · γ / n_p |
| gauss (process p) | ν_pi = C_p h_pi + x · C_p e_pi |

The width of a total bin follows the current coefficients (r, lnN, …); the errors e_pi are
the nominal ones (not morphed). The bin total is floored at 1e-9 (`CropUnderflows`), and the
extended term is the sum of the floored bins. For per-process outputs (`expected_by_process`)
the shift x_i·σ_i of a total bin is attributed to the processes in proportion to
(C_p e_pi)², as Combine's `CMSHistFuncWrapper` does; the per-process yields then add up to
the unfloored total, while `Likelihood.expected_counts` (toys, Asimov data) is floored like
the NLL.

**Minimisation.** Combine's CascadeMinimizer by default profiles the "total" parameters
analytically (`runBarlowBeeston`): for fixed other parameters, bin i contributes
ν − n ln ν + ½(x − g)², whose minimum in x solves
x² + (σ + V/σ − g)x + (V − n − gV/σ) = 0 (the larger root). That is exactly the profiled value
of x, so pymodel's fitter, which profiles these parameters with MIGRAD like any other, gives
the same profile likelihood (it also keeps x inside [−7, 7], which the analytic step does
not). No analytic shortcut is implemented.

**Combine numerical artefact (this build, ROOT 6.32).** RooRealIntegral does not use
CMSHistErrorPropagator's analytical integral (the observable also reaches the propagator
through its CMSHistFunc servers) and integrates the step function numerically. The
extended term is then off by ~2e-5 relative, in a parameter-dependent, non-smooth way;
CachingAddNLL reports "integrals don't match" and falls back to it, and MIGRAD fails on
the example card (MultiDimFit "failed", expected limits 0.0004 / 0.0009 / 0.88 / 0.88 / 0.89).
The Combine fixture (`tests/fixtures/mcstats.json`) is therefore made on a workspace patched
to integrate the propagators with RooBinIntegrator over the CMS_th1x bins, which equals the
class's own analytical integral (`tests/combine_fixtures.py` `patch_bin_integrator`).
`tests/backend_roomodel_check.py` removes the same artefact from RooFit's NLL.

### Multi-dimensional channels (`ir.Observable.axes`, `obs:multidim`)
A channel whose data_obs is a RooDataHist or RooDataSet over N = 2 or 3 variables (mumep_ana's
build_model.C `do_2d_fit_`: (obs, obs_t)) is one channel over the product binning. What Combine
does (ShapeTools.prepareAllShapes / CachingAddNLL, ROOT 6.32):

* **Binned data.** CachingAddNLL evaluates every process pdf at the coordinates of the
  RooDataHist entries (the N-D bin centres c_i) with the normalisation set of all N variables:

      NLL = sum_p nu_p - sum_i n_i log( sum_p nu_p f_p(c_i) ) + const,

  so the bin volume V_i = prod_d w_{d,i} enters only through the data-only constant
  sum_i n_i log V_i. pymodel uses nu_ip = nu_p f_p(c_i) V_i (the 1D convention with the width
  replaced by the volume); the extended term is nu_p. A 2D RooHistPdf (interpolation order 0,
  the build_model.C default) has f(c_i) = h_i / (V_i sum_j h_j), so nu_ip = nu_p h_i / sum_j h_j:
  it is exactly a template on the flattened bins. Pdfs with floating parameters (e.g. a
  RooProdPdf in (p, t0)) are evaluated at the N-D centres by roomodel.
* **Unbinned data** (RooDataSet over N variables): NLL = nu_tot - sum_j w_j log(sum_p nu_p f_p(x_j)).
* **Bin order.** Bins are flattened row-major in the order of the data_obs variables (the last
  axis fastest, as numpy and RooDataHist index them). pymodel requires every RooDataHist entry
  (data and templates) to sit at a bin centre of the variables' binning.
* **Toys and Asimov.** Binned toys are Poisson per flattened bin; Asimov data are the expected
  counts per bin; unbinned Asimov data are weighted events at the N-D bin centres
  (SinglePdfGenInfo::generateWithHisto with a TH2/TH3; Combine supports at most 3 observables,
  so pymodel refuses N > 3). Unbinned toys are generated over all N variables.
* **Refused, as in Combine or because Combine fails:** TH2/TH3 histograms in `shapes` lines
  (prepareAllShapes accepts only TH1, RooDataHist and RooAbsPdf); TH1 templates in an N-D
  channel (they would get a separate CMS_th1x observable); shape systematics in N-D channels
  (Combine would use FastVerticalInterpHistPdf2D2 for 2D RooHistPdfs, but text2workspace fails in
  ShapeTools.getPdf, `histpdf.get().second()`, which PyROOT 6.32 does not provide); a pdf that does
  not depend on every axis (RooFit would not normalise it over that axis); and a RooHistPdf
  over variables that data_obs does not have (with a 1D data_obs Combine treats the extra
  variable, e.g. obs_t, as a floating parameter).

Validation (`examples/two_dim`, fixtures `two_dim`, `two_dim_param`, `two_dim_unbinned`): NLL
differences of roomodel, hfmodel and the oracle agree with Combine's CachingSimNLL on the
text2workspace model to ~1e-13 at 12 points, and limits, intervals, grid and significance agree
within the usual fixture tolerances (observed limit 2.161 vs 2.161).

### POI
`r` scales every signal process and starts at 1. Its range is [0, 20] by default
(`--rmin/--rmax`), as in Combine's default physics model. The lower bound 0 makes r̂ ≥ 0 in
every test statistic.

## 2. Global observables and toys (`inference/toys.py`)

Every constraint term has a global observable g, whose nominal value is the datacard centre.
A `Dataset` carries both the main data and g.

| mode | CLI | nuisances used for generation | global observables of the toy |
|---|---|---|---|
| default (prior) | `-t N` | drawn from their constraint pdfs around g₀ (Gamma(N+1) for gmN) | g₀ |
| frequentist | `--toys-frequentist` | fit to data with r = `--expect-signal` | drawn from C(g ∣ θ̂) |
| bypass | `--bypass-frequentist-fit` | pre-fit values | drawn from C(g ∣ θ_prefit) |
| no systematics | `--toys-no-systematics` | nominal | g₀ |
| Asimov | `-t -1` | nominal (or θ̂ with `--toys-frequentist`) | g₀ (or g = θ̂) |

For every mode, the main data are Poisson-fluctuated per bin. For unbinned channels,
N ~ Pois(ν) and the events are sampled from the total pdf. The Asimov dataset of an unbinned
channel is a weighted dataset at the bin centres of the observable binning.

## 3. Test statistics (`inference/teststat.py`)

All are `2(NLL(conditional) − NLL(free))` with r̂ ≥ 0:

| name | used by | definition |
|---|---|---|
| q̃_μ ("LHC") | CLs limits | one-sided: 0 if r̂ > μ |
| t_μ ("PL") | Feldman–Cousins | two-sided |
| q₀ | significance | 0 if r̂ ≤ 0 |

## 4. Methods

### Asymptotic CLs (`inference/asymptotic.py`, Combine `AsymptoticLimits.cc`)
1. Fit the data freely (r ≥ 0).
2. Build the Asimov dataset at r = 0, with the nuisances from a fit to data at r = 0 and
   matching global observables (`asimovDatasetWithFit`). Fit it freely.
3. Compute CLs(r) = CLs+b / CLb, using
   - q_μ from the data and q_A from the Asimov dataset, both profiled at r;
   - CLs+b = Φc(√q_μ) and CLb = Φ(√q_A − √q_μ);
   - for q_μ > q_A, Φc((q_μ + q_A) / 2√q_A) and Φc((q_μ − q_A) / 2√q_A).
4. The observed limit comes from bracketing the crossing (starting at max(0, r̂) + 3σ and
   doubling up to 5 times) and then log-interpolated bisection. The accuracy is
   max(0.005·r, 0.0005).
5. Expected limits (newExpected) are the profiled Asimov NLL crossings at
   ½(N + Φc⁻¹(pb·(1 − CL)))², with N = Φ⁻¹(pb).

`--run blind` uses the pre-fit Asimov dataset and gives expected limits only.

### Toy CLs (`inference/hybrid.py`, HybridNew `--LHCmode LHC-limits`)
- **Test statistic.** q̃_μ evaluated identically on data and toys.
- **Toys.** s+b toys at (r, θ̂(r)), with the nuisances from a conditional fit to data.
  Background-only toys at (0, θ̂(0)); they are generated once and reused at every r. All
  toys have frequentist global observables.
- **p-values.** CLs+b = P(q_sb ≥ q_obs) and CLb = P(q_b ≥ q_obs), with binomial errors.
- **Limit.** Log-linear interpolation of CLs between the bracketing grid points, plus
  `--refine` bisection points.
- **Expected limits.** q_obs is replaced by the (1 − quantile) quantile of the b-only q
  distribution.
- **Failed toy fits** are excluded and counted in `flags`.
- **Adaptive toys** (HybridNew `--clsAcc`, `--rAbsAcc`, `--rRelAcc`). With `--cls-acc A`,
  toys are added at each point in rounds that double the count, up to
  `--max-toys-per-point` (default 20 × `--toys-per-point`). A point stops when the CLs error
  is ≤ A, or when |CLs − (1 − CL)| > 3σ, as HybridNew does. Here σ is the larger of the
  estimated error and the binomial error CLs would have at 1 − CL, so a low-statistics
  estimate far below the target cannot stop early. With `--r-abs-acc`/`--r-rel-acc`, the
  refinement continues until the propagated error of the observed limit is
  ≤ max(abs, rel · limit). It bisects while `--refine` steps remain and the bracket is
  wider than the target, and then doubles the toys at the two bracketing points. Every point
  records its toys, its toy index ranges and its `stop_reason`. Reaching the toy cap before
  the target is flagged.

### Feldman–Cousins (`inference/hybrid.py`, HybridNew `--LHCmode LHC-feldman-cousins`)
- **Test statistic.** t_μ, with toys at (μ, θ̂(μ)) and frequentist global observables.
- **Acceptance.** r is accepted if p(r) = P(t_toy ≥ t_obs) > 1 − CL.
- **Interval edges.** Interpolated linearly in p between neighbouring grid points.
- **Flags.** Holes in the accepted region, and accepted points at a grid edge, are flagged.
- **Adaptive toys.** `--p-acc A` adds toys as for CLs until the p-value error is ≤ A, or
  p is more than 3σ from 1 − CL, or the cap is reached (flagged).

### Toy seeds, parallel toys and job splitting (`inference/parallel.py`)
- **Per-toy seeds.** Toy i of a stream is drawn from
  `SeedSequence(--seed, spawn_key=(stream, point, i, purpose))`. The streams are fit/generate,
  s+b at r, b-only, FC at r and q₀. The point is the IEEE bit pattern of r. The purpose
  separates generation, the free fit, and the fit at each tested r. Minuit restarts use the
  fitter's random generator, which is reseeded from the same keys before every fit, including
  the fits of the observed data. A toy and every fit on it therefore depend only on
  (seed, key). They do not depend on the process, the order, or the chunking.
- **`--jobs N`.** Tasks, meaning chunks of toys (and one nuisance per task for `impacts`),
  run in a `multiprocessing` pool with the "spawn" start method. Every worker rebuilds the
  model and its likelihood from the command arguments, because backend objects (ROOT,
  TensorFlow) must not be forked or pickled. The fitter bounds are sent with every call. The
  fits of the observed data and all the adaptive and refinement decisions run in the main
  process. Results are bit-identical for any N. On roomodel (counting example, 5 points ×
  2000 s+b and b toys), the wall time goes from 107 s with 1 job to 20 s with 8 and 15 s with
  16. Each worker spends a few seconds on its start-up.
- **Splitting and merging** (HybridNew `--saveHybridResult` + `hadd` + `--readHybridResults`).
  - **Split jobs.** `--points r1,r2` computes only those points, without refinement.
    `--toy-chunk I/N` computes toys [I·T/N, (I+1)·T/N) of the T = `--toys-per-point` at every
    point of an explicit `--grid`. Either option needs `--save-toy-results FILE`, which
    writes the raw per-point arrays (q_obs, q_sb, q_b or t_toys, failed counts, toy index
    ranges and seed).
  - **Merging.** `merge FILES` joins points with the same r and computes the limit or
    interval with the same readout code. It refuses to merge files that share a toy
    (seed, index), which would double count identical toys, and files whose observed test
    statistics differ.
  - **Equality with a single run.** A merged result equals the single-process result on the
    same grid exactly, because the toy index sets are the same.

### Impacts (`inference/impacts.py`, `combineTool.py -M Impacts`)
1. **Initial fit.** A free fit, then the 68% interval of r from its profile.
2. **Nuisance intervals.** For every floating nuisance θ (constrained or not, including
   rateParams), θ̂ and [θ_lo, θ_hi]. By default (`--errors profile`, Combine `--algo impact`
   with `--robustFit 1`) these are the crossings of the profile 2ΔNLL(θ), found by
   bracketing and Brent's method on fits with θ fixed. `--errors hesse` uses θ̂ ∓ the Hesse
   error instead. "68%" is Combine's `--algo singles` level, 2ΔNLL = χ²₁ quantile(0.68) =
   0.98894, not 1. With 1 the crossings came out 0.55% wider than Combine's.
3. **Shifts of r.** Fits with θ fixed at θ_lo and at θ_hi, starting from the best fit, give
   Δr± = r(θ_hi/lo) − r̂. The impact is max(|Δr+|, |Δr−|), and the parameters are sorted by it.
4. **Pulls** as plotImpacts.py draws them.
   - **Pre-fit interval.** The 68% interval of the constraint alone: g ∓ σ, the bifurcated
     widths, or the likelihood interval of θ − g log θ for gmN.
   - **Pull.** (θ̂ − g) divided by the pre-fit width on its side, with the post-fit errors
     scaled the same way.
   - **Constraint.** (θ_hi − θ_lo) / (pre-fit width).
5. **Crossings outside the range.** A crossing that does not exist inside the parameter
   range is replaced by the range limit and flagged (Combine does the same). If r̂ is at its
   lower bound, the impacts are truncated there. This is flagged too: use `--rmin` < 0, as
   with Combine's `--rMin`.

Validation (`tests/fixtures/{counting,templates}_impacts.json`, `combineTool.py` with
`--robustFit 1 --cminDefaultMinimizerTolerance 0.001`):
- **Crossings.** The θ crossings agree to < 1e-3.
- **Shifts of r.** r(θ_lo/hi) agrees to ≤ 1.2e-3, and the impacts too.
- **Minimiser tolerance.** With Combine's default tolerance (0.1), the fixed-nuisance fits
  are only good to ~5e-3 in r (templates bkg_norm: r = 1.0210 at 0.1, 1.0246 at 0.001).
- **Independent check.** On counting, an independent scipy calculation agrees to 4e-4
  (semantic) and 1.2e-3 (roomodel), the limit of Minuit's tolerance.

### Scans and fits (`inference/scan.py`)
- **Scans.** `scan` follows MultiDimFit `--algo grid`: points at the centres of equal
  intervals. The 68% and 95% intervals come from the 2ΔNLL crossings at 1 and 3.84.
- **2D scans** (`--param a,b`, MultiDimFit `--algo grid -P a -P b`).
  - **Grid.** ceil(√N) points per axis at interval centres, as Combine. `--range lo:hi,lo:hi`
    sets the ranges.
  - **Fits.** Every other parameter is profiled. The points are fitted in order of their
    distance from the best fit, each starting from the nearest fitted point.
  - **Contours.** The 68% and 95% contours are the 2ΔNLL = 2.30 and 5.99 lines (χ² with 2
    dof), linearly interpolated on the grid (contourpy). A contour that is not closed inside
    the range is flagged.
  - **Validation.** On templates (r, bkg_norm; 20×20), 2ΔNLL agrees with Combine to 1e-4
    (semantic, roomodel). hfmodel differs by up to 0.03, from pyhf's interpolation.
- **Fits.** `fit` reports Hesse errors, optional MINOS errors, pulls (θ̂ − g)/σ and
  correlations.
- **Parameters at a bound** are flagged, because their Hesse errors are meaningless.

### Envelopes (discrete profiling)
The fitter minimises over every category state. It adds 0.5 per floating parameter of the
selected pdf to the NLL (RooMultiPdf's default correction), and floats only the parameters
of the selected pdf.

## 5. Minimisation (`inference/fitting.py`)
- **Minimiser.** iminuit MIGRAD, with strategy 1 and tolerance 0.01 by default (ROOT's Minuit
  default). A tolerance of 0.1 was measured to stop flat envelope fits about 0.003 in NLL above
  the minimum, which moved r̂ from 52.8 to 56.0 on the mumep_40 envelope card.
- **Retries.** A failed fit is retried with strategy 2, then from perturbed starts. If it
  still fails, the result has `valid = False` and every method that uses it adds a flag.

## 6. Known differences from Combine
- **hfmodel (pyhf) interpolation.** pyhf cannot express Combine's smooth-step morphing or
  the asymPow interior (\|θ\| < 0.5) exactly. The residual differences are listed in each
  model's notes and in `docs/backend-hfmodel.md`.
- **Toy CLs b-only toys** are reused across r. Combine regenerates them per point. This is
  statistically equivalent, but correlates the errors of neighbouring points.
- **Toy seeds.** Toys are seeded per toy (see above), so for a given `--seed` the toys differ
  from the sequential draws of Combine.
- **autoMCStats** is exact in all backends (section 1), but this Combine build integrates the
  propagator numerically (see "Combine numerical artefact" above), so unpatched Combine fits of
  autoMCStats cards can differ from pymodel or fail.
