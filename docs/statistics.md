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

### Feldman–Cousins (`inference/hybrid.py`, HybridNew `--LHCmode LHC-feldman-cousins`)
- **Test statistic.** t_μ, with toys at (μ, θ̂(μ)) and frequentist global observables.
- **Acceptance.** r is accepted if p(r) = P(t_toy ≥ t_obs) > 1 − CL.
- **Interval edges.** Interpolated linearly in p between neighbouring grid points.
- **Flags.** Holes in the accepted region, and accepted points at a grid edge, are flagged.

### Scans and fits (`inference/scan.py`)
- **Scans.** `scan` follows MultiDimFit `--algo grid`: points at the centres of equal
  intervals. The 68% and 95% intervals come from the 2ΔNLL crossings at 1 and 3.84.
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
- **autoMCStats** is not supported yet. Cards that use it are rejected.
