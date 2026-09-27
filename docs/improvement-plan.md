# pymodel improvement plan (2026-09-27)

> **Status:** implemented the same day as the v2 rewrite, with these decisions:
> - the old formats are dropped, with no legacy support;
> - zmodel is kept;
> - Combine-style toys are the default;
> - agents may run Combine, on copies only.
>
> The findings below describe the **pre-v2** code (commit `fedce97`), which is kept for the
> record. The current design is in `architecture.md` and `statistics.md`. Open items are in
> `../todo.org`.

This plan comes from a read-only review of the whole repository. Four parallel reviews covered hfmodel, zmodel, roomodel with the shared inference code, and Combine conversion. Each review ran the examples and compared the results with independent references: scipy profile likelihood, pyhf-native `upper_limit`, exact Poisson CLs, and `text2workspace.py` on the real mumep_ana cards.

The scratch reproductions are in
`/tmp/claude-53542/-nashome-m-mmackenz/1c1afe0f-ecc0-4f6b-901e-40212914739e/scratchpad/{hf,z,roo,conv}`.
That area is temporary, so copy anything you want to keep: `ref_cls.py`, `indep3.py`, `cmp_bins.py` and `synth.txt` are good seeds for tests.

## 1. Summary of the current state

| Area | Status |
|---|---|
| Counting-model likelihood (zmodel, hfmodel) | Correct. It matches an independent Poisson×Gauss NLL. |
| lnN in hfmodel (normsys code1) and in zmodel (κ^θ) | Correct. |
| hfmodel observed asymptotic CLs | Correct apart from grid error, but the expected band is 20–40% off on shape examples because of the coarse grid. |
| roomodel asymptotic CLs | **Wrong.** There is no one-sided condition, so an excess gives a limit of 0.41 instead of 7.6. The q̃ branch is missing, so n=0 gives 0.62 instead of 0.25. q_A is not profiled. |
| zmodel CLs | **Wrong by default.** It reports 11.8 instead of 3.9: the scan runs to 99 with 9 points. It also uses hepstats q_μ, not q̃, and an inconsistent Asimov, which puts the expected median 47% above Combine. |
| Toys (all backends) | **Not frequentist-correct.** In roomodel, counting and multi-channel toys are not Poisson-fluctuated at all. zmodel and roomodel do not randomise global observables. hfmodel does, but around pre-fit θ. |
| Feldman–Cousins | The ordering is right. Toys are generated at θ̂ instead of θ̂(μ), the interval is the min/max of the grid, and q_crit is a percentile from ~100 toys, which is the source of the "jitter". It fails outright on roomodel counting models. |
| Uncertainties | hfmodel's manual Hessian understates σ by √2 (it inverts the Hessian of 2NLL) and returns σ = 1e4 or 0 at the μ=0 boundary. |
| Scan edges | Every backend returns the last grid point as "the limit" when CLs never crosses α. |
| Multi-channel | zmodel merges channels that share an observable into one pdf. roomodel imports data_obs for the first channel only. |
| zmodel save/load | Constraints are lost on reload, counting models cannot be reloaded, and binning is lost. This is the cause of the "direct vs loaded" difference noted in todo.org. |
| Shape systematics | zmodel morphing can go negative, has a kink at θ=0, and nests multiple nuisances instead of adding them. hfmodel uses code0 (piecewise linear) and keeps every JSON modifier whatever the card says. roomodel skips shape systematics silently. |
| `param` lines | Ignored by roomodel, rejected by hfmodel, and partly handled by zmodel. |
| `gs` kind | Means something different in each backend (truncated Gaussian in zmodel, lnN in hfmodel). `gs 0.9` means ±90%. |
| Asymmetric lnN | Reversed in hfmodel (reads hi/lo; Combine is down/up), dropped silently in roomodel, and crashes zmodel. |
| Combine conversion on the real cards | Only roomodel runs end to end. It matches Combine exactly on hists and funcs, but is wrong on the env card: it drops `_norm`, so the yield becomes 1.0 instead of a floating 8009. zmodel and hfmodel convert none of the 6 test cards into a buildable model. |
| Error handling | 113 `except Exception:` blocks that `pass`, `return None` or `continue`, 51 of them in `roomodel/analyze_model.py`. Most of the failures above are hidden this way. |
| Reproducibility | `--seed` is not wired through in roomodel or zmodel toys. |
| Silently ignored CLI options | `--cls-toys`, `--limit-poi-min`, `--jobs`, `--signal-strength` (for Asimov runs), `--checkpoint-freq`, `--resume-from`, `--profile-scan`, `--poi-scan-points` in various backends. |
| Duplication | The per-backend `analysis_core.py` files differ in 1464 of 1812 lines. The same inference is implemented three times, and it disagrees. |
| Environment | rootana 2.5.0 has no pyhf, so hfmodel cannot run under `setup_env.sh`. |

### Problems found in the mumep_ana Combine inputs (not pymodel)
- **The energy-scale systematic does nothing in Combine.** The cards declare `mumem_75_es_nuis param 0 1`, but the pdf formulas use `mumem_75_es`, which is constant. After `text2workspace`, the only client of `es_nuis` is its own constraint. The hists workspace does not contain the variable at all.
- **The `*_80_evt_r0101_2d` card is not really 2D.** Its pdfs are over (obs_80, obs_t_80), but data_obs is 1D. Combine therefore treats obs_t_80 as a floating parameter (value 1062.5).
- **The `mumem_75_signal_norm`-style variables are ignored by Combine.** They don't match `<pdf>_norm` (`mumem_75_signal_pdf_norm`). This is harmless today because `rate` carries the yield, but it is a trap for converters.

## 2. Guiding decisions (proposed; to be confirmed)
1. **Combine is the reference semantics.** "Correct" means that on the same card, pymodel reproduces Combine's NLL, AsymptoticLimits and HybridNew (LHC-limits and LHC-FC modes) within stated tolerances. Where pymodel deliberately differs, it says so in the output.
2. **Fail loudly.** A limit that is not bracketed, a failed fit, or an unsupported card feature is an error or an explicit flag in the output. It is never a silent default.
3. **One inference implementation.** Backends provide primitives (NLL at a point, conditional and unconditional fits, toy generation with global observables, Asimov). Test statistics, CLs, FC, scans and root-finding live once in `backends/`.
4. **One model representation.** Combine datacards are parsed into a backend-neutral intermediate representation (IR). Backends build from the IR and reject IR features they don't support.
5. **The native card grammar is Combine's grammar.** Drop `observation <bin> <n>` and `gs`, or keep them only as deprecated aliases with warnings.

## 3. Phased plan

### Phase 0: stop producing wrong numbers silently (small and urgent)
- Replace the `except Exception: pass/return None/continue` blocks with narrow exceptions plus logging. Where a fallback is kept, mark the result as degraded, for example `fit_status`, `limit_flags`.
- `interpolate_cls_crossing` (`backends/analysis_common.py:153-159`) should return `None` plus a `not_bracketed` flag. Callers should widen the scan automatically, like Combine's rMax growth, or report an error.
- Enforce fit validity. Record the status of every fit. Retry with Strategy 1/2 and random restarts, and exclude a point, with a flag, if it still fails. Never clip a negative q silently.
- Error on unsupported card content: shape systematics in roomodel, `param` in hfmodel and roomodel, asymmetric lnN, `rate -1`, and a missing observation line. Remove the silent defaults that turn `rate -` into 1.0 and a missing observation into n=0 or Asimov.
- Remove or implement every CLI option that is parsed but ignored, listed in section 1.
- Wire `--seed` through to `RooRandom`, `numpy.default_rng`, `zfit.settings.set_seed` and the pyhf toy RNG. Use distinct per-toy seeds derived from one master seed.
- Fix the environment: install pyhf into a documented venv or `--target` directory layered on rootana, and make `setup_env.sh` check that pyhf, zfit, hepstats and ROOT can all be imported.

### Phase 1: reference and validation harness (before any statistics fixes)
Build this first so that every later fix is measured, not argued.
- **`tests/reference/`** is a small, independent, pure numpy/scipy implementation for counting and binned models. It covers the Poisson × lnN, Gaussian and gamma likelihood; q̃_μ and q_0; asymptotic CLs following Cowan et al. eqs. 65–67 and `AsymptoticLimits.cc`; exact Poisson CLs; and FC by enumeration without nuisances. It does not share code with pymodel.
- **Combine fixtures.** A script such as `tests/make_combine_fixtures.sh` runs `text2workspace`, `AsymptoticLimits`, `MultiDimFit --algo grid` and `HybridNew` on a fixed set of cards and stores JSON results under `tests/fixtures/`. The user runs it, since combine runs are expensive. Add a `.gitignore` exception for `tests/fixtures/*.json`: the current `*.json` rule would drop them.
- **Test matrix.** Each case runs on every backend.
  - Counting, s=12 and b=80, with n ∈ {0, 60, 80, 100, 150}: stat-only, with lnN 1.2 on bkg, and with asymmetric lnN.
  - A multi-bin counting card.
  - A single-channel binned shape model.
  - A two-channel binned shape model.
  - An unbinned shape model with a `param` nuisance.
  - A shape-systematic model.
  - A low-background Mu2e-like model with b ≈ 0.1 and n = 0.
  - Weak signal, where the limit is above 10.
- **Checks.**
  - Asymptotic CLs agrees with the reference and with Combine to 0.5%.
  - The toy count variance equals the mean.
  - Pulls have mean 0 and width 1 away from the boundary.
  - FC reproduces the FC98 table (b=3, n=0–10, 90% CL).
  - FC coverage is ≥ 1−α within its binomial error at μ_true ∈ {0, 0.5, 1, 2}.
  - Toy CLs converges to the asymptotic result at large b.
  - The build→save→load round trip gives identical NLL at 5 random points and the same constraint count.
  - Same seed gives identical output.
- **Test runner.** Keep the existing plain-script style: pytest is not in rootana. Add a single `tests/run_all.py` with fast and slow tiers.

### Phase 2: model IR and card parsing
- Replace `backends/card_parser.py` and the parsing half of `backends/datacard_convert_common.py`. Use Combine's `DatacardParser.parseCard` and `ShapeTools` from `combine/HiggsAnalysis/CombinedLimit/python` when it is importable. Otherwise use a vendored copy of that pure-Python parser, pinned to the mu2e_dev branch. This gives the full grammar for free:
  - multi-bin `bin`/`observation` pairing
  - 4-token `shapes`, `FAKE`, and `$PROCESS/$CHANNEL/$SYSTEMATIC/$MASS`
  - the precedence (ch,proc) > (ch,*) > (*,proc) > (*,*)
  - swapped process lines
  - lnN (symmetric and asymmetric), lnU, gmN, shape/shapeN/shape?, param with asymmetric widths and ranges
  - rateParam including formulas, extArg, flatParam, discrete, group, autoMCStats, and `nuisance edit`
- IR dataclasses (`backends/model_ir.py`):
  - `Channel`: observables taken from data_obs, binned or unbinned, weights.
  - `Process`: signal flag from the id; nominal object reference; normalisation expression `rate × <obj>_norm × Π(nuisance responses) × rateParams`.
  - `Nuisance`: type, per-process effects, constraint, range.
  - `Parameter`: floating, constant or range; constraint.
  - `Envelope`: a RooMultiPdf with its category.
  - `POI`: a single shared `r`. This replaces the per-process `mu_<proc>` in zmodel.
- Each backend declares a `supported_features` set. Building from an IR that uses anything else raises an error listing the unsupported items.

### Phase 3: shared, correct inference core
Backends expose one interface, for example `BackendModel` in `backends/base.py`:
- `nll(params, data)`
- `fit(data, fixed={...}) -> FitResult(status, values, cov, nll)`
- `generate(params, n_toys, rng, global_obs=...)` returning main data and global observables
- `asimov(params, global_obs)`
- `set_global_observables(...)`

Then implement the inference once in `backends/inference/`:
- **Test statistics.** q̃_μ (one-sided, μ̂ bounded at 0, fitted `r` as the reference, never a grid minimum), q_μ, q_0, and the profile-likelihood t_μ.
- **Asymptotic CLs, following `AsymptoticLimits.cc`.**
  - Fit the data.
  - Build the Asimov at μ=0 with nuisances and global observables set to θ̂₀ (`asimovDatasetWithFit`).
  - Profile nll_A separately, including the q̃ > q_A branch.
  - Find the limit by root-finding on log CLs, not a fixed grid, and grow r_max automatically.
- **Toy CLs, following HybridNew LHC-limits.**
  - Refit at μ_test to get θ̂(μ).
  - Generate with nuisances at θ̂(μ) and randomised global observables (`generateExternalMeasurements=1`, `fitNuisances=1`), using the same test statistic for data and toys.
  - Report the CLb and CLs errors and add toys adaptively.
  - Make toy-based CLs the default for Mu2e-regime backgrounds (b ≲ few).
- **Feldman–Cousins, following HybridNew LHC-feldman-cousins.**
  - Use toys at θ̂(μ_test) with randomised global observables, and reuse toys across the ordering.
  - Accept a point when p = P(t_toy ≥ t_obs) > α, with a binomial error.
  - Interpolate the interval endpoints between grid points, and flag holes and grid-edge truncation.
  - This replaces the percentile-based q_crit and fixes the jitter.
- **Other methods.**
  - Likelihood intervals and grid scans (MultiDimFit-style) for any parameter.
  - Significance.
  - Hessian and MINOS uncertainties from each backend's native minimiser. Report "at boundary" instead of a Hessian value when μ̂ = 0.
- **Toy generation modes** from todo.org: prefit, postfit, frequentist and Bayesian nuisances, and generate-only with store and reuse. These become a single `ToyConfig` in this layer.

Each backend's `analysis_core.py` and `analysis_backend.py` then shrinks to its adapter, which removes the three divergent inference implementations. Plotting consumes one result schema.

### Phase 4: backend-specific model fixes
**hfmodel**
- Fix the Hessian covariance: `cov = 2·inv(H_2NLL)`. Better, use pyhf's minuit uncertainties and drop the manual Hessian and its 1e-8 regulariser.
- Keep the POI lower bound at 0 for q̃. `--poi-max` must not set the lower bound to −1e12.
- Asymmetric lnN is `down/up`.
- Use histosys code4p and normsys code4, which are the closest match to Combine, via `modifier_settings`.
- Honour the card: attach only the listed modifiers, apply the shape scale value, and don't keep unlisted JSON modifiers.
- Implement `param` as a constrained normfactor or an auxiliary measurement.
- Add autoMCStats as staterror or shapesys.
- Asimov defaults to μ=0 (Combine's `expectSignal=0`).

**zmodel**
- Always build a simultaneous loss, one per channel, even when channels share an observable. Toy counts are per channel.
- Rewrite shape morphing as vertical morphing, Combine-style:
  - quadratic for |θ|<1 and linear outside
  - additive over nuisances
  - clipped at ≥0
  - with the normalisation effect carried by the up/down template integrals
- Keep `BinnedData` with its own edges. Remove the conversion to bin-centre points and the hard-coded 40 bins.
- Serialise the full likelihood: constraints, all `channel_models`, yields, binning, observed counts. Counting models must be reloadable. If HS3 cannot represent something, save a rebuildable IR plus parameter values instead of a partial pdf dump, which may be the simpler design.
- Replace the hepstats calculators with the shared core, or at minimum use `qtilde=True` and a self-consistent Asimov, and build the calculator once rather than once per point.
- Default the scan range from an Asimov or σ estimate, not from the POI's upper bound of 100.

**roomodel**
- Put constraint terms inside each channel pdf, or use `ExternalConstraints`, instead of wrapping the RooSimultaneous in a RooProdPdf. That wrapping breaks every counting model that has systematics on ROOT 6.32.
- Declare global observables: constraint means become RooRealVars in a `globalObservables` set.
- Use `Extended()` in toy generation and give each channel its own data_obs, combined into one dataset with the category index.
- Implement `param`, asymmetric lnN (AsymPow), shape systematics (vertical morphing, as Combine's `ShapeTools` does for histograms and param-based for parametric pdfs), rateParam, and discrete envelopes (RooMultiPdf with a penalty).
- Do not mutate the loaded workspace in place for `--set-parameters`; clone it or use a snapshot.
- Remove the fabricated expected bands, such as q97.5 = q50 + 2(q84−q50).

### Phase 5: Combine conversion rebuilt on the IR
- **Combine → IR.** Parse the datacard (Phase 2), then introspect each referenced workspace:
  - take observables from `data_obs.get()`, not `getObservables(allVars)`
  - resolve nominal and systematic objects through the card's templates
  - pick up `<pdf>_norm`
  - record floating shape parameters with their ranges
  - collect RooMultiPdf members and the category
  - keep one workspace per channel, so the multi-file `combineCards` outputs work
- **IR → roomodel.** Reuse the original pdfs through `RooWorkspace.import` with channel-qualified renaming, import `_norm`, the systematic pdfs and the category, and fix the invalid `RecycleConflictNodes` option passed on data imports. This is the most faithful backend, and it already matches Combine on the hists and funcs cards.
- **IR → pyhf.**
  - Use exact bin integrals, not bin-centre evaluation (which is 16% off in the signal peak bin).
  - Use data_obs binning when data is binned.
  - Binning unbinned data must be an explicit, reported choice.
  - Floating shape parameters must be rejected or converted to histosys, never frozen silently.
  - Take the POI from the process id, not from "sig" in the name. Name channels and samples after the card's bin and process.
  - Never replace n_obs = 0 with the expectation.
- **IR → zfit.**
  - Build an explicit class map, with a clear error for unsupported classes. Needed: RooCrystalBall, DoubleCB with both σ, RooCBShape, RooGenericPdf, RooLandauCB, RooHistPdf/CMSHistFunc, and the custom RooStitchedPdf and MyPolyPowerPdf.
  - Walk only the pdfs referenced by the IR.
  - Replace the chained-`replace` formula translator (which produces `znp.znp.abs`, among others) with a small tokeniser, or with zfit's own formula support.
  - Keep unbounded parameters unbounded.
- **Backend → Combine.** Write from the IR, so that object names, templates and `rate` semantics agree with the files written.
- **Validation (automatic).**
  - Run `text2workspace.py` into scratch.
  - Compare `n_exp_final_bin<ch>_proc_<p>` and the bin-integrated shapes per (channel, process).
  - Compare NLL(data_obs) at the nominal point, at ±1σ for each nuisance, at r ∈ {0,1,2}, and at each envelope index.
  - Fail above a set tolerance. The prototype is `scratchpad/conv/cmp_bins.py`.
- **Grammar-coverage test.** A synthetic card (seed: `scratchpad/conv/synth.txt`) that exercises every datacard directive, plus a round trip backend → Combine → backend.

### Phase 6: files and tools for AI development
The current AI context has problems:
- `ai_repo_skeleton.txt` is 2400 lines of signatures with the logic trimmed. It is regenerated by hand and is already partly stale.
- `.github/copilot-instructions.md` and `.kilo/kilo.jsonc` tell agents to "always read" it.
- The instructions describe structure but say nothing about statistical conventions, validation, or the known-bad areas. That gap is exactly why wrong statistics accumulated unnoticed.

Proposed:
- **`AGENTS.md`**, symlinked as `CLAUDE.md`, and have `.github/copilot-instructions.md` and `kilo.jsonc` point to it so there is one source. Keep it short (~100 lines):
  - environment setup, including the pyhf layer, and the Combine environment for fixture generation
  - where things live after the refactor (IR, inference core, adapters)
  - hard rules: Combine is the reference; no silent fallbacks; every statistics change must run `tests/run_all.py --fast` and quote the before/after numbers; never report an unbracketed limit
  - don't run combine fits unless asked; the fixture script is user-run
  - a known-issues list linking to this plan
- **`docs/statistics.md`**, an authoritative conventions page:
  - the likelihood form
  - nuisance response functions per datacard type, with equations
  - global observables
  - q̃, q_0 and t_μ definitions
  - asymptotic formulas, toy-generation modes, the FC acceptance rule
  - tolerances used in tests
  - pointers to the Combine source (`AsymptoticLimits.cc`, `HybridNew.cc`, `ToyMCSamplerOpt.cc`, `ShapeTools.py`, `AsymPow.cc`)

  Agents and humans review statistics changes against this page.
- **Retire `ai_repo_skeleton.txt`.** Modern agents search the code directly. If a map is still wanted, have `generate_ai_context.py` emit a short module map (one line per module with its docstring) and check freshness in the test runner.
- **A machine-readable run summary.** Every `analyze` writes a JSON with fit statuses, flags (`not_bracketed`, `at_boundary`, `degraded_fit`, `unsupported_feature`), seeds and versions. This gives agents and scripts a reliable signal instead of parsing console text.
- **Claude Code skills or commands** under `.claude/`:
  - `/validate-stats`: runs the fast test tier and compares with fixtures.
  - `/compare-combine <card>`: converts, builds on every backend, runs the NLL and yield comparison, and prints a table. It does not run limits unless asked.
  - `/new-card-feature <directive>`: a checklist of parser → IR → each adapter's `supported_features` → test card → docs.
- **An optional small stdio MCP server** exposing `build`, `analyze`, `compare_to_combine` and `inspect_workspace` with structured JSON output. This would fit the existing Mu2e MCP ecosystem (for example the mu2e-analysis server). Only worth doing after Phase 3 stabilises the result schema.

## 4. Suggested order and size
1. Phase 0 plus the environment fix: about 1–2 sessions.
2. Phase 1 harness and reference: about 2 sessions. Fixture generation needs a user combine run.
3. Phase 2 IR and parser: about 2 sessions.
4. Phase 3 inference core: about 3–4 sessions. This is the largest; migrate hfmodel first, since it is closest to correct.
5. Phase 4 backend fixes: in parallel per backend.
6. Phase 5 conversion: roomodel first, then pyhf, then zfit.
7. Phase 6 AI files: write `AGENTS.md` and `statistics.md` right after Phase 1, and update them as the phases land.

## 5. Open questions for the author
1. Is it acceptable to break the native card grammar (drop `observation <bin> <n>` and `gs`)?
2. Keep all three backends? zmodel costs the most (serialisation, morphing, class map). An option is to keep it only for unbinned custom-pdf fits.
3. Default toy mode: Combine LHC-style frequentist, or keep a hybrid option?
4. May agents run combine to generate fixtures, or will you run the fixture script yourself?
5. Should the mumep_ana `es_nuis` and 2D-card issues be fixed there as a separate task?
