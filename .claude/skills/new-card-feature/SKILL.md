---
name: new-card-feature
description: Checklist for adding support for a Combine datacard feature (e.g. autoMCStats, pdf-morph shape systematics, a new constraint type) to pymodel end to end. Use when a card is rejected with UnsupportedFeature or UnsupportedByBackend and support should be added.
---

# Adding a datacard feature

Work through these in order, and do not skip the Combine comparison.

1. **Semantics first.** Find exactly what Combine does in its source at
   `/exp/mu2e/app/users/mmackenz/mumep/combine/HiggsAnalysis/CombinedLimit`. The places to look:
   - `python/ModelTools.py`: nuisance pdfs and ranges
   - `python/ShapeTools.py`: shape handling
   - `interface/CombineMathFuncs.h`: interpolation
   - `src/`: the C++ classes

   Write the definition into `docs/statistics.md`, with equations and a pointer to the
   Combine code.
2. **IR.** Add explicit fields to `python/modelspec/ir.py`, never a free-form string. Add a
   feature string in `ModelIR.features()`, and update `ir_from_dict` if you add dataclasses.
3. **Parsing.** Update `python/modelspec/datacard.py` to fill the IR. The vendored Combine
   parser already parses the syntax: read the fields from the `Datacard` object. Anything
   partial raises `UnsupportedFeature`.
4. **Reference formula.** Put a numpy implementation in `python/modelspec/semantics.py` and
   use it in the oracle, `python/inference/semantic_likelihood.py`, adding the feature to its
   `supported` set.
5. **Backends.** Implement it in each backend that can do it *exactly*, and add the feature to
   that backend's `supported_features`. If a backend can only approximate it, add a
   `Likelihood.notes` entry giving the bound, document it in `docs/backend-<name>.md`, and
   give the test a declared-approximation bound (see the asymmetric-lnN case in
   `tests/run_all.py`).
6. **Tests.**
   - Add a grammar case to the fast tier of `tests/run_all.py`.
   - Extend the backend checks.
   - Add an example card under `examples/`, with `make_inputs.py`, and a fixture entry in
     `tests/combine_fixtures.py`.
   - Regenerate that fixture with `tests/make_combine_fixtures.sh WORK --only <name>`.
7. **Docs.** Update the support matrix in `docs/cards-and-conversion.md`, and `todo.org`.
8. Run the validate-stats skill.
