# Testing and Regression

This project currently includes regression scripts focused on CLI behavior and smoke workflows.

## Test Files

- CLI surface checks: [tests/regression_cli_surface.py](../tests/regression_cli_surface.py)
- Example smoke checks: [tests/regression_examples_smoke.py](../tests/regression_examples_smoke.py)

## Run Regressions

From repository root:

```bash
python3 tests/regression_cli_surface.py
python3 tests/regression_examples_smoke.py
```

## What They Validate

## CLI surface regression

- Top-level backend choices in help text
- Backend-specific analyze flags for both backends

## Example smoke regression

- Example shape generation scripts
- Build command output format keys
- Analyze command snapshot keys
- Ensemble report key presence

## Related Example Inputs

- hfmodel example set: [examples/hfmodel](../examples/hfmodel)
- zmodel example set: [examples/zmodel](../examples/zmodel)

## Recommended Update Practice

When changing CLI flags, model serialization keys, or analyze outputs:

1. Run both regression scripts.
2. Update docs in this directory if user-facing behavior changed.
3. Keep examples synchronized with expected test inputs.
