# Architecture and Code Map

pymodel uses a shared CLI core with backend adapters.

## High-Level Flow

1. CLI parses `<backend> <command> ...`
2. Backend adapter registers backend-specific arguments
3. Core dispatches to backend methods (`build_model`, `load_summary`, `run_analysis`)
4. Backend implementation calls concrete modules in `python/hfmodel` or `python/zmodel`

## Main Modules

- CLI and dispatch:
  - [python/pymodel](../python/pymodel)
  - [python/pymodel_core.py](../python/pymodel_core.py)
- Backend registry and interface:
  - [python/backends/__init__.py](../python/backends/__init__.py)
  - [python/backends/base.py](../python/backends/base.py)
- Backend adapters:
  - [python/backends/hfmodel/implementation.py](../python/backends/hfmodel/implementation.py)
  - [python/backends/zmodel/implementation.py](../python/backends/zmodel/implementation.py)

## Shared Utility Layers

- Card parsing dataclasses and parser: [python/backends/card_parser.py](../python/backends/card_parser.py)
- Build helpers: [python/backends/builder_common.py](../python/backends/builder_common.py)
- Shared analysis reporting and console formatting:
  - [python/backends/analysis_common.py](../python/backends/analysis_common.py)
  - [python/backends/analysis_console.py](../python/backends/analysis_console.py)
  - [python/backends/analysis_reporting.py](../python/backends/analysis_reporting.py)
- Shared plotting wrapper support: [python/backends/plot_analysis_common.py](../python/backends/plot_analysis_common.py)
- Shared conversion support: [python/backends/datacard_convert_common.py](../python/backends/datacard_convert_common.py)
- Shared zfit parameter traversal: [python/backends/zfit_parameter_utils.py](../python/backends/zfit_parameter_utils.py)

## Backend-Specific Trees

- hfmodel code: [python/hfmodel](../python/hfmodel)
- zmodel code: [python/zmodel](../python/zmodel)

## Notes on Standalone Behavior

Entry scripts under `python/hfmodel` and `python/zmodel` include path bootstrap logic so they can be invoked directly during development and conversion workflows.
