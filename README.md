# pymodel: Unified Statistical Backend Driver

pymodel is a standalone, unified CLI for statistical workflows using three backends:

- hfmodel (pyhf-based)
- zmodel (zfit-based)
- roomodel (ROOT RooFit-based)

It supports a common build/load/analyze workflow, shared card parsing utilities, and backend-specific model execution.

## What This Repository Provides

- One top-level CLI that dispatches to either backend.
- Shared infrastructure for card parsing and common reporting/analysis helpers.
- Backend-specific implementations for model construction, fitting, and serialization.
- Conversion tools for:
	- text card format conversion (Combine <-> backend card)
	- RooWorkspace shape conversion (ROOT <-> backend payload)

## Repository Layout

- Top-level CLI:
	- python/pymodel
	- bin/pymodel
- Shared backend framework:
	- python/backends/base.py
	- python/backends/common.py
	- python/backends/card_parser.py
- Backend adapters:
	- python/backends/hfmodel/implementation.py
	- python/backends/zmodel/implementation.py
	- python/backends/roomodel/implementation.py
- Backend implementations:
	- python/hfmodel/
	- python/zmodel/
	- python/roomodel/

## Installation

Use your own Python environment or the Mu2e environment shown below.

General virtual environment example:

~~~bash
python -m venv .venv
source .venv/bin/activate
pip install pyhf zfit hepstats tensorflow uproot hist dill scipy
~~~

Mu2e environment used for this repository:

~~~bash
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh
pyenv rootana 2.5.0
~~~

## Quick Start

Use the unified CLI:

~~~bash
python3 python/pymodel --help
python3 python/pymodel hfmodel --help
python3 python/pymodel zmodel --help
python3 python/pymodel roomodel --help
~~~

Build a model from a text card:

~~~bash
python3 python/pymodel hfmodel build examples/hfmodel/simple_shapes_card.txt
python3 python/pymodel zmodel build examples/zmodel/simple_shapes_card.txt
python3 python/pymodel roomodel build examples/roomodel/simple_shapes_card.txt
~~~

Load and summarize a saved model:

~~~bash
python3 python/pymodel hfmodel load model.json
python3 python/pymodel zmodel load model.pkl
python3 python/pymodel roomodel load model.root
~~~

Run analysis:

~~~bash
python3 python/pymodel hfmodel analyze --model-file model.json
python3 python/pymodel zmodel analyze --model-file model.pkl
python3 python/pymodel roomodel analyze --model-file model.root
~~~

## Backend Output Formats

- hfmodel:
	- model bundle default: model.json
	- analysis snapshot default: analysis_output.json
- zmodel:
	- model bundle default: model.pkl
	- analysis snapshot default: analysis_output.pkl
- roomodel:
	- model bundle default: model.root
	- analysis snapshot default: analysis_output_roomodel.json

Both backends also produce an ensemble evaluation report JSON (derived from output path by default, overridable with --report-file).

## Common Analyze Options

Shared options include:

- --toys N
- --plot
- --cls ALPHA
- --cls-scan-points N
- --feldman-cousins ALPHA
- --limit-poi-min X
- --checkpoint-freq N
- --output
- --set-parameters NAME=VALUE,...
- --freeze-parameters NAME,...
- --set-parameter-ranges NAME=MIN:MAX,...
- --plot (includes per-dataset plots, delta-NLL, CLs band, and Feldman-Cousins construction when requested)

Backend-specific examples:

- hfmodel:
	- --backend {scipy,minuit,jax}
	- --hessian-method {auto,manual,minuit,jax}
- zmodel:
	- --fit-mode {auto,binned,unbinned}
	- --graph-mode {auto,on,off}
	- --profile-scan
	- --poi-name
	- --output-pkl (compatibility alias for --output)
- roomodel:
	- --fit-mode {auto,binned,unbinned}

## Plotting Existing Snapshots

Plot helper scripts are backend-specific wrappers:

~~~bash
python3 python/hfmodel/plot_analysis.py analysis_output.json --plot-dir plots_hf
python3 python/zmodel/plot_analysis.py analysis_output.pkl --plot-dir plots_z
python3 python/pymodel roomodel analyze --model-file model.root --plot --ntoys-plot 1 --output analysis_output_roomodel.json
~~~

## Card Format Conversion

Convert between Combine cards and backend cards:

~~~bash
# Combine -> hfmodel
python3 python/hfmodel/convert_datacard_format.py input_combine.txt output_hfmodel.txt --shapes-file shapes/workspace.json

# hfmodel -> Combine
python3 python/hfmodel/convert_datacard_format.py input_hfmodel.txt output_combine.txt --direction hfmodel-to-combine --root-file workspace.root

# Combine -> zmodel
python3 python/zmodel/convert_datacard_format.py input_combine.txt output_zmodel.txt --shapes-file shapes/workspace.pkl

# zmodel -> Combine
python3 python/zmodel/convert_datacard_format.py input_zmodel.txt output_combine.txt --direction zmodel-to-combine --root-file workspace.root
~~~

## RooWorkspace Shape Conversion

Convert ROOT workspaces into backend shape payloads:

~~~bash
python3 python/hfmodel/convert_rooworkspace_shapes.py input.root --output-dir shapes --bins 60
python3 python/zmodel/convert_rooworkspace_shapes.py input.root --output-dir shapes
~~~

Convert saved backend outputs back to ROOT workspaces:

~~~bash
python3 python/hfmodel/convert_rooworkspace_shapes.py model.json --output-root workspace.root --workspace-name workspace
python3 python/zmodel/convert_rooworkspace_shapes.py analysis_output.pkl --output-root workspace.root
~~~

## Regression Tests

Run CLI surface regression checks:

~~~bash
python3 tests/regression_cli_surface.py
~~~

Run example smoke regressions:

~~~bash
python3 tests/regression_examples_smoke.py
~~~

## Notes

- Relative paths in cards are recommended for portability.
- `roomodel` supports direct ROOT/RooFit workflows without converting to pyhf/zfit payloads.
