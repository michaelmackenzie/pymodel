# hfmodel Backend Guide

`hfmodel` is the pyhf-backed implementation.

## Main Code Paths

- Backend adapter: [python/backends/hfmodel/implementation.py](../python/backends/hfmodel/implementation.py)
- Build: [python/hfmodel/build_model_from_text.py](../python/hfmodel/build_model_from_text.py)
- Load summary: [python/hfmodel/load_model.py](../python/hfmodel/load_model.py)
- Analyze CLI + orchestration: [python/hfmodel/analyze_model.py](../python/hfmodel/analyze_model.py)
- Analysis core: [python/hfmodel/analysis_core.py](../python/hfmodel/analysis_core.py)
- Model I/O: [python/hfmodel/model_io.py](../python/hfmodel/model_io.py)

## Default Artifacts

- Model bundle: `model.json`
- Analysis snapshot: `analysis_output.json`
- Ensemble report: derived from output (for example `analysis_output_ensemble_report.json`)

## Typical Workflow

```bash
python3 python/pymodel hfmodel build examples/hfmodel/simple_shapes_card.txt model.json
python3 python/pymodel hfmodel load model.json
python3 python/pymodel hfmodel analyze --model-file model.json --toys 10 --cls 0.05 --output analysis_output.json
```

## Example Inputs

- [examples/hfmodel/simple_shapes_card.txt](../examples/hfmodel/simple_shapes_card.txt)
- [examples/hfmodel/simple_shapes_two_channel_card.txt](../examples/hfmodel/simple_shapes_two_channel_card.txt)
- [examples/hfmodel/counting_example.txt](../examples/hfmodel/counting_example.txt)

## Conversion Utilities

- Card conversion: [python/hfmodel/convert_datacard_format.py](../python/hfmodel/convert_datacard_format.py)
- ROOT workspace shape conversion: [python/hfmodel/convert_rooworkspace_shapes.py](../python/hfmodel/convert_rooworkspace_shapes.py)
