# zmodel Backend Guide

`zmodel` is the zfit-backed implementation.

## Main Code Paths

- Backend adapter: [python/backends/zmodel/implementation.py](../python/backends/zmodel/implementation.py)
- Build: [python/zmodel/build_model_from_text.py](../python/zmodel/build_model_from_text.py)
- Load summary: [python/zmodel/load_model.py](../python/zmodel/load_model.py)
- Analyze CLI + orchestration: [python/zmodel/analyze_model.py](../python/zmodel/analyze_model.py)
- Analysis core: [python/zmodel/analysis_core.py](../python/zmodel/analysis_core.py)
- Model I/O: [python/zmodel/model_io.py](../python/zmodel/model_io.py)

## Default Artifacts

- Model bundle: `model.pkl`
- Analysis snapshot: `analysis_output.pkl`
- Ensemble report: derived from output (for example `analysis_output_ensemble_report.json`)

## Typical Workflow

```bash
python3 python/pymodel zmodel build examples/zmodel/simple_shapes_card.txt model.pkl
python3 python/pymodel zmodel load model.pkl
python3 python/pymodel zmodel analyze --model-file model.pkl --toys 10 --fit-mode auto --cls 0.05 --output analysis_output.pkl
```

## Example Inputs

- [examples/zmodel/simple_shapes_card.txt](../examples/zmodel/simple_shapes_card.txt)
- [examples/zmodel/two_category_shapes_card.txt](../examples/zmodel/two_category_shapes_card.txt)
- [examples/zmodel/counting_example.txt](../examples/zmodel/counting_example.txt)
- [examples/zmodel/mixed_observable_shapes_card.txt](../examples/zmodel/mixed_observable_shapes_card.txt)

## Conversion Utilities

- Card conversion: [python/zmodel/convert_datacard_format.py](../python/zmodel/convert_datacard_format.py)
- ROOT workspace shape conversion: [python/zmodel/convert_rooworkspace_shapes.py](../python/zmodel/convert_rooworkspace_shapes.py)
