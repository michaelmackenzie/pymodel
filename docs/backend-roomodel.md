# roomodel Backend Guide

`roomodel` is the ROOT RooFit-based implementation.

## Key Modules

- Backend adapter: [python/backends/roomodel/implementation.py](../python/backends/roomodel/implementation.py)
- Build: [python/roomodel/build_model_from_text.py](../python/roomodel/build_model_from_text.py)
- Load summary: [python/roomodel/load_model.py](../python/roomodel/load_model.py)
- Analyze CLI + orchestration: [python/roomodel/analyze_model.py](../python/roomodel/analyze_model.py)
- Plotting helpers: [python/roomodel/analyze_plotting.py](../python/roomodel/analyze_plotting.py)
- Model I/O: [python/roomodel/model_io.py](../python/roomodel/model_io.py)

## Typical Workflow

```bash
python3 python/pymodel roomodel build examples/roomodel/simple_shapes_card.txt model.root
python3 python/pymodel roomodel load model.root
python3 python/pymodel roomodel analyze --model-file model.root --toys 10 --output analysis_output_roomodel.json
```

## Common Analyze Flags

In addition to shared analyze options:

- `--fit-mode {auto,binned,unbinned}`
- `--plot`
- `--set-parameters NAME=VALUE,...`
- `--freeze-parameters NAME,...`
- `--set-parameter-ranges NAME=MIN:MAX,...`

For limit-style scans:

- `--cls ALPHA`
- `--cls-smart-scan`
- `--feldman-cousins ALPHA` (`-fc`)
- `--fc-toys N`
- `--limit-poi-min X` (defaults to `0.0`)

`--limit-poi-min` restricts CLs and Feldman-Cousins scan domains. Keep default `0.0` for physical non-negative signal strength limits, or set a negative value to include signed POI regions.

## Plot Outputs

When `--plot` is enabled, roomodel writes artifacts under `--plot-dir` (default `plots`), including:

- dataset overlays (`dataset_XXXX*.png`)
- profile scan (`delta_nll_XXXX.png`)
- CLs curve (`dataset_XXXX_cls_band.png`) when CLs info is available
- Feldman-Cousins construction (`dataset_XXXX_feldman_cousins.png`) when FC info is available

## Example Inputs

- [examples/roomodel/simple_shapes_card.txt](../examples/roomodel/simple_shapes_card.txt)
- [examples/roomodel/simple_shapes_two_channel_card.txt](../examples/roomodel/simple_shapes_two_channel_card.txt)
- [examples/roomodel/counting_example.txt](../examples/roomodel/counting_example.txt)
- [examples/roomodel/counting_two_channel_example.txt](../examples/roomodel/counting_two_channel_example.txt)
