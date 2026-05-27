# Cards and Conversion Workflows

This guide covers card parsing and format conversion between Combine-style cards and backend cards.

## Shared Parsing and Conversion Code

- Shared card dataclasses/parser: [python/backends/card_parser.py](../python/backends/card_parser.py)
- Shared conversion logic: [python/backends/datacard_convert_common.py](../python/backends/datacard_convert_common.py)

## Build from Card

Both backends consume cards parsed into `CardSpec` through the shared parser:

- hfmodel build path: [python/hfmodel/build_model_from_text.py](../python/hfmodel/build_model_from_text.py)
- zmodel build path: [python/zmodel/build_model_from_text.py](../python/zmodel/build_model_from_text.py)

## Combine -> Backend Card

```bash
python3 python/hfmodel/convert_datacard_format.py input_combine.txt output_hfmodel.txt --shapes-file shapes/workspace.json
python3 python/zmodel/convert_datacard_format.py input_combine.txt output_zmodel.txt --shapes-file shapes/workspace.pkl
```

## Backend Card -> Combine

```bash
python3 python/hfmodel/convert_datacard_format.py input_hfmodel.txt output_combine.txt --direction hfmodel-to-combine --root-file workspace.root
python3 python/zmodel/convert_datacard_format.py input_zmodel.txt output_combine.txt --direction zmodel-to-combine --root-file workspace.root
```

## RooWorkspace Shape Conversion

Generate backend shapes from ROOT workspaces:

```bash
python3 python/hfmodel/convert_rooworkspace_shapes.py input.root --output-dir shapes --bins 60
python3 python/zmodel/convert_rooworkspace_shapes.py input.root --output-dir shapes
```

Convert backend outputs back into ROOT workspaces:

```bash
python3 python/hfmodel/convert_rooworkspace_shapes.py model.json --output-root workspace.root --workspace-name workspace
python3 python/zmodel/convert_rooworkspace_shapes.py analysis_output.pkl --output-root workspace.root
```

## Example Cards

- hfmodel cards: [examples/hfmodel](../examples/hfmodel)
- zmodel cards: [examples/zmodel](../examples/zmodel)
