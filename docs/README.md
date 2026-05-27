# pymodel Documentation

This directory contains detailed documentation for the unified `pymodel` workflow.

## Documentation Map

- [Getting Started](getting-started.md)
- [CLI Reference](cli-reference.md)
- [hfmodel Backend Guide](backend-hfmodel.md)
- [zmodel Backend Guide](backend-zmodel.md)
- [roomodel Backend Guide](backend-roomodel.md)
- [Cards and Conversion Workflows](cards-and-conversion.md)
- [Architecture and Code Map](architecture.md)
- [Testing and Regression](testing-and-regression.md)

## Primary Entrypoints

- Unified CLI script: [python/pymodel](../python/pymodel)
- Core dispatcher: [python/pymodel_core.py](../python/pymodel_core.py)
- Shell wrapper: [bin/pymodel](../bin/pymodel)

## Source Roots

- Shared backend infrastructure: [python/backends](../python/backends)
- hfmodel implementation: [python/hfmodel](../python/hfmodel)
- zmodel implementation: [python/zmodel](../python/zmodel)
- roomodel implementation: [python/roomodel](../python/roomodel)

## Example Data and Cards

- hfmodel examples: [examples/hfmodel](../examples/hfmodel)
- zmodel examples: [examples/zmodel](../examples/zmodel)
- roomodel examples: [examples/roomodel](../examples/roomodel)

## Suggested Read Order

1. [Getting Started](getting-started.md)
2. [CLI Reference](cli-reference.md)
3. Backend-specific guide:
   - [hfmodel Backend Guide](backend-hfmodel.md)
   - [zmodel Backend Guide](backend-zmodel.md)
   - [roomodel Backend Guide](backend-roomodel.md)
4. [Cards and Conversion Workflows](cards-and-conversion.md)
5. [Testing and Regression](testing-and-regression.md)
6. [Architecture and Code Map](architecture.md)
