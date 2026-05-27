# pymodel: Combined pyhf + zfit Model Driver

This package provides one central CLI that dispatches to either the pyhf-backed
implementation (`hfmodel`) or the zfit-backed implementation (`zmodel`).

Backend-specific logic lives under:

- `python/backends/hfmodel/` for pyhf (`hfmodel`) behavior
- `python/backends/zmodel/` for zfit (`zmodel`) behavior

Convenience backend namespaces are also exposed under:

- `python/hfmodel/`
- `python/zmodel/`

Shared parser and dispatch logic lives in `python/pymodel_core.py`.

Backends implement a common interface in `python/backends/base.py`.
The core runner calls backend hooks for each command parser:

- `add_build_arguments(parser)`
- `add_load_arguments(parser)`
- `add_analyze_arguments(parser)`

## Usage

Run backend-specific commands through one entrypoint:

```bash
python python/pymodel hfmodel build examples/simple_model_card_example.txt
python python/pymodel hfmodel analyze --model-file model.json

python python/pymodel zmodel build examples/simple_model_card_example.txt
python python/pymodel zmodel analyze --model-file model.pkl
```
