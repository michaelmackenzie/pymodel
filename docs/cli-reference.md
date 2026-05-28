# CLI Reference

The unified CLI uses this shape:

```text
python3 python/pymodel <backend> <command> [options]
```

Where:

- `<backend>` is `hfmodel`, `zmodel`, or `roomodel`
- `<command>` is `build`, `load`, or `analyze`

## Backend and Command Routing

Implemented in:

- [python/pymodel_core.py](../python/pymodel_core.py)
- [python/backends/__init__.py](../python/backends/__init__.py)
- [python/backends/base.py](../python/backends/base.py)

## Common Commands

## build

Create model bundle from card:

```bash
python3 python/pymodel hfmodel build <card.txt> [output.json]
python3 python/pymodel zmodel build <card.txt> [output.pkl]
python3 python/pymodel roomodel build <card.txt> [output.root]
```

## load

Inspect saved model bundle:

```bash
python3 python/pymodel hfmodel load <model.json> [-v|-vv]
python3 python/pymodel zmodel load <model.pkl> [-v|-vv]
python3 python/pymodel roomodel load <model.root> [-v|-vv]
```

## analyze

Run fit/scan/toy workflows:

```bash
python3 python/pymodel hfmodel analyze --model-file model.json --toys 10 --cls 0.05
python3 python/pymodel zmodel analyze --model-file model.pkl --toys 10 --cls 0.05
python3 python/pymodel roomodel analyze --model-file model.root --toys 10 --cls 0.05
```

## Shared Analyze Options

Most shared options are added in [python/backends/common.py](../python/backends/common.py).

Commonly used flags:

- `--model-file` or `--input-card`
- `--toys`
- `--plot`
- `--output`
- `--report-file`
- `--cls`
- `--cls-scan-points`
- `--limit-poi-min`
- `--feldman-cousins`
- `--checkpoint-freq`

`--limit-poi-min` defaults to `0.0`, so CLs and Feldman-Cousins scans are restricted to non-negative POI by default. Use a negative value to allow negative POI limits.

## hfmodel-Specific Analyze Flags

Defined by [python/backends/hfmodel/implementation.py](../python/backends/hfmodel/implementation.py):

- `--backend {scipy,minuit,jax}`
- `--hessian-method {auto,manual,minuit,jax}`

## zmodel-Specific Analyze Flags

Defined by [python/backends/zmodel/implementation.py](../python/backends/zmodel/implementation.py):

- `--fit-mode {auto,binned,unbinned}`
- `--graph-mode {auto,on,off}`
- `--profile-scan`
- `--poi-name`
- `--promote-poi`

## roomodel-Specific Analyze Flags

Defined by [python/backends/roomodel/implementation.py](../python/backends/roomodel/implementation.py):

- `--fit-mode {auto,binned,unbinned}`
- `--set-parameters NAME=VALUE,...`
- `--freeze-parameters NAME,...`
- `--set-parameter-ranges NAME=MIN:MAX,...`
- `--plot` (saves dataset plots and profile-scan artifacts including CLs/FC plots when available)
