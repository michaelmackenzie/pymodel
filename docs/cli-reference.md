# CLI Reference

The unified CLI uses this shape:

```text
python3 python/pymodel <backend> <command> [options]
```

Where:

- `<backend>` is `hfmodel` or `zmodel`
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
```

## load

Inspect saved model bundle:

```bash
python3 python/pymodel hfmodel load <model.json> [-v|-vv]
python3 python/pymodel zmodel load <model.pkl> [-v|-vv]
```

## analyze

Run fit/scan/toy workflows:

```bash
python3 python/pymodel hfmodel analyze --model-file model.json --toys 10 --cls 0.05
python3 python/pymodel zmodel analyze --model-file model.pkl --toys 10 --cls 0.05
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
- `--feldman-cousins`
- `--checkpoint-freq`

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
- `--output-pkl` (compatibility alias for `--output`)
