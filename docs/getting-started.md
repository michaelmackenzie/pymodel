# Getting Started

This guide shows the shortest path from environment setup to a working build/load/analyze cycle for all backends.

## 1. Environment

Mu2e environment used by this repository:

```bash
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh
pyenv rootana 2.5.0
```

## 2. Basic CLI Checks

From the repository root:

```bash
python3 python/pymodel --help
python3 python/pymodel hfmodel --help
python3 python/pymodel zmodel --help
python3 python/pymodel roomodel --help
```

Relevant source:

- [python/pymodel](../python/pymodel)
- [python/pymodel_core.py](../python/pymodel_core.py)

## 3. hfmodel Quickstart

Build, load, and analyze:

```bash
python3 python/pymodel hfmodel build examples/hfmodel/simple_shapes_card.txt
python3 python/pymodel hfmodel load model.json
python3 python/pymodel hfmodel analyze --model-file model.json --toys 1 --output analysis_output.json
```

Examples and generators:

- [examples/hfmodel/simple_shapes_card.txt](../examples/hfmodel/simple_shapes_card.txt)
- [examples/hfmodel/simple_shapes.py](../examples/hfmodel/simple_shapes.py)

## 4. zmodel Quickstart

Build, load, and analyze:

```bash
python3 python/pymodel zmodel build examples/zmodel/simple_shapes_card.txt
python3 python/pymodel zmodel load model.pkl
python3 python/pymodel zmodel analyze --model-file model.pkl --toys 1 --output analysis_output.pkl
```

Examples and generators:

- [examples/zmodel/simple_shapes_card.txt](../examples/zmodel/simple_shapes_card.txt)
- [examples/zmodel/simple_shapes.py](../examples/zmodel/simple_shapes.py)

## 5. Plot Existing Analysis Snapshots

```bash
python3 python/hfmodel/plot_analysis.py analysis_output.json --plot-dir plots_hf
python3 python/zmodel/plot_analysis.py analysis_output.pkl --plot-dir plots_z
python3 python/pymodel roomodel analyze --model-file model.root --plot --ntoys-plot 1 --output analysis_output_roomodel.json
```

Source:

- [python/hfmodel/plot_analysis.py](../python/hfmodel/plot_analysis.py)
- [python/zmodel/plot_analysis.py](../python/zmodel/plot_analysis.py)

## 6. roomodel Quickstart

Build, load, and analyze:

```bash
python3 python/pymodel roomodel build examples/roomodel/simple_shapes_card.txt
python3 python/pymodel roomodel load model.root
python3 python/pymodel roomodel analyze --model-file model.root --toys 1 --output analysis_output_roomodel.json
```

Examples and generators:

- [examples/roomodel/simple_shapes_card.txt](../examples/roomodel/simple_shapes_card.txt)
- [examples/roomodel/simple_shapes.py](../examples/roomodel/simple_shapes.py)
- [examples/roomodel/simple_shapes_two_channel_card.txt](../examples/roomodel/simple_shapes_two_channel_card.txt)
