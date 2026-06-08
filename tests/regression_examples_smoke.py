#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path

import dill

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))


def _run(command, cwd):
    print("+", " ".join(command))
    proc = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise RuntimeError(f"Command failed with code {proc.returncode}: {' '.join(command)}")
    return proc.stdout


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_pickle(path):
    with open(path, "rb") as handle:
        return dill.load(handle)


def _assert_keys(payload, keys, label):
    missing = [key for key in keys if key not in payload]
    if missing:
        raise AssertionError(f"{label}: missing keys {missing}")


def _hfmodel_smoke(repo, cli):
    hf_examples = repo / "examples" / "hfmodel"

    _run([sys.executable, "simple_shapes.py"], cwd=hf_examples)
    _run([sys.executable, "simple_shapes_two_channel.py"], cwd=hf_examples)

    model_path = "examples/hfmodel/simple_shapes_model_regtest.json"
    output_snapshot = "examples/hfmodel/analysis_simple_regtest.json"

    _run(cli + ["hfmodel", "build", "examples/hfmodel/simple_shapes_card.txt", model_path], cwd=repo)
    model_payload = _load_json(repo / model_path)
    _assert_keys(model_payload, ["format", "workspace", "fit_metadata", "card"], "hfmodel build output")

    _run(
        cli
        + [
            "hfmodel",
            "analyze",
            "--model-file",
            model_path,
            "--toys",
            "1",
            "--cls",
            "0.05",
            "--cls-scan-points",
            "7",
            "--output",
            output_snapshot,
        ],
        cwd=repo,
    )

    snapshot = _load_json(repo / output_snapshot)
    _assert_keys(
        snapshot,
        ["format", "workspace", "model_metadata", "observed_counts_by_channel", "summaries", "config"],
        "hfmodel analysis snapshot",
    )

    summaries = snapshot.get("summaries", [])
    if len(summaries) != 1:
        raise AssertionError("hfmodel snapshot: expected one summary entry")

    report = _load_json(repo / "examples/hfmodel/analysis_simple_regtest_ensemble_report.json")
    _assert_keys(report, ["n_datasets", "runtime", "fit_quality", "poi_name"], "hfmodel ensemble report")


def _zmodel_smoke(repo, cli):
    z_examples = repo / "examples" / "zmodel"

    _run([sys.executable, "simple_shapes.py"], cwd=z_examples)

    model_path = "examples/zmodel/simple_shapes_model_regtest.pkl"
    output_snapshot = "examples/zmodel/analysis_simple_regtest.pkl"

    _run(cli + ["zmodel", "build", "examples/zmodel/simple_shapes_card.txt", model_path], cwd=repo)
    model_payload = _load_pickle(repo / model_path)
    _assert_keys(model_payload, ["format", "hs3_model", "fit_metadata", "card"], "zmodel build output")

    _run(
        cli
        + [
            "zmodel",
            "analyze",
            "--model-file",
            model_path,
            "--toys",
            "1",
            "--cls",
            "0.05",
            "--cls-scan-points",
            "7",
            "--output",
            output_snapshot,
        ],
        cwd=repo,
    )

    snapshot = _load_pickle(repo / output_snapshot)
    _assert_keys(snapshot, ["format", "fit_model", "input_data", "summaries", "config"], "zmodel analysis snapshot")

    summaries = snapshot.get("summaries", [])
    if len(summaries) != 1:
        raise AssertionError("zmodel snapshot: expected one summary entry")

    report = _load_json(repo / "examples/zmodel/analysis_simple_regtest_ensemble_report.json")
    _assert_keys(report, ["n_datasets", "runtime", "fit_quality", "poi_name"], "zmodel ensemble report")


def _roomodel_smoke(repo, cli):
    roo_examples = repo / "examples" / "roomodel"

    _run([sys.executable, "simple_shapes.py"], cwd=roo_examples)

    model_path = "examples/roomodel/simple_shapes_model_regtest.root"
    output_snapshot = "examples/roomodel/analysis_simple_regtest.json"

    _run(cli + ["roomodel", "build", "examples/roomodel/simple_shapes_card.txt", model_path], cwd=repo)

    _run(
        cli
        + [
            "roomodel",
            "analyze",
            "--model-file",
            model_path,
            "--toys",
            "1",
            "--output",
            output_snapshot,
        ],
        cwd=repo,
    )

    snapshot = _load_json(repo / output_snapshot)
    _assert_keys(
        snapshot,
        ["format", "model_file", "workspace_name", "model_name", "channels", "process_names", "summaries", "config"],
        "roomodel analysis snapshot",
    )

    summaries = snapshot.get("summaries", [])
    if len(summaries) != 1:
        raise AssertionError("roomodel snapshot: expected one summary entry")

    report_path = repo / "examples/roomodel/analysis_simple_regtest_ensemble_report.json"
    report = _load_json(report_path)
    _assert_keys(report, ["n_datasets", "runtime", "fit_quality", "poi_name"], "roomodel ensemble report")


def main():
    repo = Path(__file__).resolve().parents[1]
    cli = [sys.executable, "python/pymodel"]

    _hfmodel_smoke(repo, cli)
    _zmodel_smoke(repo, cli)
    _roomodel_smoke(repo, cli)

    print("Example smoke regression checks passed for hfmodel, zmodel, and roomodel backends.")


if __name__ == "__main__":
    main()
