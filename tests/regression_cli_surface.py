#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path


def _run(command, cwd):
    print("+", " ".join(command))
    proc = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise RuntimeError(f"Command failed with code {proc.returncode}: {' '.join(command)}")
    return proc.stdout


def _assert_in(text, expected, label):
    if expected not in text:
        raise AssertionError(f"Missing '{expected}' in {label}")


def main():
    repo = Path(__file__).resolve().parents[1]
    cli = [sys.executable, "python/pymodel"]

    top_help = _run(cli + ["--help"], cwd=repo)
    _assert_in(top_help, "{hfmodel,zmodel}", "top-level help")

    hf_help = _run(cli + ["hfmodel", "analyze", "--help"], cwd=repo)
    for token in ["--backend", "--hessian-method", "--output", "--plot"]:
        _assert_in(hf_help, token, "hfmodel analyze help")

    z_help = _run(cli + ["zmodel", "analyze", "--help"], cwd=repo)
    for token in ["--fit-mode", "--graph-mode", "--output-pkl", "--profile-scan"]:
        _assert_in(z_help, token, "zmodel analyze help")

    print("CLI surface regression checks passed.")


if __name__ == "__main__":
    main()
