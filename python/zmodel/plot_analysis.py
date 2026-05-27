#!/usr/bin/env python3
import sys

# Allow importing project modules when running from zmodel.
import pathlib
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backends.plot_analysis_common import run_plot_snapshot_cli
from zmodel.analyze_plotting import plot_summary_artifacts


def main():
    run_plot_snapshot_cli(plot_summary_artifacts)


if __name__ == "__main__":
    main()
