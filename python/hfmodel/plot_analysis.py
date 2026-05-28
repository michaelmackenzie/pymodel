#!/usr/bin/env python3

# Allow importing project modules when running from hfmodel.
from backends.path_bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path(__file__)

from backends.plot_analysis_common import run_backend_plot_snapshot_cli


def main():
    run_backend_plot_snapshot_cli("hfmodel")


if __name__ == "__main__":
    main()
