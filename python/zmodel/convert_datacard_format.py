#!/usr/bin/env python3
import dill

# Allow importing project modules when running from zmodel.
from backends.path_bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path(__file__)

from backends.datacard_convert_common import (
    build_backend_converter_functions,
    run_converter_cli,
)


def _load_workspace_payload_dill(shape_path: str):
    with open(shape_path, "rb") as handle:
        return dill.load(handle)


convert_combine_to_zmodel, convert_zmodel_to_combine = build_backend_converter_functions(
    backend_name="zmodel",
    backend_shape_ext=".pkl",
)


def main() -> None:
    run_converter_cli(
        backend_name="zmodel",
        backend_shape_ext=".pkl",
        payload_loader=_load_workspace_payload_dill,
    )


if __name__ == "__main__":
    main()
