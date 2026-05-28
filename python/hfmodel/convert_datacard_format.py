#!/usr/bin/env python3
import json

# Allow importing project modules when running from hfmodel.
from backends.path_bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path(__file__)

from backends.datacard_convert_common import (
    build_backend_converter_functions,
    run_converter_cli,
)


def _load_workspace_payload_json(shape_path: str):
    with open(shape_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


convert_combine_to_hfmodel, convert_hfmodel_to_combine = build_backend_converter_functions(
    backend_name="hfmodel",
    backend_shape_ext=".json",
)


def main() -> None:
    run_converter_cli(
        backend_name="hfmodel",
        backend_shape_ext=".json",
        payload_loader=_load_workspace_payload_json,
    )


if __name__ == "__main__":
    main()
