#!/usr/bin/env python3
import json
from typing import List

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


def _convert_workspace_hfmodel(
    root_paths: List[str],
    card_path: str,
    output_dir: str,
    output_prefix: str,
) -> str:
    """Convert ROOT workspace(s) to a pyhf JSON file for hfmodel.

    Uses the first ROOT file as the primary input (multi-workspace files are
    handled internally by ``convert_root_workspaces_to_pyhf``).  Returns the
    path to the written ``.json`` file.
    """
    from hfmodel.convert_rooworkspace_shapes import convert_root_workspaces_to_pyhf

    if len(root_paths) > 1:
        print(
            f"Warning: {len(root_paths)} ROOT workspace files referenced; "
            "only the first will be converted for hfmodel."
        )

    results = convert_root_workspaces_to_pyhf(
        root_path=root_paths[0],
        output_dir=output_dir,
        output_prefix=output_prefix,
        bins=40,
        bin_edges=None,
        workspace_prefix=False,
    )

    if not results:
        raise RuntimeError(
            f"hfmodel workspace conversion produced no output files from '{root_paths[0]}'"
        )

    return results[0].output_file


convert_combine_to_hfmodel, convert_hfmodel_to_combine = build_backend_converter_functions(
    backend_name="hfmodel",
    backend_shape_ext=".json",
)


def main() -> None:
    run_converter_cli(
        backend_name="hfmodel",
        backend_shape_ext=".json",
        payload_loader=_load_workspace_payload_json,
        workspace_converter=_convert_workspace_hfmodel,
    )


if __name__ == "__main__":
    main()
