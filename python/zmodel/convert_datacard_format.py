#!/usr/bin/env python3
import dill
from typing import List

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


def _convert_workspace_zmodel(
    root_paths: List[str],
    card_path: str,
    output_dir: str,
    output_prefix: str,
) -> str:
    """Convert ROOT workspace(s) to a zfit pickle file for zmodel.

    Uses the first ROOT file (``convert_root_file`` merges all workspaces
    inside a single file into one ``.pkl`` payload).  Returns the path to
    the written ``.pkl`` file.
    """
    from zmodel.convert_rooworkspace_shapes import convert_root_file

    if len(root_paths) > 1:
        print(
            f"Warning: {len(root_paths)} ROOT workspace files referenced; "
            "only the first will be converted for zmodel."
        )

    results = convert_root_file(
        root_path=root_paths[0],
        output_dir=output_dir,
        output_prefix=output_prefix,
        default_rate=1.0,
        include_prefix=False,
    )

    if not results:
        raise RuntimeError(
            f"zmodel workspace conversion produced no output files from '{root_paths[0]}'"
        )

    # All workspaces inside a single ROOT file are merged into one .pkl.
    return results[0].output_file


convert_combine_to_zmodel, convert_zmodel_to_combine = build_backend_converter_functions(
    backend_name="zmodel",
    backend_shape_ext=".pkl",
)


def main() -> None:
    run_converter_cli(
        backend_name="zmodel",
        backend_shape_ext=".pkl",
        payload_loader=_load_workspace_payload_dill,
        workspace_converter=_convert_workspace_zmodel,
    )


if __name__ == "__main__":
    main()
