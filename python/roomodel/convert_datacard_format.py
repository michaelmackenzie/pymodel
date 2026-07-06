#!/usr/bin/env python3
"""Convert between Combine and roomodel text card formats.

Usage
-----
    python -m roomodel.convert_datacard_format <input_card> <output_card> [--direction auto|combine-to-roomodel|roomodel-to-combine]

The Combine card format uses ``shapes`` lines that reference ROOT workspaces with
``workspace:$PROCESS_pdf`` template expressions, ``imax/jmax/kmax`` headers, and
observation lines with bare counts.  The roomodel card format uses simple
``shapes * * file.root`` lines with process names that directly match PDF names
in the workspace.
"""

import os
from typing import List

from backends.path_bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path(__file__)

from backends.datacard_convert_common import (
    build_backend_converter_functions,
    run_converter_cli,
)


def _load_workspace_payload_root(shape_path: str):
    """Load a ROOT shapes file and return a dict with workspace_name_mapping.

    For roomodel, the shapes payload *is* the ROOT file itself.  We only need
    to extract the workspace-name mapping (workspace name -> name prefix used
    when objects were imported).  Since roomodel clones RooFit objects directly
    without prefixing, the mapping is typically trivial (empty prefix), but we
    still build it for completeness so ``convert_combine_to_backend`` can
    resolve process-name remapping.
    """
    import ROOT

    tf = ROOT.TFile.Open(shape_path)
    if tf is None or tf.IsZombie():
        return {}

    mapping = {}
    try:
        for key in tf.GetListOfKeys() or []:
            obj = tf.Get(key.GetName())
            if obj is not None and obj.InheritsFrom("RooWorkspace"):
                # No prefix renaming in roomodel
                mapping[str(obj.GetName())] = ""
    finally:
        tf.Close()

    return {"workspace_name_mapping": mapping}


def _convert_workspace_roomodel(
    root_paths: List[str],
    card_path: str,
    output_dir: str,
    output_prefix: str,
) -> str:
    """Convert a Combine workspace to roomodel naming conventions.

    roomodel's workspace converter reads the Combine card directly to learn
    the PDF name templates, so ``card_path`` is used as the primary input
    rather than the raw ``root_paths``.  The ``root_paths`` argument is
    accepted for interface consistency but not used directly (the card
    already references the ROOT file).

    Returns the path to the written ``.root`` output file.
    """
    from roomodel.convert_rooworkspace_shapes import convert_combine_workspace_to_roomodel

    output_root = os.path.join(output_dir, f"{output_prefix}.root")
    os.makedirs(output_dir, exist_ok=True)

    result = convert_combine_workspace_to_roomodel(
        combine_card_path=card_path,
        output_root=output_root,
    )
    return result.output_file


convert_combine_to_roomodel, convert_roomodel_to_combine = build_backend_converter_functions(
    backend_name="roomodel",
    backend_shape_ext=".root",
)


def main() -> None:
    run_converter_cli(
        backend_name="roomodel",
        backend_shape_ext=".root",
        payload_loader=_load_workspace_payload_root,
        workspace_converter=_convert_workspace_roomodel,
    )


if __name__ == "__main__":
    main()
