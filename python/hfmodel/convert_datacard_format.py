#!/usr/bin/env python3
import json
import sys
from typing import Dict, Optional

# Allow importing project modules when running from hfmodel.
import pathlib
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backends.datacard_convert_common import (
    ParsedCard,
    convert_backend_to_combine,
    convert_combine_to_backend,
    run_converter_cli,
)


def _load_workspace_payload_json(shape_path: str):
    with open(shape_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def convert_combine_to_hfmodel(
    parsed: ParsedCard,
    shapes_file: Optional[str],
    map_process_names: bool = True,
    workspace_name_mapping: Optional[Dict[str, str]] = None,
) -> str:
    return convert_combine_to_backend(
        parsed=parsed,
        shapes_file=shapes_file,
        backend_name="hfmodel",
        backend_shape_ext=".json",
        map_process_names=map_process_names,
        workspace_name_mapping=workspace_name_mapping,
    )


def convert_hfmodel_to_combine(
    parsed: ParsedCard,
    root_file: str,
    workspace_name: str,
    pdf_template: str,
    syst_template: str,
) -> str:
    return convert_backend_to_combine(
        parsed=parsed,
        backend_name="hfmodel",
        root_file=root_file,
        workspace_name=workspace_name,
        pdf_template=pdf_template,
        syst_template=syst_template,
    )


def main() -> None:
    run_converter_cli(
        backend_name="hfmodel",
        backend_shape_ext=".json",
        payload_loader=_load_workspace_payload_json,
    )


if __name__ == "__main__":
    main()
