#!/usr/bin/env python3
"""Convert RooWorkspace shapes between Combine and roomodel layouts.

Combine workspaces name PDFs via channel-specific templates such as
``mumem_20_$PROCESS_pdf`` (resolved per process), while roomodel expects
PDFs named directly by their process name (e.g. ``signal``, ``dio``).

This tool reads a Combine datacard to learn the template mapping, opens
the referenced ROOT workspace, and clones the relevant objects into a
new workspace with the names roomodel expects.  The reverse direction
(roomodel workspace -> Combine workspace) is also supported.

Usage
-----
  # Combine -> roomodel
  python -m roomodel.convert_rooworkspace_shapes \\
      --combine-card combine_mumem_20_evt_r0101.txt \\
      --output-root converted_shapes.root

  # roomodel -> Combine
  python -m roomodel.convert_rooworkspace_shapes \\
      --roomodel-root shapes.root \\
      --output-root combine_workspace.root \\
      --channel-prefix mumem_20 \\
      --pdf-suffix _pdf

  # Inspect a Combine workspace (list PDFs, datasets, variables)
  python -m roomodel.convert_rooworkspace_shapes \\
      --combine-card combine_mumem_20_evt_r0101.txt \\
      --inspect
"""

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from backends.path_bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path(__file__)

from backends.datacard_convert_common import (
    parse_card,
    ParsedCard,
    render_process_expr,
    resolve_shape_file,
    workspace_from_expr,
    choose_shape_line,
)


class ConversionError(RuntimeError):
    pass


_ROOT_LOGGING_CONFIGURED = False


def _configure_root_logging() -> None:
    global _ROOT_LOGGING_CONFIGURED
    if _ROOT_LOGGING_CONFIGURED:
        return

    import ROOT

    try:
        ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
    except Exception:
        pass

    try:
        ROOT.gErrorIgnoreLevel = ROOT.kError
    except Exception:
        pass

    _ROOT_LOGGING_CONFIGURED = True


def _iter_roo_collection(collection):
    """Iterate over a RooFit collection (RooArgSet, RooArgList, etc.)."""
    if collection is None:
        return
    try:
        for obj in collection:
            yield obj
    except TypeError:
        create_iterator = getattr(collection, "createIterator", None)
        if not callable(create_iterator):
            return
        iterator = create_iterator()
        while True:
            obj = iterator.Next()
            if not obj:
                break
            yield obj


def _collect_workspaces(root_file) -> List:
    """Scan a TFile for all RooWorkspace objects."""
    workspaces = []
    for key in root_file.GetListOfKeys() or []:
        obj = root_file.Get(key.GetName())
        if obj is not None and obj.InheritsFrom("RooWorkspace"):
            workspaces.append(obj)
    return workspaces


@dataclass
class ProcessMapping:
    """Mapping from a Combine template-expanded PDF name to a roomodel process name."""
    combine_pdf_name: str
    roomodel_name: str
    channel: str
    rate: float


@dataclass
class WorkspaceConversionResult:
    output_file: str
    workspace_name: str
    n_pdfs: int
    n_datasets: int
    process_mappings: List[ProcessMapping]


# ---------------------------------------------------------------------------
# Combine -> roomodel
# ---------------------------------------------------------------------------

def _resolve_combine_mappings(
    parsed: ParsedCard,
    card_dir: str,
) -> Tuple[str, str, List[ProcessMapping]]:
    """Parse a Combine card and build the PDF name mapping.

    Returns (root_file_path, workspace_name, mappings).
    """
    if not parsed.shapes:
        raise ConversionError(
            "Combine card has no 'shapes' lines; cannot determine workspace file or PDF templates."
        )

    # Find the first non-data_obs shapes line to determine root file and workspace
    root_file_rel = None
    ws_name = None
    for shape in parsed.shapes:
        if shape.process.lower() == "data_obs":
            continue
        root_file_rel = shape.file_path
        if shape.extras:
            ws_name = workspace_from_expr(shape.extras[0])
        break

    if root_file_rel is None:
        raise ConversionError("No non-data_obs shapes line found in Combine card.")

    root_file_path = resolve_shape_file(root_file_rel, card_dir)

    if ws_name is None:
        ws_name = "workspace"

    mappings = []
    for process, channel, rate in zip(parsed.process_names, parsed.bin_names, parsed.rates):
        shape = choose_shape_line(parsed, process, channel)
        if shape is None or not shape.extras:
            # Counting experiment or no template; use process name directly
            mappings.append(ProcessMapping(
                combine_pdf_name=process,
                roomodel_name=process,
                channel=channel,
                rate=float(rate),
            ))
            continue

        nominal_expr = shape.extras[0]
        combine_pdf_name = render_process_expr(nominal_expr, process=process, channel=channel)
        mappings.append(ProcessMapping(
            combine_pdf_name=combine_pdf_name,
            roomodel_name=process,
            channel=channel,
            rate=float(rate),
        ))

    return root_file_path, ws_name, mappings


def _clone_pdf_recursive(src_ws, dst_ws, pdf_name: str, new_name: str) -> bool:
    """Clone a PDF and all its dependencies from src_ws into dst_ws with a new name.

    Uses RooWorkspace.import with RenameVariable to map the PDF name.
    Returns True if the PDF was successfully cloned.
    """
    import ROOT

    src_pdf = src_ws.pdf(pdf_name)
    if src_pdf is None:
        return False

    ws_import = getattr(dst_ws, "import")

    if pdf_name == new_name:
        # No renaming needed; just import with RecycleConflictNodes
        ws_import(src_pdf, ROOT.RooFit.RecycleConflictNodes())
    else:
        ws_import(
            src_pdf,
            ROOT.RooFit.RenameVariable(pdf_name, new_name),
            ROOT.RooFit.RecycleConflictNodes(),
        )

    return dst_ws.pdf(new_name) is not None


def convert_combine_workspace_to_roomodel(
    combine_card_path: str,
    output_root: str,
    output_ws_name: str = "workspace",
) -> WorkspaceConversionResult:
    """Read a Combine datacard and its workspace, produce a roomodel-style workspace.

    PDFs are cloned from the source workspace and renamed from the Combine
    template-expanded names (e.g. ``mumem_20_signal_pdf``) to the bare
    process names (e.g. ``signal``).  Datasets (``data_obs``) are cloned
    as-is.
    """
    import ROOT

    _configure_root_logging()

    combine_card_path = os.path.abspath(combine_card_path)
    card_dir = os.path.dirname(combine_card_path)
    parsed = parse_card(combine_card_path)
    root_path, src_ws_name, mappings = _resolve_combine_mappings(parsed, card_dir)

    if not os.path.isfile(root_path):
        raise ConversionError(f"Workspace ROOT file not found: {root_path}")

    src_file = ROOT.TFile.Open(root_path)
    if src_file is None or src_file.IsZombie():
        raise ConversionError(f"Could not open ROOT file: {root_path}")

    src_ws = src_file.Get(src_ws_name)
    if src_ws is None:
        # Try to find any workspace
        workspaces = _collect_workspaces(src_file)
        if workspaces:
            src_ws = workspaces[0]
            print(f"Warning: workspace '{src_ws_name}' not found, using '{src_ws.GetName()}'")
        else:
            src_file.Close()
            raise ConversionError(
                f"No RooWorkspace named '{src_ws_name}' (or any workspace) in {root_path}"
            )

    dst_ws = ROOT.RooWorkspace(output_ws_name)
    ws_import = getattr(dst_ws, "import")

    # Clone PDFs with renaming
    n_pdfs = 0
    for mapping in mappings:
        if mapping.combine_pdf_name == mapping.roomodel_name:
            ok = _clone_pdf_recursive(src_ws, dst_ws, mapping.combine_pdf_name, mapping.roomodel_name)
        else:
            ok = _clone_pdf_recursive(src_ws, dst_ws, mapping.combine_pdf_name, mapping.roomodel_name)
        if ok:
            n_pdfs += 1
        else:
            print(
                f"Warning: PDF '{mapping.combine_pdf_name}' not found in source workspace; "
                f"skipping process '{mapping.roomodel_name}'"
            )

    # Clone data_obs datasets
    n_datasets = 0
    for shape in parsed.shapes:
        if shape.process.lower() != "data_obs":
            continue
        # Resolve the data_obs name from the shapes line
        if shape.extras:
            data_name = render_process_expr(shape.extras[0], process="data_obs", channel=shape.channel)
        else:
            data_name = "data_obs"

        src_data = src_ws.data(data_name)
        if src_data is not None and bool(src_data):
            ws_import(src_data, ROOT.RooFit.RecycleConflictNodes())
            n_datasets += 1
        else:
            # Try the generic "data_obs" name
            src_data = src_ws.data("data_obs")
            if src_data is not None and bool(src_data):
                ws_import(src_data, ROOT.RooFit.RecycleConflictNodes())
                n_datasets += 1

    # If no data_obs was imported via shapes lines, try to clone it directly
    if n_datasets == 0:
        src_data = src_ws.data("data_obs")
        if src_data is not None and bool(src_data):
            ws_import(src_data, ROOT.RooFit.RecycleConflictNodes())
            n_datasets += 1

    # Write output
    output_root = os.path.abspath(output_root)
    os.makedirs(os.path.dirname(output_root) or ".", exist_ok=True)
    out_file = ROOT.TFile.Open(output_root, "RECREATE")
    if out_file is None or out_file.IsZombie():
        raise ConversionError(f"Could not create output ROOT file: {output_root}")
    out_file.cd()
    dst_ws.Write(output_ws_name)
    out_file.Close()
    src_file.Close()

    return WorkspaceConversionResult(
        output_file=output_root,
        workspace_name=output_ws_name,
        n_pdfs=n_pdfs,
        n_datasets=n_datasets,
        process_mappings=mappings,
    )


# ---------------------------------------------------------------------------
# roomodel -> Combine
# ---------------------------------------------------------------------------

def convert_roomodel_workspace_to_combine(
    roomodel_root: str,
    output_root: str,
    channel_prefix: str = "",
    pdf_suffix: str = "_pdf",
    output_ws_name: str = "workspace",
) -> WorkspaceConversionResult:
    """Convert a roomodel-style workspace to a Combine-style workspace.

    PDFs named by process (e.g. ``signal``) are cloned and renamed
    to the Combine template form (e.g. ``mumem_20_signal_pdf``).
    """
    import ROOT

    _configure_root_logging()

    roomodel_root = os.path.abspath(roomodel_root)
    src_file = ROOT.TFile.Open(roomodel_root)
    if src_file is None or src_file.IsZombie():
        raise ConversionError(f"Could not open ROOT file: {roomodel_root}")

    workspaces = _collect_workspaces(src_file)
    if not workspaces:
        src_file.Close()
        raise ConversionError(f"No RooWorkspace found in {roomodel_root}")

    src_ws = workspaces[0]
    dst_ws = ROOT.RooWorkspace(output_ws_name)
    ws_import = getattr(dst_ws, "import")

    prefix = f"{channel_prefix}_" if channel_prefix else ""
    mappings = []
    n_pdfs = 0

    for pdf_obj in _iter_roo_collection(src_ws.allPdfs()):
        process_name = str(pdf_obj.GetName())
        combine_name = f"{prefix}{process_name}{pdf_suffix}"

        ok = _clone_pdf_recursive(src_ws, dst_ws, process_name, combine_name)
        if ok:
            n_pdfs += 1
            mappings.append(ProcessMapping(
                combine_pdf_name=combine_name,
                roomodel_name=process_name,
                channel="",
                rate=1.0,
            ))

    # Clone data_obs
    n_datasets = 0
    src_data = src_ws.data("data_obs")
    if src_data is not None and bool(src_data):
        ws_import(src_data, ROOT.RooFit.RecycleConflictNodes())
        n_datasets += 1

    # Also clone any other datasets
    for data_obj in _iter_roo_collection(src_ws.allData()):
        data_name = str(data_obj.GetName())
        if data_name == "data_obs":
            continue
        ws_import(data_obj, ROOT.RooFit.RecycleConflictNodes())
        n_datasets += 1

    # Write output
    output_root = os.path.abspath(output_root)
    os.makedirs(os.path.dirname(output_root) or ".", exist_ok=True)
    out_file = ROOT.TFile.Open(output_root, "RECREATE")
    if out_file is None or out_file.IsZombie():
        raise ConversionError(f"Could not create output ROOT file: {output_root}")
    out_file.cd()
    dst_ws.Write(output_ws_name)
    out_file.Close()
    src_file.Close()

    return WorkspaceConversionResult(
        output_file=output_root,
        workspace_name=output_ws_name,
        n_pdfs=n_pdfs,
        n_datasets=n_datasets,
        process_mappings=mappings,
    )


# ---------------------------------------------------------------------------
# Inspect mode
# ---------------------------------------------------------------------------

def inspect_combine_workspace(combine_card_path: str) -> None:
    """Print the contents of a Combine workspace referenced by a datacard."""
    import ROOT

    _configure_root_logging()

    combine_card_path = os.path.abspath(combine_card_path)
    card_dir = os.path.dirname(combine_card_path)
    parsed = parse_card(combine_card_path)
    root_path, src_ws_name, mappings = _resolve_combine_mappings(parsed, card_dir)

    if not os.path.isfile(root_path):
        print(f"ERROR: Workspace ROOT file not found: {root_path}")
        return

    src_file = ROOT.TFile.Open(root_path)
    if src_file is None or src_file.IsZombie():
        print(f"ERROR: Could not open ROOT file: {root_path}")
        return

    src_ws = src_file.Get(src_ws_name)
    if src_ws is None:
        workspaces = _collect_workspaces(src_file)
        if workspaces:
            src_ws = workspaces[0]
            print(f"Warning: workspace '{src_ws_name}' not found, using '{src_ws.GetName()}'")
        else:
            print(f"ERROR: No workspace found in {root_path}")
            src_file.Close()
            return

    print(f"\n=== Workspace: {src_ws.GetName()} in {root_path} ===\n")

    print("--- PDFs ---")
    for pdf in _iter_roo_collection(src_ws.allPdfs()):
        print(f"  {pdf.GetName():40s}  {pdf.ClassName()}")

    print("\n--- Datasets ---")
    for data in _iter_roo_collection(src_ws.allData()):
        entries = data.numEntries() if hasattr(data, "numEntries") else "?"
        print(f"  {data.GetName():40s}  {data.ClassName()}  entries={entries}")

    print("\n--- Variables ---")
    for var in _iter_roo_collection(src_ws.allVars()):
        const = "const" if hasattr(var, "isConstant") and var.isConstant() else "float"
        val = var.getVal() if hasattr(var, "getVal") else "?"
        print(f"  {var.GetName():40s}  val={val}  ({const})")

    print("\n--- Combine -> roomodel process mapping ---")
    for m in mappings:
        status = "OK" if src_ws.pdf(m.combine_pdf_name) is not None else "MISSING"
        print(f"  {m.combine_pdf_name:40s} -> {m.roomodel_name:20s}  [{status}]")

    src_file.Close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert RooWorkspace shapes between Combine and roomodel layouts.  "
            "Clones PDFs and datasets with renaming as needed."
        )
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--combine-card",
        help="Path to a Combine datacard (for combine->roomodel conversion or --inspect)",
    )
    mode.add_argument(
        "--roomodel-root",
        help="Path to a roomodel ROOT workspace (for roomodel->combine conversion)",
    )

    parser.add_argument(
        "--output-root",
        default=None,
        help="Output ROOT file path (required for conversion, not for --inspect)",
    )
    parser.add_argument(
        "--output-ws-name",
        default="workspace",
        help="Name of the output RooWorkspace (default: workspace)",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print workspace contents and mapping without converting",
    )

    # roomodel -> Combine options
    parser.add_argument(
        "--channel-prefix",
        default="",
        help=(
            "Prefix for Combine PDF names when converting roomodel -> Combine "
            "(e.g. 'mumem_20' produces 'mumem_20_signal_pdf')"
        ),
    )
    parser.add_argument(
        "--pdf-suffix",
        default="_pdf",
        help="Suffix appended to process names for Combine PDF names (default: _pdf)",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.combine_card is not None:
        if args.inspect:
            inspect_combine_workspace(args.combine_card)
            return

        if args.output_root is None:
            parser.error("--output-root is required for conversion")

        result = convert_combine_workspace_to_roomodel(
            combine_card_path=args.combine_card,
            output_root=args.output_root,
            output_ws_name=args.output_ws_name,
        )
        print(f"Wrote roomodel workspace '{result.workspace_name}' to {result.output_file}")
        print(f"  PDFs cloned: {result.n_pdfs}")
        print(f"  Datasets cloned: {result.n_datasets}")
        print("  Process mappings:")
        for m in result.process_mappings:
            print(f"    {m.combine_pdf_name} -> {m.roomodel_name}")

    elif args.roomodel_root is not None:
        if args.output_root is None:
            parser.error("--output-root is required for conversion")

        result = convert_roomodel_workspace_to_combine(
            roomodel_root=args.roomodel_root,
            output_root=args.output_root,
            channel_prefix=args.channel_prefix,
            pdf_suffix=args.pdf_suffix,
            output_ws_name=args.output_ws_name,
        )
        print(f"Wrote Combine workspace '{result.workspace_name}' to {result.output_file}")
        print(f"  PDFs cloned: {result.n_pdfs}")
        print(f"  Datasets cloned: {result.n_datasets}")
        print("  Process mappings:")
        for m in result.process_mappings:
            print(f"    {m.roomodel_name} -> {m.combine_pdf_name}")


if __name__ == "__main__":
    main()
