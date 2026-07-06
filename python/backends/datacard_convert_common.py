import argparse
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple


@dataclass
class ShapeLine:
    process: str
    channel: str
    file_path: str
    extras: List[str]


@dataclass
class ParsedCard:
    shapes: List[ShapeLine]
    bin_names: List[str]
    process_names: List[str]
    process_ids: List[str]
    rates: List[str]
    observations: Dict[str, str]
    nuisances: List[List[str]]
    params: List[List[str]]


def tokenize_card_line(line: str) -> List[str]:
    text = line.strip()
    if not text or text.startswith("#"):
        return []
    if set(text) <= {"-"}:
        return []
    if "#" in text:
        text = text.split("#", 1)[0].strip()
    return text.split()


def read_tokens(card_path: str) -> List[List[str]]:
    with open(card_path, "r", encoding="utf-8") as handle:
        return [tok for tok in (tokenize_card_line(line) for line in handle) if tok]


def format_row(key: str, values: List[str]) -> str:
    key_width = max(12, len(key) + 1)
    return f"{key:<{key_width}}" + " ".join(f"{value:<10}" for value in values)


def parse_observation(fields: List[str]) -> Optional[Tuple[str, str]]:
    if fields[0].lower() != "observation":
        return None
    if len(fields) == 3:
        return fields[1], fields[2]
    if len(fields) == 2:
        return "*", fields[1]
    return None


def parse_card(card_path: str) -> ParsedCard:
    tokens = read_tokens(card_path)

    shapes: List[ShapeLine] = []
    bin_lines: List[List[str]] = []
    process_lines: List[List[str]] = []
    rate_line: Optional[List[str]] = None
    observations: Dict[str, str] = {}
    nuisances: List[List[str]] = []
    params: List[List[str]] = []

    # Recognised systematic/param types for nuisance lines (<name> <type> <values...>).
    # Any line whose second token is not in this set and that doesn't match another
    # known keyword is silently ignored (e.g. "Combination of ...").
    _NUISANCE_TYPES = {
        "lnn", "shape", "lnu", "gmn", "gmc", "rateparam", "param",
        "flatparam", "extarg", "nuisance", "edit",
    }

    for fields in tokens:
        key = fields[0].lower()

        if key == "shapes":
            if len(fields) < 4:
                raise ValueError(f"Invalid shapes line: {' '.join(fields)}")
            process = fields[1]
            channel = "*"
            file_path = fields[2]
            extras = fields[3:]
            if len(fields) >= 5:
                channel = fields[2]
                file_path = fields[3]
                extras = fields[4:]
            shapes.append(ShapeLine(process=process, channel=channel, file_path=file_path, extras=extras))
            continue

        if key == "bin":
            if len(fields) > 1:
                bin_lines.append(fields[1:])
            continue

        if key == "process":
            process_lines.append(fields[1:])
            continue

        if key == "rate":
            rate_line = fields[1:]
            continue

        obs = parse_observation(fields)
        if obs is not None:
            observations[obs[0]] = obs[1]
            continue

        if len(fields) >= 4 and fields[1].lower() == "param":
            params.append(fields)
            continue

        if key in ("imax", "jmax", "kmax"):
            continue

        if len(fields) >= 2 and fields[1].lower() in _NUISANCE_TYPES:
            nuisances.append(fields)
            continue
        # Unrecognised line (e.g. "Combination of ...") — silently skip.

    if len(process_lines) < 2:
        raise ValueError("Could not find two process lines")
    if rate_line is None:
        raise ValueError("Could not find rate line")

    process_names = process_lines[0]
    process_ids = process_lines[1]

    if len(process_names) != len(process_ids) or len(process_names) != len(rate_line):
        raise ValueError("process/process-id/rate columns have inconsistent lengths")

    process_count = len(process_names)
    bin_names: List[str] = []
    for line in reversed(bin_lines):
        if len(line) == process_count:
            bin_names = line
            break

    if not bin_names:
        if len(bin_lines) == 1 and len(bin_lines[0]) == 1:
            bin_names = [bin_lines[0][0]] * process_count
        else:
            raise ValueError("Could not find bin line matching process columns")

    return ParsedCard(
        shapes=shapes,
        bin_names=bin_names,
        process_names=process_names,
        process_ids=process_ids,
        rates=rate_line,
        observations=observations,
        nuisances=nuisances,
        params=params,
    )


def detect_card_flavor(card_path: str, backend_shape_ext: str, backend_name: str) -> str:
    tokens = read_tokens(card_path)
    has_header = any(fields[0].lower() in ("imax", "jmax", "kmax") for fields in tokens)
    has_root_shapes = any(
        fields[0].lower() == "shapes" and any(token.endswith(".root") for token in fields)
        for fields in tokens
    )
    has_backend_shapes = any(
        fields[0].lower() == "shapes" and any(token.endswith(backend_shape_ext) for token in fields)
        for fields in tokens
    )

    if has_header or has_root_shapes:
        return "combine"
    if has_backend_shapes:
        return backend_name
    raise ValueError("Could not infer card flavor; use --direction explicitly")


def default_shapes_file_from_combine(parsed: ParsedCard, backend_shape_ext: str) -> str:
    for shape in parsed.shapes:
        if shape.process.lower() == "data_obs":
            continue
        if shape.file_path.endswith(".root"):
            return os.path.splitext(shape.file_path)[0] + backend_shape_ext
    return f"converted_shapes{backend_shape_ext}"


def resolve_shape_file(rel_path: str, card_dir: str) -> str:
    """Resolve a shape file path, trying both card-relative and CWD-relative lookups.

    For absolute paths the path is returned as-is.  For relative paths the
    resolution order is:

    1. Relative to *card_dir* (the directory containing the datacard) — this
       is the Combine convention for files stored alongside the card.
    2. Relative to the current working directory — useful when the user runs
       the converter from the workspace directory rather than the card directory.

    The first candidate that exists on disk is returned.  When neither exists
    the card-relative path is returned so that callers can surface a clear
    "file not found" error with the expected location.
    """
    if os.path.isabs(rel_path):
        return rel_path

    card_relative = os.path.abspath(os.path.join(card_dir, rel_path))
    if os.path.isfile(card_relative):
        return card_relative

    cwd_relative = os.path.abspath(rel_path)
    if os.path.isfile(cwd_relative):
        return cwd_relative

    # Neither found — return the card-relative path so errors are informative.
    return card_relative


def collect_root_shape_files(parsed: ParsedCard, card_dir: str) -> List[str]:
    """Return the unique .root workspace files referenced by non-data_obs shapes lines.

    Paths are resolved via :func:`resolve_shape_file`, which tries the card
    directory first and falls back to CWD.  The order matches the first
    occurrence of each unique resolved path so callers can use the first entry
    as the primary workspace file.

    Returns an empty list when the card has no ``shapes`` lines that point
    to ``.root`` files (i.e. counting experiments).
    """
    seen: List[str] = []
    for shape in parsed.shapes:
        if shape.process.lower() == "data_obs":
            continue
        if not shape.file_path.endswith(".root"):
            continue
        abs_path = resolve_shape_file(shape.file_path, card_dir)
        if abs_path not in seen:
            seen.append(abs_path)
    return seen


def clean_workspace_expr(expr: str) -> str:
    if ":" in expr:
        return expr.split(":", 1)[1]
    return expr


def render_process_expr(expr: str, process: str, channel: str) -> str:
    rendered = str(expr)
    rendered = rendered.replace("$PROCESS", process)
    rendered = rendered.replace("$CHANNEL", channel)
    rendered = rendered.replace("{channel}", channel)
    return clean_workspace_expr(rendered)


def workspace_from_expr(expr: str) -> Optional[str]:
    text = str(expr)
    if ":" not in text:
        return None
    return text.split(":", 1)[0]


def load_workspace_name_mapping(shapes_file: str, card_dir: str, payload_loader) -> Dict[str, str]:
    shape_path = shapes_file
    if not os.path.isabs(shape_path):
        shape_path = os.path.abspath(os.path.join(card_dir, shape_path))

    try:
        payload = payload_loader(shape_path)
    except Exception:
        return {}

    if not isinstance(payload, dict):
        return {}

    mapping = payload.get("workspace_name_mapping", {})
    if not isinstance(mapping, dict):
        return {}

    clean_map: Dict[str, str] = {}
    for key, value in mapping.items():
        clean_map[str(key)] = str(value)
    return clean_map


def choose_shape_line(parsed: ParsedCard, process: str, channel: str) -> Optional[ShapeLine]:
    best: Optional[ShapeLine] = None
    best_rank = -1

    for shape in parsed.shapes:
        if shape.process.lower() == "data_obs":
            continue
        process_match = shape.process == "*" or shape.process == process
        channel_match = shape.channel == "*" or shape.channel == channel
        if not (process_match and channel_match):
            continue
        rank = int(shape.process != "*") + int(shape.channel != "*")
        if rank > best_rank:
            best_rank = rank
            best = shape

    return best


def map_process_names_from_shapes(parsed: ParsedCard, workspace_name_mapping: Optional[Dict[str, str]] = None) -> List[str]:
    workspace_name_mapping = workspace_name_mapping or {}
    mapped: List[str] = []
    for process, channel in zip(parsed.process_names, parsed.bin_names):
        shape = choose_shape_line(parsed, process, channel)
        if shape is None or not shape.extras:
            mapped.append(process)
            continue
        nominal_expr = shape.extras[0]
        mapped_name = render_process_expr(nominal_expr, process=process, channel=channel)
        workspace_name = workspace_from_expr(nominal_expr)
        prefix = workspace_name_mapping.get(workspace_name, "") if workspace_name is not None else ""
        mapped.append(f"{prefix}{mapped_name}" if prefix else mapped_name)
    return mapped


def first_channel(bin_names: List[str]) -> str:
    if not bin_names:
        return "ch1"
    return bin_names[0]


def channel_list(bin_names: List[str]) -> List[str]:
    channels: List[str] = []
    for name in bin_names:
        if name not in channels:
            channels.append(name)
    return channels


def observation_values(parsed: ParsedCard, channels: List[str]) -> List[str]:
    if not parsed.observations:
        return ["-1"] * len(channels)

    if "*" in parsed.observations and len(channels) == 1:
        return [parsed.observations["*"]]

    values: List[str] = []
    for channel in channels:
        values.append(parsed.observations.get(channel, parsed.observations.get("*", "-1")))
    return values


def is_counting_experiment(parsed: ParsedCard) -> bool:
    """Return True when the card has no shape workspace references.

    A card is a pure counting experiment when its ``shapes`` list is empty
    or contains only ``data_obs`` entries — i.e. all yields are given
    directly by the ``rate`` row and no workspace PDF is needed.
    """
    for shape in parsed.shapes:
        if shape.process.lower() != "data_obs":
            return False
    return True


def convert_combine_to_backend(
    parsed: ParsedCard,
    shapes_file: Optional[str],
    backend_name: str,
    backend_shape_ext: str,
    map_process_names: bool = True,
    workspace_name_mapping: Optional[Dict[str, str]] = None,
) -> str:
    counting = is_counting_experiment(parsed)
    channels = channel_list(parsed.bin_names)
    # For combine-to-backend conversion, use the bare process names from the
    # Combine card's process row directly.  map_process_names_from_shapes
    # expands Combine PDF templates (e.g. mumem_20_$PROCESS_pdf -> mumem_20_signal_pdf)
    # which is the wrong direction here; the backend card should use the bare
    # names (signal, dio, cosmic) that the workspace conversion will produce.
    process_names = list(parsed.process_names)

    lines: List[str] = []
    lines.append(f"# Auto-generated {backend_name} card converted from a Combine card")
    if not counting:
        target_shapes = shapes_file or default_shapes_file_from_combine(parsed, backend_shape_ext)
        lines.append(f"shapes * * {target_shapes}")
        lines.append(f"shapes data_obs * {target_shapes}")
    lines.append("")
    lines.append(format_row("bin", parsed.bin_names))
    lines.append(format_row("process", process_names))
    lines.append(format_row("process", parsed.process_ids))
    lines.append(format_row("rate", parsed.rates))

    if parsed.observations:
        if "*" in parsed.observations and len(channels) == 1:
            lines.append(f"observation {channels[0]} {parsed.observations['*']}")
        else:
            for channel in channels:
                value = parsed.observations.get(channel, parsed.observations.get("*"))
                if value is not None:
                    lines.append(f"observation {channel} {value}")

    if parsed.nuisances:
        lines.append("")
        for nuisance in parsed.nuisances:
            rendered = format_row(nuisance[0], nuisance[1:])
            # rateParam and 'nuisance edit' are Combine-specific; comment them
            # out in backend cards until they are supported natively.
            nuisance_type = nuisance[1].lower() if len(nuisance) > 1 else ""
            if nuisance_type == "rateparam" or nuisance[0].lower() == "nuisance":
                rendered = "# " + rendered
            lines.append(rendered)

    if parsed.params:
        lines.append("")
        for param in parsed.params:
            lines.append(" ".join(param))

    return "\n".join(lines) + "\n"


def convert_backend_to_combine(
    parsed: ParsedCard,
    backend_name: str,
    root_file: str,
    workspace_name: str,
    pdf_template: str,
    syst_template: str,
) -> str:
    channels = channel_list(parsed.bin_names)
    if not channels:
        channels = [first_channel(parsed.bin_names)]

    lines: List[str] = []
    lines.append(f"# Auto-generated Combine card converted from a {backend_name} card")
    lines.append("imax *")
    lines.append("jmax *")
    lines.append("kmax *")
    lines.append("-")

    for channel in channels:
        process_expr = pdf_template.format(channel=channel)
        syst_expr = syst_template.format(channel=channel)
        lines.append(
            f"shapes * {channel} {root_file} {workspace_name}:{process_expr} {workspace_name}:{syst_expr}"
        )
        lines.append(
            f"shapes data_obs {channel} {root_file} {workspace_name}:data_obs"
        )

    lines.append("-")
    if len(channels) > 1:
        lines.append(format_row("bin", channels))
    lines.append(format_row("observation", observation_values(parsed, channels)))

    lines.append("-")
    lines.append(format_row("bin", parsed.bin_names))
    lines.append(format_row("process", parsed.process_names))
    lines.append(format_row("process", parsed.process_ids))
    lines.append(format_row("rate", parsed.rates))

    if parsed.nuisances:
        lines.append("-")
        for nuisance in parsed.nuisances:
            lines.append(format_row(nuisance[0], nuisance[1:]))

    if parsed.params:
        lines.append("-")
        for param in parsed.params:
            lines.append(" ".join(param))

    return "\n".join(lines) + "\n"


def default_root_file_from_backend(parsed: ParsedCard, backend_shape_ext: str) -> str:
    for shape in parsed.shapes:
        if shape.process.lower() == "data_obs":
            continue
        if shape.file_path.endswith(backend_shape_ext):
            return os.path.splitext(shape.file_path)[0] + ".root"
    return "converted_workspace.root"


def build_converter_parser(backend_name: str, backend_shape_ext: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            f"Convert a Higgs Combine text card to a {backend_name} card, or reverse-convert a {backend_name} card "
            "to a Combine-style card."
        )
    )
    parser.add_argument("input_card", help="Input card path")
    parser.add_argument("output_card", help="Output card path")
    parser.add_argument(
        "--direction",
        choices=["auto", f"combine-to-{backend_name}", f"{backend_name}-to-combine"],
        default="auto",
        help="Conversion direction (default: auto-detect)",
    )

    parser.add_argument(
        "--shapes-file",
        default=None,
        help=(
            f"Target {backend_shape_ext} shapes file for combine-to-{backend_name} conversion. "
            f"If omitted, defaults to the first Combine shapes ROOT file with {backend_shape_ext} extension."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory where converted workspace files are written during "
            f"combine-to-{backend_name} conversion. "
            "Defaults to the directory containing the output card."
        ),
    )
    parser.add_argument(
        "--skip-workspace-conversion",
        action="store_true",
        help=(
            "Do not run workspace conversion even when ROOT shape files are "
            "referenced in the Combine card.  Use this when the backend workspace "
            "file already exists or will be created separately."
        ),
    )
    parser.add_argument(
        "--no-process-name-mapping",
        action="store_true",
        help=(
            f"Do not map Combine process labels using shapes templates when converting to {backend_name}. "
            "By default, process names are mapped to workspace PDF names."
        ),
    )

    parser.add_argument(
        "--root-file",
        default=None,
        help=(
            f"Target ROOT file for {backend_name}-to-combine conversion. "
            f"If omitted, defaults to replacing the first {backend_name} shapes {backend_shape_ext} path with .root."
        ),
    )
    parser.add_argument(
        "--workspace-name",
        default="workspace",
        help="RooWorkspace name to use in Combine shapes expressions (default: workspace)",
    )
    parser.add_argument(
        "--pdf-template",
        default="$PROCESS",
        help=(
            "Combine PDF name template for nominal shapes. "
            "Use {channel} to inject the channel name (default: $PROCESS)."
        ),
    )
    parser.add_argument(
        "--syst-template",
        default="$PROCESS_$SYSTEMATIC",
        help=(
            "Combine PDF name template for systematic shapes. "
            "Use {channel} to inject the channel name (default: $PROCESS_$SYSTEMATIC)."
        ),
    )
    return parser


def run_converter_cli(
    backend_name: str,
    backend_shape_ext: str,
    payload_loader: Callable[[str], object],
    workspace_converter: Optional[Callable] = None,
) -> None:
    """Run the datacard conversion CLI.

    Parameters
    ----------
    backend_name:
        Short name of the backend (e.g. ``"hfmodel"``).
    backend_shape_ext:
        File extension used by the backend's shape workspace (e.g. ``".json"``).
    payload_loader:
        Callable that loads a backend workspace file and returns a payload
        object (used to extract ``workspace_name_mapping`` for reverse
        conversion).
    workspace_converter:
        Optional callable invoked during ``combine-to-{backend_name}``
        conversion when the Combine card references one or more ``.root``
        shape workspace files.  Signature::

            workspace_converter(
                root_paths: List[str],
                card_path: str,
                output_dir: str,
                output_prefix: str,
            ) -> str

        It must return the path to the converted backend workspace file.
        When *None* (or ``--skip-workspace-conversion`` is given), workspace
        conversion is skipped and the ``shapes_file`` used in the output card
        is derived from the first ``.root`` file with the backend extension
        substituted — the same behaviour as before this feature was added.
    """
    args = build_converter_parser(backend_name, backend_shape_ext).parse_args()

    input_card = os.path.abspath(args.input_card)
    output_card = os.path.abspath(args.output_card)

    direction = args.direction
    if direction == "auto":
        flavor = detect_card_flavor(
            input_card,
            backend_shape_ext=backend_shape_ext,
            backend_name=backend_name,
        )
        direction = f"combine-to-{backend_name}" if flavor == "combine" else f"{backend_name}-to-combine"

    parsed = parse_card(input_card)

    if direction == f"combine-to-{backend_name}":
        counting = is_counting_experiment(parsed)

        output_dir = (
            os.path.abspath(args.output_dir)
            if args.output_dir is not None
            else os.path.dirname(output_card)
        )

        # Determine the target shapes file for the backend card.
        # For counting experiments there are no shapes, so shapes_file stays None
        # unless the user forced one with --shapes-file.
        if args.shapes_file is not None:
            shapes_file: Optional[str] = args.shapes_file
        elif counting:
            shapes_file = None
        else:
            shapes_file = default_shapes_file_from_combine(parsed, backend_shape_ext)
            # Place the workspace file alongside the output card when no
            # explicit --shapes-file was given and no output-dir was set.
            if args.output_dir is None and not os.path.isabs(shapes_file):
                shapes_file = os.path.join(output_dir, os.path.basename(shapes_file))

        # Run workspace conversion unless the card is a counting experiment,
        # the caller opted out, or no workspace_converter was supplied.
        skip_ws = counting or args.skip_workspace_conversion or workspace_converter is None
        if not skip_ws:
            card_dir = os.path.dirname(input_card)
            root_paths = collect_root_shape_files(parsed, card_dir)
            if root_paths:
                output_prefix = os.path.splitext(os.path.basename(shapes_file))[0]
                print(
                    f"Converting {len(root_paths)} ROOT workspace file(s) to "
                    f"{backend_name} format in '{output_dir}' ..."
                )
                converted_shapes = workspace_converter(
                    root_paths=root_paths,
                    card_path=input_card,
                    output_dir=output_dir,
                    output_prefix=output_prefix,
                )
                # Use the path returned by the converter as the shapes file.
                shapes_file = converted_shapes
                print(f"Workspace conversion complete: {shapes_file}")
            else:
                print(
                    "No ROOT shape workspace files found in card; "
                    "skipping workspace conversion."
                )

        output_text = convert_combine_to_backend(
            parsed=parsed,
            shapes_file=shapes_file,
            backend_name=backend_name,
            backend_shape_ext=backend_shape_ext,
        )
    else:
        root_file = args.root_file or default_root_file_from_backend(parsed, backend_shape_ext)
        output_text = convert_backend_to_combine(
            parsed=parsed,
            backend_name=backend_name,
            root_file=root_file,
            workspace_name=args.workspace_name,
            pdf_template=args.pdf_template,
            syst_template=args.syst_template,
        )

    with open(output_card, "w", encoding="utf-8") as handle:
        handle.write(output_text)

    print(f"Wrote converted card: {output_card}")


def build_backend_converter_functions(backend_name: str, backend_shape_ext: str):
    def convert_to_backend(
        parsed: ParsedCard,
        shapes_file: Optional[str],
        map_process_names: bool = True,
        workspace_name_mapping: Optional[Dict[str, str]] = None,
    ) -> str:
        return convert_combine_to_backend(
            parsed=parsed,
            shapes_file=shapes_file,
            backend_name=backend_name,
            backend_shape_ext=backend_shape_ext,
            map_process_names=map_process_names,
            workspace_name_mapping=workspace_name_mapping,
        )

    def convert_to_combine(
        parsed: ParsedCard,
        root_file: str,
        workspace_name: str,
        pdf_template: str,
        syst_template: str,
    ) -> str:
        return convert_backend_to_combine(
            parsed=parsed,
            backend_name=backend_name,
            root_file=root_file,
            workspace_name=workspace_name,
            pdf_template=pdf_template,
            syst_template=syst_template,
        )

    return convert_to_backend, convert_to_combine