from typing import List

from backends.card_parser import CardSpec


def _shape_mapping_rank(process_target: str, channel_target: str, process: str, channel: str):
    process_match = process_target == "*" or process_target == process
    channel_match = channel_target == "*" or channel_target == channel
    if not (process_match and channel_match):
        return None
    specificity = int(process_target != "*") + int(channel_target != "*")
    return specificity


def resolve_shape_file_for_term(card: CardSpec, process: str, channel: str) -> str:
    best_spec = None
    best_rank = None

    for idx, spec in enumerate(card.shape_specs):
        specificity = _shape_mapping_rank(spec.process, spec.channel, process, channel)
        if specificity is None:
            continue
        ranked = (specificity, idx)
        if best_rank is None or ranked > best_rank:
            best_rank = ranked
            best_spec = spec

    if best_spec is None:
        raise ValueError(f"No shape mapping found for process/channel '{process}/{channel}'")
    return best_spec.file


def kind_token(kind: str) -> str:
    token = kind.strip()
    if token == "lnN":
        return "lnN"
    lowered = token.lower()
    if lowered == "gs":
        return "gs"
    if lowered == "shape":
        return "shape"
    raise ValueError(f"Unknown uncertainty type '{kind}'. Use lnN, gs, or shape.")


def make_term_names(process_names: List[str], bin_names: List[str]) -> List[str]:
    term_names: List[str] = []
    name_counts = {}

    for process, channel in zip(process_names, bin_names):
        base = f"{process}__{channel}"
        safe_base = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in base)
        index = name_counts.get(safe_base, 0)
        name_counts[safe_base] = index + 1
        if index:
            term_names.append(f"{safe_base}_{index}")
        else:
            term_names.append(safe_base)

    return term_names


def make_term_mappings(term_names: List[str], process_names: List[str], bin_names: List[str]):
    term_channels = {
        term_name: channel
        for term_name, channel in zip(term_names, bin_names)
    }
    term_processes = {
        term_name: process
        for term_name, process in zip(term_names, process_names)
    }
    return term_channels, term_processes