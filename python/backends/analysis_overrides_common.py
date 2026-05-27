from typing import List


def parse_parameter_value_map(spec):
    if spec is None:
        return {}

    assignments = {}
    for raw_item in spec.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid parameter assignment '{item}'. Expected format name=value")
        name, value_text = item.split("=", 1)
        name = name.strip()
        value_text = value_text.strip()
        if not name:
            raise ValueError(f"Invalid parameter assignment '{item}'")
        assignments[name] = float(value_text)
    return assignments


def parse_parameter_range_map(spec):
    if spec is None:
        return {}

    ranges = {}
    for raw_item in spec.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "=" not in item or ":" not in item:
            raise ValueError(f"Invalid range assignment '{item}'. Expected format name=low:high")
        name, bounds_text = item.split("=", 1)
        low_text, high_text = bounds_text.split(":", 1)
        name = name.strip()
        low = float(low_text.strip())
        high = float(high_text.strip())
        if not name:
            raise ValueError(f"Invalid range assignment '{item}'")
        if high <= low:
            raise ValueError(f"Invalid range for '{name}': high ({high}) must be > low ({low})")
        ranges[name] = (low, high)
    return ranges


def parse_parameter_name_list(spec):
    if spec is None:
        return []
    return [item.strip() for item in spec.split(",") if item.strip()]


def validate_override_names(available_names: List[str], requested_names: List[str]):
    available = set(available_names)
    missing = sorted(set(requested_names) - available)
    if missing:
        preview = ", ".join(available_names[:30])
        more = "" if len(available_names) <= 30 else f", ... (+{len(available_names) - 30} more)"
        raise ValueError(
            f"Unknown parameters in overrides: {', '.join(missing)}. Available parameters include: {preview}{more}"
        )
