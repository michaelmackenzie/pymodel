from typing import Dict, List, Tuple


def _parse_parameter_value_map(spec):
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


def _parse_parameter_range_map(spec):
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


def _parse_parameter_name_list(spec):
    if spec is None:
        return []
    return [item.strip() for item in spec.split(",") if item.strip()]


def _validate_override_names(available_names: List[str], requested_names: List[str]):
    available = set(available_names)
    missing = sorted(set(requested_names) - available)
    if missing:
        preview = ", ".join(available_names[:30])
        more = "" if len(available_names) <= 30 else f", ... (+{len(available_names) - 30} more)"
        raise ValueError(
            f"Unknown parameters in overrides: {', '.join(missing)}. Available parameters include: {preview}{more}"
        )


def apply_parameter_overrides(fit_model, set_values_spec, set_ranges_spec, freeze_spec):
    model = fit_model.model
    par_order = list(model.config.par_order)

    value_updates: Dict[str, float] = _parse_parameter_value_map(set_values_spec)
    range_updates: Dict[str, Tuple[float, float]] = _parse_parameter_range_map(set_ranges_spec)
    freeze_names: List[str] = _parse_parameter_name_list(freeze_spec)

    required_names = list(value_updates.keys()) + list(range_updates.keys()) + list(freeze_names)
    _validate_override_names(par_order, required_names)

    fit_model.analysis_overrides = {
        "set_values": value_updates,
        "set_ranges": range_updates,
        "freeze": freeze_names,
    }
