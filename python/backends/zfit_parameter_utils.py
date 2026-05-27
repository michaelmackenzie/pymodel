from typing import Dict, Iterable, List, Optional, Tuple


def channel_models(fit_model) -> Dict:
    return getattr(fit_model, "channel_models", {}) or {}


def all_models(fit_model) -> List:
    models = channel_models(fit_model)
    if models:
        return list(models.values())
    return [fit_model.model]


def _iter_child_params(param):
    children = getattr(param, "params", None)
    if children is None:
        return
    if isinstance(children, dict):
        iterable = children.values()
    else:
        iterable = children
    for child in iterable:
        candidate = child
        if isinstance(child, tuple) and len(child) >= 2:
            candidate = child[1]
        if hasattr(candidate, "value"):
            yield candidate


def iter_unique_params(fit_model):
    seen = set()

    def _yield_param(param):
        ident = id(param)
        if ident in seen:
            return
        seen.add(ident)
        yield param
        for child in _iter_child_params(param):
            yield from _yield_param(child)

    for model in all_models(fit_model):
        for kwargs in ({}, {"floating": None}, {"floating": None, "is_yield": None}):
            try:
                params = list(model.get_params(**kwargs))
            except Exception:
                continue
            for param in params:
                yield from _yield_param(param)

    for param in (getattr(fit_model, "yields", {}) or {}).values():
        if hasattr(param, "value"):
            yield from _yield_param(param)


def all_params_list(fit_model) -> List:
    return list(iter_unique_params(fit_model))


def find_parameter_with_error(fit_model, parameter_name: str):
    params = all_params_list(fit_model)

    exact_matches = [param for param in params if param.name == parameter_name]
    if len(exact_matches) == 1:
        return exact_matches[0], None
    if len(exact_matches) > 1:
        names = sorted(param.name for param in exact_matches)
        return None, (
            f"Parameter name '{parameter_name}' is ambiguous; exact matches: {', '.join(names)}"
        )

    suffixed_matches = [
        param for param in params if param.name.startswith(f"{parameter_name}__")
    ]
    if len(suffixed_matches) == 1:
        return suffixed_matches[0], None
    if len(suffixed_matches) > 1:
        names = sorted(param.name for param in suffixed_matches)
        return None, (
            f"Parameter '{parameter_name}' matches multiple channel-specific parameters: "
            f"{', '.join(names)}. Use the full parameter name."
        )

    if "__" in parameter_name:
        base_name = parameter_name.split("__", 1)[0]
        base_matches = [param for param in params if param.name == base_name]
        if len(base_matches) == 1:
            return base_matches[0], None

    available_names = sorted(param.name for param in params)
    preview = ", ".join(available_names[:20])
    more = "" if len(available_names) <= 20 else f", ... (+{len(available_names) - 20} more)"
    return None, (
        f"Parameter '{parameter_name}' was not found in the model. "
        f"Available parameters include: {preview}{more}"
    )


def find_parameter_by_name(fit_model, parameter_name: str):
    param, _ = find_parameter_with_error(fit_model, parameter_name)
    return param


def capture_fit_model_parameter_values(fit_model) -> Dict:
    values = {}
    for param in iter_unique_params(fit_model):
        if not hasattr(param, "set_value"):
            continue
        try:
            values[param] = float(param.value())
        except Exception:
            continue
    return values


def capture_parameter_values(model) -> Dict:
    values = {}
    for kwargs in ({}, {"floating": None}, {"floating": None, "is_yield": None}):
        try:
            params = list(model.get_params(**kwargs))
        except Exception:
            continue
        for param in params:
            if not hasattr(param, "set_value"):
                continue
            try:
                values[param] = float(param.value())
            except Exception:
                continue
    return values


def restore_parameter_values(saved_values: Dict) -> None:
    for param, value in saved_values.items():
        try:
            param.set_value(value)
        except Exception:
            continue
