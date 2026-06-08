from backends.analysis_overrides_common import (
    parse_parameter_name_list,
    parse_parameter_range_map,
    parse_parameter_value_map,
)
from backends.zfit_parameter_utils import (
    find_parameter_by_name,
    find_parameter_with_error,
)


def _resolve_required_parameters(fit_model, required_names):
    params_by_name = {}
    for name in required_names:
        param, error = find_parameter_with_error(fit_model, name)
        if param is None:
            raise ValueError(error)
        params_by_name[name] = param
    return params_by_name


def apply_parameter_overrides(fit_model, set_values_spec, set_ranges_spec, freeze_spec):
    value_updates = parse_parameter_value_map(set_values_spec)
    range_updates = parse_parameter_range_map(set_ranges_spec)
    freeze_names = parse_parameter_name_list(freeze_spec)

    required_names = set(value_updates) | set(range_updates) | set(freeze_names)
    params_by_name = _resolve_required_parameters(fit_model, required_names)

    for name, value in value_updates.items():
        params_by_name[name].set_value(value)

    for name, (low, high) in range_updates.items():
        param = params_by_name[name]
        # Resolve None to the existing bound so we never pass None to zfit.
        effective_low  = low  if low  is not None else float(getattr(param, "lower", low))
        effective_high = high if high is not None else float(getattr(param, "upper", high))
        if hasattr(param, "set_limits"):
            param.set_limits(low=effective_low, high=effective_high)
        else:
            param.lower = effective_low
            param.upper = effective_high

    for name in freeze_names:
        param = params_by_name[name]
        if not hasattr(param, "floating"):
            raise ValueError(f"Parameter '{name}' does not support floating/fixed state")
        param.floating = False
