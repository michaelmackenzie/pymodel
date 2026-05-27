from backends.analysis_overrides_common import (
    parse_parameter_name_list,
    parse_parameter_range_map,
    parse_parameter_value_map,
    validate_override_names,
)


def apply_parameter_overrides(fit_model, set_values_spec, set_ranges_spec, freeze_spec):
    model = fit_model.model
    par_order = list(model.config.par_order)

    value_updates = parse_parameter_value_map(set_values_spec)
    range_updates = parse_parameter_range_map(set_ranges_spec)
    freeze_names = parse_parameter_name_list(freeze_spec)

    required_names = list(value_updates.keys()) + list(range_updates.keys()) + list(freeze_names)
    validate_override_names(par_order, required_names)

    fit_model.analysis_overrides = {
        "set_values": value_updates,
        "set_ranges": range_updates,
        "freeze": freeze_names,
    }
