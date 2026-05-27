import os

from hfmodel.model_io import load_fit_model


def _format_parameter_entry(param_set):
    fields = []
    name = param_set.get("name")
    if name:
        fields.append(name)
    if "n_parameters" in param_set:
        fields.append(f"n={param_set['n_parameters']}")
    if "inits" in param_set:
        fields.append(f"init={param_set['inits']}")
    if "bounds" in param_set:
        fields.append(f"bounds={param_set['bounds']}")
    if "fixed" in param_set:
        fields.append(f"fixed={param_set['fixed']}")
    return ", ".join(fields)


def _parset_metadata(parset):
    if parset is None:
        return {
            "n_parameters": None,
            "inits": None,
            "bounds": None,
            "fixed": None,
            "constrained": None,
            "pdf_type": None,
            "auxdata": None,
            "kind": None,
        }

    def _maybe_value(attr_name):
        value = getattr(parset, attr_name, None)
        if callable(value):
            try:
                return value()
            except Exception:
                return None
        return value

    return {
        "n_parameters": _maybe_value("n_parameters"),
        "inits": _maybe_value("suggested_init"),
        "bounds": _maybe_value("suggested_bounds"),
        "fixed": _maybe_value("suggested_fixed"),
        "constrained": _maybe_value("constrained"),
        "pdf_type": _maybe_value("pdf_type"),
        "auxdata": _maybe_value("auxdata"),
        "kind": type(parset).__name__,
    }


def load_and_summarize_model(model_file: str, verbose: int = 0):
    model_path = os.path.abspath(model_file)
    fit_model = load_fit_model(model_path)

    model = fit_model.model
    config = model.config

    channels = list(getattr(fit_model, "channels", []) or list(config.channels))
    process_names = list(getattr(fit_model, "process_names", []) or list(config.samples))
    observed_count = sum(float(v) for v in fit_model.observed_counts_by_channel.values())

    suggested_fixed = config.suggested_fixed() if callable(getattr(config, "suggested_fixed", None)) else getattr(config, "suggested_fixed", [])
    floating = len(config.par_order)
    if isinstance(suggested_fixed, (list, tuple)) and len(suggested_fixed) == len(config.par_order):
        floating = int(sum(0 if bool(flag) else 1 for flag in suggested_fixed))

    detail_lines = []
    if verbose:
        init_all = list(config.suggested_init())
        bounds_all = list(config.suggested_bounds())
        fixed_all = list(suggested_fixed) if isinstance(suggested_fixed, (list, tuple)) else [None] * len(config.par_order)

        detail_lines.append("Parameters:")
        for idx, par_name in enumerate(config.par_order):
            entry = config.par_map.get(par_name, {}) if isinstance(config.par_map, dict) else {}
            parset = entry.get("paramset") if isinstance(entry, dict) else None
            meta = _parset_metadata(parset)

            kind_bits = []
            if meta.get("kind"):
                kind_bits.append(meta["kind"])
            if meta.get("pdf_type"):
                kind_bits.append(str(meta["pdf_type"]))
            kind_label = "/".join(kind_bits) if kind_bits else "unknown"

            init_val = init_all[idx] if idx < len(init_all) else None
            bounds_val = bounds_all[idx] if idx < len(bounds_all) else None
            fixed_val = fixed_all[idx] if idx < len(fixed_all) else None

            detail_lines.append(
                f"  - {par_name}: init={init_val}, bounds={bounds_val}, fixed={fixed_val}, kind={kind_label}"
            )

        if verbose >= 2:
            detail_lines.append("Parameter-set internals:")
            for par_name in config.par_order:
                entry = config.par_map.get(par_name, {}) if isinstance(config.par_map, dict) else {}
                parset = entry.get("paramset") if isinstance(entry, dict) else None
                meta = _parset_metadata(parset)
                detail_lines.append(
                    f"  - {par_name}: n={meta.get('n_parameters')}, constrained={meta.get('constrained')}, auxdata={meta.get('auxdata')}"
                )

    signal_processes = list(getattr(fit_model, "signal_processes", []))
    signal_text = ", ".join(signal_processes) if signal_processes else "none"

    return {
        "model_path": model_path,
        "model_name": "pyhf_workspace_model",
        "obs_range": "binned",
        "channels": channels,
        "processes": process_names,
        "signal_process": signal_text,
        "poi_name": getattr(fit_model, "poi_name", config.poi_name),
        "constraints": int(config.nauxdata),
        "floating_params": floating,
        "observed_count": observed_count,
        "pdf_lines": detail_lines,
    }
