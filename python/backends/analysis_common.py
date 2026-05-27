import os


def load_analysis_model(model_file, input_card, load_fit_model_fn, parse_model_card_fn, build_model_from_card_fn):
    if model_file is not None:
        return load_fit_model_fn(os.path.abspath(model_file))

    card_path = os.path.abspath(input_card)
    card = parse_model_card_fn(card_path)
    return build_model_from_card_fn(card, os.path.dirname(card_path))


def resolve_data_mode(use_observed_data, use_asimov_data):
    if use_observed_data:
        return "observed"
    if use_asimov_data:
        return "asimov"
    return "toy"


def resolve_dataset_mode(toys, has_observed_data, *, error_suffix=""):
    if toys is None:
        return has_observed_data, False, 1
    if toys == -1:
        return False, True, 1
    if toys < -1:
        suffix = f" {error_suffix}" if error_suffix else ""
        raise ValueError(f"Only --toys -1 is supported as a special Asimov mode{suffix}")
    return False, False, int(toys)


def checkpoint_mismatches(checkpoint, expected):
    mismatches = []
    for key, expected_value in expected.items():
        if key not in checkpoint:
            mismatches.append((key, "<missing>", expected_value))
            continue
        if checkpoint.get(key) != expected_value:
            mismatches.append((key, checkpoint.get(key), expected_value))
    return mismatches


def normalize_output_path(output_path, extension):
    normalized_ext = extension if extension.startswith(".") else f".{extension}"
    abs_out = os.path.abspath(output_path)
    if abs_out.lower().endswith(normalized_ext.lower()):
        return abs_out
    base, _ = os.path.splitext(abs_out)
    return f"{base}{normalized_ext}"