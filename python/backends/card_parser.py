from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class UncertaintySpec:
    name: str
    kind: str
    values: List[str]


@dataclass
class ConstraintSpec:
    name: str
    mean: float
    width: float


@dataclass
class ShapeSpec:
    process: str
    channel: str
    file: str


@dataclass
class CardSpec:
    shape_specs: List[ShapeSpec]
    is_counting: bool
    channels: List[str]
    bin_names: List[str]
    process_names: List[str]
    process_ids: List[int]
    rates: List[Optional[float]]
    uncertainties: List[UncertaintySpec]
    observations: Dict[str, float]
    data_obs_files: Dict[str, str]
    category: Optional[str] = None
    observation_count: Optional[float] = None
    param_constraints: List[ConstraintSpec] = None

    def __post_init__(self):
        if self.param_constraints is None:
            self.param_constraints = []
        if self.category is None and self.channels:
            self.category = self.channels[0]
        if self.observation_count is None and self.observations:
            self.observation_count = float(sum(self.observations.values()))


def _has_shape_mapping(shape_specs: List[ShapeSpec], process: str, channel: str) -> bool:
    for spec in shape_specs:
        if spec.process.lower() == "data_obs":
            continue
        process_match = spec.process == "*" or spec.process == process
        channel_match = spec.channel == "*" or spec.channel == channel
        if process_match and channel_match:
            return True
    return False


def _tokenize_card_line(line: str) -> List[str]:
    text = line.strip()
    if not text or text.startswith("#"):
        return []
    if "#" in text:
        text = text.split("#", 1)[0].strip()
    return text.split()


def parse_model_card(
    card_path: str,
    *,
    shape_extension: str,
    shape_description: str,
) -> CardSpec:
    with open(card_path, "r", encoding="utf-8") as handle:
        lines = [_tokenize_card_line(line) for line in handle]

    tokens = [line for line in lines if line]

    shape_specs: List[ShapeSpec] = []
    bin_names: Optional[List[str]] = None
    process_names: Optional[List[str]] = None
    process_ids: Optional[List[int]] = None
    rates: Optional[List[Optional[float]]] = None
    uncertainties: List[UncertaintySpec] = []
    param_constraints: List[ConstraintSpec] = []
    process_line_count = 0
    observations: Dict[str, float] = {}
    data_obs_files: Dict[str, str] = {}
    comment_markers = {"#", "//", "--"}

    normalized_ext = shape_extension.lower()
    if not normalized_ext.startswith("."):
        normalized_ext = f".{normalized_ext}"

    for fields in tokens:
        key = fields[0].lower()
        for marker in comment_markers:
            key = key.split(marker, 1)[0].strip()
        if not key:
            continue

        if key == "shapes":
            if len(fields) not in (3, 4):
                raise ValueError(f"Invalid shapes line: {' '.join(fields)}")

            if len(fields) == 3:
                process_target = fields[1]
                channel_target = "*"
                file_name = fields[2]
            else:
                process_target = fields[1]
                channel_target = fields[2]
                file_name = fields[3]

            if not file_name.lower().endswith(normalized_ext):
                raise ValueError(
                    f"Shape file '{file_name}' must be {shape_description} ({normalized_ext})"
                )

            if process_target.lower() == "data_obs":
                data_obs_files[channel_target] = file_name
            else:
                shape_specs.append(
                    ShapeSpec(process=process_target, channel=channel_target, file=file_name)
                )
            continue

        if key == "bin":
            if len(fields) < 2:
                raise ValueError(f"Invalid bin line: {' '.join(fields)}")
            bin_names = fields[1:]
            continue

        if key == "process":
            process_line_count += 1
            if process_line_count == 1:
                process_names = fields[1:]
            elif process_line_count == 2:
                process_ids = [int(item) for item in fields[1:]]
            else:
                raise ValueError("Model card has more than two process lines")
            continue

        if key == "rate":
            if process_names is None:
                raise ValueError("rate line appears before process names")
            values = fields[1:]
            if len(values) != len(process_names):
                raise ValueError("rate line length does not match process count")
            rates = [None if value == "-" else float(value) for value in values]
            continue

        if key == "observation":
            if len(fields) != 3:
                raise ValueError(
                    f"Invalid observation line: {' '.join(fields)}. Expected 'observation <category> <count>'"
                )
            observations[fields[1]] = float(fields[2])
            continue

        if len(fields) >= 4 and fields[1].lower() == "param":
            try:
                mean = float(fields[2])
                width = float(fields[3])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid param constraint line: {' '.join(fields)}. Expected '<name> param <mean> <width>'"
                ) from exc
            param_constraints.append(ConstraintSpec(name=fields[0], mean=mean, width=width))
            continue

        if len(fields) < 3:
            raise ValueError(f"Invalid uncertainty line: {' '.join(fields)}")

        uncertainties.append(UncertaintySpec(name=fields[0], kind=fields[1], values=fields[2:]))

    if bin_names is None:
        raise ValueError("Missing bin line")
    if process_names is None:
        raise ValueError("Missing process names line")
    if process_ids is None:
        raise ValueError("Missing process id line")
    if rates is None:
        raise ValueError("Missing rate line")
    if len(process_names) != len(process_ids):
        raise ValueError("process names and IDs length mismatch")
    if len(bin_names) == 1 and len(process_names) > 1:
        bin_names = [bin_names[0]] * len(process_names)
    if len(bin_names) != len(process_names):
        raise ValueError("bin line length does not match process count")

    channels = list(dict.fromkeys(bin_names))

    if observations:
        unknown_obs = [name for name in observations if name not in channels]
        if unknown_obs:
            raise ValueError(f"Observation category not present in bin line: {unknown_obs}")

    is_counting = len(shape_specs) == 0
    if not is_counting:
        for process, channel in zip(process_names, bin_names):
            if not _has_shape_mapping(shape_specs, process, channel):
                raise ValueError(
                    f"Missing shape mapping for process/channel '{process}/{channel}'. "
                    "Expected a matching line: shapes <process|*> <channel|*> <file>"
                )

    for unc in uncertainties:
        if len(unc.values) != len(process_names):
            raise ValueError(
                f"Uncertainty '{unc.name}' has {len(unc.values)} values, expected {len(process_names)}"
            )
        if is_counting and unc.kind.strip().lower() == "shape":
            raise ValueError(
                f"Shape uncertainty '{unc.name}' is not allowed for counting models (no shapes section provided)"
            )

    return CardSpec(
        shape_specs=shape_specs,
        is_counting=is_counting,
        channels=channels,
        bin_names=bin_names,
        process_names=process_names,
        process_ids=process_ids,
        rates=rates,
        uncertainties=uncertainties,
        observations=observations,
        data_obs_files=data_obs_files,
        param_constraints=param_constraints,
    )