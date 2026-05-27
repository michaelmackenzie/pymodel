from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class FitModel:
    workspace: Dict[str, Any]
    model: Any
    data: Any
    process_names: List[str] = field(default_factory=list)
    process_ids: List[int] = field(default_factory=list)
    signal_processes: List[str] = field(default_factory=list)
    channels: List[str] = field(default_factory=list)
    term_channels: Dict[str, str] = field(default_factory=dict)
    term_processes: Dict[str, str] = field(default_factory=dict)
    observed_counts_by_channel: Dict[str, float] = field(default_factory=dict)
    measurement_name: Optional[str] = None
    poi_name: Optional[str] = None
