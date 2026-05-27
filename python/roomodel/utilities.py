from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class FitModel:
    model_file: str
    workspace_name: str
    model_name: str
    data_name: Optional[str]
    channels: List[str] = field(default_factory=list)
    process_names: List[str] = field(default_factory=list)
    signal_processes: List[str] = field(default_factory=list)
    observed_counts_by_channel: Dict[str, float] = field(default_factory=dict)
    poi_name: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
