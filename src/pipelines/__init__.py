"""Scheduled reporting pipeline contracts."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScheduledReport:
    report_type: str
    period_key: str
    title: str
    markdown: str
    data_cutoff: str
    validated: bool = True
    results: dict[str, Any] = field(default_factory=dict)
