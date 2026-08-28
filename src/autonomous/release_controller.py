"""Final release gate for autonomous report email delivery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ReleaseDecision:
    approved: bool
    recipients: tuple[str, ...]
    reason: str


class ReleaseController:
    def __init__(self, approved_recipients: Iterable[str]) -> None:
        self.approved_recipients = frozenset(
            item.strip().lower() for item in approved_recipients if item.strip()
        )

    def evaluate(
        self,
        *,
        recipients: Iterable[str],
        validated: bool,
        pdf_path: str | Path,
        human_approved: bool,
    ) -> ReleaseDecision:
        normalized = tuple(dict.fromkeys(item.strip().lower() for item in recipients if item.strip()))
        if not validated:
            return ReleaseDecision(False, normalized, "Report validation has not passed.")
        if not Path(pdf_path).is_file():
            return ReleaseDecision(False, normalized, "PDF attachment does not exist.")
        unauthorized = set(normalized) - self.approved_recipients
        if unauthorized:
            return ReleaseDecision(False, normalized, "One or more recipients are not approved.")
        if not human_approved:
            return ReleaseDecision(False, normalized, "Email release requires approval.")
        return ReleaseDecision(True, normalized, "All report release controls passed.")
