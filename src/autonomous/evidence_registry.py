"""In-memory evidence registry for one autonomous workflow."""

from __future__ import annotations

import copy
from collections.abc import Iterator
from typing import Any

import pandas as pd

from src.autonomous.schemas import EvidenceRecord, ToolResult


class EvidenceRegistry:
    """Store compact evidence separately from full internal results."""

    def __init__(self) -> None:
        self._records: dict[str, EvidenceRecord] = {}
        self._full_results: dict[str, Any] = {}
        self._counter = 0

    def register(
        self,
        tool_result: ToolResult,
        *,
        result_type: str,
        full_result: Any = None,
        evidence_id: str | None = None,
        period: str | None = None,
        category: str | None = None,
    ) -> EvidenceRecord:
        """Register an independent copy of one deterministic tool result."""

        if not isinstance(tool_result, ToolResult):
            raise TypeError("tool_result must be ToolResult.")
        cleaned_type = _required_text(result_type, "result_type")
        _reject_dataframes(tool_result.payload)

        resolved_id = evidence_id or tool_result.evidence_id
        if resolved_id is None:
            self._counter += 1
            resolved_id = f"{cleaned_type}-{self._counter:03d}"
        resolved_id = _required_text(resolved_id, "evidence_id")

        if resolved_id in self._records:
            raise ValueError(
                f"Evidence ID {resolved_id!r} is already registered."
            )

        record = EvidenceRecord(
            evidence_id=resolved_id,
            source_tool=tool_result.tool_name,
            result_type=cleaned_type,
            tool_status=tool_result.status,
            compact_payload=copy.deepcopy(tool_result.payload),
            period=period,
            category=category,
            warnings=tool_result.warnings,
        )
        self._records[resolved_id] = record
        self._full_results[resolved_id] = full_result
        return record.model_copy(deep=True)

    def get(self, evidence_id: str) -> EvidenceRecord:
        """Return a defensive copy of registered compact evidence."""

        resolved_id = _required_text(evidence_id, "evidence_id")
        try:
            record = self._records[resolved_id]
        except KeyError as exc:
            raise KeyError(
                f"Unknown evidence ID: {resolved_id!r}."
            ) from exc
        return record.model_copy(deep=True)

    def get_full_result(self, evidence_id: str) -> Any:
        """Return the internal deterministic result for trusted code."""

        resolved_id = _required_text(evidence_id, "evidence_id")
        if resolved_id not in self._full_results:
            raise KeyError(f"Unknown evidence ID: {resolved_id!r}.")
        return self._full_results[resolved_id]

    def list(self, *, result_type: str | None = None) -> tuple[EvidenceRecord, ...]:
        """List defensive copies, optionally filtered by result type."""

        records: Iterator[EvidenceRecord] = iter(self._records.values())
        if result_type is not None:
            cleaned_type = _required_text(result_type, "result_type")
            records = (
                item
                for item in records
                if item.result_type == cleaned_type
            )
        return tuple(item.model_copy(deep=True) for item in records)

    def mark_reconciled(
        self,
        evidence_id: str,
        *,
        passed: bool,
    ) -> EvidenceRecord:
        """Record deterministic reconciliation status."""

        if not isinstance(passed, bool):
            raise TypeError("passed must be a boolean.")
        record = self.get(evidence_id)
        if record.tool_status != "completed" and passed:
            raise ValueError("Failed or skipped evidence cannot reconcile.")
        updated = record.model_copy(
            update={"reconciled": passed, "verified": False},
            deep=True,
        )
        self._records[record.evidence_id] = updated
        return updated.model_copy(deep=True)

    def mark_verified(self, evidence_id: str) -> EvidenceRecord:
        """Mark completed, reconciled evidence ready for reviewer use."""

        record = self.get(evidence_id)
        if record.tool_status != "completed":
            raise ValueError("Only completed evidence can be verified.")
        if not record.reconciled:
            raise ValueError("Evidence must reconcile before verification.")
        updated = record.model_copy(
            update={"verified": True},
            deep=True,
        )
        self._records[record.evidence_id] = updated
        return updated.model_copy(deep=True)


def _reject_dataframes(value: Any) -> None:
    if isinstance(value, pd.DataFrame):
        raise TypeError("Evidence payloads cannot contain DataFrames.")
    if isinstance(value, dict):
        for item in value.values():
            _reject_dataframes(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_dataframes(item)


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty.")
    return cleaned
