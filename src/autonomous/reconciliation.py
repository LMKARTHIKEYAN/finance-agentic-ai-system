"""Deterministic reconciliation of registered finance-tool evidence."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from src.autonomous.evidence_registry import EvidenceRegistry
from src.autonomous.schemas import (
    EvidenceRecord,
    ReconciliationCheck,
    ReconciliationResult,
)


class AutonomousReconciler:
    """Validate existing result contracts without recalculating finance."""

    def __init__(
        self,
        *,
        revenue_variance_tolerance: float = 0.01,
        gp_tolerance: float = 0.0001,
    ) -> None:
        self._revenue_variance_tolerance = _non_negative_number(
            revenue_variance_tolerance,
            "revenue_variance_tolerance",
        )
        self._gp_tolerance = _non_negative_number(
            gp_tolerance,
            "gp_tolerance",
        )

    def reconcile(
        self,
        registry: EvidenceRegistry,
        evidence_ids: Iterable[str] | None = None,
    ) -> ReconciliationResult:
        """Reconcile selected evidence and update registry status."""

        if not isinstance(registry, EvidenceRegistry):
            raise TypeError("registry must be EvidenceRegistry.")

        records = (
            registry.list()
            if evidence_ids is None
            else tuple(registry.get(item) for item in evidence_ids)
        )
        if not records:
            raise ValueError("At least one evidence record is required.")

        checks: list[ReconciliationCheck] = []
        warnings: list[str] = []
        record_statuses: dict[str, bool] = {}

        for record in records:
            record_checks, record_warnings = self._check_record(record)
            checks.extend(record_checks)
            warnings.extend(record_warnings)
            record_statuses[record.evidence_id] = all(
                item.passed for item in record_checks
            )

        period_check = _scope_check(records, "period")
        category_check = _scope_check(records, "category")
        checks.extend((period_check, category_check))
        scope_passed = period_check.passed and category_check.passed

        for record in records:
            registry.mark_reconciled(
                record.evidence_id,
                passed=(
                    record_statuses[record.evidence_id]
                    and scope_passed
                ),
            )

        return ReconciliationResult(
            passed=all(item.passed for item in checks),
            checks=tuple(checks),
            warnings=tuple(warnings),
        )

    def _check_record(
        self,
        record: EvidenceRecord,
    ) -> tuple[list[ReconciliationCheck], list[str]]:
        checks = [
            ReconciliationCheck(
                name=f"{record.evidence_id}:tool_status",
                passed=record.tool_status == "completed",
                details=(
                    "Tool completed."
                    if record.tool_status == "completed"
                    else f"Tool status is {record.tool_status!r}."
                ),
                evidence_ids=(record.evidence_id,),
            )
        ]
        warnings = list(record.warnings)

        if record.tool_status != "completed":
            return checks, warnings

        payload = record.compact_payload
        if record.result_type == "kpi":
            checks.append(
                _required_fields_check(
                    record,
                    payload,
                    (
                        "requested_kpis",
                        "selected_kpis",
                        "unavailable_kpis",
                        "unknown_kpis",
                    ),
                )
            )
        elif record.result_type == "pnl":
            checks.append(
                _required_fields_check(
                    record,
                    payload,
                    (
                        "actual_pnl",
                        "budget_pnl",
                        "variance_pnl",
                        "pnl_summary",
                        "available_months",
                        "excluded_actual_months",
                        "excluded_budget_months",
                    ),
                )
            )
            for field_name in (
                "excluded_actual_months",
                "excluded_budget_months",
            ):
                if payload.get(field_name):
                    warnings.append(
                        f"{record.evidence_id} reports {field_name}."
                    )
        elif record.result_type == "revenue_variance":
            value = payload.get("variance_check")
            passed = _within_tolerance(
                value,
                self._revenue_variance_tolerance,
            )
            checks.append(
                ReconciliationCheck(
                    name=f"{record.evidence_id}:revenue_variance",
                    passed=passed,
                    details="Existing variance_check is within tolerance."
                    if passed
                    else "Existing variance_check is missing or outside tolerance.",
                    evidence_ids=(record.evidence_id,),
                )
            )
        elif record.result_type == "gp_decomposition":
            required = _required_fields_check(
                record,
                payload,
                (
                    "product_level",
                    "portfolio_level",
                    "reconciliation_status",
                    "reconciliation_difference",
                ),
            )
            checks.append(required)
            levels_passed = (
                isinstance(payload.get("product_level"), list)
                and bool(payload.get("product_level"))
                and isinstance(payload.get("portfolio_level"), dict)
                and bool(payload.get("portfolio_level"))
            )
            checks.append(
                ReconciliationCheck(
                    name=f"{record.evidence_id}:gp_levels",
                    passed=levels_passed,
                    details=(
                        "Product- and Portfolio-Level GP% outputs are present."
                        if levels_passed
                        else "Product- or Portfolio-Level GP% output is empty."
                    ),
                    evidence_ids=(record.evidence_id,),
                )
            )
            reconciliation_status = payload.get(
                "reconciliation_status"
            )
            status_passed = (
                isinstance(reconciliation_status, str)
                and reconciliation_status.strip().lower()
                in {"pass", "passed"}
                and _within_tolerance(
                    payload.get("reconciliation_difference"),
                    self._gp_tolerance,
                )
            )
            checks.append(
                ReconciliationCheck(
                    name=f"{record.evidence_id}:gp_bridge",
                    passed=status_passed,
                    details="Existing GP% reconciliation passed."
                    if status_passed
                    else "Existing GP% reconciliation failed.",
                    evidence_ids=(record.evidence_id,),
                )
            )

        return checks, warnings


def _required_fields_check(
    record: EvidenceRecord,
    payload: dict[str, Any],
    required_fields: tuple[str, ...],
) -> ReconciliationCheck:
    missing = [item for item in required_fields if item not in payload]
    return ReconciliationCheck(
        name=f"{record.evidence_id}:required_fields",
        passed=not missing,
        details=(
            "Required result fields are present."
            if not missing
            else f"Missing result fields: {', '.join(missing)}."
        ),
        evidence_ids=(record.evidence_id,),
    )


def _scope_check(
    records: tuple[EvidenceRecord, ...],
    field_name: str,
) -> ReconciliationCheck:
    values = {
        getattr(record, field_name)
        for record in records
        if getattr(record, field_name) is not None
    }
    passed = len(values) <= 1
    return ReconciliationCheck(
        name=f"scope:{field_name}",
        passed=passed,
        details=(
            f"Evidence {field_name} is consistent."
            if passed
            else f"Evidence {field_name} values are inconsistent."
        ),
        evidence_ids=tuple(item.evidence_id for item in records),
    )


def _within_tolerance(value: Any, tolerance: float) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return abs(float(value)) <= tolerance


def _non_negative_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric.")
    numeric = float(value)
    if numeric < 0:
        raise ValueError(f"{field_name} cannot be negative.")
    return numeric
