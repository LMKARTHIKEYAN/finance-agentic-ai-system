"""Tests for the in-memory evidence registry."""

import pandas as pd
import pytest

from src.autonomous.evidence_registry import EvidenceRegistry
from src.autonomous.schemas import ToolResult


def _tool_result(
    *,
    status: str = "completed",
    payload: dict | None = None,
) -> ToolResult:
    return ToolResult(
        call_id="call-1",
        tool_name="generate_validated_pnl_analysis",
        status=status,  # type: ignore[arg-type]
        payload=payload or {"actual_pnl": []},
    )


def test_registry_registers_and_retrieves_evidence() -> None:
    registry = EvidenceRegistry()

    registered = registry.register(
        _tool_result(),
        result_type="pnl",
        full_result=object(),
        period="2026-04",
    )

    assert registered.evidence_id == "pnl-001"
    assert registry.get("pnl-001").period == "2026-04"


def test_registry_rejects_duplicate_evidence_id() -> None:
    registry = EvidenceRegistry()
    registry.register(
        _tool_result(),
        result_type="pnl",
        evidence_id="same",
    )

    with pytest.raises(ValueError, match="already"):
        registry.register(
            _tool_result(),
            result_type="pnl",
            evidence_id="same",
        )


def test_registry_returns_defensive_compact_copy() -> None:
    registry = EvidenceRegistry()
    registry.register(
        _tool_result(payload={"items": [{"value": 1}]}),
        result_type="pnl",
    )

    first = registry.get("pnl-001")
    first.compact_payload["items"][0]["value"] = 999

    assert (
        registry.get("pnl-001").compact_payload["items"][0]["value"]
        == 1
    )


def test_registry_retains_full_result_separately() -> None:
    registry = EvidenceRegistry()
    full_result = object()
    registry.register(
        _tool_result(),
        result_type="pnl",
        full_result=full_result,
    )

    assert registry.get_full_result("pnl-001") is full_result
    assert "full_result" not in registry.get("pnl-001").model_dump()


def test_registry_filters_by_result_type() -> None:
    registry = EvidenceRegistry()
    registry.register(_tool_result(), result_type="pnl")
    registry.register(
        ToolResult(
            call_id="call-2",
            tool_name="calculate_validated_kpis",
            status="completed",
            payload={"requested_kpis": []},
        ),
        result_type="kpi",
    )

    assert len(registry.list(result_type="pnl")) == 1
    assert registry.list(result_type="pnl")[0].result_type == "pnl"


def test_registry_rejects_dataframe_payload() -> None:
    registry = EvidenceRegistry()

    with pytest.raises(TypeError, match="DataFrames"):
        registry.register(
            _tool_result(payload={"data": pd.DataFrame({"value": [1]})}),
            result_type="pnl",
        )


def test_failed_evidence_cannot_be_reconciled_as_passed() -> None:
    registry = EvidenceRegistry()
    registry.register(
        _tool_result(status="failed"),
        result_type="pnl",
    )

    with pytest.raises(ValueError, match="cannot reconcile"):
        registry.mark_reconciled("pnl-001", passed=True)


def test_unreconciled_evidence_cannot_be_verified() -> None:
    registry = EvidenceRegistry()
    registry.register(_tool_result(), result_type="pnl")

    with pytest.raises(ValueError, match="reconcile"):
        registry.mark_verified("pnl-001")


def test_reconciled_evidence_can_be_verified() -> None:
    registry = EvidenceRegistry()
    registry.register(_tool_result(), result_type="pnl")
    registry.mark_reconciled("pnl-001", passed=True)

    verified = registry.mark_verified("pnl-001")

    assert verified.verified is True


def test_registry_rejects_unknown_evidence() -> None:
    with pytest.raises(KeyError, match="Unknown"):
        EvidenceRegistry().get("missing")
