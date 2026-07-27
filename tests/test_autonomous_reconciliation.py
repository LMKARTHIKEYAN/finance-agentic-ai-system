"""Tests for deterministic autonomous evidence reconciliation."""

from src.autonomous.evidence_registry import EvidenceRegistry
from src.autonomous.reconciliation import AutonomousReconciler
from src.autonomous.schemas import ToolResult


def _register(
    registry: EvidenceRegistry,
    result_type: str,
    payload: dict,
    *,
    period: str = "2026-04",
    category: str | None = None,
    status: str = "completed",
) -> str:
    record = registry.register(
        ToolResult(
            call_id=f"{result_type}-call",
            tool_name=f"{result_type}-tool",
            status=status,  # type: ignore[arg-type]
            payload=payload,
        ),
        result_type=result_type,
        period=period,
        category=category,
    )
    return record.evidence_id


def test_valid_kpi_structure_passes() -> None:
    registry = EvidenceRegistry()
    evidence_id = _register(
        registry,
        "kpi",
        {
            "requested_kpis": ["actual_revenue"],
            "selected_kpis": [{"value": 100}],
            "unavailable_kpis": [],
            "unknown_kpis": [],
        },
    )

    result = AutonomousReconciler().reconcile(registry)

    assert result.passed is True
    assert registry.get(evidence_id).reconciled is True
    assert registry.get(evidence_id).verified is False


def test_missing_kpi_fields_fail() -> None:
    registry = EvidenceRegistry()
    _register(registry, "kpi", {"selected_kpis": []})

    result = AutonomousReconciler().reconcile(registry)

    assert result.passed is False


def test_valid_pnl_structure_passes_and_exclusions_warn() -> None:
    registry = EvidenceRegistry()
    _register(
        registry,
        "pnl",
        {
            "actual_pnl": [],
            "budget_pnl": [],
            "variance_pnl": [],
            "pnl_summary": {},
            "available_months": ["2026-04"],
            "excluded_actual_months": ["2026-03"],
            "excluded_budget_months": [],
        },
    )

    result = AutonomousReconciler().reconcile(registry)

    assert result.passed is True
    assert any("excluded_actual_months" in item for item in result.warnings)


def test_missing_pnl_variance_fails() -> None:
    registry = EvidenceRegistry()
    _register(
        registry,
        "pnl",
        {
            "actual_pnl": [],
            "budget_pnl": [],
            "pnl_summary": {},
            "available_months": [],
            "excluded_actual_months": [],
            "excluded_budget_months": [],
        },
    )

    assert AutonomousReconciler().reconcile(registry).passed is False


def test_revenue_variance_uses_existing_check() -> None:
    registry = EvidenceRegistry()
    _register(
        registry,
        "revenue_variance",
        {"variance_check": 0.001},
    )

    assert AutonomousReconciler(
        revenue_variance_tolerance=0.01
    ).reconcile(registry).passed is True


def test_revenue_variance_outside_tolerance_fails() -> None:
    registry = EvidenceRegistry()
    _register(
        registry,
        "revenue_variance",
        {"variance_check": 1.0},
    )

    assert AutonomousReconciler().reconcile(registry).passed is False


def test_gp_decomposition_requires_both_levels_and_passed_bridge() -> None:
    registry = EvidenceRegistry()
    _register(
        registry,
        "gp_decomposition",
        {
            "product_level": [{"category": "2W"}],
            "portfolio_level": {"actual_gp_percentage": 30},
            "reconciliation_status": "passed",
            "reconciliation_difference": 0,
        },
    )

    assert AutonomousReconciler().reconcile(registry).passed is True


def test_failed_gp_bridge_fails() -> None:
    registry = EvidenceRegistry()
    _register(
        registry,
        "gp_decomposition",
        {
            "product_level": [],
            "portfolio_level": {},
            "reconciliation_status": "failed",
            "reconciliation_difference": 1,
        },
    )

    assert AutonomousReconciler().reconcile(registry).passed is False


def test_empty_gp_analysis_level_fails() -> None:
    registry = EvidenceRegistry()
    _register(
        registry,
        "gp_decomposition",
        {
            "product_level": [],
            "portfolio_level": {"actual_gp_percentage": 30},
            "reconciliation_status": "passed",
            "reconciliation_difference": 0,
        },
    )

    assert AutonomousReconciler().reconcile(registry).passed is False


def test_period_mismatch_fails_combined_reconciliation() -> None:
    registry = EvidenceRegistry()
    payload = {
        "requested_kpis": [],
        "selected_kpis": [],
        "unavailable_kpis": [],
        "unknown_kpis": [],
    }
    _register(registry, "kpi", payload, period="2026-04")
    _register(registry, "kpi", payload, period="2026-05")

    result = AutonomousReconciler().reconcile(registry)

    assert result.passed is False
    assert any(
        item.name == "scope:period" and not item.passed
        for item in result.checks
    )
    assert all(not item.reconciled for item in registry.list())


def test_category_mismatch_fails_combined_reconciliation() -> None:
    registry = EvidenceRegistry()
    payload = {
        "requested_kpis": [],
        "selected_kpis": [],
        "unavailable_kpis": [],
        "unknown_kpis": [],
    }
    _register(registry, "kpi", payload, category="2W")
    _register(registry, "kpi", payload, category="3W")

    assert AutonomousReconciler().reconcile(registry).passed is False


def test_failed_tool_result_fails_reconciliation() -> None:
    registry = EvidenceRegistry()
    _register(
        registry,
        "kpi",
        {},
        status="failed",
    )

    assert AutonomousReconciler().reconcile(registry).passed is False
