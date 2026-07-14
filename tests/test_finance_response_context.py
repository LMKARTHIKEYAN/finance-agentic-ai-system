"""Unit tests for structured finance response context generation."""

from __future__ import annotations

from copy import deepcopy

import pytest

from src.api.finance_response_context import (
    FinanceResponseContextBuilder,
    build_finance_response_context,
)


def test_variance_flow_groups_primary_and_supporting_results() -> None:
    """Variance flow should prioritize variance, operations, and budget outputs."""

    finance_analysis = {
        "variance_result": {"variance": 100.0},
        "operations_result": {"actual_revenue": 500.0},
        "budget_result": {"budget_revenue": 400.0},
        "root_cause_result": {"causes": ["price"]},
        "recommendation_result": {"actions": ["Monitor pricing"]},
    }

    context = build_finance_response_context(
        selected_flow="variance",
        finance_analysis=finance_analysis,
    )

    assert list(context["primary_analysis"]) == [
        "variance_result",
        "operations_result",
        "budget_result",
    ]
    assert list(context["supporting_analysis"]) == [
        "root_cause_result",
        "recommendation_result",
    ]
    assert context["selected_flow"] == "variance"
    assert context["data_availability"]["has_primary_analysis"] is True
    assert context["data_availability"]["has_supporting_analysis"] is True


@pytest.mark.parametrize(
    ("selected_flow", "expected_primary_keys"),
    [
        ("kpi", ["kpi_result", "operations_result"]),
        ("budget", ["budget_result"]),
        ("forecast", ["forecast_result", "operations_result"]),
        ("variance", ["variance_result", "operations_result", "budget_result"]),
        ("scenario", ["scenario_result", "forecast_result"]),
        (
            "full",
            [
                "report_result",
                "commentary_result",
                "kpi_result",
                "variance_result",
                "forecast_result",
                "scenario_result",
                "budget_result",
                "operations_result",
            ],
        ),
    ],
)
def test_each_known_flow_uses_expected_primary_results(
    selected_flow: str,
    expected_primary_keys: list[str],
) -> None:
    """Each supported flow should use its configured primary result order."""

    finance_analysis = {
        "operations_result": {"value": "operations"},
        "budget_result": {"value": "budget"},
        "forecast_result": {"value": "forecast"},
        "scenario_result": {"value": "scenario"},
        "variance_result": {"value": "variance"},
        "finance_rules_result": {"value": "rules"},
        "anomaly_result": {"value": "anomaly"},
        "root_cause_result": {"value": "root cause"},
        "recommendation_result": {"value": "recommendation"},
        "kpi_result": {"value": "kpi"},
        "commentary_result": {"value": "commentary"},
        "report_result": {"value": "report"},
    }

    context = build_finance_response_context(
        selected_flow=selected_flow,
        finance_analysis=finance_analysis,
    )

    assert list(context["primary_analysis"]) == expected_primary_keys


def test_unknown_flow_uses_safe_default_grouping() -> None:
    """Unknown flows should fall back to the default primary/supporting order."""

    finance_analysis = {
        "report_result": {"summary": "Executive summary"},
        "commentary_result": {"commentary": "Management commentary"},
        "anomaly_result": {"items": ["Revenue spike"]},
        "finance_rules_result": {"status": "passed"},
    }

    context = build_finance_response_context(
        selected_flow="custom-analysis",
        finance_analysis=finance_analysis,
    )

    assert context["selected_flow"] == "custom-analysis"
    assert list(context["primary_analysis"]) == [
        "report_result",
        "commentary_result",
    ]
    assert list(context["supporting_analysis"]) == [
        "anomaly_result",
        "finance_rules_result",
    ]


def test_flow_name_is_normalized() -> None:
    """Flow names should be stripped and converted to lowercase."""

    context = build_finance_response_context(
        selected_flow="  VARIANCE  ",
        finance_analysis={
            "variance_result": {"variance": 25.0},
        },
    )

    assert context["selected_flow"] == "variance"
    assert list(context["primary_analysis"]) == ["variance_result"]


def test_missing_and_empty_outputs_are_reported_as_unavailable() -> None:
    """Missing, None, blank, and empty outputs should not be treated as evidence."""

    finance_analysis = {
        "variance_result": {"variance": 25.0},
        "operations_result": None,
        "budget_result": {},
        "root_cause_result": [],
        "commentary_result": "   ",
    }

    context = build_finance_response_context(
        selected_flow="variance",
        finance_analysis=finance_analysis,
    )

    availability = context["data_availability"]

    assert availability["available_results"] == ["variance_result"]
    assert "operations_result" in availability["unavailable_results"]
    assert "budget_result" in availability["unavailable_results"]
    assert "root_cause_result" in availability["unavailable_results"]
    assert "commentary_result" in availability["unavailable_results"]
    assert list(context["primary_analysis"]) == ["variance_result"]
    assert context["supporting_analysis"] == {}
    assert availability["has_primary_analysis"] is True
    assert availability["has_supporting_analysis"] is False


def test_internal_sources_match_included_agent_outputs() -> None:
    """Internal source metadata should identify each included agent result."""

    context = build_finance_response_context(
        selected_flow="variance",
        finance_analysis={
            "variance_result": {"variance": 100.0},
            "root_cause_result": {"causes": ["volume"]},
        },
    )

    sources = context["internal_sources"]

    assert [source["result_key"] for source in sources] == [
        "variance_result",
        "root_cause_result",
    ]
    assert sources[0]["source_name"] == "Revenue Variance Agent"
    assert sources[1]["source_name"] == "Root Cause Agent"
    assert all(source["purpose"] for source in sources)


def test_unknown_agent_output_is_preserved_as_supporting_analysis() -> None:
    """Additional supplied outputs should be preserved instead of discarded."""

    context = build_finance_response_context(
        selected_flow="variance",
        finance_analysis={
            "variance_result": {"variance": 100.0},
            "custom_agent_result": {"custom_metric": 42},
        },
    )

    assert context["supporting_analysis"]["custom_agent_result"] == {
        "custom_metric": 42
    }

    custom_source = next(
        source
        for source in context["internal_sources"]
        if source["result_key"] == "custom_agent_result"
    )
    assert custom_source["source_name"] == "custom_agent_result"
    assert custom_source["purpose"] == (
        "Provides additional supplied finance analysis."
    )


def test_builder_does_not_modify_input_values() -> None:
    """Building context should not mutate the original finance analysis."""

    finance_analysis = {
        "variance_result": {
            "summary": {
                "actual_revenue": 940_200_000.0,
                "budget_revenue": 60_400_000.0,
            },
            "drivers": ["price", "volume"],
        },
        "recommendation_result": {
            "actions": ["Monitor pricing"],
        },
    }
    original = deepcopy(finance_analysis)

    context = build_finance_response_context(
        selected_flow="variance",
        finance_analysis=finance_analysis,
    )

    context["primary_analysis"]["variance_result"]["summary"][
        "actual_revenue"
    ] = 0.0
    context["supporting_analysis"]["recommendation_result"]["actions"].append(
        "Added later"
    )

    assert finance_analysis == original


def test_empty_analysis_returns_explicit_no_data_context() -> None:
    """An empty analysis mapping should return a valid context with no evidence."""

    context = build_finance_response_context(
        selected_flow="forecast",
        finance_analysis={},
    )

    assert context["primary_analysis"] == {}
    assert context["supporting_analysis"] == {}
    assert context["internal_sources"] == []
    assert context["data_availability"]["available_results"] == []
    assert context["data_availability"]["has_primary_analysis"] is False
    assert context["data_availability"]["has_supporting_analysis"] is False


def test_context_contains_guardrail_instruction_and_version() -> None:
    """The context should include its schema version and anti-invention guardrail."""

    context = build_finance_response_context(
        selected_flow="budget",
        finance_analysis={"budget_result": {"budget": 500.0}},
    )

    assert context["context_version"] == "1.0"
    assert "Use only the supplied values" in context["response_instruction"]
    assert "Do not invent" in context["response_instruction"]


@pytest.mark.parametrize(
    "invalid_flow",
    [None, 123, [], {}],
)
def test_non_string_flow_is_rejected(invalid_flow: object) -> None:
    """The builder should reject non-string flow values."""

    with pytest.raises(
        TypeError,
        match="selected_flow must be a string",
    ):
        build_finance_response_context(
            selected_flow=invalid_flow,  # type: ignore[arg-type]
            finance_analysis={},
        )


@pytest.mark.parametrize("empty_flow", ["", " ", "\t\n"])
def test_empty_flow_is_rejected(empty_flow: str) -> None:
    """The builder should reject empty flow names."""

    with pytest.raises(
        ValueError,
        match="selected_flow cannot be empty",
    ):
        build_finance_response_context(
            selected_flow=empty_flow,
            finance_analysis={},
        )


@pytest.mark.parametrize(
    "invalid_analysis",
    [None, [], "not-a-mapping", 123],
)
def test_non_mapping_finance_analysis_is_rejected(
    invalid_analysis: object,
) -> None:
    """The builder should reject finance-analysis values that are not mappings."""

    with pytest.raises(
        TypeError,
        match="finance_analysis must be a mapping",
    ):
        build_finance_response_context(
            selected_flow="variance",
            finance_analysis=invalid_analysis,  # type: ignore[arg-type]
        )


def test_builder_class_and_convenience_function_return_same_result() -> None:
    """The class API and convenience function should behave identically."""

    finance_analysis = {
        "kpi_result": {"revenue": 500.0},
        "operations_result": {"orders": 25},
    }

    class_result = FinanceResponseContextBuilder().build(
        selected_flow="kpi",
        finance_analysis=finance_analysis,
    )
    function_result = build_finance_response_context(
        selected_flow="kpi",
        finance_analysis=finance_analysis,
    )

    assert class_result == function_result