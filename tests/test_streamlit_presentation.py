"""Tests for concise Streamlit management presentation."""

import pandas as pd

from src.ui.streamlit import (
    _build_hybrid_status_messages,
    _has_chart_values,
    _prepare_table_dataframe,
    _select_visible_chat_answer,
    _shorten_management_summary,
)


def test_dashboard_response_uses_concise_chat_message() -> None:
    result = _select_visible_chat_answer(
        response={"dashboard": {"kpi_cards": []}},
        answer="A long detailed analysis.",
        clarification_required=False,
    )

    assert result == (
        "Analysis completed. Review the management dashboard below "
        "for the key results and supporting details."
    )


def test_non_dashboard_response_keeps_answer() -> None:
    result = _select_visible_chat_answer(
        response={"dashboard": {}},
        answer="Please provide a reporting period.",
        clarification_required=True,
    )

    assert result == "Please provide a reporting period."


def test_management_summary_is_limited_to_500_characters() -> None:
    result = _shorten_management_summary("word " * 150)

    assert len(result) <= 501
    assert result.endswith("…")


def test_short_management_summary_is_unchanged() -> None:
    summary = "April revenue exceeded budget."

    assert _shorten_management_summary(summary) == summary


def test_chart_values_reject_empty_placeholder_records() -> None:
    assert not _has_chart_values(
        [{"period": "2026-04", "actual": None}],
        value_fields=("actual", "budget"),
    )


def test_chart_values_accept_numeric_records() -> None:
    assert _has_chart_values(
        [{"period": "2026-04", "actual": 100.0}],
        value_fields=("actual", "budget"),
    )


def test_hybrid_status_presents_review_and_reconciliation() -> None:
    messages = _build_hybrid_status_messages(
        {
            "hybrid_metadata": {
                "execution_mode": "autonomous",
                "autonomous_status": "completed",
                "review_decision": "approved_with_caveats",
                "reconciliation_warnings": ["Excluded category."],
            }
        }
    )

    text = " ".join(message for _, message in messages)
    assert "Autonomous" in text
    assert "Approved With Caveats" in text
    assert "Excluded category" in text


def test_hybrid_status_presents_deterministic_fallback() -> None:
    messages = _build_hybrid_status_messages(
        {
            "hybrid_metadata": {
                "execution_mode": "deterministic",
                "autonomous_status": "fallback",
                "fallback_used": True,
                "fallback_reason": "Reviewer requested replan.",
            }
        }
    )

    assert any(
        level == "warning" and "Reviewer requested replan" in message
        for level, message in messages
    )


def test_hybrid_status_presents_safe_reviewer_failure_counts() -> None:
    messages = _build_hybrid_status_messages(
        {
            "hybrid_metadata": {
                "review_decision": "failed",
                "review_issue_counts": {
                    "unsupported_claims": 2,
                    "missing_evidence": 1,
                },
            }
        }
    )

    assert (
        "error",
        "Reviewer status: Failed",
    ) in messages
    assert (
        "warning",
        "Reviewer finding: Unsupported Claims (2).",
    ) in messages
    assert (
        "warning",
        "Reviewer finding: Missing Evidence (1).",
    ) in messages


def test_pnl_table_is_presented_vertically() -> None:
    dataframe = pd.DataFrame(
        [
            {
                "month": "2026-05",
                "revenue_actual": 4_678_564.85,
                "revenue_budget": 5_579_766.62,
                "revenue_variance": -901_201.77,
                "revenue_variance_percentage": -16.15,
            }
        ]
    )

    result = _prepare_table_dataframe(
        table_payload={
            "title": "Actual vs Budget P&L",
        },
        dataframe=dataframe,
    )

    assert list(result.columns) == [
        "Metric",
        "Actual",
        "Budget",
        "Variance",
        "Variance %",
    ]
    assert result.to_dict("records") == [
        {
            "Metric": "Revenue",
            "Actual": "4,678,564.85",
            "Budget": "5,579,766.62",
            "Variance": "-901,201.77",
            "Variance %": "-16.15%",
        },
    ]


def test_non_pnl_table_keeps_horizontal_layout() -> None:
    dataframe = pd.DataFrame(
        [{"month": "2026-05", "revenue": 100.0}]
    )

    result = _prepare_table_dataframe(
        table_payload={"title": "KPI Detail"},
        dataframe=dataframe,
    )

    pd.testing.assert_frame_equal(
        result,
        dataframe,
    )
