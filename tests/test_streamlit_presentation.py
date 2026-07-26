"""Tests for concise Streamlit management presentation."""

from src.ui.streamlit import (
    _has_chart_values,
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
