"""
Tests for the P&L Commentary Agent.

These tests verify that PnlCommentaryAgent:

- Uses the PnlAgent summary without recalculating P&L
- Generates revenue commentary
- Generates profitability commentary
- Interprets cost variances correctly
- Interprets gross-margin movement as percentage points
- Identifies favourable drivers
- Identifies financial risks
- Identifies material variances
- Validates missing and invalid input
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import pytest

from src.agents.reporting.pnl_commentary_agent import (
    PnlCommentaryAgent,
    PnlCommentaryResult,
)


@dataclass
class SamplePnlResult:
    """Simple P&L result used for unit testing."""

    summary: dict[str, Any]


@pytest.fixture
def agent() -> PnlCommentaryAgent:
    """Return a P&L Commentary Agent instance."""

    return PnlCommentaryAgent()


@pytest.fixture
def favourable_summary() -> dict[str, Any]:
    """
    Return a P&L summary containing mostly favourable results.

    Revenue and profits are above budget, while most costs are below
    budget.
    """

    return {
        "actual": {
            "revenue": 1_100_000.0,
            "direct_cost": 620_000.0,
            "gross_profit": 480_000.0,
            "gross_margin_percentage": 43.64,
            "sales_marketing": 70_000.0,
            "other_opex": 90_000.0,
            "ebitda": 320_000.0,
            "depreciation": 35_000.0,
            "ebit": 285_000.0,
            "interest": 15_000.0,
            "ebt": 270_000.0,
        },
        "budget": {
            "revenue": 1_000_000.0,
            "direct_cost": 650_000.0,
            "gross_profit": 350_000.0,
            "gross_margin_percentage": 35.0,
            "sales_marketing": 75_000.0,
            "other_opex": 100_000.0,
            "ebitda": 175_000.0,
            "depreciation": 40_000.0,
            "ebit": 135_000.0,
            "interest": 20_000.0,
            "ebt": 115_000.0,
        },
        "variance": {
            "revenue_variance": 100_000.0,
            "revenue_variance_percentage": 10.0,
            "direct_cost_variance": -30_000.0,
            "direct_cost_variance_percentage": -4.62,
            "gross_profit_variance": 130_000.0,
            "gross_profit_variance_percentage": 37.14,
            "gross_margin_percentage_point_variance": 8.64,
            "sales_marketing_variance": -5_000.0,
            "sales_marketing_variance_percentage": -6.67,
            "other_opex_variance": -10_000.0,
            "other_opex_variance_percentage": -10.0,
            "ebitda_variance": 145_000.0,
            "ebitda_variance_percentage": 82.86,
            "depreciation_variance": -5_000.0,
            "depreciation_variance_percentage": -12.5,
            "ebit_variance": 150_000.0,
            "ebit_variance_percentage": 111.11,
            "interest_variance": -5_000.0,
            "interest_variance_percentage": -25.0,
            "ebt_variance": 155_000.0,
            "ebt_variance_percentage": 134.78,
        },
    }


@pytest.fixture
def unfavourable_summary() -> dict[str, Any]:
    """
    Return a P&L summary containing mostly unfavourable results.

    Revenue and profits are below budget, while costs are above budget.
    """

    return {
        "actual": {
            "revenue": 800_000.0,
            "direct_cost": 580_000.0,
            "gross_profit": 220_000.0,
            "gross_margin_percentage": 27.5,
            "sales_marketing": 90_000.0,
            "other_opex": 110_000.0,
            "ebitda": 20_000.0,
            "depreciation": 45_000.0,
            "ebit": -25_000.0,
            "interest": 25_000.0,
            "ebt": -50_000.0,
        },
        "budget": {
            "revenue": 1_000_000.0,
            "direct_cost": 550_000.0,
            "gross_profit": 450_000.0,
            "gross_margin_percentage": 45.0,
            "sales_marketing": 75_000.0,
            "other_opex": 90_000.0,
            "ebitda": 285_000.0,
            "depreciation": 40_000.0,
            "ebit": 245_000.0,
            "interest": 20_000.0,
            "ebt": 225_000.0,
        },
        "variance": {
            "revenue_variance": -200_000.0,
            "revenue_variance_percentage": -20.0,
            "direct_cost_variance": 30_000.0,
            "direct_cost_variance_percentage": 5.45,
            "gross_profit_variance": -230_000.0,
            "gross_profit_variance_percentage": -51.11,
            "gross_margin_percentage_point_variance": -17.5,
            "sales_marketing_variance": 15_000.0,
            "sales_marketing_variance_percentage": 20.0,
            "other_opex_variance": 20_000.0,
            "other_opex_variance_percentage": 22.22,
            "ebitda_variance": -265_000.0,
            "ebitda_variance_percentage": -92.98,
            "depreciation_variance": 5_000.0,
            "depreciation_variance_percentage": 12.5,
            "ebit_variance": -270_000.0,
            "ebit_variance_percentage": -110.2,
            "interest_variance": 5_000.0,
            "interest_variance_percentage": 25.0,
            "ebt_variance": -275_000.0,
            "ebt_variance_percentage": -122.22,
        },
    }


@pytest.fixture
def neutral_summary() -> dict[str, Any]:
    """Return a P&L summary where Actual equals Budget."""

    actual = {
        "revenue": 1_000_000.0,
        "direct_cost": 600_000.0,
        "gross_profit": 400_000.0,
        "gross_margin_percentage": 40.0,
        "sales_marketing": 80_000.0,
        "other_opex": 100_000.0,
        "ebitda": 220_000.0,
        "depreciation": 40_000.0,
        "ebit": 180_000.0,
        "interest": 20_000.0,
        "ebt": 160_000.0,
    }

    return {
        "actual": actual,
        "budget": deepcopy(actual),
        "variance": {
            "revenue_variance": 0.0,
            "revenue_variance_percentage": 0.0,
            "direct_cost_variance": 0.0,
            "direct_cost_variance_percentage": 0.0,
            "gross_profit_variance": 0.0,
            "gross_profit_variance_percentage": 0.0,
            "gross_margin_percentage_point_variance": 0.0,
            "sales_marketing_variance": 0.0,
            "sales_marketing_variance_percentage": 0.0,
            "other_opex_variance": 0.0,
            "other_opex_variance_percentage": 0.0,
            "ebitda_variance": 0.0,
            "ebitda_variance_percentage": 0.0,
            "depreciation_variance": 0.0,
            "depreciation_variance_percentage": 0.0,
            "ebit_variance": 0.0,
            "ebit_variance_percentage": 0.0,
            "interest_variance": 0.0,
            "interest_variance_percentage": 0.0,
            "ebt_variance": 0.0,
            "ebt_variance_percentage": 0.0,
        },
    }


def test_analyze_returns_pnl_commentary_result(
    agent: PnlCommentaryAgent,
    favourable_summary: dict[str, Any],
) -> None:
    """Analyze should return the expected result type."""

    pnl_result = SamplePnlResult(summary=favourable_summary)

    result = agent.analyze(pnl_result)

    assert isinstance(result, PnlCommentaryResult)
    assert result.executive_summary
    assert result.source_summary == favourable_summary


def test_analyze_accepts_dictionary_result(
    agent: PnlCommentaryAgent,
    favourable_summary: dict[str, Any],
) -> None:
    """The agent should accept a dictionary containing summary."""

    result = agent.analyze(
        {
            "summary": favourable_summary,
        }
    )

    assert isinstance(result, PnlCommentaryResult)
    assert result.revenue_commentary


def test_analyze_does_not_modify_source_summary(
    agent: PnlCommentaryAgent,
    favourable_summary: dict[str, Any],
) -> None:
    """Commentary generation must not modify the P&L summary."""

    original_summary = deepcopy(favourable_summary)
    pnl_result = SamplePnlResult(summary=favourable_summary)

    agent.analyze(pnl_result)

    assert favourable_summary == original_summary


def test_source_summary_is_copied(
    agent: PnlCommentaryAgent,
    favourable_summary: dict[str, Any],
) -> None:
    """The result should contain a separate summary dictionary."""

    pnl_result = SamplePnlResult(summary=favourable_summary)

    result = agent.analyze(pnl_result)

    assert result.source_summary == favourable_summary
    assert result.source_summary is not favourable_summary
    assert (
        result.source_summary["actual"]
        is not favourable_summary["actual"]
    )


def test_favourable_revenue_commentary(
    agent: PnlCommentaryAgent,
    favourable_summary: dict[str, Any],
) -> None:
    """Positive revenue variance should be favourable."""

    result = agent.analyze(
        SamplePnlResult(summary=favourable_summary)
    )

    combined_commentary = " ".join(result.revenue_commentary)

    assert "above budget" in combined_commentary
    assert "favourable variance" in combined_commentary
    assert "₹1.00 lakh" in combined_commentary


def test_unfavourable_revenue_commentary(
    agent: PnlCommentaryAgent,
    unfavourable_summary: dict[str, Any],
) -> None:
    """Negative revenue variance should be unfavourable."""

    result = agent.analyze(
        SamplePnlResult(summary=unfavourable_summary)
    )

    combined_commentary = " ".join(result.revenue_commentary)

    assert "below budget" in combined_commentary
    assert "unfavourable variance" in combined_commentary
    assert "₹2.00 lakh" in combined_commentary


def test_neutral_revenue_commentary(
    agent: PnlCommentaryAgent,
    neutral_summary: dict[str, Any],
) -> None:
    """Zero revenue variance should be described as in line."""

    result = agent.analyze(
        SamplePnlResult(summary=neutral_summary)
    )

    assert result.revenue_commentary == [
        "Actual revenue was in line with budget at ₹10.00 lakh."
    ]


def test_profitability_commentary_contains_all_metrics(
    agent: PnlCommentaryAgent,
    favourable_summary: dict[str, Any],
) -> None:
    """Profitability commentary should include all major profit lines."""

    result = agent.analyze(
        SamplePnlResult(summary=favourable_summary)
    )

    combined_commentary = " ".join(
        result.profitability_commentary
    )

    assert len(result.profitability_commentary) == 4
    assert "Gross profit" in combined_commentary
    assert "EBITDA" in combined_commentary
    assert "EBIT" in combined_commentary
    assert "EBT" in combined_commentary


def test_positive_profit_variances_are_favourable(
    agent: PnlCommentaryAgent,
    favourable_summary: dict[str, Any],
) -> None:
    """Positive profit variance should be classified as favourable."""

    result = agent.analyze(
        SamplePnlResult(summary=favourable_summary)
    )

    for comment in result.profitability_commentary:
        assert "above budget" in comment
        assert "favourable variance" in comment


def test_negative_profit_variances_are_unfavourable(
    agent: PnlCommentaryAgent,
    unfavourable_summary: dict[str, Any],
) -> None:
    """Negative profit variance should be classified as unfavourable."""

    result = agent.analyze(
        SamplePnlResult(summary=unfavourable_summary)
    )

    for comment in result.profitability_commentary:
        assert "below budget" in comment
        assert "unfavourable variance" in comment


def test_cost_commentary_contains_all_cost_metrics(
    agent: PnlCommentaryAgent,
    favourable_summary: dict[str, Any],
) -> None:
    """Cost commentary should include all relevant expense lines."""

    result = agent.analyze(
        SamplePnlResult(summary=favourable_summary)
    )

    combined_commentary = " ".join(result.cost_commentary)

    assert len(result.cost_commentary) == 5
    assert "Direct cost" in combined_commentary
    assert "Sales and marketing expense" in combined_commentary
    assert "Other operating expense" in combined_commentary
    assert "Depreciation" in combined_commentary
    assert "Interest expense" in combined_commentary


def test_negative_cost_variances_are_favourable(
    agent: PnlCommentaryAgent,
    favourable_summary: dict[str, Any],
) -> None:
    """Actual cost below budget should be favourable."""

    result = agent.analyze(
        SamplePnlResult(summary=favourable_summary)
    )

    for comment in result.cost_commentary:
        assert "below budget" in comment
        assert "favourable variance" in comment


def test_positive_cost_variances_are_unfavourable(
    agent: PnlCommentaryAgent,
    unfavourable_summary: dict[str, Any],
) -> None:
    """Actual cost above budget should be unfavourable."""

    result = agent.analyze(
        SamplePnlResult(summary=unfavourable_summary)
    )

    for comment in result.cost_commentary:
        assert "above budget" in comment
        assert "unfavourable variance" in comment


def test_favourable_margin_commentary(
    agent: PnlCommentaryAgent,
    favourable_summary: dict[str, Any],
) -> None:
    """Positive margin movement should be favourable."""

    result = agent.analyze(
        SamplePnlResult(summary=favourable_summary)
    )

    comment = result.margin_commentary[0]

    assert "Gross margin improved" in comment
    assert "favourable movement" in comment
    assert "8.64 percentage points" in comment


def test_unfavourable_margin_commentary(
    agent: PnlCommentaryAgent,
    unfavourable_summary: dict[str, Any],
) -> None:
    """Negative margin movement should be unfavourable."""

    result = agent.analyze(
        SamplePnlResult(summary=unfavourable_summary)
    )

    comment = result.margin_commentary[0]

    assert "Gross margin declined" in comment
    assert "unfavourable movement" in comment
    assert "17.50 percentage points" in comment


def test_neutral_margin_commentary(
    agent: PnlCommentaryAgent,
    neutral_summary: dict[str, Any],
) -> None:
    """Zero margin movement should be described as in line."""

    result = agent.analyze(
        SamplePnlResult(summary=neutral_summary)
    )

    assert result.margin_commentary == [
        "Gross margin was in line with budget at 40.00%."
    ]


def test_margin_percentage_points_are_not_multiplied_by_100(
    agent: PnlCommentaryAgent,
    favourable_summary: dict[str, Any],
) -> None:
    """
    A margin variance of 8.64 must remain 8.64 percentage points.

    It must not be formatted as 864 percentage points.
    """

    result = agent.analyze(
        SamplePnlResult(summary=favourable_summary)
    )

    comment = result.margin_commentary[0]

    assert "8.64 percentage points" in comment
    assert "864.00 percentage points" not in comment


def test_positive_drivers_are_identified(
    agent: PnlCommentaryAgent,
    favourable_summary: dict[str, Any],
) -> None:
    """Favourable profit and cost outcomes should be identified."""

    result = agent.analyze(
        SamplePnlResult(summary=favourable_summary)
    )

    assert "Revenue exceeded budget." in result.positive_drivers
    assert "Gross profit exceeded budget." in result.positive_drivers
    assert "EBITDA exceeded budget." in result.positive_drivers
    assert "Direct cost was below budget." in result.positive_drivers
    assert (
        "Gross margin improved compared with budget."
        in result.positive_drivers
    )


def test_unfavourable_results_are_identified_as_risks(
    agent: PnlCommentaryAgent,
    unfavourable_summary: dict[str, Any],
) -> None:
    """Unfavourable profit and cost outcomes should be risks."""

    result = agent.analyze(
        SamplePnlResult(summary=unfavourable_summary)
    )

    assert "Revenue was below budget." in result.risks
    assert "Gross profit was below budget." in result.risks
    assert "EBITDA was below budget." in result.risks
    assert "Direct cost exceeded budget." in result.risks
    assert (
        "Gross margin declined compared with budget."
        in result.risks
    )


def test_neutral_summary_has_no_positive_drivers_or_risks(
    agent: PnlCommentaryAgent,
    neutral_summary: dict[str, Any],
) -> None:
    """Neutral P&L results should produce no drivers or risks."""

    result = agent.analyze(
        SamplePnlResult(summary=neutral_summary)
    )

    assert result.positive_drivers == []
    assert result.risks == []


def test_material_variances_require_management_attention(
    agent: PnlCommentaryAgent,
    unfavourable_summary: dict[str, Any],
) -> None:
    """Material variance percentages should trigger review items."""

    result = agent.analyze(
        SamplePnlResult(summary=unfavourable_summary)
    )

    assert (
        "Review the material revenue variance."
        in result.management_attention
    )

    assert (
        "Review the material gross-profit variance."
        in result.management_attention
    )

    assert (
        "Review the material EBITDA variance."
        in result.management_attention
    )

    assert (
        "Review the material gross-margin movement."
        in result.management_attention
    )


def test_material_variance_threshold_is_inclusive(
    agent: PnlCommentaryAgent,
    favourable_summary: dict[str, Any],
) -> None:
    """A variance exactly at 10 percent should be material."""

    result = agent.analyze(
        SamplePnlResult(summary=favourable_summary)
    )

    assert (
        "Review the material revenue variance."
        in result.management_attention
    )


def test_small_variances_do_not_require_attention(
    agent: PnlCommentaryAgent,
    neutral_summary: dict[str, Any],
) -> None:
    """Small or zero variances should not trigger attention items."""

    result = agent.analyze(
        SamplePnlResult(summary=neutral_summary)
    )

    assert result.management_attention == []


def test_executive_summary_contains_key_pnl_metrics(
    agent: PnlCommentaryAgent,
    favourable_summary: dict[str, Any],
) -> None:
    """Executive summary should contain core P&L information."""

    result = agent.analyze(
        SamplePnlResult(summary=favourable_summary)
    )

    assert "Actual revenue was" in result.executive_summary
    assert "Gross profit reached" in result.executive_summary
    assert "gross margin of 43.64%" in result.executive_summary
    assert "EBITDA was" in result.executive_summary
    assert "EBT was" in result.executive_summary


def test_none_pnl_result_raises_value_error(
    agent: PnlCommentaryAgent,
) -> None:
    """None should not be accepted as a P&L result."""

    with pytest.raises(
        ValueError,
        match="pnl_result is required",
    ):
        agent.analyze(None)


def test_missing_summary_raises_value_error(
    agent: PnlCommentaryAgent,
) -> None:
    """A result without summary should fail validation."""

    class InvalidResult:
        pass

    with pytest.raises(
        ValueError,
        match="valid summary dictionary",
    ):
        agent.analyze(InvalidResult())


@pytest.mark.parametrize(
    "missing_section",
    [
        "actual",
        "budget",
        "variance",
    ],
)
def test_missing_summary_section_raises_value_error(
    agent: PnlCommentaryAgent,
    favourable_summary: dict[str, Any],
    missing_section: str,
) -> None:
    """Each required summary section must be present."""

    invalid_summary = deepcopy(favourable_summary)
    invalid_summary.pop(missing_section)

    with pytest.raises(
        ValueError,
        match="missing required sections",
    ):
        agent.analyze(
            SamplePnlResult(summary=invalid_summary)
        )


@pytest.mark.parametrize(
    "invalid_section",
    [
        "actual",
        "budget",
        "variance",
    ],
)
def test_non_mapping_summary_section_raises_type_error(
    agent: PnlCommentaryAgent,
    favourable_summary: dict[str, Any],
    invalid_section: str,
) -> None:
    """Summary sections must be dictionary-like mappings."""

    invalid_summary = deepcopy(favourable_summary)
    invalid_summary[invalid_section] = []

    with pytest.raises(
        TypeError,
        match=f"section '{invalid_section}' must be a mapping",
    ):
        agent.analyze(
            SamplePnlResult(summary=invalid_summary)
        )


def test_missing_required_financial_field_raises_value_error(
    agent: PnlCommentaryAgent,
    favourable_summary: dict[str, Any],
) -> None:
    """Missing financial values should fail instead of being invented."""

    invalid_summary = deepcopy(favourable_summary)
    invalid_summary["actual"].pop("revenue")

    with pytest.raises(
        ValueError,
        match="missing required field 'revenue'",
    ):
        agent.analyze(
            SamplePnlResult(summary=invalid_summary)
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        None,
        "not-a-number",
        [],
        {},
        True,
    ],
)
def test_invalid_financial_value_raises_type_error(
    agent: PnlCommentaryAgent,
    favourable_summary: dict[str, Any],
    invalid_value: Any,
) -> None:
    """Invalid financial values should fail validation."""

    invalid_summary = deepcopy(favourable_summary)
    invalid_summary["actual"]["revenue"] = invalid_value

    with pytest.raises(
        TypeError,
        match="must be numeric",
    ):
        agent.analyze(
            SamplePnlResult(summary=invalid_summary)
        )


def test_numeric_strings_are_supported(
    agent: PnlCommentaryAgent,
    favourable_summary: dict[str, Any],
) -> None:
    """Numeric strings should be converted safely to floats."""

    string_summary = deepcopy(favourable_summary)

    string_summary["actual"]["revenue"] = "1100000"
    string_summary["budget"]["revenue"] = "1000000"
    string_summary["variance"]["revenue_variance"] = "100000"
    string_summary["variance"][
        "revenue_variance_percentage"
    ] = "10"

    result = agent.analyze(
        SamplePnlResult(summary=string_summary)
    )

    assert result.revenue_commentary
    assert "₹11.00 lakh" in result.revenue_commentary[0]