from types import SimpleNamespace

import pytest

from src.agents.reporting.commentary_agent import CommentaryAgent, CommentaryResult


def make_kpi(
    kpi: str,
    display_name: str,
    value,
    unit: str,
    *,
    source: str = "test",
    period: str | None = "2026-06",
    dimension: str | None = None,
    dimension_value: str | None = None,
):
    return SimpleNamespace(
        kpi=kpi,
        display_name=display_name,
        value=value,
        unit=unit,
        source=source,
        period=period,
        dimension=dimension,
        dimension_value=dimension_value,
    )


def make_kpi_result(*kpis, unavailable_kpis=None, unknown_kpis=None):
    return SimpleNamespace(
        selected_kpis=list(kpis),
        unavailable_kpis=list(unavailable_kpis or []),
        unknown_kpis=list(unknown_kpis or []),
    )


def test_requires_kpi_result():
    agent = CommentaryAgent()

    with pytest.raises(ValueError, match="kpi_result is required"):
        agent.analyze(None)


def test_requires_selected_kpis_attribute():
    agent = CommentaryAgent()

    with pytest.raises(ValueError, match="selected_kpis"):
        agent.analyze(SimpleNamespace())


def test_selected_kpis_must_be_list():
    agent = CommentaryAgent()

    with pytest.raises(TypeError, match="selected_kpis must be a list"):
        agent.analyze(SimpleNamespace(selected_kpis="actual_revenue"))


def test_requires_at_least_one_selected_kpi():
    agent = CommentaryAgent()

    with pytest.raises(ValueError, match="At least one selected KPI"):
        agent.analyze(make_kpi_result())


def test_generates_basic_kpi_commentary_and_source_rows():
    agent = CommentaryAgent()
    kpi_result = make_kpi_result(
        make_kpi(
            "actual_revenue",
            "Actual Revenue",
            2_500_000,
            "currency",
            dimension="vehicle_category",
            dimension_value="Tata Ace",
        ),
        make_kpi(
            "fulfillment_percentage",
            "Fulfillment Percentage",
            94.5,
            "percentage",
        ),
    )

    result = agent.analyze(kpi_result)

    assert isinstance(result, CommentaryResult)
    assert "Actual revenue reached ₹25.00 lakh." in result.executive_summary
    assert "Fulfillment was 94.50%." in result.executive_summary
    assert any("Actual Revenue" in item for item in result.kpi_commentary)
    assert any("Tata Ace" in item for item in result.kpi_commentary)
    assert len(result.source_kpis) == 2
    assert result.source_kpis[0]["kpi"] == "actual_revenue"


def test_positive_revenue_variance_commentary():
    agent = CommentaryAgent()
    kpi_result = make_kpi_result(
        make_kpi("actual_revenue", "Actual Revenue", 12_000_000, "currency"),
        make_kpi("revenue_variance", "Revenue Variance", 1_000_000, "currency"),
    )
    variance_result = SimpleNamespace(
        revenue_variance=1_000_000,
        price_effect=600_000,
        volume_effect=400_000,
        new_discontinued_effect=0,
        variance_check=0.0,
    )

    result = agent.analyze(
        kpi_result,
        revenue_variance_result=variance_result,
    )

    assert any("favourable compared with budget" in x for x in result.variance_commentary)
    assert any("Price/AOV effect was favourable" in x for x in result.variance_commentary)
    assert any("Volume effect was favourable" in x for x in result.variance_commentary)
    assert "Favourable price/AOV movement supported revenue." in result.positive_drivers
    assert "Higher completed-order volume supported revenue." in result.positive_drivers


def test_negative_revenue_variance_is_reported_as_risk():
    agent = CommentaryAgent()
    kpi_result = make_kpi_result(
        make_kpi("actual_revenue", "Actual Revenue", 8_000_000, "currency"),
        make_kpi("revenue_variance", "Revenue Variance", -2_000_000, "currency"),
    )
    variance_result = SimpleNamespace(
        revenue_variance=-2_000_000,
        price_effect=-500_000,
        volume_effect=-1_500_000,
        new_discontinued_effect=0,
        variance_check=0.0,
    )

    result = agent.analyze(
        kpi_result,
        revenue_variance_result=variance_result,
    )

    assert "Revenue is below budget." in result.risks
    assert "Price/AOV movement reduced revenue." in result.risks
    assert "Lower completed-order volume reduced revenue." in result.risks
    assert any("Revenue was below budget" in x for x in [result.executive_summary])


def test_forecast_commentary_for_multiple_periods():
    agent = CommentaryAgent()
    kpi_result = make_kpi_result(
        make_kpi(
            "forecast_revenue",
            "Forecast Revenue",
            1_000_000,
            "currency",
            period="2026-07",
        ),
    )
    forecast_result = SimpleNamespace(
        method="three-month moving average",
        forecast_summary=[
            {
                "forecast_period": "2026-07",
                "forecast_orders": 1_000,
                "forecast_revenue": 1_000_000,
            },
            {
                "forecast_period": "2026-08",
                "forecast_orders": 1_100,
                "forecast_revenue": 1_150_000,
            },
        ],
    )

    result = agent.analyze(
        kpi_result,
        forecast_result=forecast_result,
    )

    assert any("three-month moving average" in x for x in result.forecast_commentary)
    assert any("forecast revenue increases" in x.lower() for x in result.forecast_commentary)
    assert "Base forecast revenue for 2026-07 is ₹10.00 lakh." in result.executive_summary


def test_scenario_commentary_and_unapplied_assumption_attention():
    agent = CommentaryAgent()
    kpi_result = make_kpi_result(
        make_kpi("actual_revenue", "Actual Revenue", 1_000_000, "currency")
    )
    scenario_result = SimpleNamespace(
        scenario_name="Management Upside",
        applied_assumption_count=1,
        total_assumptions=2,
        adjusted_forecast=[
            {
                "forecast_period": "2026-07",
                "orders_adjustment": 100,
                "revenue_adjustment": 200_000,
            }
        ],
        unapplied_assumptions=[{"name": "Unknown assumption"}],
    )

    result = agent.analyze(
        kpi_result,
        scenario_result=scenario_result,
    )

    assert any("Management Upside applied 1 of 2" in x for x in result.scenario_commentary)
    assert "Some business assumptions could not be applied." in result.risks
    assert "Review unapplied business assumptions." in result.management_attention
    assert any("1 business assumptions" in result.executive_summary for _ in [0])


@pytest.mark.parametrize(
    ("status", "expected_comment", "expected_attention"),
    [
        ("PASS", "No finance-control warnings or errors were identified.", None),
        ("WARNING", "warning(s) require management review.", "Review finance-control warnings"),
        ("FAIL", "finance-control error(s) must be resolved", "Resolve finance-control errors"),
    ],
)
def test_finance_control_commentary(status, expected_comment, expected_attention):
    agent = CommentaryAgent()
    kpi_result = make_kpi_result(
        make_kpi("actual_revenue", "Actual Revenue", 1_000_000, "currency")
    )
    finance_rules_result = SimpleNamespace(
        overall_status=status,
        rules_checked=5,
        passed_rules=3 if status != "PASS" else 5,
        warning_count=2 if status == "WARNING" else 0,
        error_count=2 if status == "FAIL" else 0,
    )

    result = agent.analyze(
        kpi_result,
        finance_rules_result=finance_rules_result,
    )

    assert any(expected_comment in x for x in result.control_commentary)

    if expected_attention is None:
        assert not any("finance-control" in x.lower() for x in result.management_attention)
    else:
        assert any(expected_attention in x for x in result.management_attention)


def test_operational_thresholds_create_positive_driver_and_risk():
    agent = CommentaryAgent()
    kpi_result = make_kpi_result(
        make_kpi(
            "fulfillment_percentage",
            "Fulfillment Percentage",
            96.0,
            "percentage",
        ),
        make_kpi(
            "cancellation_percentage",
            "Cancellation Percentage",
            12.0,
            "percentage",
        ),
    )

    result = agent.analyze(kpi_result)

    assert "Fulfillment performance was strong." in result.positive_drivers
    assert "Cancellation is high and may require operational action." in result.risks


def test_unknown_and_unavailable_kpis_require_management_attention():
    agent = CommentaryAgent()
    kpi_result = make_kpi_result(
        make_kpi("actual_revenue", "Actual Revenue", 1_000_000, "currency"),
        unavailable_kpis=["gross_profit"],
        unknown_kpis=["abc_metric"],
    )

    result = agent.analyze(kpi_result)

    assert "Some requested KPIs were unavailable." in result.management_attention
    assert "Some requested KPI names were not recognized." in result.management_attention