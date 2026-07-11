from types import SimpleNamespace

import pytest

from src.agents.reporting.report_agent import (
    ReportAgent,
    ReportResult,
    ReportSection,
)


def make_commentary_result():
    return SimpleNamespace(
        executive_summary=(
            "Revenue was below budget and requires management attention."
        ),
        kpi_commentary=[
            "Actual Revenue for 2026-06 was ₹10.00 lakh.",
            "Fulfillment Percentage for 2026-06 was 88.00%.",
        ],
        variance_commentary=[
            "Revenue performance was unfavourable compared with budget."
        ],
        forecast_commentary=[
            "The base forecast was prepared using moving average."
        ],
        scenario_commentary=[
            "Management Upside applied 1 of 2 supplied assumptions."
        ],
        control_commentary=[
            "Finance controls checked 5 rules, with 4 passing."
        ],
        positive_drivers=[
            "Favourable price/AOV movement supported revenue."
        ],
        risks=[
            "Revenue is below budget.",
            "Fulfillment is below the acceptable threshold.",
        ],
        management_attention=[
            "Review finance-control warnings before final reporting."
        ],
        source_kpis=[
            {
                "kpi": "actual_revenue",
                "display_name": "Actual Revenue",
                "value": 1_000_000,
                "unit": "currency",
                "period": "2026-06",
            },
            {
                "kpi": "fulfillment_percentage",
                "display_name": "Fulfillment Percentage",
                "value": 88.0,
                "unit": "percentage",
                "period": "2026-06",
            },
        ],
    )


def make_operations_result():
    return SimpleNamespace(
        total_orders=1_000,
        completed_orders=880,
        cancelled_orders=120,
        fulfillment_percentage=88.0,
        cancellation_percentage=12.0,
        total_revenue=1_000_000,
        average_order_value=1_136.36,
        period_summary=[
            {
                "period": "2026-06",
                "total_orders": 1_000,
                "completed_orders": 880,
            }
        ],
        vehicle_summary=[
            {
                "vehicle_category": "Tata Ace",
                "total_orders": 400,
                "completed_orders": 330,
            }
        ],
        cluster_summary=[],
    )


def make_kpi_result():
    return SimpleNamespace(
        selected_kpis=[
            SimpleNamespace(
                kpi="actual_revenue",
                display_name="Actual Revenue",
                value=1_000_000,
                unit="currency",
            )
        ],
        unavailable_kpis=[],
        unknown_kpis=[],
    )


def make_budget_result():
    return SimpleNamespace(
        total_budget_orders=1_100,
        total_budget_revenue=1_200_000,
        budget_average_order_value=1_090.91,
        period_summary=[
            {
                "period": "2026-06",
                "budget_orders": 1_100,
                "budget_revenue": 1_200_000,
            }
        ],
        vehicle_summary=[],
    )


def make_forecast_result():
    return SimpleNamespace(
        method="moving average",
        forecast_summary=[
            {
                "forecast_period": "2026-07",
                "forecast_orders": 1_050,
                "forecast_revenue": 1_100_000,
            }
        ],
    )


def make_variance_result():
    return SimpleNamespace(
        actual_orders=1_000,
        budget_orders=1_100,
        actual_revenue=1_000_000,
        budget_revenue=1_200_000,
        actual_aov=1_000.0,
        budget_aov=1_090.91,
        order_variance=-100,
        revenue_variance=-200_000,
        aov_variance=-90.91,
        price_effect=-90_000,
        volume_effect=-110_000,
        new_discontinued_effect=0,
        variance_check=0.0,
        vehicle_variance_summary=[
            {
                "vehicle_category": "Tata Ace",
                "revenue_variance": -100_000,
            }
        ],
    )


def make_anomaly_result():
    return SimpleNamespace(
        overall_status="REVIEW",
        anomaly_count=1,
        high_priority_count=1,
        findings=[
            SimpleNamespace(
                anomaly_code="ANOM-001",
                metric="total_revenue",
                dimension_value="Tata Ace",
                severity="high",
                message="Revenue decreased materially.",
            )
        ],
    )


def make_root_cause_result():
    return SimpleNamespace(
        overall_status="ACTION_REQUIRED",
        findings=[
            SimpleNamespace(
                root_cause_code="RC-001",
                cause_description=(
                    "Lower completed-order volume reduced revenue."
                ),
                confidence="high",
                impact="unfavorable",
                dimension_value="Tata Ace",
                recommended_next_check=(
                    "Review demand and fulfillment by cluster."
                ),
            )
        ],
        unresolved_anomalies=[],
    )


def make_recommendation_result():
    return SimpleNamespace(
        overall_status="ACTION_REQUIRED",
        critical_priority_count=1,
        high_priority_count=0,
        recommendations=[
            SimpleNamespace(
                recommendation_code="REC-001",
                priority="critical",
                owner="Operations Team",
                recommended_action=(
                    "Increase partner availability during peak hours."
                ),
            )
        ],
    )


def make_scenario_result():
    return SimpleNamespace(
        applied_assumption_count=1,
        adjusted_forecast=[
            {
                "forecast_period": "2026-07",
                "orders_adjustment": 100,
                "revenue_adjustment": 150_000,
            }
        ],
        unapplied_assumptions=[{"name": "Unknown assumption"}],
    )


def make_finance_rules_result(status="WARNING"):
    return SimpleNamespace(
        overall_status=status,
        rules_checked=5,
        passed_rules=4,
        warning_count=1 if status == "WARNING" else 0,
        error_count=1 if status == "FAIL" else 0,
        issues=[
            {
                "rule": "variance_reconciliation",
                "status": status,
            }
        ],
    )


def test_constructor_validation():
    with pytest.raises(
        ValueError,
        match="max_items_per_section must be positive",
    ):
        ReportAgent(max_items_per_section=0)

    with pytest.raises(
        ValueError,
        match="max_table_rows must be positive",
    ):
        ReportAgent(max_table_rows=0)


def test_requires_commentary_result():
    agent = ReportAgent()

    with pytest.raises(
        ValueError,
        match="commentary_result is required",
    ):
        agent.analyze(None)


def test_requires_executive_summary_attribute():
    agent = ReportAgent()

    with pytest.raises(
        ValueError,
        match="executive_summary",
    ):
        agent.analyze(SimpleNamespace())


def test_validates_report_title_and_type():
    agent = ReportAgent()
    commentary = make_commentary_result()

    with pytest.raises(
        ValueError,
        match="report_title must be a non-empty string",
    ):
        agent.analyze(commentary, report_title="")

    with pytest.raises(
        ValueError,
        match="report_type must be one of",
    ):
        agent.analyze(commentary, report_type="invalid")


def test_minimum_report_contains_executive_section():
    agent = ReportAgent()
    result = agent.analyze(make_commentary_result())

    assert isinstance(result, ReportResult)
    assert result.report_title == "Finance Agentic AI Management Report"
    assert result.report_type == "management"
    assert result.section_count >= 1
    assert result.sections[0].section_code == "EXECUTIVE_SUMMARY"
    assert "Executive Summary" in result.markdown_report


def test_full_report_builds_all_major_sections():
    agent = ReportAgent()

    result = agent.analyze(
        commentary_result=make_commentary_result(),
        operations_result=make_operations_result(),
        kpi_result=make_kpi_result(),
        budget_result=make_budget_result(),
        forecast_result=make_forecast_result(),
        variance_result=make_variance_result(),
        anomaly_result=make_anomaly_result(),
        root_cause_result=make_root_cause_result(),
        recommendation_result=make_recommendation_result(),
        finance_rules_result=make_finance_rules_result(),
        scenario_result=make_scenario_result(),
        report_title="June 2026 Management Report",
        report_type="monthly",
    )

    section_codes = {
        section.section_code
        for section in result.sections
    }

    expected_sections = {
        "EXECUTIVE_SUMMARY",
        "KPI_SUMMARY",
        "OPERATIONS_SUMMARY",
        "BUDGET_SUMMARY",
        "FORECAST_SUMMARY",
        "VARIANCE_SUMMARY",
        "ANOMALY_SUMMARY",
        "ROOT_CAUSE_SUMMARY",
        "RECOMMENDATION_SUMMARY",
        "SCENARIO_SUMMARY",
        "FINANCE_CONTROL_SUMMARY",
        "KEY_RISKS",
        "MANAGEMENT_ATTENTION",
    }

    assert expected_sections.issubset(section_codes)
    assert result.report_title == "June 2026 Management Report"
    assert result.report_type == "monthly"
    assert result.source_availability["operations"] is True
    assert result.source_availability["recommendation"] is True
    assert "Recommendations and Actions" in result.markdown_report
    assert "Consolidated Management Actions" in result.markdown_report


def test_operations_section_is_review_when_thresholds_are_weak():
    agent = ReportAgent()

    result = agent.analyze(
        commentary_result=make_commentary_result(),
        operations_result=make_operations_result(),
    )

    section = next(
        item
        for item in result.sections
        if item.section_code == "OPERATIONS_SUMMARY"
    )

    assert section.status == "REVIEW"
    assert any("Fulfillment" in item for item in section.items)
    assert any("Cancellation" in item for item in section.items)


def test_variance_section_is_review_for_negative_variance():
    agent = ReportAgent()

    result = agent.analyze(
        commentary_result=make_commentary_result(),
        variance_result=make_variance_result(),
    )

    section = next(
        item
        for item in result.sections
        if item.section_code == "VARIANCE_SUMMARY"
    )

    assert section.status == "REVIEW"
    assert "₹2.00 lakh" in section.summary


def test_anomaly_root_cause_and_recommendation_sections():
    agent = ReportAgent()

    result = agent.analyze(
        commentary_result=make_commentary_result(),
        anomaly_result=make_anomaly_result(),
        root_cause_result=make_root_cause_result(),
        recommendation_result=make_recommendation_result(),
    )

    anomaly_section = next(
        item
        for item in result.sections
        if item.section_code == "ANOMALY_SUMMARY"
    )
    root_section = next(
        item
        for item in result.sections
        if item.section_code == "ROOT_CAUSE_SUMMARY"
    )
    recommendation_section = next(
        item
        for item in result.sections
        if item.section_code == "RECOMMENDATION_SUMMARY"
    )

    assert "[HIGH]" in anomaly_section.items[0]
    assert "[HIGH]" in root_section.items[0]
    assert "[CRITICAL]" in recommendation_section.items[0]
    assert "Operations Team" in recommendation_section.items[0]


def test_management_actions_include_recommendations_and_next_checks():
    agent = ReportAgent()

    result = agent.analyze(
        commentary_result=make_commentary_result(),
        root_cause_result=make_root_cause_result(),
        recommendation_result=make_recommendation_result(),
    )

    assert any(
        "Increase partner availability" in action
        for action in result.management_actions
    )
    assert any(
        action.startswith("Investigation:")
        for action in result.management_actions
    )


@pytest.mark.parametrize(
    ("finance_status", "expected_status"),
    [
        ("PASS", "PASS"),
        ("WARNING", "WARNING"),
        ("FAIL", "FAIL"),
    ],
)
def test_overall_status_uses_most_serious_upstream_status(
    finance_status,
    expected_status,
):
    agent = ReportAgent()
    commentary = make_commentary_result()
    commentary.risks = []

    result = agent.analyze(
        commentary_result=commentary,
        finance_rules_result=make_finance_rules_result(
            status=finance_status
        ),
    )

    assert result.overall_status == expected_status


def test_overall_status_is_action_required_for_root_cause():
    agent = ReportAgent()

    result = agent.analyze(
        commentary_result=make_commentary_result(),
        root_cause_result=make_root_cause_result(),
    )

    assert result.overall_status == "ACTION_REQUIRED"


def test_row_and_item_limits_are_applied():
    agent = ReportAgent(
        max_items_per_section=1,
        max_table_rows=1,
    )

    result = agent.analyze(
        commentary_result=make_commentary_result(),
        operations_result=make_operations_result(),
    )

    executive = next(
        item
        for item in result.sections
        if item.section_code == "EXECUTIVE_SUMMARY"
    )
    operations = next(
        item
        for item in result.sections
        if item.section_code == "OPERATIONS_SUMMARY"
    )

    assert len(executive.items) <= 1
    assert len(operations.data) <= 1


def test_markdown_report_ends_with_newline():
    agent = ReportAgent()

    result = agent.analyze(make_commentary_result())

    assert result.markdown_report.endswith("\n")
    assert "# Finance Agentic AI Management Report" in (
        result.markdown_report
    )