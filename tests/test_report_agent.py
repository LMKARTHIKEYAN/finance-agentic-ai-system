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




def make_pnl_summary():
    """Return a deterministic P&L actual-versus-budget summary."""

    return {
        "actual": {
            "revenue": 1_000_000.0,
            "direct_cost": 650_000.0,
            "gross_profit": 350_000.0,
            "gross_margin_percentage": 35.0,
            "sales_marketing": 80_000.0,
            "other_opex": 100_000.0,
            "ebitda": 170_000.0,
            "depreciation": 40_000.0,
            "ebit": 130_000.0,
            "interest": 20_000.0,
            "ebt": 110_000.0,
        },
        "budget": {
            "revenue": 1_200_000.0,
            "direct_cost": 720_000.0,
            "gross_profit": 480_000.0,
            "gross_margin_percentage": 40.0,
            "sales_marketing": 75_000.0,
            "other_opex": 90_000.0,
            "ebitda": 315_000.0,
            "depreciation": 35_000.0,
            "ebit": 280_000.0,
            "interest": 15_000.0,
            "ebt": 265_000.0,
        },
        "variance": {
            "revenue_variance": -200_000.0,
            "revenue_variance_percentage": -16.67,
            "direct_cost_variance": -70_000.0,
            "direct_cost_variance_percentage": -9.72,
            "gross_profit_variance": -130_000.0,
            "gross_profit_variance_percentage": -27.08,
            "gross_margin_percentage_point_variance": -5.0,
            "sales_marketing_variance": 5_000.0,
            "sales_marketing_variance_percentage": 6.67,
            "other_opex_variance": 10_000.0,
            "other_opex_variance_percentage": 11.11,
            "ebitda_variance": -145_000.0,
            "ebitda_variance_percentage": -46.03,
            "depreciation_variance": 5_000.0,
            "depreciation_variance_percentage": 14.29,
            "ebit_variance": -150_000.0,
            "ebit_variance_percentage": -53.57,
            "interest_variance": 5_000.0,
            "interest_variance_percentage": 33.33,
            "ebt_variance": -155_000.0,
            "ebt_variance_percentage": -58.49,
        },
    }


def make_pnl_result():
    """Return a P&L result containing the source summary."""

    return SimpleNamespace(summary=make_pnl_summary())


def make_pnl_commentary_result():
    """Return structured deterministic P&L commentary."""

    return SimpleNamespace(
        executive_summary=(
            "Revenue, gross profit, EBITDA, EBIT, and EBT were below "
            "budget, while gross margin declined."
        ),
        revenue_commentary=[
            "Revenue was below budget by ₹2.00 lakh."
        ],
        profitability_commentary=[
            "Gross profit was below budget.",
            "EBITDA was below budget.",
            "EBIT was below budget.",
            "EBT was below budget.",
        ],
        cost_commentary=[
            "Direct cost was below budget.",
            "Sales and marketing expense exceeded budget.",
            "Other operating expense exceeded budget.",
            "Depreciation exceeded budget.",
            "Interest expense exceeded budget.",
        ],
        margin_commentary=[
            "Gross margin declined by 5.00 percentage points."
        ],
        positive_drivers=[
            "Direct cost was below budget."
        ],
        risks=[
            "Revenue was below budget.",
            "Gross margin declined compared with budget.",
        ],
        management_attention=[
            "Review the material revenue variance.",
            "Review the material gross-margin movement.",
        ],
        source_summary=make_pnl_summary(),
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


def test_report_without_pnl_preserves_existing_flow():
    """P&L sections should not appear when no P&L input is supplied."""

    agent = ReportAgent()

    result = agent.analyze(make_commentary_result())

    section_codes = {
        section.section_code
        for section in result.sections
    }

    assert "PNL_EXECUTIVE_SUMMARY" not in section_codes
    assert "PNL_REVENUE_SUMMARY" not in section_codes
    assert result.source_availability["pnl"] is False
    assert result.source_availability["pnl_commentary"] is False


def test_pnl_integration_builds_all_pnl_sections():
    """P&L input should create every dedicated P&L report section."""

    agent = ReportAgent()

    result = agent.analyze(
        commentary_result=make_commentary_result(),
        pnl_result=make_pnl_result(),
        pnl_commentary_result=make_pnl_commentary_result(),
    )

    section_codes = {
        section.section_code
        for section in result.sections
    }

    expected_pnl_sections = {
        "PNL_EXECUTIVE_SUMMARY",
        "PNL_REVENUE_SUMMARY",
        "PNL_PROFITABILITY_SUMMARY",
        "PNL_COST_SUMMARY",
        "PNL_MARGIN_SUMMARY",
    }

    assert expected_pnl_sections.issubset(section_codes)
    assert result.source_availability["pnl"] is True
    assert result.source_availability["pnl_commentary"] is True


def test_pnl_executive_section_uses_pnl_commentary():
    """P&L executive commentary and positive drivers should be retained."""

    agent = ReportAgent()

    result = agent.analyze(
        commentary_result=make_commentary_result(),
        pnl_commentary_result=make_pnl_commentary_result(),
    )

    section = next(
        item
        for item in result.sections
        if item.section_code == "PNL_EXECUTIVE_SUMMARY"
    )

    assert section.status == "REVIEW"
    assert "Revenue, gross profit" in section.summary
    assert section.items == [
        "Positive driver: Direct cost was below budget."
    ]


def test_report_result_combines_operational_and_pnl_summaries():
    """The top-level executive summary should combine both domains."""

    commentary = make_commentary_result()
    pnl_commentary = make_pnl_commentary_result()
    agent = ReportAgent()

    result = agent.analyze(
        commentary_result=commentary,
        pnl_commentary_result=pnl_commentary,
    )

    assert commentary.executive_summary in result.executive_summary
    assert pnl_commentary.executive_summary in result.executive_summary


def test_pnl_revenue_section_is_review_for_negative_variance():
    """Revenue below budget should produce REVIEW status."""

    agent = ReportAgent()

    result = agent.analyze(
        commentary_result=make_commentary_result(),
        pnl_result=make_pnl_result(),
        pnl_commentary_result=make_pnl_commentary_result(),
    )

    section = next(
        item
        for item in result.sections
        if item.section_code == "PNL_REVENUE_SUMMARY"
    )

    assert section.status == "REVIEW"
    assert section.items == [
        "Revenue was below budget by ₹2.00 lakh."
    ]
    assert len(section.data) == 1
    assert section.data[0]["metric"] == "revenue"
    assert section.data[0]["actual"] == 1_000_000.0
    assert section.data[0]["budget"] == 1_200_000.0
    assert section.data[0]["variance"] == -200_000.0


def test_pnl_profitability_section_contains_major_profit_lines():
    """Profitability section should contain GP, EBITDA, EBIT, and EBT."""

    agent = ReportAgent()

    result = agent.analyze(
        commentary_result=make_commentary_result(),
        pnl_result=make_pnl_result(),
        pnl_commentary_result=make_pnl_commentary_result(),
    )

    section = next(
        item
        for item in result.sections
        if item.section_code == "PNL_PROFITABILITY_SUMMARY"
    )

    metrics = {
        row["metric"]
        for row in section.data
    }

    assert section.status == "REVIEW"
    assert metrics == {
        "gross_profit",
        "ebitda",
        "ebit",
        "ebt",
    }
    assert len(section.items) == 4


def test_pnl_cost_section_treats_positive_cost_variance_as_review():
    """Any cost above budget should make the cost section REVIEW."""

    agent = ReportAgent()

    result = agent.analyze(
        commentary_result=make_commentary_result(),
        pnl_result=make_pnl_result(),
        pnl_commentary_result=make_pnl_commentary_result(),
    )

    section = next(
        item
        for item in result.sections
        if item.section_code == "PNL_COST_SUMMARY"
    )

    metrics = {
        row["metric"]
        for row in section.data
    }

    assert section.status == "REVIEW"
    assert metrics == {
        "direct_cost",
        "sales_marketing",
        "other_opex",
        "depreciation",
        "interest",
    }


def test_pnl_margin_section_preserves_percentage_point_unit():
    """Gross-margin movement must remain in percentage points."""

    agent = ReportAgent()

    result = agent.analyze(
        commentary_result=make_commentary_result(),
        pnl_result=make_pnl_result(),
        pnl_commentary_result=make_pnl_commentary_result(),
    )

    section = next(
        item
        for item in result.sections
        if item.section_code == "PNL_MARGIN_SUMMARY"
    )

    assert section.status == "REVIEW"
    assert len(section.data) == 1
    assert section.data[0]["variance"] == -5.0
    assert section.data[0]["variance_unit"] == "percentage_points"
    assert "5.00 percentage points" in section.items[0]


def test_pnl_sections_can_use_commentary_source_summary_without_pnl_result():
    """P&L commentary source_summary should be sufficient for tables."""

    agent = ReportAgent()

    result = agent.analyze(
        commentary_result=make_commentary_result(),
        pnl_commentary_result=make_pnl_commentary_result(),
    )

    revenue_section = next(
        item
        for item in result.sections
        if item.section_code == "PNL_REVENUE_SUMMARY"
    )

    assert revenue_section.data[0]["actual"] == 1_000_000.0
    assert result.source_availability["pnl"] is False
    assert result.source_availability["pnl_commentary"] is True


def test_pnl_sections_can_use_pnl_result_without_pnl_commentary():
    """Raw P&L summary should build structured sections without comments."""

    agent = ReportAgent()

    result = agent.analyze(
        commentary_result=make_commentary_result(),
        pnl_result=make_pnl_result(),
    )

    section_codes = {
        section.section_code
        for section in result.sections
    }

    assert "PNL_EXECUTIVE_SUMMARY" not in section_codes
    assert "PNL_REVENUE_SUMMARY" in section_codes
    assert "PNL_PROFITABILITY_SUMMARY" in section_codes
    assert "PNL_COST_SUMMARY" in section_codes
    assert "PNL_MARGIN_SUMMARY" in section_codes

    revenue_section = next(
        item
        for item in result.sections
        if item.section_code == "PNL_REVENUE_SUMMARY"
    )

    assert revenue_section.items == []
    assert revenue_section.data


def test_pnl_risks_are_added_to_consolidated_key_risks():
    """P&L risks should be merged with operational risks."""

    agent = ReportAgent()

    result = agent.analyze(
        commentary_result=make_commentary_result(),
        pnl_commentary_result=make_pnl_commentary_result(),
    )

    assert "Revenue was below budget." in result.key_risks
    assert (
        "Gross margin declined compared with budget."
        in result.key_risks
    )
    assert "Revenue is below budget." in result.key_risks


def test_pnl_management_attention_is_added_to_actions():
    """P&L management-attention items should become consolidated actions."""

    agent = ReportAgent()

    result = agent.analyze(
        commentary_result=make_commentary_result(),
        pnl_commentary_result=make_pnl_commentary_result(),
    )

    assert (
        "Review the material revenue variance."
        in result.management_actions
    )
    assert (
        "Review the material gross-margin movement."
        in result.management_actions
    )


def test_duplicate_pnl_risks_and_actions_are_removed():
    """Consolidated outputs should not contain duplicate entries."""

    commentary = make_commentary_result()
    commentary.risks.append("Revenue was below budget.")
    commentary.management_attention.append(
        "Review the material revenue variance."
    )

    agent = ReportAgent()

    result = agent.analyze(
        commentary_result=commentary,
        pnl_commentary_result=make_pnl_commentary_result(),
    )

    assert result.key_risks.count(
        "Revenue was below budget."
    ) == 1
    assert result.management_actions.count(
        "Review the material revenue variance."
    ) == 1


def test_pnl_item_and_table_limits_are_applied():
    """P&L sections should respect configured output limits."""

    agent = ReportAgent(
        max_items_per_section=1,
        max_table_rows=1,
    )

    result = agent.analyze(
        commentary_result=make_commentary_result(),
        pnl_result=make_pnl_result(),
        pnl_commentary_result=make_pnl_commentary_result(),
    )

    profitability = next(
        item
        for item in result.sections
        if item.section_code == "PNL_PROFITABILITY_SUMMARY"
    )
    cost = next(
        item
        for item in result.sections
        if item.section_code == "PNL_COST_SUMMARY"
    )

    assert len(profitability.items) == 1
    assert len(profitability.data) == 1
    assert len(cost.items) == 1
    assert len(cost.data) == 1


def test_pnl_sections_are_rendered_in_markdown():
    """Dedicated P&L headings should appear in the Markdown report."""

    agent = ReportAgent()

    result = agent.analyze(
        commentary_result=make_commentary_result(),
        pnl_result=make_pnl_result(),
        pnl_commentary_result=make_pnl_commentary_result(),
    )

    assert "## P&L Executive Summary" in result.markdown_report
    assert "## P&L Revenue Summary" in result.markdown_report
    assert "## P&L Profitability Summary" in result.markdown_report
    assert "## P&L Cost Summary" in result.markdown_report
    assert "## P&L Margin Summary" in result.markdown_report


def test_pnl_favourable_results_produce_normal_section_statuses():
    """Favourable P&L variances should keep P&L sections NORMAL."""

    summary = make_pnl_summary()

    summary["variance"]["revenue_variance"] = 200_000.0
    summary["variance"]["gross_profit_variance"] = 130_000.0
    summary["variance"]["ebitda_variance"] = 145_000.0
    summary["variance"]["ebit_variance"] = 150_000.0
    summary["variance"]["ebt_variance"] = 155_000.0

    for field in (
        "direct_cost_variance",
        "sales_marketing_variance",
        "other_opex_variance",
        "depreciation_variance",
        "interest_variance",
    ):
        summary["variance"][field] = -abs(
            summary["variance"][field]
        )

    summary["variance"][
        "gross_margin_percentage_point_variance"
    ] = 5.0

    pnl_result = SimpleNamespace(summary=summary)
    agent = ReportAgent()

    result = agent.analyze(
        commentary_result=make_commentary_result(),
        pnl_result=pnl_result,
    )

    statuses = {
        section.section_code: section.status
        for section in result.sections
    }

    assert statuses["PNL_REVENUE_SUMMARY"] == "NORMAL"
    assert statuses["PNL_PROFITABILITY_SUMMARY"] == "NORMAL"
    assert statuses["PNL_COST_SUMMARY"] == "NORMAL"
    assert statuses["PNL_MARGIN_SUMMARY"] == "NORMAL"