"""
Report Agent for the Finance Agentic AI System.

This module combines outputs from finance, analytics, and reporting agents into
one structured management report.

The agent is deterministic and does not recalculate financial metrics. It only:
- Validates supplied agent results
- Converts supported results into report sections
- Preserves source data in simple dictionaries
- Produces a management-ready Markdown report
"""

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from typing import Any


@dataclass
class ReportSection:
    """Stores one section of the final management report."""

    section_code: str
    title: str
    status: str
    summary: str
    items: list[str] = field(default_factory=list)
    data: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ReportResult:
    """Stores the complete assembled report."""

    report_title: str
    report_type: str
    generated_at: str
    overall_status: str
    section_count: int
    sections: list[ReportSection] = field(default_factory=list)
    executive_summary: str = ""
    key_risks: list[str] = field(default_factory=list)
    management_actions: list[str] = field(default_factory=list)
    markdown_report: str = ""
    source_availability: dict[str, bool] = field(default_factory=dict)


class ReportAgent:
    """
    Combines agent outputs into one management-ready report.

    Required input:
        commentary_result

    Optional inputs:
        operations_result
        kpi_result
        budget_result
        forecast_result
        variance_result
        anomaly_result
        root_cause_result
        recommendation_result
        finance_rules_result
        scenario_result
        pnl_result
        pnl_commentary_result
    """

    VALID_REPORT_TYPES = {
        "management",
        "daily",
        "weekly",
        "monthly",
        "full",
    }

    STATUS_RANK = {
        "NO_DATA": 0,
        "PASS": 1,
        "NORMAL": 1,
        "NO_ANOMALY": 1,
        "NO_ROOT_CAUSE_REQUIRED": 1,
        "NO_ACTION_REQUIRED": 1,
        "MONITOR": 2,
        "REVIEW": 3,
        "WARNING": 3,
        "ACTION_REQUIRED": 4,
        "FAIL": 5,
        "ERROR": 5,
    }

    def __init__(
        self,
        max_items_per_section: int = 10,
        max_table_rows: int = 25,
    ) -> None:
        if max_items_per_section <= 0:
            raise ValueError("max_items_per_section must be positive.")

        if max_table_rows <= 0:
            raise ValueError("max_table_rows must be positive.")

        self.max_items_per_section = int(max_items_per_section)
        self.max_table_rows = int(max_table_rows)

    def analyze(
        self,
        commentary_result: Any,
        operations_result: Any | None = None,
        kpi_result: Any | None = None,
        budget_result: Any | None = None,
        forecast_result: Any | None = None,
        variance_result: Any | None = None,
        anomaly_result: Any | None = None,
        root_cause_result: Any | None = None,
        recommendation_result: Any | None = None,
        finance_rules_result: Any | None = None,
        scenario_result: Any | None = None,
        report_title: str = "Finance Agentic AI Management Report",
        report_type: str = "management",
        pnl_result: Any | None = None,
        pnl_commentary_result: Any | None = None,
    ) -> ReportResult:
        """Assemble supplied agent outputs into a complete report."""

        self._validate_inputs(
            commentary_result=commentary_result,
            report_title=report_title,
            report_type=report_type,
        )

        source_availability = {
            "operations": operations_result is not None,
            "kpi": kpi_result is not None,
            "budget": budget_result is not None,
            "forecast": forecast_result is not None,
            "variance": variance_result is not None,
            "anomaly": anomaly_result is not None,
            "root_cause": root_cause_result is not None,
            "recommendation": recommendation_result is not None,
            "commentary": commentary_result is not None,
            "finance_rules": finance_rules_result is not None,
            "scenario": scenario_result is not None,
            "pnl": pnl_result is not None,
            "pnl_commentary": pnl_commentary_result is not None,
        }

        sections: list[ReportSection] = []

        sections.append(
            self._build_executive_section(commentary_result)
        )

        pnl_executive_section = self._build_pnl_executive_section(
            pnl_commentary_result
        )
        if pnl_executive_section is not None:
            sections.append(pnl_executive_section)

        optional_sections = [
            self._build_pnl_revenue_section(
                pnl_result,
                pnl_commentary_result,
            ),
            self._build_pnl_profitability_section(
                pnl_result,
                pnl_commentary_result,
            ),
            self._build_pnl_cost_section(
                pnl_result,
                pnl_commentary_result,
            ),
            self._build_pnl_margin_section(
                pnl_result,
                pnl_commentary_result,
            ),
            self._build_kpi_section(kpi_result, commentary_result),
            self._build_operations_section(operations_result),
            self._build_budget_section(budget_result),
            self._build_forecast_section(forecast_result, commentary_result),
            self._build_variance_section(variance_result, commentary_result),
            self._build_anomaly_section(anomaly_result),
            self._build_root_cause_section(root_cause_result),
            self._build_recommendation_section(recommendation_result),
            self._build_scenario_section(scenario_result, commentary_result),
            self._build_finance_control_section(
                finance_rules_result,
                commentary_result,
            ),
            self._build_risk_section(commentary_result),
            self._build_management_attention_section(commentary_result),
        ]

        sections.extend(
            section
            for section in optional_sections
            if section is not None
        )

        key_risks = self._collect_key_risks(
            commentary_result=commentary_result,
            anomaly_result=anomaly_result,
            root_cause_result=root_cause_result,
            finance_rules_result=finance_rules_result,
            pnl_commentary_result=pnl_commentary_result,
        )

        management_actions = self._collect_management_actions(
            commentary_result=commentary_result,
            recommendation_result=recommendation_result,
            root_cause_result=root_cause_result,
            pnl_commentary_result=pnl_commentary_result,
        )

        overall_status = self._determine_overall_status(
            anomaly_result=anomaly_result,
            root_cause_result=root_cause_result,
            recommendation_result=recommendation_result,
            finance_rules_result=finance_rules_result,
            risks=key_risks,
        )

        generated_at = datetime.now(timezone.utc).isoformat()

        result = ReportResult(
            report_title=report_title.strip(),
            report_type=report_type,
            generated_at=generated_at,
            overall_status=overall_status,
            section_count=len(sections),
            sections=sections,
            executive_summary=self._combine_executive_summaries(
                commentary_result=commentary_result,
                pnl_commentary_result=pnl_commentary_result,
            ),
            key_risks=key_risks,
            management_actions=management_actions,
            source_availability=source_availability,
        )

        result.markdown_report = self._render_markdown(result)
        return result

    def _validate_inputs(
        self,
        commentary_result: Any,
        report_title: str,
        report_type: str,
    ) -> None:
        if commentary_result is None:
            raise ValueError("commentary_result is required.")

        if not hasattr(commentary_result, "executive_summary"):
            raise ValueError(
                "commentary_result must contain an executive_summary attribute."
            )

        if not isinstance(report_title, str) or not report_title.strip():
            raise ValueError("report_title must be a non-empty string.")

        if report_type not in self.VALID_REPORT_TYPES:
            raise ValueError(
                "report_type must be one of "
                f"{sorted(self.VALID_REPORT_TYPES)}."
            )

    def _build_executive_section(
        self,
        commentary_result: Any,
    ) -> ReportSection:
        summary = str(
            getattr(
                commentary_result,
                "executive_summary",
                "Executive summary is unavailable.",
            )
        ).strip()

        positive_drivers = self._string_list(
            getattr(commentary_result, "positive_drivers", [])
        )

        items = [
            f"Positive driver: {item}"
            for item in positive_drivers[: self.max_items_per_section]
        ]

        return ReportSection(
            section_code="EXECUTIVE_SUMMARY",
            title="Executive Summary",
            status="AVAILABLE",
            summary=summary or "Executive summary is unavailable.",
            items=items,
        )

    def _build_pnl_executive_section(
        self,
        pnl_commentary_result: Any | None,
    ) -> ReportSection | None:
        if pnl_commentary_result is None:
            return None

        summary = str(
            getattr(
                pnl_commentary_result,
                "executive_summary",
                "",
            )
        ).strip()

        positive_drivers = self._string_list(
            getattr(
                pnl_commentary_result,
                "positive_drivers",
                [],
            )
        )

        risks = self._string_list(
            getattr(
                pnl_commentary_result,
                "risks",
                [],
            )
        )

        status = "REVIEW" if risks else "NORMAL"

        items = [
            f"Positive driver: {item}"
            for item in positive_drivers[
                : self.max_items_per_section
            ]
        ]

        return ReportSection(
            section_code="PNL_EXECUTIVE_SUMMARY",
            title="P&L Executive Summary",
            status=status,
            summary=summary or "P&L executive summary is unavailable.",
            items=items,
        )

    def _build_pnl_revenue_section(
        self,
        pnl_result: Any | None,
        pnl_commentary_result: Any | None,
    ) -> ReportSection | None:
        comments = self._string_list(
            getattr(
                pnl_commentary_result,
                "revenue_commentary",
                [],
            )
        )

        summary = self._extract_pnl_summary(
            pnl_result=pnl_result,
            pnl_commentary_result=pnl_commentary_result,
        )

        if summary is None and not comments:
            return None

        rows = self._build_pnl_metric_rows(
            summary=summary,
            metrics=(("revenue", "Revenue"),),
        )

        revenue_variance = self._pnl_variance_value(
            summary,
            "revenue_variance",
        )

        status = (
            "REVIEW"
            if revenue_variance is not None and revenue_variance < 0
            else "NORMAL"
        )

        return ReportSection(
            section_code="PNL_REVENUE_SUMMARY",
            title="P&L Revenue Summary",
            status=status,
            summary="Actual revenue performance compared with budget.",
            items=comments[: self.max_items_per_section],
            data=self._limit_rows(rows),
        )

    def _build_pnl_profitability_section(
        self,
        pnl_result: Any | None,
        pnl_commentary_result: Any | None,
    ) -> ReportSection | None:
        comments = self._string_list(
            getattr(
                pnl_commentary_result,
                "profitability_commentary",
                [],
            )
        )

        summary = self._extract_pnl_summary(
            pnl_result=pnl_result,
            pnl_commentary_result=pnl_commentary_result,
        )

        if summary is None and not comments:
            return None

        metrics = (
            ("gross_profit", "Gross Profit"),
            ("ebitda", "EBITDA"),
            ("ebit", "EBIT"),
            ("ebt", "EBT"),
        )

        rows = self._build_pnl_metric_rows(
            summary=summary,
            metrics=metrics,
        )

        status = "NORMAL"
        for metric, _ in metrics:
            value = self._pnl_variance_value(
                summary,
                f"{metric}_variance",
            )
            if value is not None and value < 0:
                status = "REVIEW"
                break

        return ReportSection(
            section_code="PNL_PROFITABILITY_SUMMARY",
            title="P&L Profitability Summary",
            status=status,
            summary=(
                "Gross profit, EBITDA, EBIT, and EBT performance "
                "compared with budget."
            ),
            items=comments[: self.max_items_per_section],
            data=self._limit_rows(rows),
        )

    def _build_pnl_cost_section(
        self,
        pnl_result: Any | None,
        pnl_commentary_result: Any | None,
    ) -> ReportSection | None:
        comments = self._string_list(
            getattr(
                pnl_commentary_result,
                "cost_commentary",
                [],
            )
        )

        summary = self._extract_pnl_summary(
            pnl_result=pnl_result,
            pnl_commentary_result=pnl_commentary_result,
        )

        if summary is None and not comments:
            return None

        metrics = (
            ("direct_cost", "Direct Cost"),
            ("sales_marketing", "Sales and Marketing"),
            ("other_opex", "Other OPEX"),
            ("depreciation", "Depreciation"),
            ("interest", "Interest"),
        )

        rows = self._build_pnl_metric_rows(
            summary=summary,
            metrics=metrics,
        )

        status = "NORMAL"
        for metric, _ in metrics:
            value = self._pnl_variance_value(
                summary,
                f"{metric}_variance",
            )
            if value is not None and value > 0:
                status = "REVIEW"
                break

        return ReportSection(
            section_code="PNL_COST_SUMMARY",
            title="P&L Cost Summary",
            status=status,
            summary=(
                "Direct costs and operating expenses compared with budget."
            ),
            items=comments[: self.max_items_per_section],
            data=self._limit_rows(rows),
        )

    def _build_pnl_margin_section(
        self,
        pnl_result: Any | None,
        pnl_commentary_result: Any | None,
    ) -> ReportSection | None:
        comments = self._string_list(
            getattr(
                pnl_commentary_result,
                "margin_commentary",
                [],
            )
        )

        summary = self._extract_pnl_summary(
            pnl_result=pnl_result,
            pnl_commentary_result=pnl_commentary_result,
        )

        if summary is None and not comments:
            return None

        rows = self._build_pnl_metric_rows(
            summary=summary,
            metrics=(
                (
                    "gross_margin_percentage",
                    "Gross Margin Percentage",
                ),
            ),
            variance_field_overrides={
                "gross_margin_percentage": (
                    "gross_margin_percentage_point_variance"
                )
            },
        )

        margin_variance = self._pnl_variance_value(
            summary,
            "gross_margin_percentage_point_variance",
        )

        status = (
            "REVIEW"
            if margin_variance is not None and margin_variance < 0
            else "NORMAL"
        )

        return ReportSection(
            section_code="PNL_MARGIN_SUMMARY",
            title="P&L Margin Summary",
            status=status,
            summary=(
                "Gross-margin performance and percentage-point movement "
                "compared with budget."
            ),
            items=comments[: self.max_items_per_section],
            data=self._limit_rows(rows),
        )

    def _extract_pnl_summary(
        self,
        pnl_result: Any | None,
        pnl_commentary_result: Any | None,
    ) -> dict[str, Any] | None:
        summary = getattr(
            pnl_commentary_result,
            "source_summary",
            None,
        )

        if not isinstance(summary, dict) and pnl_result is not None:
            summary = getattr(pnl_result, "summary", None)

        if not isinstance(summary, dict):
            return None

        actual = summary.get("actual")
        budget = summary.get("budget")
        variance = summary.get("variance")

        if not all(
            isinstance(section, dict)
            for section in (actual, budget, variance)
        ):
            return None

        return {
            "actual": dict(actual),
            "budget": dict(budget),
            "variance": dict(variance),
        }

    def _build_pnl_metric_rows(
        self,
        summary: dict[str, Any] | None,
        metrics: tuple[tuple[str, str], ...],
        variance_field_overrides: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        if summary is None:
            return []

        actual = summary["actual"]
        budget = summary["budget"]
        variance = summary["variance"]
        overrides = variance_field_overrides or {}

        rows: list[dict[str, Any]] = []

        for metric, display_name in metrics:
            variance_field = overrides.get(
                metric,
                f"{metric}_variance",
            )

            percentage_field = f"{metric}_variance_percentage"

            rows.append(
                {
                    "metric": metric,
                    "display_name": display_name,
                    "actual": actual.get(metric),
                    "budget": budget.get(metric),
                    "variance": variance.get(variance_field),
                    "variance_percentage": variance.get(
                        percentage_field
                    ),
                    "variance_unit": (
                        "percentage_points"
                        if variance_field
                        == "gross_margin_percentage_point_variance"
                        else "currency"
                    ),
                }
            )

        return rows

    def _pnl_variance_value(
        self,
        summary: dict[str, Any] | None,
        field_name: str,
    ) -> float | None:
        if summary is None:
            return None

        variance = summary.get("variance", {})
        return self._number(variance.get(field_name))

    def _combine_executive_summaries(
        self,
        commentary_result: Any,
        pnl_commentary_result: Any | None,
    ) -> str:
        summaries = [
            str(
                getattr(
                    commentary_result,
                    "executive_summary",
                    "",
                )
            ).strip()
        ]

        if pnl_commentary_result is not None:
            summaries.append(
                str(
                    getattr(
                        pnl_commentary_result,
                        "executive_summary",
                        "",
                    )
                ).strip()
            )

        return " ".join(
            summary
            for summary in summaries
            if summary
        )

    def _build_kpi_section(
        self,
        kpi_result: Any | None,
        commentary_result: Any,
    ) -> ReportSection | None:
        source_rows = getattr(commentary_result, "source_kpis", [])

        if not source_rows and kpi_result is not None:
            source_rows = [
                self._to_dict(item)
                for item in getattr(kpi_result, "selected_kpis", [])
            ]

        comments = self._string_list(
            getattr(commentary_result, "kpi_commentary", [])
        )

        if not source_rows and not comments:
            return None

        unavailable = list(
            getattr(kpi_result, "unavailable_kpis", [])
            if kpi_result is not None
            else []
        )
        unknown = list(
            getattr(kpi_result, "unknown_kpis", [])
            if kpi_result is not None
            else []
        )

        status = "REVIEW" if unavailable or unknown else "NORMAL"
        summary = (
            f"{len(source_rows)} KPI(s) were included in the report."
        )

        return ReportSection(
            section_code="KPI_SUMMARY",
            title="KPI Summary",
            status=status,
            summary=summary,
            items=comments[: self.max_items_per_section],
            data=self._limit_rows(source_rows),
        )

    def _build_operations_section(
        self,
        operations_result: Any | None,
    ) -> ReportSection | None:
        if operations_result is None:
            return None

        summary_row = {
            "total_orders": getattr(operations_result, "total_orders", None),
            "completed_orders": getattr(
                operations_result,
                "completed_orders",
                None,
            ),
            "cancelled_orders": getattr(
                operations_result,
                "cancelled_orders",
                None,
            ),
            "fulfillment_percentage": getattr(
                operations_result,
                "fulfillment_percentage",
                None,
            ),
            "cancellation_percentage": getattr(
                operations_result,
                "cancellation_percentage",
                None,
            ),
            "total_revenue": getattr(
                operations_result,
                "total_revenue",
                None,
            ),
            "average_order_value": getattr(
                operations_result,
                "average_order_value",
                None,
            ),
        }

        fulfillment = self._number(
            summary_row["fulfillment_percentage"]
        )
        cancellation = self._number(
            summary_row["cancellation_percentage"]
        )

        status = "NORMAL"
        if (
            fulfillment is not None
            and fulfillment < 90
        ) or (
            cancellation is not None
            and cancellation >= 10
        ):
            status = "REVIEW"

        items = [
            f"Total orders: {self._format_number(summary_row['total_orders'])}",
            (
                "Completed orders: "
                f"{self._format_number(summary_row['completed_orders'])}"
            ),
            (
                "Total revenue: "
                f"{self._format_currency(summary_row['total_revenue'])}"
            ),
            (
                "Fulfillment: "
                f"{self._format_percentage(summary_row['fulfillment_percentage'])}"
            ),
            (
                "Cancellation: "
                f"{self._format_percentage(summary_row['cancellation_percentage'])}"
            ),
        ]

        detail_rows: list[dict[str, Any]] = []
        for attribute in (
            "period_summary",
            "vehicle_summary",
            "cluster_summary",
        ):
            rows = getattr(operations_result, attribute, [])
            if isinstance(rows, list):
                for row in rows:
                    item = dict(row) if isinstance(row, dict) else self._to_dict(row)
                    item["_summary_type"] = attribute
                    detail_rows.append(item)

        return ReportSection(
            section_code="OPERATIONS_SUMMARY",
            title="Operations Summary",
            status=status,
            summary="Operational performance summary from the analysed dataset.",
            items=items,
            data=self._limit_rows(detail_rows),
        )

    def _build_budget_section(
        self,
        budget_result: Any | None,
    ) -> ReportSection | None:
        if budget_result is None:
            return None

        items = [
            (
                "Budget orders: "
                f"{self._format_number(getattr(budget_result, 'total_budget_orders', None))}"
            ),
            (
                "Budget revenue: "
                f"{self._format_currency(getattr(budget_result, 'total_budget_revenue', None))}"
            ),
            (
                "Budget AOV: "
                f"{self._format_currency(getattr(budget_result, 'budget_average_order_value', None))}"
            ),
        ]

        rows: list[dict[str, Any]] = []
        for attribute in ("period_summary", "vehicle_summary"):
            values = getattr(budget_result, attribute, [])
            if isinstance(values, list):
                for row in values:
                    item = dict(row) if isinstance(row, dict) else self._to_dict(row)
                    item["_summary_type"] = attribute
                    rows.append(item)

        return ReportSection(
            section_code="BUDGET_SUMMARY",
            title="Budget Summary",
            status="NORMAL",
            summary="Budget targets used for financial comparison.",
            items=items,
            data=self._limit_rows(rows),
        )

    def _build_forecast_section(
        self,
        forecast_result: Any | None,
        commentary_result: Any,
    ) -> ReportSection | None:
        comments = self._string_list(
            getattr(commentary_result, "forecast_commentary", [])
        )

        if forecast_result is None and not comments:
            return None

        rows = list(
            getattr(forecast_result, "forecast_summary", [])
            if forecast_result is not None
            else []
        )

        method = (
            str(getattr(forecast_result, "method", "Unknown"))
            if forecast_result is not None
            else "Unknown"
        )

        return ReportSection(
            section_code="FORECAST_SUMMARY",
            title="Forecast Summary",
            status="NORMAL" if rows else "REVIEW",
            summary=(
                f"Forecast prepared using {method}; "
                f"{len(rows)} forecast period(s) are available."
            ),
            items=comments[: self.max_items_per_section],
            data=self._limit_rows(rows),
        )

    def _build_variance_section(
        self,
        variance_result: Any | None,
        commentary_result: Any,
    ) -> ReportSection | None:
        comments = self._string_list(
            getattr(commentary_result, "variance_commentary", [])
        )

        if variance_result is None and not comments:
            return None

        data: list[dict[str, Any]] = []
        status = "NORMAL"
        summary = "Variance commentary is available."

        if variance_result is not None:
            revenue_variance = self._number(
                getattr(variance_result, "revenue_variance", None)
            )
            variance_check = self._number(
                getattr(variance_result, "variance_check", None)
            )

            if revenue_variance is not None and revenue_variance < 0:
                status = "REVIEW"

            if variance_check is not None and abs(variance_check) > 0.05:
                status = "REVIEW"

            summary = (
                "Actual revenue variance was "
                f"{self._format_currency(revenue_variance)}."
            )

            overall_row = {
                key: getattr(variance_result, key, None)
                for key in (
                    "actual_orders",
                    "budget_orders",
                    "actual_revenue",
                    "budget_revenue",
                    "actual_aov",
                    "budget_aov",
                    "order_variance",
                    "revenue_variance",
                    "aov_variance",
                    "price_effect",
                    "volume_effect",
                    "new_discontinued_effect",
                    "variance_check",
                )
            }
            data.append(overall_row)

            vehicle_rows = getattr(
                variance_result,
                "vehicle_variance_summary",
                [],
            )
            if isinstance(vehicle_rows, list):
                data.extend(
                    dict(row) if isinstance(row, dict) else self._to_dict(row)
                    for row in vehicle_rows
                )

        return ReportSection(
            section_code="VARIANCE_SUMMARY",
            title="Variance Summary",
            status=status,
            summary=summary,
            items=comments[: self.max_items_per_section],
            data=self._limit_rows(data),
        )

    def _build_anomaly_section(
        self,
        anomaly_result: Any | None,
    ) -> ReportSection | None:
        if anomaly_result is None:
            return None

        findings = list(getattr(anomaly_result, "findings", []))
        anomaly_count = int(
            getattr(anomaly_result, "anomaly_count", len(findings))
        )
        high_priority_count = int(
            getattr(anomaly_result, "high_priority_count", 0)
        )

        status = str(
            getattr(
                anomaly_result,
                "overall_status",
                "REVIEW" if anomaly_count else "NO_ANOMALY",
            )
        ).upper()

        items = []
        for finding in findings[: self.max_items_per_section]:
            metric = str(getattr(finding, "metric", "metric"))
            dimension_value = str(
                getattr(finding, "dimension_value", "overall")
            )
            severity = str(getattr(finding, "severity", "unknown"))
            message = str(getattr(finding, "message", "")).strip()
            items.append(
                f"[{severity.upper()}] {metric} for {dimension_value}: "
                f"{message or 'Anomaly detected.'}"
            )

        summary = (
            f"{anomaly_count} anomaly/anomalies were detected, including "
            f"{high_priority_count} high-priority item(s)."
        )

        return ReportSection(
            section_code="ANOMALY_SUMMARY",
            title="Anomaly Summary",
            status=status,
            summary=summary,
            items=items,
            data=self._limit_rows(
                [self._to_dict(item) for item in findings]
            ),
        )

    def _build_root_cause_section(
        self,
        root_cause_result: Any | None,
    ) -> ReportSection | None:
        if root_cause_result is None:
            return None

        findings = list(getattr(root_cause_result, "findings", []))
        unresolved = list(
            getattr(root_cause_result, "unresolved_anomalies", [])
        )

        items = []
        for finding in findings[: self.max_items_per_section]:
            cause = str(
                getattr(finding, "cause_description", "Cause identified.")
            )
            confidence = str(
                getattr(finding, "confidence", "unknown")
            )
            dimension_value = str(
                getattr(finding, "dimension_value", "overall")
            )
            items.append(
                f"[{confidence.upper()}] {dimension_value}: {cause}"
            )

        summary = (
            f"{len(findings)} root-cause hypothesis/hypotheses were identified; "
            f"{len(unresolved)} anomaly/anomalies remain unresolved."
        )

        return ReportSection(
            section_code="ROOT_CAUSE_SUMMARY",
            title="Root Cause Summary",
            status=str(
                getattr(root_cause_result, "overall_status", "REVIEW")
            ).upper(),
            summary=summary,
            items=items,
            data=self._limit_rows(
                [self._to_dict(item) for item in findings]
                + [
                    {
                        **(
                            dict(item)
                            if isinstance(item, dict)
                            else self._to_dict(item)
                        ),
                        "_record_type": "unresolved_anomaly",
                    }
                    for item in unresolved
                ]
            ),
        )

    def _build_recommendation_section(
        self,
        recommendation_result: Any | None,
    ) -> ReportSection | None:
        if recommendation_result is None:
            return None

        recommendations = list(
            getattr(recommendation_result, "recommendations", [])
        )

        items = []
        for recommendation in recommendations[: self.max_items_per_section]:
            priority = str(
                getattr(recommendation, "priority", "medium")
            )
            owner = str(
                getattr(recommendation, "owner", "Unassigned")
            )
            action = str(
                getattr(
                    recommendation,
                    "recommended_action",
                    "Review the finding.",
                )
            )
            items.append(
                f"[{priority.upper()}] {owner}: {action}"
            )

        summary = (
            f"{len(recommendations)} recommendation(s) were generated, "
            f"including "
            f"{getattr(recommendation_result, 'critical_priority_count', 0)} "
            "critical-priority and "
            f"{getattr(recommendation_result, 'high_priority_count', 0)} "
            "high-priority item(s)."
        )

        return ReportSection(
            section_code="RECOMMENDATION_SUMMARY",
            title="Recommendations and Actions",
            status=str(
                getattr(
                    recommendation_result,
                    "overall_status",
                    "REVIEW",
                )
            ).upper(),
            summary=summary,
            items=items,
            data=self._limit_rows(
                [self._to_dict(item) for item in recommendations]
            ),
        )

    def _build_scenario_section(
        self,
        scenario_result: Any | None,
        commentary_result: Any,
    ) -> ReportSection | None:
        comments = self._string_list(
            getattr(commentary_result, "scenario_commentary", [])
        )

        if scenario_result is None and not comments:
            return None

        rows = list(
            getattr(scenario_result, "adjusted_forecast", [])
            if scenario_result is not None
            else []
        )

        unapplied = list(
            getattr(scenario_result, "unapplied_assumptions", [])
            if scenario_result is not None
            else []
        )

        return ReportSection(
            section_code="SCENARIO_SUMMARY",
            title="Scenario Summary",
            status="REVIEW" if unapplied else "NORMAL",
            summary=(
                f"{getattr(scenario_result, 'applied_assumption_count', 0)} "
                "assumption(s) were applied and "
                f"{len(unapplied)} remained unapplied."
            ),
            items=comments[: self.max_items_per_section],
            data=self._limit_rows(rows),
        )

    def _build_finance_control_section(
        self,
        finance_rules_result: Any | None,
        commentary_result: Any,
    ) -> ReportSection | None:
        comments = self._string_list(
            getattr(commentary_result, "control_commentary", [])
        )

        if finance_rules_result is None and not comments:
            return None

        issues = list(
            getattr(finance_rules_result, "issues", [])
            if finance_rules_result is not None
            else []
        )

        status = str(
            getattr(finance_rules_result, "overall_status", "REVIEW")
            if finance_rules_result is not None
            else "REVIEW"
        ).upper()

        summary = (
            f"{getattr(finance_rules_result, 'rules_checked', 0)} rule(s) "
            f"were checked; {getattr(finance_rules_result, 'warning_count', 0)} "
            f"warning(s) and {getattr(finance_rules_result, 'error_count', 0)} "
            "error(s) were reported."
        )

        return ReportSection(
            section_code="FINANCE_CONTROL_SUMMARY",
            title="Finance Control Summary",
            status=status,
            summary=summary,
            items=comments[: self.max_items_per_section],
            data=self._limit_rows(
                [self._to_dict(item) for item in issues]
            ),
        )

    def _build_risk_section(
        self,
        commentary_result: Any,
    ) -> ReportSection | None:
        risks = self._string_list(
            getattr(commentary_result, "risks", [])
        )

        if not risks:
            return None

        return ReportSection(
            section_code="KEY_RISKS",
            title="Key Risks",
            status="REVIEW",
            summary=f"{len(risks)} management risk item(s) were identified.",
            items=risks[: self.max_items_per_section],
        )

    def _build_management_attention_section(
        self,
        commentary_result: Any,
    ) -> ReportSection | None:
        items = self._string_list(
            getattr(commentary_result, "management_attention", [])
        )

        if not items:
            return None

        return ReportSection(
            section_code="MANAGEMENT_ATTENTION",
            title="Management Attention",
            status="REVIEW",
            summary=f"{len(items)} item(s) require management review.",
            items=items[: self.max_items_per_section],
        )

    def _collect_key_risks(
        self,
        commentary_result: Any,
        anomaly_result: Any | None,
        root_cause_result: Any | None,
        finance_rules_result: Any | None,
        pnl_commentary_result: Any | None,
    ) -> list[str]:
        risks = self._string_list(
            getattr(commentary_result, "risks", [])
        )

        if pnl_commentary_result is not None:
            risks.extend(
                self._string_list(
                    getattr(
                        pnl_commentary_result,
                        "risks",
                        [],
                    )
                )
            )

        if anomaly_result is not None:
            for finding in getattr(anomaly_result, "findings", []):
                severity = str(
                    getattr(finding, "severity", "")
                ).lower()
                if severity in {"high", "critical"}:
                    message = str(
                        getattr(finding, "message", "")
                    ).strip()
                    if message:
                        risks.append(message)

        if root_cause_result is not None:
            for finding in getattr(root_cause_result, "findings", []):
                if (
                    str(getattr(finding, "confidence", "")).lower()
                    == "high"
                    and str(
                        getattr(finding, "impact", "")
                    ).lower()
                    == "unfavorable"
                ):
                    description = str(
                        getattr(finding, "cause_description", "")
                    ).strip()
                    if description:
                        risks.append(description)

        if finance_rules_result is not None:
            status = str(
                getattr(finance_rules_result, "overall_status", "")
            ).upper()
            if status == "FAIL":
                risks.append(
                    "Finance-control errors must be resolved before release."
                )
            elif status == "WARNING":
                risks.append(
                    "Finance-control warnings require management review."
                )

        return self._remove_duplicates(risks)[
            : self.max_items_per_section
        ]

    def _collect_management_actions(
        self,
        commentary_result: Any,
        recommendation_result: Any | None,
        root_cause_result: Any | None,
        pnl_commentary_result: Any | None,
    ) -> list[str]:
        actions = self._string_list(
            getattr(commentary_result, "management_attention", [])
        )

        if pnl_commentary_result is not None:
            actions.extend(
                self._string_list(
                    getattr(
                        pnl_commentary_result,
                        "management_attention",
                        [],
                    )
                )
            )

        if recommendation_result is not None:
            for item in getattr(
                recommendation_result,
                "recommendations",
                [],
            ):
                priority = str(
                    getattr(item, "priority", "medium")
                ).upper()
                owner = str(
                    getattr(item, "owner", "Unassigned")
                )
                action = str(
                    getattr(item, "recommended_action", "")
                ).strip()

                if action:
                    actions.append(
                        f"[{priority}] {owner}: {action}"
                    )

        if root_cause_result is not None:
            for item in getattr(root_cause_result, "findings", []):
                next_check = str(
                    getattr(item, "recommended_next_check", "")
                ).strip()
                if next_check:
                    actions.append(
                        f"Investigation: {next_check}"
                    )

        return self._remove_duplicates(actions)[
            : self.max_items_per_section
        ]

    def _determine_overall_status(
        self,
        anomaly_result: Any | None,
        root_cause_result: Any | None,
        recommendation_result: Any | None,
        finance_rules_result: Any | None,
        risks: list[str],
    ) -> str:
        candidate_statuses: list[str] = []

        for result in (
            anomaly_result,
            root_cause_result,
            recommendation_result,
            finance_rules_result,
        ):
            if result is not None:
                candidate_statuses.append(
                    str(
                        getattr(result, "overall_status", "REVIEW")
                    ).upper()
                )

        if not candidate_statuses:
            return "REVIEW" if risks else "NORMAL"

        return max(
            candidate_statuses,
            key=lambda status: self.STATUS_RANK.get(status, 3),
        )

    def _render_markdown(self, report: ReportResult) -> str:
        lines = [
            f"# {report.report_title}",
            "",
            f"**Report type:** {report.report_type}",
            f"**Generated at:** {report.generated_at}",
            f"**Overall status:** {report.overall_status}",
            "",
        ]

        for section in report.sections:
            lines.extend(
                [
                    f"## {section.title}",
                    "",
                    f"**Status:** {section.status}",
                    "",
                    section.summary,
                    "",
                ]
            )

            for item in section.items:
                lines.append(f"- {item}")

            if section.items:
                lines.append("")

            if section.data:
                lines.append(
                    f"_Structured records available: {len(section.data)}_"
                )
                lines.append("")

        if report.management_actions:
            lines.extend(
                [
                    "## Consolidated Management Actions",
                    "",
                ]
            )
            lines.extend(
                f"- {item}"
                for item in report.management_actions
            )
            lines.append("")

        return "\n".join(lines).strip() + "\n"

    def _to_dict(self, value: Any) -> dict[str, Any]:
        if value is None:
            return {}

        if isinstance(value, dict):
            return {
                str(key): self._serialize(item)
                for key, item in value.items()
            }

        if is_dataclass(value):
            return {
                str(key): self._serialize(item)
                for key, item in asdict(value).items()
            }

        if hasattr(value, "__dict__"):
            return {
                str(key): self._serialize(item)
                for key, item in vars(value).items()
                if not str(key).startswith("_")
            }

        return {"value": self._serialize(value)}

    def _serialize(self, value: Any) -> Any:
        if isinstance(value, Enum):
            return value.value

        if is_dataclass(value):
            return {
                str(key): self._serialize(item)
                for key, item in asdict(value).items()
            }

        if isinstance(value, dict):
            return {
                str(key): self._serialize(item)
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple, set)):
            return [self._serialize(item) for item in value]

        return value

    def _limit_rows(
        self,
        rows: list[Any],
    ) -> list[dict[str, Any]]:
        normalized = [
            dict(row) if isinstance(row, dict) else self._to_dict(row)
            for row in rows
        ]
        return normalized[: self.max_table_rows]

    def _string_list(self, values: Any) -> list[str]:
        if not isinstance(values, list):
            return []

        return [
            str(value).strip()
            for value in values
            if str(value).strip()
        ]

    def _number(self, value: Any) -> float | None:
        if value is None or isinstance(value, bool):
            return None

        try:
            number = float(value)
        except (TypeError, ValueError):
            return None

        return number if isfinite(number) else None

    def _format_currency(self, value: Any) -> str:
        number = self._number(value)
        if number is None:
            return "N/A"

        absolute = abs(number)
        sign = "-" if number < 0 else ""

        if absolute >= 10_000_000:
            return f"{sign}₹{absolute / 10_000_000:,.2f} crore"

        if absolute >= 100_000:
            return f"{sign}₹{absolute / 100_000:,.2f} lakh"

        return f"{sign}₹{absolute:,.2f}"

    def _format_percentage(self, value: Any) -> str:
        number = self._number(value)
        return "N/A" if number is None else f"{number:,.2f}%"

    def _format_number(self, value: Any) -> str:
        number = self._number(value)
        return "N/A" if number is None else f"{int(round(number)):,}"

    def _remove_duplicates(
        self,
        values: list[str],
    ) -> list[str]:
        return list(dict.fromkeys(values))