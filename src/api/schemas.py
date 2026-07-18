"""
Request and response schemas for the Finance Agentic AI API.

This module contains only API data models.

It must not contain:

- Finance calculations
- LangGraph routing logic
- RAG retrieval logic
- PostgreSQL queries
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """
    Request body accepted by the POST /ask endpoint.

    Attributes:
        question:
            Finance-related question entered by the user.

        top_k:
            Maximum number of relevant RAG documents to retrieve.

        metadata_filter:
            Optional metadata values used to filter retrieved documents.
    """

    question: str = Field(
        ...,
        min_length=1,
        description="Finance question submitted by the user.",
        examples=["What is the revenue variance?"],
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of RAG sources to retrieve.",
    )

    metadata_filter: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata filters for document retrieval.",
    )


class SourceResponse(BaseModel):
    """
    A source returned by the RAG retrieval process.

    Attributes:
        id:
            Unique identifier of the retrieved document or chunk.

        score:
            Similarity score assigned by the retriever.

        rank:
            Position of the source in the retrieval result.

        metadata:
            Metadata attached to the source document.

        excerpt:
            Short text extracted from the retrieved source.
    """

    id: str

    score: float | None = None

    rank: int

    metadata: dict[str, Any] = Field(default_factory=dict)

    excerpt: str


class DashboardReportMetadata(BaseModel):
    """
    Metadata describing the generated dashboard report.

    Attributes:
        title:
            Dashboard or management report title.

        period:
            Reporting period selected by the user.

        analysis_type:
            Type of finance analysis displayed.

        comparison:
            Comparison used in the analysis.

        category:
            Selected business category.

        generated_at:
            Report generation timestamp.

        overall_status:
            Overall financial performance status.
    """

    title: str = "Finance Performance Dashboard"

    period: str | None = None

    analysis_type: str | None = None

    comparison: str | None = None

    category: str | None = None

    generated_at: str | None = None

    overall_status: str | None = None


class DashboardKpiCard(BaseModel):
    """
    A KPI card displayed at the top of the dashboard.

    Attributes:
        key:
            Machine-readable KPI identifier.

        label:
            User-facing KPI name.

        value:
            Current numeric KPI value.

        formatted_value:
            Display-ready KPI value.

        unit:
            KPI unit such as INR, percentage, count or ratio.

        comparison_label:
            Name of the comparison value.

        comparison_value:
            Budget, forecast or prior-period comparison value.

        delta:
            Absolute difference from the comparison value.

        delta_percentage:
            Percentage difference from the comparison value.

        status:
            KPI performance status.

        favourability:
            Indicates whether the result is favourable, unfavourable
            or neutral.
    """

    key: str

    label: str

    value: float | int | None = None

    formatted_value: str | None = None

    unit: str | None = None

    comparison_label: str | None = None

    comparison_value: float | int | None = None

    delta: float | int | None = None

    delta_percentage: float | None = None

    status: str | None = None

    favourability: str | None = None


class DashboardTable(BaseModel):
    """
    A structured table displayed in the dashboard.

    Attributes:
        title:
            User-facing table title.

        columns:
            Ordered table column names.

        rows:
            Table records represented as dictionaries.

        empty_message:
            Message displayed when no table data is available.
    """

    title: str

    columns: list[str] = Field(default_factory=list)

    rows: list[dict[str, Any]] = Field(default_factory=list)

    empty_message: str | None = None


class DashboardTrendPoint(BaseModel):
    """
    A single period used in a trend chart.

    Attributes:
        period:
            Time period represented by the point.

        actual:
            Actual financial value.

        budget:
            Budget financial value.

        forecast:
            Forecast financial value.

        prior_year:
            Comparable prior-year value.

        metadata:
            Optional additional chart dimensions.
    """

    period: str

    actual: float | int | None = None

    budget: float | int | None = None

    forecast: float | int | None = None

    prior_year: float | int | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)


class DashboardWaterfallPoint(BaseModel):
    """
    A single component of a variance waterfall chart.

    Attributes:
        label:
            Name of the waterfall component.

        value:
            Numeric value of the component.

        measure:
            Waterfall measure type such as absolute, relative or total.

        sequence:
            Display order of the component.

        favourability:
            Financial favourability of the component.

        unit:
            Unit associated with the value.
    """

    label: str

    value: float | int

    measure: str = "relative"

    sequence: int = Field(ge=1)

    favourability: str | None = None

    unit: str | None = None


class DashboardRecommendation(BaseModel):
    """
    A management recommendation displayed by the dashboard.

    Attributes:
        priority:
            Recommendation priority.

        title:
            Short recommendation heading.

        action:
            Recommended management action.

        owner:
            Suggested action owner when available.

        time_horizon:
            Suggested completion period.

        expected_impact:
            Expected business or financial impact.

        source:
            Backend result that produced the recommendation.
    """

    priority: str | None = None

    title: str

    action: str

    owner: str | None = None

    time_horizon: str | None = None

    expected_impact: str | None = None

    source: str | None = None


class DashboardRisk(BaseModel):
    """
    A business or financial risk displayed by the dashboard.

    Attributes:
        severity:
            Risk severity level.

        title:
            Short risk heading.

        description:
            Detailed risk explanation.

        metric:
            KPI associated with the risk.

        value:
            Current value of the affected KPI.

        source:
            Backend result that produced the risk.
    """

    severity: str | None = None

    title: str

    description: str

    metric: str | None = None

    value: float | int | str | None = None

    source: str | None = None


class DashboardCommentary(BaseModel):
    """
    Structured finance commentary displayed in the dashboard.

    Attributes:
        executive_summary:
            High-level management summary.

        financial_performance:
            Commentary on overall financial performance.

        kpi_commentary:
            Commentary on important KPI movements.

        variance_commentary:
            Commentary explaining financial variances.

        forecast_commentary:
            Commentary on forecast performance.

        scenario_commentary:
            Commentary on scenario analysis.

        management_attention:
            Issues requiring management attention.
    """

    executive_summary: str | None = None

    financial_performance: str | None = None

    kpi_commentary: str | None = None

    variance_commentary: str | None = None

    forecast_commentary: str | None = None

    scenario_commentary: str | None = None

    management_attention: list[str] = Field(default_factory=list)


class DashboardPayload(BaseModel):
    """
    Structured data required by the enterprise FP&A dashboard.

    Attributes:
        report_metadata:
            Report title, filters, period and generation information.

        kpi_cards:
            Summary KPI cards displayed at the top of the dashboard.

        kpi_table:
            Detailed KPI performance table.

        variance_table:
            Actual, budget and forecast variance table.

        category_table:
            Category-level financial performance table.

        trend_data:
            Chart-ready revenue, budget and forecast trend data.

        waterfall_data:
            Chart-ready variance waterfall data.

        recommendations:
            Management recommendations.

        risks:
            Identified financial and operational risks.

        executive_summary:
            High-level management summary.

        commentary:
            Structured finance commentary.

        data_limitations:
            Missing data and analysis limitations.

        available_sections:
            Dashboard sections successfully populated.

        unavailable_sections:
            Dashboard sections without sufficient data.
    """

    report_metadata: DashboardReportMetadata = Field(
        default_factory=DashboardReportMetadata
    )

    kpi_cards: list[DashboardKpiCard] = Field(default_factory=list)

    kpi_table: DashboardTable | None = None

    variance_table: DashboardTable | None = None

    category_table: DashboardTable | None = None

    trend_data: list[DashboardTrendPoint] = Field(default_factory=list)

    waterfall_data: list[DashboardWaterfallPoint] = Field(
        default_factory=list
    )

    recommendations: list[DashboardRecommendation] = Field(
        default_factory=list
    )

    risks: list[DashboardRisk] = Field(default_factory=list)

    executive_summary: str | None = None

    commentary: DashboardCommentary = Field(
        default_factory=DashboardCommentary
    )

    data_limitations: list[str] = Field(default_factory=list)

    available_sections: list[str] = Field(default_factory=list)

    unavailable_sections: list[str] = Field(default_factory=list)

class ParsedPeriodResponse(BaseModel):
    """Period parsed from the user's natural-language request."""

    start_date: str | None = None
    end_date: str | None = None
    display_value: str = ""
    granularity: str = "unknown"


class FinanceIntentResponse(BaseModel):
    """Structured interpretation of a finance request."""

    original_request: str = ""
    selected_flow: str = "unknown"
    comparison: str = "none"
    period: ParsedPeriodResponse = Field(
        default_factory=ParsedPeriodResponse
    )
    category: str | None = None
    scenario_name: str | None = None
    requested_kpis: list[str] = Field(
        default_factory=list
    )
    missing_fields: list[str] = Field(
        default_factory=list
    )
    clarification_question: str | None = None
    is_complete: bool = False
    filters: dict[str, Any] = Field(
        default_factory=dict
    )


class AskResponse(BaseModel):
    """
    Response returned by the POST /ask endpoint.

    Attributes:
        answer:
            Final answer produced by the Finance AI system.

        sources:
            RAG sources used to support the response.

        selected_flow:
            LangGraph flow selected for the question.

        execution_status:
            Final orchestration execution status.

        used_fallback:
            Indicates whether a fallback response was required.

        dashboard:
            Optional structured dashboard data. This remains optional
            to preserve backward compatibility with existing API clients.
    """

    answer: str

    sources: list[SourceResponse] = Field(default_factory=list)

    selected_flow: str | None = None

    execution_status: str

    used_fallback: bool = False

    dashboard: DashboardPayload | None = None