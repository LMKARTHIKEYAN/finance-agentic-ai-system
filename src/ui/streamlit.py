"""
Streamlit dashboard for the Finance Agentic AI platform.

This module provides the presentation layer for the existing FastAPI
backend.

It is responsible only for:

- Dashboard filters
- Report selection
- API request submission
- KPI card presentation
- Finance table presentation
- Commentary presentation
- Recommendation and risk presentation
- Source presentation

It must not contain finance calculations, LangGraph execution, RAG logic
or direct database access.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import streamlit as st

from src.ui.api_client import (
    FinanceApiConnectionError,
    FinanceApiResponseError,
    ask_finance_question,
)


PAGE_TITLE = "Finance Agentic AI Dashboard"
PAGE_ICON = "📊"
DEFAULT_TOP_K = 5

ANALYSIS_TYPES = (
    "Management Report",
    "KPI Analysis",
    "Variance Analysis",
    "Budget Analysis",
    "Forecast Analysis",
    "Scenario Analysis",
)

COMPARISON_TYPES = (
    "Actual vs Budget",
    "Actual vs Forecast",
    "Actual vs Last Year",
)

REPORT_FREQUENCIES = (
    "Daily",
    "Weekly",
    "Monthly",
)

SCENARIO_TYPES = (
    "Management Case",
    "Base Case",
    "Upside Case",
    "Downside Case",
)

DEFAULT_CATEGORIES = (
    "All Categories",
    "2W",
    "3W",
    "Tata Ace",
    "Packer & Movers",
)


def configure_page() -> None:
    """Configure the Streamlit browser page."""

    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )


def initialize_session_state() -> None:
    """Initialize values retained across Streamlit reruns."""

    defaults = {
        "finance_response": None,
        "finance_error": None,
        "selected_period": date.today().replace(day=1),
        "selected_analysis": "Management Report",
        "selected_comparison": "Actual vs Budget",
        "selected_frequency": "Monthly",
        "selected_category": "All Categories",
        "selected_scenario": "Management Case",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_header() -> None:
    """Render the dashboard title and current scope."""

    left_column, right_column = st.columns(
        [4, 1]
    )

    with left_column:
        st.title(
            f"{PAGE_ICON} {PAGE_TITLE}"
        )
        st.caption(
            "Chennai Branch | FP&A Management Dashboard"
        )

    with right_column:
        response = st.session_state.finance_response

        if isinstance(response, dict):
            status = response.get(
                "execution_status",
                "unknown",
            )
            st.metric(
                label="Workflow Status",
                value=str(status).title(),
            )


def render_sidebar() -> dict[str, Any]:
    """
    Render dashboard controls.

    Returns:
        Selected filter values and action status.
    """

    with st.sidebar:
        st.header("Dashboard Filters")

        selected_period = st.date_input(
            label="Period",
            value=st.session_state.selected_period,
            help="Select any date within the reporting month.",
        )

        selected_analysis = st.selectbox(
            label="Analysis Type",
            options=ANALYSIS_TYPES,
            index=ANALYSIS_TYPES.index(
                st.session_state.selected_analysis
            ),
        )

        selected_comparison = st.selectbox(
            label="Comparison",
            options=COMPARISON_TYPES,
            index=COMPARISON_TYPES.index(
                st.session_state.selected_comparison
            ),
        )

        selected_frequency = st.selectbox(
            label="Report Frequency",
            options=REPORT_FREQUENCIES,
            index=REPORT_FREQUENCIES.index(
                st.session_state.selected_frequency
            ),
        )

        selected_category = st.selectbox(
            label="Category",
            options=DEFAULT_CATEGORIES,
            index=DEFAULT_CATEGORIES.index(
                st.session_state.selected_category
            ),
        )

        selected_scenario = (
            st.selectbox(
                label="Scenario",
                options=SCENARIO_TYPES,
                index=SCENARIO_TYPES.index(
                    st.session_state.selected_scenario
                ),
            )
            if selected_analysis == "Scenario Analysis"
            else st.session_state.selected_scenario
        )

        st.divider()

        run_analysis = st.button(
            label="Run Analysis",
            type="primary",
            use_container_width=True,
        )

        st.subheader("Quick Reports")

        daily_report = st.button(
            label="Daily Report",
            use_container_width=True,
        )

        weekly_report = st.button(
            label="Weekly Report",
            use_container_width=True,
        )

        monthly_report = st.button(
            label="Monthly Report",
            use_container_width=True,
        )

        st.divider()

        st.caption(
            "All dashboard data currently represents "
            "the Chennai branch."
        )

    st.session_state.selected_period = selected_period
    st.session_state.selected_analysis = selected_analysis
    st.session_state.selected_comparison = selected_comparison
    st.session_state.selected_frequency = selected_frequency
    st.session_state.selected_category = selected_category
    st.session_state.selected_scenario = selected_scenario

    selected_action = None

    if run_analysis:
        selected_action = "analysis"
    elif daily_report:
        selected_action = "daily"
    elif weekly_report:
        selected_action = "weekly"
    elif monthly_report:
        selected_action = "monthly"

    return {
        "period": selected_period,
        "analysis_type": selected_analysis,
        "comparison": selected_comparison,
        "frequency": selected_frequency,
        "category": selected_category,
        "scenario": selected_scenario,
        "action": selected_action,
    }


def build_dashboard_question(
    filters: dict[str, Any],
) -> str:
    """
    Build a controlled finance request from dashboard filters.

    Args:
        filters:
            Selected dashboard filter values.

    Returns:
        Natural-language finance request sent to FastAPI.
    """

    selected_period = filters["period"]
    period_text = selected_period.strftime(
        "%B %Y"
    )

    action = filters.get("action")

    frequency = filters["frequency"]

    if action == "daily":
        frequency = "Daily"
    elif action == "weekly":
        frequency = "Weekly"
    elif action == "monthly":
        frequency = "Monthly"

    request_parts = [
        f"Generate a {frequency.lower()} "
        f"{filters['analysis_type'].lower()}",
        f"for Chennai branch for {period_text}",
        f"using {filters['comparison']}",
    ]

    if filters["category"] != "All Categories":
        request_parts.append(
            f"for the {filters['category']} category"
        )

    if filters["analysis_type"] == "Scenario Analysis":
        request_parts.append(
            f"using the {filters['scenario']} scenario"
        )

    request_parts.append(
        "Include KPI performance, financial tables, "
        "variance analysis, executive summary, risks, "
        "recommendations and management commentary."
    )

    return ". ".join(request_parts)


def process_dashboard_request(
    filters: dict[str, Any],
) -> None:
    """
    Submit a dashboard request to FastAPI.

    Args:
        filters:
            Selected dashboard filter values.
    """

    question = build_dashboard_question(
        filters
    )

    st.session_state.finance_error = None

    try:
        with st.spinner(
            "Running finance analysis..."
        ):
            response = ask_finance_question(
                question=question,
                top_k=DEFAULT_TOP_K,
                metadata_filter={},
            )

    except ValueError as exc:
        st.session_state.finance_error = str(exc)
        return

    except FinanceApiConnectionError as exc:
        st.session_state.finance_error = (
            "The Finance AI backend could not be reached. "
            "Confirm that FastAPI is running.\n\n"
            f"{exc}"
        )
        return

    except FinanceApiResponseError as exc:
        st.session_state.finance_error = (
            "The Finance AI backend returned an error.\n\n"
            f"{exc}"
        )
        return

    except Exception as exc:
        st.session_state.finance_error = (
            "An unexpected error occurred.\n\n"
            f"{exc}"
        )
        return

    st.session_state.finance_response = response


def render_error() -> None:
    """Display the latest dashboard error."""

    error = st.session_state.finance_error

    if error:
        st.error(error)


def render_report_context(
    filters: dict[str, Any],
) -> None:
    """Display the selected reporting context."""

    period_text = filters["period"].strftime(
        "%B %Y"
    )

    columns = st.columns(4)

    columns[0].metric(
        label="Branch",
        value="Chennai",
    )

    columns[1].metric(
        label="Period",
        value=period_text,
    )

    columns[2].metric(
        label="Analysis",
        value=filters["analysis_type"],
    )

    columns[3].metric(
        label="Comparison",
        value=filters["comparison"],
    )


def get_dashboard() -> dict[str, Any] | None:
    """Return the current dashboard response."""

    response = st.session_state.finance_response

    if not isinstance(response, dict):
        return None

    dashboard = response.get(
        "dashboard"
    )

    if isinstance(dashboard, dict):
        return dashboard

    return None


def render_empty_state() -> None:
    """Display instructions before the first analysis."""

    st.info(
        "Select the period and analysis options in the sidebar, "
        "then click **Run Analysis**."
    )


def render_dashboard_metadata(
    dashboard: dict[str, Any],
) -> None:
    """Render dashboard report metadata."""

    metadata = dashboard.get(
        "report_metadata",
        {},
    )

    if not isinstance(metadata, dict):
        return

    title = metadata.get(
        "title"
    )

    if title:
        st.subheader(str(title))

    metadata_values = []

    generated_at = metadata.get(
        "generated_at"
    )

    overall_status = metadata.get(
        "overall_status"
    )

    if generated_at:
        metadata_values.append(
            f"Generated: {generated_at}"
        )

    if overall_status:
        metadata_values.append(
            f"Status: {overall_status}"
        )

    if metadata_values:
        st.caption(
            " | ".join(metadata_values)
        )


def render_kpi_cards(
    dashboard: dict[str, Any],
) -> None:
    """Render dashboard KPI cards."""

    cards = dashboard.get(
        "kpi_cards",
        [],
    )

    if not isinstance(cards, list) or not cards:
        st.info(
            "No KPI card data is available "
            "for this analysis."
        )
        return

    st.subheader("KPI Performance")

    maximum_cards_per_row = 5

    for start_index in range(
        0,
        len(cards),
        maximum_cards_per_row,
    ):
        card_group = cards[
            start_index:
            start_index + maximum_cards_per_row
        ]

        columns = st.columns(
            len(card_group)
        )

        for column, card in zip(
            columns,
            card_group,
        ):
            if not isinstance(card, dict):
                continue

            label = str(
                card.get(
                    "label",
                    "KPI",
                )
            )

            value = (
                card.get("formatted_value")
                or card.get("value")
                or "N/A"
            )

            delta = format_card_delta(
                card
            )

            column.metric(
                label=label,
                value=value,
                delta=delta,
            )


def format_card_delta(
    card: dict[str, Any],
) -> str | None:
    """Create a readable KPI comparison delta."""

    delta = card.get(
        "delta"
    )

    delta_percentage = card.get(
        "delta_percentage"
    )

    if delta is None:
        return None

    unit = str(
        card.get(
            "unit",
            "",
        )
    ).lower()

    if unit in {
        "currency",
        "inr",
        "rupees",
    }:
        delta_text = f"₹{float(delta):,.2f}"
    elif unit in {
        "percentage",
        "percent",
        "%",
    }:
        delta_text = f"{float(delta):,.2f}%"
    else:
        delta_text = f"{float(delta):,.2f}"

    if delta_percentage is not None:
        delta_text += (
            f" ({float(delta_percentage):,.2f}%)"
        )

    comparison_label = card.get(
        "comparison_label"
    )

    if comparison_label:
        delta_text += (
            f" vs {comparison_label}"
        )

    return delta_text


def render_executive_summary(
    dashboard: dict[str, Any],
) -> None:
    """Render the management executive summary."""

    summary = dashboard.get(
        "executive_summary"
    )

    if not summary:
        commentary = dashboard.get(
            "commentary",
            {},
        )

        if isinstance(commentary, dict):
            summary = commentary.get(
                "executive_summary"
            )

    if not summary:
        return

    st.subheader("Executive Summary")
    st.markdown(str(summary))


def render_finance_tables(
    dashboard: dict[str, Any],
) -> None:
    """Render available structured finance tables."""

    table_definitions = (
        (
            "kpi_table",
            "KPI Summary",
        ),
        (
            "variance_table",
            "Actual vs Budget Variance",
        ),
        (
            "category_table",
            "Category Performance",
        ),
    )

    for field_name, default_title in table_definitions:
        table = dashboard.get(
            field_name
        )

        if not isinstance(table, dict):
            continue

        rows = table.get(
            "rows",
            [],
        )

        if not isinstance(rows, list) or not rows:
            continue

        title = table.get(
            "title",
            default_title,
        )

        st.subheader(
            str(title)
        )

        dataframe = pd.DataFrame(
            rows
        )

        declared_columns = table.get(
            "columns",
            [],
        )

        if isinstance(
            declared_columns,
            list,
        ):
            available_columns = [
                column
                for column in declared_columns
                if column in dataframe.columns
            ]

            if available_columns:
                dataframe = dataframe[
                    available_columns
                ]

        st.dataframe(
            dataframe,
            use_container_width=True,
            hide_index=True,
        )


def render_recommendations_and_risks(
    dashboard: dict[str, Any],
) -> None:
    """Render management recommendations and risks."""

    recommendations = dashboard.get(
        "recommendations",
        [],
    )

    risks = dashboard.get(
        "risks",
        [],
    )

    left_column, right_column = st.columns(2)

    with left_column:
        st.subheader("Recommendations")

        if not recommendations:
            st.info(
                "No recommendations were returned."
            )
        else:
            for recommendation in recommendations:
                render_recommendation(
                    recommendation
                )

    with right_column:
        st.subheader("Risks")

        if not risks:
            st.info(
                "No material risks were returned."
            )
        else:
            for risk in risks:
                render_risk(
                    risk
                )


def render_recommendation(
    recommendation: Any,
) -> None:
    """Render one management recommendation."""

    if not isinstance(
        recommendation,
        dict,
    ):
        st.markdown(
            f"- {recommendation}"
        )
        return

    priority = recommendation.get(
        "priority"
    )
    title = recommendation.get(
        "title",
        "Management Recommendation",
    )
    action = recommendation.get(
        "action",
        "",
    )

    heading = str(title)

    if priority:
        heading = (
            f"[{priority}] {heading}"
        )

    with st.container(border=True):
        st.markdown(
            f"**{heading}**"
        )
        st.write(action)

        additional_details = []

        owner = recommendation.get(
            "owner"
        )

        time_horizon = recommendation.get(
            "time_horizon"
        )

        expected_impact = recommendation.get(
            "expected_impact"
        )

        if owner:
            additional_details.append(
                f"Owner: {owner}"
            )

        if time_horizon:
            additional_details.append(
                f"Timeline: {time_horizon}"
            )

        if expected_impact:
            additional_details.append(
                f"Impact: {expected_impact}"
            )

        if additional_details:
            st.caption(
                " | ".join(
                    additional_details
                )
            )


def render_risk(
    risk: Any,
) -> None:
    """Render one dashboard risk."""

    if not isinstance(
        risk,
        dict,
    ):
        st.warning(str(risk))
        return

    severity = risk.get(
        "severity"
    )
    title = risk.get(
        "title",
        "Financial or Operational Risk",
    )
    description = risk.get(
        "description",
        "",
    )

    heading = str(title)

    if severity:
        heading = (
            f"[{severity}] {heading}"
        )

    with st.container(border=True):
        st.markdown(
            f"**{heading}**"
        )
        st.write(description)

        metric = risk.get(
            "metric"
        )

        value = risk.get(
            "value"
        )

        if metric:
            st.caption(
                f"Metric: {metric}"
                + (
                    f" | Value: {value}"
                    if value is not None
                    else ""
                )
            )


def render_commentary(
    dashboard: dict[str, Any],
) -> None:
    """Render structured finance commentary."""

    commentary = dashboard.get(
        "commentary",
        {},
    )

    if not isinstance(
        commentary,
        dict,
    ):
        return

    commentary_sections = (
        (
            "financial_performance",
            "Financial Performance",
        ),
        (
            "kpi_commentary",
            "KPI Commentary",
        ),
        (
            "variance_commentary",
            "Variance Commentary",
        ),
        (
            "forecast_commentary",
            "Forecast Commentary",
        ),
        (
            "scenario_commentary",
            "Scenario Commentary",
        ),
    )

    populated_sections = [
        (
            key,
            title,
            commentary.get(key),
        )
        for key, title in commentary_sections
        if commentary.get(key)
    ]

    management_attention = commentary.get(
        "management_attention",
        [],
    )

    if (
        not populated_sections
        and not management_attention
    ):
        return

    st.subheader("Management Commentary")

    for _, title, value in populated_sections:
        with st.expander(
            title,
            expanded=False,
        ):
            st.markdown(str(value))

    if isinstance(
        management_attention,
        list,
    ) and management_attention:
        with st.expander(
            "Management Attention",
            expanded=True,
        ):
            for item in management_attention:
                st.markdown(
                    f"- {item}"
                )


def render_ai_answer() -> None:
    """Render the complete AI-generated management answer."""

    response = st.session_state.finance_response

    if not isinstance(
        response,
        dict,
    ):
        return

    answer = response.get(
        "answer"
    )

    if not answer:
        return

    with st.expander(
        "Full AI Management Report",
        expanded=False,
    ):
        st.markdown(str(answer))


def render_data_limitations(
    dashboard: dict[str, Any],
) -> None:
    """Render transparent data limitations."""

    limitations = dashboard.get(
        "data_limitations",
        [],
    )

    if not isinstance(
        limitations,
        list,
    ) or not limitations:
        return

    with st.expander(
        "Data Availability and Limitations",
        expanded=False,
    ):
        for limitation in limitations:
            st.markdown(
                f"- {limitation}"
            )


def render_sources() -> None:
    """Display source references returned by FastAPI."""

    response = st.session_state.finance_response

    if not isinstance(
        response,
        dict,
    ):
        return

    sources = response.get(
        "sources",
        [],
    )

    with st.expander(
        "Supporting Sources",
        expanded=False,
    ):
        if not sources:
            st.info(
                "No supporting sources were returned."
            )
            return

        for position, source in enumerate(
            sources,
            start=1,
        ):
            render_source(
                source,
                position,
            )


def render_source(
    source: Any,
    position: int,
) -> None:
    """Display one source record."""

    if not isinstance(
        source,
        dict,
    ):
        st.markdown(
            f"**Source {position}:** {source}"
        )
        return

    title = resolve_source_title(
        source,
        position,
    )

    st.markdown(
        f"**{title}**"
    )

    excerpt = source.get(
        "excerpt"
    )

    if excerpt:
        st.write(excerpt)

    score = source.get(
        "score"
    )

    rank = source.get(
        "rank"
    )

    details = []

    if rank is not None:
        details.append(
            f"Rank: {rank}"
        )

    if score is not None:
        details.append(
            f"Score: {float(score):.4f}"
        )

    if details:
        st.caption(
            " | ".join(details)
        )


def resolve_source_title(
    source: dict[str, Any],
    position: int,
) -> str:
    """Resolve a readable source title."""

    metadata = source.get(
        "metadata",
        {},
    )

    candidate_sources = [
        source,
        metadata
        if isinstance(metadata, dict)
        else {},
    ]

    candidate_fields = (
        "document_id",
        "source",
        "file_name",
        "filename",
        "title",
        "id",
    )

    for candidate in candidate_sources:
        for field_name in candidate_fields:
            value = candidate.get(
                field_name
            )

            if value:
                return (
                    f"Source {position}: {value}"
                )

    return f"Source {position}"


def render_custom_question() -> None:
    """Render an optional natural-language finance question form."""

    with st.expander(
        "Ask Finance AI",
        expanded=False,
    ):
        with st.form(
            key="custom_finance_question_form",
            clear_on_submit=False,
        ):
            question = st.text_area(
                label="Finance Question",
                placeholder=(
                    "Example: Why was revenue below budget "
                    "for the selected month?"
                ),
                height=100,
            )

            submitted = st.form_submit_button(
                label="Ask Finance AI",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            process_custom_question(
                question
            )


def process_custom_question(
    question: str,
) -> None:
    """Submit a natural-language finance question."""

    normalized_question = question.strip()

    if not normalized_question:
        st.warning(
            "Enter a finance question."
        )
        return

    try:
        with st.spinner(
            "Analysing your finance question..."
        ):
            response = ask_finance_question(
                question=normalized_question,
                top_k=DEFAULT_TOP_K,
                metadata_filter={},
            )

    except (
        ValueError,
        FinanceApiConnectionError,
        FinanceApiResponseError,
    ) as exc:
        st.error(str(exc))
        return

    except Exception as exc:
        st.error(
            "An unexpected error occurred."
        )
        st.caption(str(exc))
        return

    st.session_state.finance_response = response
    st.session_state.finance_error = None


def render_dashboard(
    filters: dict[str, Any],
) -> None:
    """Render the complete structured dashboard."""

    render_report_context(
        filters
    )

    dashboard = get_dashboard()

    if dashboard is None:
        render_empty_state()
        return

    render_dashboard_metadata(
        dashboard
    )

    render_kpi_cards(
        dashboard
    )

    st.divider()

    render_executive_summary(
        dashboard
    )

    render_finance_tables(
        dashboard
    )

    render_recommendations_and_risks(
        dashboard
    )

    render_commentary(
        dashboard
    )

    render_ai_answer()

    render_data_limitations(
        dashboard
    )

    render_sources()


def main() -> None:
    """Run the Streamlit dashboard."""

    configure_page()
    initialize_session_state()
    render_header()

    filters = render_sidebar()

    if filters["action"]:
        process_dashboard_request(
            filters
        )

    render_error()

    render_dashboard(
        filters
    )

    st.divider()

    render_custom_question()


if __name__ == "__main__":
    main()