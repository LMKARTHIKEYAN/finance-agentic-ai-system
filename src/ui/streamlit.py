"""
Streamlit interface for the Finance Agentic AI System.

The UI supports:

- Natural-language finance requests
- Clarification and temporary slot filling
- KPI cards
- Executive summaries
- Trend charts
- Variance bridge charts
- Finance tables
- Recommendations
- Risks
- Management commentary
- RAG source references

The Streamlit application communicates only with the FastAPI backend.
It must not contain finance calculations, LangGraph execution or data access.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.ui.api_client import (
    FinanceApiClientError,
    FinanceApiConnectionError,
    FinanceApiResponseError,
    ask_finance_question,
)


APP_TITLE = "Finance Agentic AI"

APP_SUBTITLE = (
    "Natural-language FP&A analysis using the project's local sample data"
)

DATA_SOURCE_LABEL = "Local project data"
DATA_SOURCE_DESCRIPTION = (
    "The FastAPI backend automatically loads the existing local CSV, "
    "Excel and assumptions files. No user upload is required."
)

DEFAULT_TOP_K = 5

EXAMPLE_QUESTIONS = (
    "Show actual vs budget revenue variance for January 2026",
    "Show revenue, orders, AOV and fulfillment for March 2026",
    "Show KPI performance for 3W for January 2026",
    "Show forecast performance for 2026",
    (
        "Show actual vs budget variance from "
        "1 January 2026 to 31 March 2026"
    ),
)


def main() -> None:
    """Run the Streamlit Finance Agentic AI application."""

    _configure_page()
    _initialize_session_state()
    _render_sidebar()
    _render_header()
    _render_conversation()

    pending_request = st.session_state.get(
        "pending_request"
    )

    if pending_request:
        input_placeholder = (
            "Enter the missing period, for example: January 2026"
        )
    else:
        input_placeholder = "Ask a finance question..."

    submitted_question = st.chat_input(
        input_placeholder
    )

    if submitted_question:
        _process_user_message(
            submitted_question
        )
        return

    latest_response = st.session_state.get(
        "latest_response"
    )

    if (
        isinstance(latest_response, dict)
        and latest_response
        and not latest_response.get(
            "clarification_required",
            False,
        )
    ):
        _render_response_metadata(
            latest_response
        )

        dashboard = latest_response.get(
            "dashboard"
        )

        if (
            isinstance(dashboard, dict)
            and dashboard
        ):
            _render_dashboard(
                dashboard
            )

        _render_ai_answer(
            latest_response
        )

        _render_sources(
            latest_response.get(
                "sources",
                [],
            )
        )


def _configure_page() -> None:
    """Configure the Streamlit browser page."""

    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1600px;
        }

        [data-testid="stMetric"] {
            background-color: rgba(128, 128, 128, 0.06);
            border: 1px solid rgba(128, 128, 128, 0.20);
            border-radius: 10px;
            padding: 1rem;
        }

        [data-testid="stMetricLabel"] {
            font-weight: 600;
        }

        .finance-subtitle {
            color: rgba(128, 128, 128, 0.95);
            margin-top: -0.7rem;
            margin-bottom: 1.2rem;
        }

        .clarification-box {
            background-color: rgba(255, 193, 7, 0.08);
            border: 1px solid rgba(255, 193, 7, 0.40);
            border-radius: 10px;
            padding: 1rem;
            margin-top: 0.5rem;
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _initialize_session_state() -> None:
    """Initialize session-state values used by the application."""

    defaults: dict[str, Any] = {
        "messages": [],
        "pending_request": None,
        "pending_intent": {},
        "latest_response": None,
        "top_k": DEFAULT_TOP_K,
        "show_intent_details": False,
        "request_in_progress": False,
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def _render_sidebar() -> None:
    """Render application controls and example questions."""

    with st.sidebar:
        st.header("Finance AI Controls")

        st.success(f"Data source: {DATA_SOURCE_LABEL}")
        st.caption(DATA_SOURCE_DESCRIPTION)

        st.session_state.top_k = st.slider(
            "RAG sources",
            min_value=1,
            max_value=20,
            value=int(
                st.session_state.top_k
            ),
            help=(
                "Maximum number of reference documents "
                "retrieved for the AI response."
            ),
        )

        st.session_state.show_intent_details = st.toggle(
            "Show interpreted request",
            value=bool(
                st.session_state.show_intent_details
            ),
        )

        st.divider()

        st.subheader("Example questions")

        for index, example in enumerate(
            EXAMPLE_QUESTIONS
        ):
            if st.button(
                example,
                key=f"example_question_{index}",
                use_container_width=True,
                disabled=bool(
                    st.session_state.request_in_progress
                ),
            ):
                _process_user_message(
                    example
                )

        st.divider()

        if st.button(
            "Start new analysis session",
            use_container_width=True,
        ):
            _clear_conversation()

        pending_request = st.session_state.get(
            "pending_request"
        )

        if pending_request:
            st.warning(
                "Waiting for reporting period"
            )

            st.caption(
                f"Original request: {pending_request}"
            )

            st.caption(
                "Enter only the missing information in the chat box."
            )

            st.code(
                "January 2026",
                language=None,
            )


def _render_header() -> None:
    """Render the application title and introduction."""

    st.title(APP_TITLE)

    st.markdown(
        f'<p class="finance-subtitle">{APP_SUBTITLE}</p>',
        unsafe_allow_html=True,
    )

    if not st.session_state.messages:
        st.info(
            "Ask for KPI, budget, forecast, variance, scenario or "
            "management commentary. The backend uses the existing "
            "local project data automatically. Include a reporting "
            "period when possible."
        )

    pending_request = st.session_state.get(
        "pending_request"
    )

    if pending_request:
        st.markdown(
            """
            <div class="clarification-box">
                <strong>Clarification required</strong><br>
                Enter only the missing reporting period below.<br><br>
                Examples: <code>January 2026</code>,
                <code>20 October 2026</code>, or
                <code>1 January 2026 to 31 March 2026</code>.
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_conversation() -> None:
    """Render user and assistant chat history."""

    for message in st.session_state.messages:
        role = str(
            message.get(
                "role",
                "assistant",
            )
        )

        content = str(
            message.get(
                "content",
                "",
            )
        )

        with st.chat_message(role):
            st.markdown(content)


def _process_user_message(
    user_message: str,
) -> None:
    """
    Submit a user request to FastAPI.

    If a clarification is pending, the new message is combined with the
    original incomplete request.
    """

    normalized_message = " ".join(
        str(user_message).split()
    )

    if not normalized_message:
        return

    if st.session_state.request_in_progress:
        return

    pending_request = st.session_state.get(
        "pending_request"
    )

    if (
        pending_request
        and _is_repeated_pending_request(
            original_request=str(
                pending_request
            ),
            new_message=normalized_message,
        )
    ):
        st.session_state.messages.append(
            {
                "role": "user",
                "content": normalized_message,
            }
        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (
                    "Please enter only the missing reporting period. "
                    "For example: **January 2026**."
                ),
            }
        )

        st.rerun()
        return

    st.session_state.messages.append(
        {
            "role": "user",
            "content": normalized_message,
        }
    )

    if pending_request:
        api_question = _combine_pending_request(
            original_request=str(
                pending_request
            ),
            clarification_answer=normalized_message,
        )
    else:
        api_question = normalized_message

    st.session_state.request_in_progress = True

    try:
        with st.spinner(
            "Running finance analysis..."
        ):
            response = ask_finance_question(
                question=api_question,
                top_k=int(
                    st.session_state.top_k
                ),
            )

    except FinanceApiConnectionError as exc:
        _record_ui_error(
            title="Finance API connection failed",
            message=str(exc),
        )
        return

    except FinanceApiResponseError as exc:
        _record_ui_error(
            title="Finance API returned an invalid response",
            message=str(exc),
        )
        return

    except FinanceApiClientError as exc:
        _record_ui_error(
            title="Finance request failed",
            message=str(exc),
        )
        return

    except (TypeError, ValueError) as exc:
        _record_ui_error(
            title="Invalid finance request",
            message=str(exc),
        )
        return

    finally:
        st.session_state.request_in_progress = False

    _handle_api_response(
        response=response,
        submitted_request=api_question,
    )

    st.rerun()


def _is_repeated_pending_request(
    *,
    original_request: str,
    new_message: str,
) -> bool:
    """
    Determine whether the user repeated the incomplete request.

    This prevents the same clarification question from being submitted again.
    """

    normalized_original = _normalize_comparison_text(
        original_request
    )

    normalized_new_message = _normalize_comparison_text(
        new_message
    )

    if normalized_original == normalized_new_message:
        return True

    if (
        len(normalized_new_message) > 20
        and normalized_new_message
        in normalized_original
    ):
        return True

    return False


def _normalize_comparison_text(
    value: str,
) -> str:
    """Normalize text for simple duplicate-request comparison."""

    return (
        " ".join(
            str(value).lower().split()
        )
        .strip()
        .rstrip(".")
    )


def _combine_pending_request(
    *,
    original_request: str,
    clarification_answer: str,
) -> str:
    """
    Combine an incomplete request with the user's clarification.

    Example:

        Original:
            Show actual vs budget revenue variance

        Clarification:
            January 2026

        Combined:
            Show actual vs budget revenue variance for January 2026
    """

    cleaned_original = (
        original_request
        .strip()
        .rstrip(".")
    )

    cleaned_answer = (
        clarification_answer
        .strip()
        .rstrip(".")
    )

    return (
        f"{cleaned_original} for "
        f"{cleaned_answer}"
    )


def _handle_api_response(
    *,
    response: dict[str, Any],
    submitted_request: str,
) -> None:
    """Store the API response and manage clarification state."""

    if not isinstance(response, dict):
        _record_ui_error(
            title="Invalid API response",
            message=(
                "The Finance API response must be a dictionary."
            ),
        )
        return

    clarification_required = bool(
        response.get(
            "clarification_required",
            False,
        )
    )

    answer = str(
        response.get(
            "answer",
            "",
        )
    ).strip()

    if not answer:
        answer = (
            "The Finance API completed the request but "
            "did not return a response message."
        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

    intent = response.get(
        "intent",
        {},
    )

    if not isinstance(intent, dict):
        intent = {}

    if clarification_required:
        original_request = intent.get(
            "original_request"
        )

        pending_request = (
            str(original_request).strip()
            if original_request
            else submitted_request
        )

        st.session_state.pending_request = pending_request
        st.session_state.pending_intent = intent

        # Clarification responses must not display dashboard cards.
        st.session_state.latest_response = None
        return

    st.session_state.pending_request = None
    st.session_state.pending_intent = {}
    st.session_state.latest_response = response


def _record_ui_error(
    *,
    title: str,
    message: str,
) -> None:
    """Save a recoverable UI error in the conversation."""

    st.session_state.request_in_progress = False

    error_text = (
        f"**{title}**\n\n{message}"
    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": error_text,
        }
    )

    st.session_state.latest_response = None
    st.rerun()


def _clear_conversation() -> None:
    """Clear temporary conversation and dashboard state."""

    st.session_state.messages = []
    st.session_state.pending_request = None
    st.session_state.pending_intent = {}
    st.session_state.latest_response = None
    st.session_state.request_in_progress = False

    st.rerun()


def _render_response_metadata(
    response: dict[str, Any],
) -> None:
    """Render execution and workflow metadata."""

    if response.get(
        "clarification_required",
        False,
    ):
        return

    selected_flow = response.get(
        "selected_flow",
        "unknown",
    )

    execution_status = response.get(
        "execution_status",
        "unknown",
    )

    used_fallback = bool(
        response.get(
            "used_fallback",
            False,
        )
    )

    columns = st.columns(3)

    columns[0].metric(
        "Analysis",
        _humanize(
            selected_flow
        ),
    )

    columns[1].metric(
        "Execution Status",
        _humanize(
            execution_status
        ),
    )

    columns[2].metric(
        "AI Response",
        (
            "Fallback"
            if used_fallback
            else "Grounded"
        ),
    )

    if st.session_state.show_intent_details:
        intent = response.get(
            "intent",
            {},
        )

        if isinstance(intent, dict) and intent:
            with st.expander(
                "Interpreted finance request",
                expanded=False,
            ):
                _render_intent(
                    intent
                )


def _render_intent(
    intent: dict[str, Any],
) -> None:
    """Render structured intent information."""

    period = intent.get(
        "period",
        {},
    )

    if not isinstance(period, dict):
        period = {}

    requested_kpis = intent.get(
        "requested_kpis",
        [],
    )

    if not isinstance(requested_kpis, list):
        requested_kpis = []

    details = {
        "Selected flow": _humanize(
            intent.get(
                "selected_flow",
                "unknown",
            )
        ),
        "Comparison": _humanize(
            intent.get(
                "comparison",
                "none",
            )
        ),
        "Period": (
            period.get(
                "display_value"
            )
            or "Not provided"
        ),
        "Category": (
            intent.get(
                "category"
            )
            or "All Categories"
        ),
        "Scenario": (
            intent.get(
                "scenario_name"
            )
            or "Not applicable"
        ),
        "Requested KPIs": (
            ", ".join(
                _humanize(item)
                for item in requested_kpis
            )
            or "Not specifically requested"
        ),
    }

    for label, value in details.items():
        st.markdown(
            f"**{label}:** {value}"
        )


def _render_dashboard(
    dashboard: dict[str, Any],
) -> None:
    """Render the complete management dashboard."""

    st.divider()
    st.header("FP&A Management Dashboard")

    _render_report_metadata(
        dashboard.get(
            "report_metadata",
            {},
        )
    )

    _render_executive_summary(
        dashboard.get(
            "executive_summary"
        )
    )

    _render_kpi_cards(
        dashboard.get(
            "kpi_cards",
            [],
        )
    )

    _render_charts(
        trend_data=dashboard.get(
            "trend_data",
            [],
        ),
        waterfall_data=dashboard.get(
            "waterfall_data",
            [],
        ),
    )

    _render_tables(
        dashboard
    )

    _render_management_sections(
        recommendations=dashboard.get(
            "recommendations",
            [],
        ),
        risks=dashboard.get(
            "risks",
            [],
        ),
    )

    _render_commentary(
        dashboard.get(
            "commentary",
            {},
        )
    )

    _render_data_limitations(
        dashboard.get(
            "data_limitations",
            [],
        )
    )


def _render_report_metadata(
    metadata: Any,
) -> None:
    """Render dashboard reporting metadata."""

    if not isinstance(metadata, dict) or not metadata:
        return

    period = (
        metadata.get("period")
        or metadata.get("reporting_period")
        or metadata.get("period_label")
    )

    comparison = metadata.get(
        "comparison"
    )

    category = metadata.get(
        "category"
    )

    generated_at = metadata.get(
        "generated_at"
    )

    metadata_items: list[str] = []

    if period:
        metadata_items.append(
            f"**Period:** {period}"
        )

    if comparison:
        metadata_items.append(
            f"**Comparison:** {_humanize(comparison)}"
        )

    if category:
        metadata_items.append(
            f"**Category:** {category}"
        )

    if generated_at:
        metadata_items.append(
            f"**Generated:** {generated_at}"
        )

    if metadata_items:
        st.caption(
            "  |  ".join(
                metadata_items
            )
        )


def _render_executive_summary(
    executive_summary: Any,
) -> None:
    """Render the management executive summary."""

    if not isinstance(
        executive_summary,
        str,
    ):
        return

    summary = executive_summary.strip()

    if not summary:
        return

    st.subheader("Executive Summary")
    st.info(summary)


def _render_kpi_cards(
    kpi_cards: Any,
) -> None:
    """Render responsive KPI cards."""

    if not isinstance(
        kpi_cards,
        list,
    ) or not kpi_cards:
        return

    st.subheader("Key Performance Indicators")

    for start_index in range(
        0,
        len(kpi_cards),
        4,
    ):
        card_group = kpi_cards[
            start_index:
            start_index + 4
        ]

        columns = st.columns(
            len(card_group)
        )

        for column, card in zip(
            columns,
            card_group,
            strict=False,
        ):
            if not isinstance(
                card,
                dict,
            ):
                continue

            label = (
                card.get("label")
                or card.get("title")
                or card.get("name")
                or card.get("metric")
                or "KPI"
            )

            column.metric(
                label=str(label),
                value=_resolve_display_value(
                    card
                ),
                delta=_resolve_delta(
                    card
                ),
                delta_color=_resolve_delta_color(
                    card
                ),
            )


def _resolve_display_value(
    card: dict[str, Any],
) -> str:
    """Resolve the display value for a KPI card."""

    for field_name in (
        "display_value",
        "formatted_value",
        "value",
    ):
        value = card.get(
            field_name
        )

        if value is not None:
            return str(value)

    return "N/A"


def _resolve_delta(
    card: dict[str, Any],
) -> str | None:
    """Resolve a KPI card delta value."""

    for field_name in (
        "delta",
        "display_delta",
        "variance_display",
        "variance",
        "change",
    ):
        value = card.get(
            field_name
        )

        if (
            value is not None
            and str(value).strip()
        ):
            return str(value)

    return None


def _resolve_delta_color(
    card: dict[str, Any],
) -> str:
    """Resolve Streamlit metric delta direction."""

    status = str(
        card.get(
            "status",
            "",
        )
    ).strip().lower()

    direction = str(
        card.get(
            "direction",
            "",
        )
    ).strip().lower()

    if status in {
        "unfavourable",
        "unfavorable",
        "negative",
        "warning",
        "fail",
    }:
        return "inverse"

    if direction in {
        "lower_is_better",
        "inverse",
    }:
        return "inverse"

    if status in {
        "neutral",
        "no_change",
    }:
        return "off"

    return "normal"


def _render_charts(
    *,
    trend_data: Any,
    waterfall_data: Any,
) -> None:
    """Render trend and variance bridge charts."""

    has_trend = (
        isinstance(trend_data, list)
        and bool(trend_data)
    )

    has_waterfall = (
        isinstance(waterfall_data, list)
        and bool(waterfall_data)
    )

    if not has_trend and not has_waterfall:
        return

    st.subheader("Performance Visuals")

    if has_trend and has_waterfall:
        trend_column, bridge_column = st.columns(2)

        with trend_column:
            _render_trend_chart(
                trend_data
            )

        with bridge_column:
            _render_waterfall_chart(
                waterfall_data
            )

    elif has_trend:
        _render_trend_chart(
            trend_data
        )

    else:
        _render_waterfall_chart(
            waterfall_data
        )


def _render_trend_chart(
    trend_data: list[Any],
) -> None:
    """Render available time-series data."""

    dataframe = _records_to_dataframe(
        trend_data
    )

    if dataframe.empty:
        return

    st.markdown(
        "**Trend Analysis**"
    )

    label_column = _find_first_column(
        dataframe,
        (
            "period",
            "date",
            "month",
            "label",
            "category",
        ),
    )

    numeric_columns = list(
        dataframe.select_dtypes(
            include="number"
        ).columns
    )

    if not numeric_columns:
        st.dataframe(
            dataframe,
            use_container_width=True,
            hide_index=True,
        )
        return

    chart_dataframe = dataframe.copy()

    if label_column:
        chart_dataframe = (
            chart_dataframe.set_index(
                label_column
            )
        )

    st.line_chart(
        chart_dataframe[
            numeric_columns
        ],
        use_container_width=True,
    )


def _render_waterfall_chart(
    waterfall_data: list[Any],
) -> None:
    """Render variance bridge data using a native bar chart."""

    dataframe = _records_to_dataframe(
        waterfall_data
    )

    if dataframe.empty:
        return

    st.markdown(
        "**Variance Bridge**"
    )

    label_column = _find_first_column(
        dataframe,
        (
            "label",
            "driver",
            "effect",
            "category",
            "name",
        ),
    )

    value_column = _find_first_column(
        dataframe,
        (
            "value",
            "amount",
            "variance",
            "effect_value",
        ),
    )

    if (
        label_column is None
        or value_column is None
    ):
        st.dataframe(
            dataframe,
            use_container_width=True,
            hide_index=True,
        )
        return

    chart_dataframe = dataframe[
        [
            label_column,
            value_column,
        ]
    ].copy()

    chart_dataframe = chart_dataframe.set_index(
        label_column
    )

    st.bar_chart(
        chart_dataframe,
        use_container_width=True,
    )


def _render_tables(
    dashboard: dict[str, Any],
) -> None:
    """Render available dashboard tables."""

    table_definitions = (
        (
            "kpi_table",
            "KPI Detail",
        ),
        (
            "variance_table",
            "Variance Analysis",
        ),
        (
            "category_table",
            "Category Performance",
        ),
    )

    available_tables = [
        (
            field_name,
            title,
            dashboard.get(
                field_name
            ),
        )
        for field_name, title in table_definitions
        if _table_has_data(
            dashboard.get(
                field_name
            )
        )
    ]

    if not available_tables:
        return

    st.subheader("Detailed Analysis")

    tabs = st.tabs(
        [
            title
            for _, title, _ in available_tables
        ]
    )

    for tab, (
        _,
        _,
        table_payload,
    ) in zip(
        tabs,
        available_tables,
        strict=False,
    ):
        with tab:
            _render_table_payload(
                table_payload
            )


def _table_has_data(
    table_payload: Any,
) -> bool:
    """Return whether a dashboard table contains records."""

    if not isinstance(
        table_payload,
        dict,
    ):
        return False

    rows = (
        table_payload.get("rows")
        or table_payload.get("data")
        or table_payload.get("records")
        or []
    )

    return (
        isinstance(rows, list)
        and bool(rows)
    )


def _render_table_payload(
    table_payload: Any,
) -> None:
    """Render one structured dashboard table."""

    if not isinstance(
        table_payload,
        dict,
    ):
        return

    title = table_payload.get(
        "title"
    )

    if title:
        st.markdown(
            f"**{title}**"
        )

    rows = (
        table_payload.get("rows")
        or table_payload.get("data")
        or table_payload.get("records")
        or []
    )

    if not isinstance(rows, list):
        st.warning(
            "The API returned an invalid table structure."
        )
        return

    dataframe = _records_to_dataframe(
        rows
    )

    if dataframe.empty:
        st.caption(
            "No table records are available."
        )
        return

    columns = table_payload.get(
        "columns"
    )

    if isinstance(columns, list) and columns:
        valid_columns = [
            column
            for column in columns
            if column in dataframe.columns
        ]

        if valid_columns:
            dataframe = dataframe[
                valid_columns
            ]

    st.dataframe(
        dataframe,
        use_container_width=True,
        hide_index=True,
    )


def _render_management_sections(
    *,
    recommendations: Any,
    risks: Any,
) -> None:
    """Render management recommendations and risks."""

    has_recommendations = (
        isinstance(recommendations, list)
        and bool(recommendations)
    )

    has_risks = (
        isinstance(risks, list)
        and bool(risks)
    )

    if not has_recommendations and not has_risks:
        return

    st.subheader("Management Review")

    recommendation_column, risk_column = st.columns(2)

    with recommendation_column:
        st.markdown(
            "#### Recommendations"
        )

        if has_recommendations:
            _render_item_list(
                recommendations
            )
        else:
            st.caption(
                "No recommendations were generated."
            )

    with risk_column:
        st.markdown(
            "#### Risks and Exceptions"
        )

        if has_risks:
            _render_item_list(
                risks
            )
        else:
            st.caption(
                "No material risks were identified."
            )


def _render_item_list(
    items: list[Any],
) -> None:
    """Render management items."""

    for item in items:
        if isinstance(item, str):
            st.markdown(
                f"- {item}"
            )
            continue

        if not isinstance(item, dict):
            st.markdown(
                f"- {item}"
            )
            continue

        title = (
            item.get("title")
            or item.get("action")
            or item.get("risk")
            or item.get("description")
            or item.get("message")
            or "Item"
        )

        priority = (
            item.get("priority")
            or item.get("severity")
        )

        owner = item.get(
            "owner"
        )

        prefix_parts: list[str] = []

        if priority:
            prefix_parts.append(
                str(priority).upper()
            )

        if owner:
            prefix_parts.append(
                str(owner)
            )

        prefix = (
            f"[{' | '.join(prefix_parts)}] "
            if prefix_parts
            else ""
        )

        st.markdown(
            f"- {prefix}{title}"
        )


def _render_commentary(
    commentary: Any,
) -> None:
    """Render management commentary."""

    if not isinstance(
        commentary,
        dict,
    ) or not commentary:
        return

    st.subheader("Management Commentary")

    commentary_text = (
        commentary.get("text")
        or commentary.get("summary")
        or commentary.get(
            "executive_commentary"
        )
        or commentary.get(
            "management_commentary"
        )
    )

    if commentary_text:
        st.markdown(
            str(commentary_text)
        )

    remaining_sections = {
        key: value
        for key, value in commentary.items()
        if key
        not in {
            "text",
            "summary",
            "executive_commentary",
            "management_commentary",
        }
        and value not in (
            None,
            "",
            [],
            {},
        )
    }

    if remaining_sections:
        with st.expander(
            "Additional commentary",
            expanded=False,
        ):
            for title, value in remaining_sections.items():
                st.markdown(
                    f"**{_humanize(title)}**"
                )

                if isinstance(value, list):
                    _render_item_list(
                        value
                    )
                elif isinstance(value, dict):
                    st.json(value)
                else:
                    st.markdown(
                        str(value)
                    )


def _render_data_limitations(
    limitations: Any,
) -> None:
    """Render disclosed data limitations."""

    if not isinstance(
        limitations,
        list,
    ) or not limitations:
        return

    with st.expander(
        "Data limitations",
        expanded=False,
    ):
        _render_item_list(
            limitations
        )


def _render_ai_answer(
    response: dict[str, Any],
) -> None:
    """Render the grounded AI finance response."""

    answer = response.get(
        "answer"
    )

    if not isinstance(answer, str):
        return

    cleaned_answer = answer.strip()

    if not cleaned_answer:
        return

    st.subheader("AI Finance Analysis")
    st.markdown(cleaned_answer)


def _render_sources(
    sources: Any,
) -> None:
    """Render RAG reference sources."""

    if not isinstance(
        sources,
        list,
    ) or not sources:
        return

    with st.expander(
        f"Sources ({len(sources)})",
        expanded=False,
    ):
        for source_index, source in enumerate(
            sources,
            start=1,
        ):
            if not isinstance(
                source,
                dict,
            ):
                continue

            metadata = source.get(
                "metadata",
                {},
            )

            if not isinstance(
                metadata,
                dict,
            ):
                metadata = {}

            source_name = (
                metadata.get("filename")
                or metadata.get("source")
                or metadata.get("title")
                or source.get("id")
                or f"Source {source_index}"
            )

            score = source.get(
                "score"
            )

            rank = source.get(
                "rank",
                source_index,
            )

            excerpt = str(
                source.get(
                    "excerpt",
                    "",
                )
            ).strip()

            heading = (
                f"**{rank}. {source_name}**"
            )

            if isinstance(
                score,
                (int, float),
            ):
                heading += (
                    f" — relevance {score:.3f}"
                )

            st.markdown(
                heading
            )

            if excerpt:
                st.caption(
                    excerpt
                )

            if source_index < len(
                sources
            ):
                st.divider()


def _records_to_dataframe(
    records: list[Any],
) -> pd.DataFrame:
    """Convert API records into a safe pandas dataframe."""

    valid_records = [
        record
        for record in records
        if isinstance(record, dict)
    ]

    if not valid_records:
        return pd.DataFrame()

    return pd.DataFrame(
        valid_records
    )


def _find_first_column(
    dataframe: pd.DataFrame,
    candidates: tuple[str, ...],
) -> str | None:
    """Find the first candidate column in a dataframe."""

    normalized_mapping = {
        str(column).strip().lower(): str(
            column
        )
        for column in dataframe.columns
    }

    for candidate in candidates:
        matched_column = normalized_mapping.get(
            candidate.lower()
        )

        if matched_column:
            return matched_column

    return None


def _humanize(
    value: Any,
) -> str:
    """Convert identifiers into user-friendly labels."""

    cleaned_value = str(
        value
    ).strip()

    if not cleaned_value:
        return "Unknown"

    return (
        cleaned_value
        .replace(
            "_",
            " ",
        )
        .replace(
            "-",
            " ",
        )
        .title()
    )


if __name__ == "__main__":
    main()