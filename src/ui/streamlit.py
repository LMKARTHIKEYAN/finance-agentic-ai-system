"""
Streamlit interface for the Finance Agentic AI Assistant.

This module provides the initial user interface for submitting finance
questions to the existing FastAPI backend and displaying the returned
answer and source references.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.ui.api_client import (
    FinanceApiConnectionError,
    FinanceApiResponseError,
    ask_finance_question,
)


PAGE_TITLE = "Finance AI Assistant"
PAGE_ICON = "📊"
DEFAULT_TOP_K = 5


def configure_page() -> None:
    """Configure the Streamlit browser page."""

    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout="centered",
        initial_sidebar_state="collapsed",
    )


def initialize_session_state() -> None:
    """Initialize values retained across Streamlit reruns."""

    if "finance_answer" not in st.session_state:
        st.session_state.finance_answer = None

    if "finance_sources" not in st.session_state:
        st.session_state.finance_sources = []


def render_header() -> None:
    """Render the page title and description."""

    st.title(f"{PAGE_ICON} {PAGE_TITLE}")

    st.write(
        "Ask a finance question and receive an answer from the "
        "Finance Agentic AI system."
    )


def render_question_form() -> tuple[str, bool]:
    """
    Render the question input form.

    Returns:
        A tuple containing the entered question and submission status.
    """

    with st.form(
        key="finance_question_form",
        clear_on_submit=False,
    ):
        question = st.text_area(
            label="Question",
            placeholder="Example: What is the revenue variance?",
            height=120,
        )

        submitted = st.form_submit_button(
            label="Ask",
            type="primary",
            use_container_width=True,
        )

    return question, submitted


def process_question(question: str) -> None:
    """
    Send the question to FastAPI and store the response.

    Args:
        question:
            Finance question entered by the user.
    """

    normalized_question = question.strip()

    if not normalized_question:
        st.warning("Enter a finance question before clicking Ask.")
        return

    try:
        with st.spinner("Analysing your finance question..."):
            response = ask_finance_question(
                question=normalized_question,
                top_k=DEFAULT_TOP_K,
                metadata_filter={},
            )

    except ValueError as exc:
        st.error(str(exc))
        return

    except FinanceApiConnectionError as exc:
        st.error(
            "The Finance AI backend could not be reached. "
            "Confirm that FastAPI is running."
        )
        st.caption(str(exc))
        return

    except FinanceApiResponseError as exc:
        st.error(
            "The Finance AI backend returned an error while processing "
            "the question."
        )
        st.caption(str(exc))
        return

    except Exception as exc:
        st.error("An unexpected error occurred.")
        st.caption(str(exc))
        return

    st.session_state.finance_answer = response.get(
        "answer",
        "No answer was returned.",
    )
    st.session_state.finance_sources = response.get("sources", [])


def render_answer() -> None:
    """Display the most recent finance answer."""

    answer = st.session_state.finance_answer

    if answer is None:
        return

    st.divider()
    st.subheader("Answer")
    st.markdown(answer)


def render_sources() -> None:
    """Display source references returned by FastAPI."""

    if st.session_state.finance_answer is None:
        return

    sources = st.session_state.finance_sources

    st.subheader("Sources")

    if not sources:
        st.info("No supporting sources were returned for this answer.")
        return

    for position, source in enumerate(sources, start=1):
        render_source(source, position)


def render_source(source: Any, position: int) -> None:
    """
    Display one source.

    Args:
        source:
            Source returned by the FastAPI response.

        position:
            Source number displayed to the user.
    """

    if isinstance(source, dict):
        title = resolve_source_title(source, position)

        with st.expander(title):
            st.json(source)

        return

    st.markdown(f"**Source {position}:** {source}")


def resolve_source_title(
    source: dict[str, Any],
    position: int,
) -> str:
    """
    Resolve a readable source title.

    Args:
        source:
            Source metadata dictionary.

        position:
            Source number.

    Returns:
        Readable source title.
    """

    candidate_fields = (
        "document_id",
        "source",
        "file_name",
        "filename",
        "title",
    )

    for field_name in candidate_fields:
        value = source.get(field_name)

        if value:
            return f"Source {position}: {value}"

    return f"Source {position}"


def main() -> None:
    """Run the Streamlit application."""

    configure_page()
    initialize_session_state()
    render_header()

    question, submitted = render_question_form()

    if submitted:
        process_question(question)

    render_answer()
    render_sources()


if __name__ == "__main__":
    main()