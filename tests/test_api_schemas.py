"""
Tests for API request and response schemas.
"""

from pydantic import ValidationError

from src.api.schemas import (
    AskRequest,
    AskResponse,
    SourceResponse,
)


def test_default_request_values():
    """Default values should be populated."""

    request = AskRequest(question="Revenue variance")

    assert request.question == "Revenue variance"
    assert request.top_k == 5
    assert request.metadata_filter == {}


def test_custom_request_values():
    """Custom values should override defaults."""

    request = AskRequest(
        question="Budget variance",
        top_k=10,
        metadata_filter={"department": "Finance"},
    )

    assert request.top_k == 10
    assert request.metadata_filter["department"] == "Finance"


def test_empty_question_validation():
    """Empty questions should not be allowed."""

    try:
        AskRequest(question="")
        assert False
    except ValidationError:
        assert True


def test_invalid_top_k():
    """top_k must be >= 1."""

    try:
        AskRequest(
            question="Revenue",
            top_k=0,
        )
        assert False
    except ValidationError:
        assert True


def test_source_response():
    """Source model should store values correctly."""

    source = SourceResponse(
        id="chunk_001",
        score=0.95,
        rank=1,
        metadata={"file": "finance.pdf"},
        excerpt="Revenue increased by 12%",
    )

    assert source.id == "chunk_001"
    assert source.rank == 1
    assert source.score == 0.95


def test_ask_response():
    """AskResponse should serialize correctly."""

    response = AskResponse(
        answer="Revenue increased.",
        execution_status="completed",
        selected_flow="variance",
        used_fallback=False,
    )

    assert response.answer == "Revenue increased."
    assert response.execution_status == "completed"
    assert response.sources == []