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
    """

    answer: str

    sources: list[SourceResponse] = Field(default_factory=list)

    selected_flow: str | None = None

    execution_status: str

    used_fallback: bool = False