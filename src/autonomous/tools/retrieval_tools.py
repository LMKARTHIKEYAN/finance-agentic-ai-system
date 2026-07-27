"""Read-only company-context retrieval tool wrapper."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.autonomous.schemas import ToolResult


def retrieve_company_context(
    retriever: Any,
    query: str,
    *,
    top_k: int = 3,
    score_threshold: float | None = None,
    metadata_filter: Mapping[str, Any] | None = None,
    max_excerpt_characters: int = 500,
) -> ToolResult:
    """Retrieve compact ranked excerpts without invoking generation."""

    if not hasattr(retriever, "retrieve") or not callable(
        retriever.retrieve
    ):
        raise TypeError("retriever must provide a callable retrieve method.")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    if (
        isinstance(max_excerpt_characters, bool)
        or not isinstance(max_excerpt_characters, int)
        or max_excerpt_characters <= 0
    ):
        raise ValueError(
            "max_excerpt_characters must be a positive integer."
        )

    result = retriever.retrieve(
        query.strip(),
        top_k=top_k,
        score_threshold=score_threshold,
        metadata_filter=dict(metadata_filter or {}),
    )
    documents = []

    for item in result.documents:
        text = item.document.text.strip()
        documents.append(
            {
                "document_id": item.document.id,
                "rank": item.rank,
                "score": float(item.score),
                "metadata": dict(item.document.metadata),
                "excerpt": text[:max_excerpt_characters],
            }
        )

    return ToolResult(
        call_id="retrieve_company_context",
        tool_name="retrieve_company_context",
        status="completed",
        payload={
            "query": result.query,
            "total_results": result.total_results,
            "documents": documents,
        },
    )
