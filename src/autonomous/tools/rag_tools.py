"""Agentic RAG retrieval with compact, traceable document evidence."""

from __future__ import annotations

from typing import Any


def search_approved_documents(
    *, retriever: Any, query: str, top_k: int = 5
) -> dict[str, Any]:
    """Retrieve approved narrative evidence; never calculate finance values."""

    result = retriever.retrieve(query, top_k=top_k)
    sources = []
    for item in result.documents:
        metadata = dict(item.document.metadata)
        sources.append(
            {
                "source_id": str(item.document.id),
                "title": str(
                    metadata.get("title")
                    or metadata.get("filename")
                    or metadata.get("source")
                    or item.document.id
                ),
                "excerpt": item.document.text[:500],
                "score": float(item.score),
                "metadata": metadata,
            }
        )
    return {
        "query": result.query,
        "context": result.context,
        "sources": sources,
        "total_results": result.total_results,
        "summary": f"Retrieved {result.total_results} approved document source(s).",
    }
