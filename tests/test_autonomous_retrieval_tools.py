"""Tests for the read-only retrieval wrapper."""

from types import SimpleNamespace
from typing import Any

import pytest

from src.autonomous.tools.retrieval_tools import retrieve_company_context


class FakeRetriever:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def retrieve(self, query: str, **kwargs: Any) -> SimpleNamespace:
        self.kwargs = {"query": query, **kwargs}
        document = SimpleNamespace(
            id="doc-1",
            text="A" * 100,
            metadata={"type": "policy"},
        )
        return SimpleNamespace(
            query=query,
            total_results=1,
            documents=(
                SimpleNamespace(document=document, rank=1, score=0.9),
            ),
        )


def test_retrieval_wrapper_returns_compact_ranked_documents() -> None:
    retriever = FakeRetriever()

    result = retrieve_company_context(
        retriever,
        "margin policy",
        top_k=2,
        metadata_filter={"type": "policy"},
        max_excerpt_characters=20,
    )

    document = result.payload["documents"][0]
    assert document["document_id"] == "doc-1"
    assert document["rank"] == 1
    assert len(document["excerpt"]) == 20
    assert retriever.kwargs["top_k"] == 2
    assert retriever.kwargs["metadata_filter"] == {"type": "policy"}


def test_retrieval_wrapper_rejects_invalid_retriever() -> None:
    with pytest.raises(TypeError):
        retrieve_company_context(object(), "policy")


def test_retrieval_wrapper_rejects_empty_query() -> None:
    with pytest.raises(ValueError):
        retrieve_company_context(FakeRetriever(), " ")
