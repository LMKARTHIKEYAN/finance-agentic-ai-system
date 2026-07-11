"""
Tests for the Finance Agentic AI document retriever.
"""

from __future__ import annotations

from typing import Sequence

import pytest

from src.rag.embeddings import BaseEmbeddingService
from src.rag.retriever import (
    DEFAULT_MAX_CONTEXT_CHARACTERS,
    DEFAULT_TOP_K,
    FinanceRetriever,
    RetrievedDocument,
    RetrievalConfig,
    RetrievalResult,
    RetrieverError,
    create_retriever,
    format_retrieval_context,
)
from src.rag.vector_store import (
    Document,
    InMemoryVectorStore,
    VectorStoreError,
)


class FixedEmbeddingService(BaseEmbeddingService):
    """Predictable embedding service for retriever tests."""

    def __init__(
        self,
        vectors: dict[str, list[float]] | None = None,
        dimension: int = 3,
    ) -> None:
        self._dimension = dimension
        self.vectors = vectors or {}

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> list[float]:
        return list(
            self.vectors.get(
                text,
                [1.0] + [0.0] * (self._dimension - 1),
            )
        )

    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return list(
            self.vectors.get(
                query,
                [1.0] + [0.0] * (self._dimension - 1),
            )
        )


class FailingVectorStore(InMemoryVectorStore):
    """Vector store that always fails during search."""

    def similarity_search(self, *args: object, **kwargs: object) -> list:
        raise VectorStoreError("Mock vector-store failure")


def build_test_store() -> InMemoryVectorStore:
    """Build a predictable populated vector store."""

    vectors = {
        "revenue query": [1.0, 0.0, 0.0],
        "Revenue increased because volume exceeded budget.": [
            1.0,
            0.0,
            0.0,
        ],
        "Gross profit improved because cost decreased.": [
            0.8,
            0.2,
            0.0,
        ],
        "Headcount policy requires manager approval.": [
            0.0,
            1.0,
            0.0,
        ],
    }

    store = InMemoryVectorStore(
        FixedEmbeddingService(vectors=vectors)
    )

    store.add_documents(
        [
            {
                "id": "revenue-doc",
                "text": (
                    "Revenue increased because volume "
                    "exceeded budget."
                ),
                "metadata": {
                    "category": "revenue",
                    "year": 2026,
                },
            },
            {
                "id": "profit-doc",
                "text": (
                    "Gross profit improved because "
                    "cost decreased."
                ),
                "metadata": {
                    "category": "profit",
                    "year": 2026,
                },
            },
            {
                "id": "headcount-doc",
                "text": (
                    "Headcount policy requires "
                    "manager approval."
                ),
                "metadata": {
                    "category": "policy",
                    "year": 2025,
                },
            },
        ]
    )

    return store


def test_default_retrieval_config() -> None:
    config = RetrievalConfig()

    assert config.top_k == DEFAULT_TOP_K
    assert config.score_threshold is None
    assert (
        config.max_context_characters
        == DEFAULT_MAX_CONTEXT_CHARACTERS
    )
    assert config.include_metadata is True
    assert config.include_scores is True
    assert config.deduplicate is True


@pytest.mark.parametrize("top_k", [0, -1, -10])
def test_retrieval_config_rejects_invalid_top_k(
    top_k: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        RetrievalConfig(top_k=top_k)


@pytest.mark.parametrize("top_k", [1.5, "5", True])
def test_retrieval_config_rejects_non_integer_top_k(
    top_k: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="top_k must be an integer",
    ):
        RetrievalConfig(top_k=top_k)  # type: ignore[arg-type]


@pytest.mark.parametrize("threshold", [-1.1, 1.1, -5.0, 5.0])
def test_retrieval_config_rejects_invalid_threshold(
    threshold: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="between -1.0 and 1.0",
    ):
        RetrievalConfig(score_threshold=threshold)


@pytest.mark.parametrize("threshold", ["0.5", True, []])
def test_retrieval_config_rejects_non_numeric_threshold(
    threshold: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="score_threshold must be numeric",
    ):
        RetrievalConfig(
            score_threshold=threshold  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("limit", [0, -1, -100])
def test_retrieval_config_rejects_invalid_context_limit(
    limit: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="max_context_characters must be greater than zero",
    ):
        RetrievalConfig(max_context_characters=limit)


@pytest.mark.parametrize("limit", [1.5, "100", True])
def test_retrieval_config_rejects_non_integer_context_limit(
    limit: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="max_context_characters must be an integer",
    ):
        RetrievalConfig(
            max_context_characters=limit  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "include_metadata",
        "include_scores",
        "deduplicate",
    ],
)
def test_retrieval_config_rejects_non_boolean_fields(
    field_name: str,
) -> None:
    kwargs = {field_name: "yes"}

    with pytest.raises(TypeError, match="must be a boolean"):
        RetrievalConfig(**kwargs)  # type: ignore[arg-type]


def test_retrieved_document_creation() -> None:
    document = Document(
        id="doc-1",
        text="Revenue policy",
        metadata={"category": "revenue"},
    )

    result = RetrievedDocument(
        document=document,
        score=0.9,
        rank=1,
    )

    assert result.document == document
    assert result.score == 0.9
    assert result.rank == 1


def test_retrieved_document_deep_copies_metadata() -> None:
    metadata = {"details": {"year": 2026}}

    document = Document(
        id="doc-1",
        text="Revenue policy",
        metadata=metadata,
    )

    result = RetrievedDocument(
        document=document,
        score=0.9,
        rank=1,
    )

    metadata["details"]["year"] = 2030

    assert result.document.metadata["details"]["year"] == 2026


def test_retrieved_document_rejects_invalid_document() -> None:
    with pytest.raises(
        TypeError,
        match="document must be a Document",
    ):
        RetrievedDocument(
            document="invalid",  # type: ignore[arg-type]
            score=0.5,
            rank=1,
        )


@pytest.mark.parametrize("score", [-1.1, 1.1])
def test_retrieved_document_rejects_invalid_score(
    score: float,
) -> None:
    document = Document(
        id="doc-1",
        text="Revenue policy",
    )

    with pytest.raises(
        ValueError,
        match="between -1.0 and 1.0",
    ):
        RetrievedDocument(
            document=document,
            score=score,
            rank=1,
        )


@pytest.mark.parametrize("score", ["0.5", True, []])
def test_retrieved_document_rejects_non_numeric_score(
    score: object,
) -> None:
    document = Document(
        id="doc-1",
        text="Revenue policy",
    )

    with pytest.raises(TypeError, match="score must be numeric"):
        RetrievedDocument(
            document=document,
            score=score,  # type: ignore[arg-type]
            rank=1,
        )


@pytest.mark.parametrize("rank", [0, -1, -10])
def test_retrieved_document_rejects_invalid_rank(
    rank: int,
) -> None:
    document = Document(
        id="doc-1",
        text="Revenue policy",
    )

    with pytest.raises(
        ValueError,
        match="rank must be greater than zero",
    ):
        RetrievedDocument(
            document=document,
            score=0.5,
            rank=rank,
        )


def test_retrieval_result_creation() -> None:
    document = RetrievedDocument(
        document=Document(
            id="doc-1",
            text="Revenue policy",
        ),
        score=0.8,
        rank=1,
    )

    result = RetrievalResult(
        query="revenue query",
        documents=(document,),
        context="Source 1",
        metadata_filter={"category": "revenue"},
        total_results=1,
    )

    assert result.query == "revenue query"
    assert result.total_results == 1
    assert result.has_results is True


def test_retrieval_result_no_results() -> None:
    result = RetrievalResult(
        query="revenue query",
        documents=(),
        context="",
        total_results=0,
    )

    assert result.has_results is False
    assert result.source_documents == []


def test_retrieval_result_rejects_count_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="must match the number of documents",
    ):
        RetrievalResult(
            query="revenue query",
            documents=(),
            context="",
            total_results=1,
        )


def test_retrieval_result_source_documents_are_copies() -> None:
    retrieved = RetrievedDocument(
        document=Document(
            id="doc-1",
            text="Revenue policy",
            metadata={"details": {"year": 2026}},
        ),
        score=0.8,
        rank=1,
    )

    result = RetrievalResult(
        query="revenue query",
        documents=(retrieved,),
        context="Source 1",
        total_results=1,
    )

    sources = result.source_documents
    sources[0].metadata["details"]["year"] = 2030

    second_sources = result.source_documents

    assert second_sources[0].metadata["details"]["year"] == 2026


def test_finance_retriever_creation() -> None:
    store = build_test_store()
    retriever = FinanceRetriever(store)

    assert retriever.vector_store is store
    assert isinstance(retriever.config, RetrievalConfig)


def test_finance_retriever_rejects_invalid_store() -> None:
    with pytest.raises(
        TypeError,
        match="InMemoryVectorStore",
    ):
        FinanceRetriever("invalid")  # type: ignore[arg-type]


def test_retrieve_returns_ranked_documents() -> None:
    retriever = FinanceRetriever(build_test_store())

    result = retriever.retrieve(
        query="revenue query",
        top_k=3,
    )

    assert result.total_results == 3
    assert result.documents[0].document.id == "revenue-doc"
    assert [item.rank for item in result.documents] == [1, 2, 3]


def test_retrieve_builds_context() -> None:
    retriever = FinanceRetriever(build_test_store())

    result = retriever.retrieve(
        query="revenue query",
        top_k=1,
    )

    assert "Source 1" in result.context
    assert "Document ID: revenue-doc" in result.context
    assert "Similarity Score:" in result.context
    assert "category=revenue" in result.context
    assert "Revenue increased" in result.context


def test_retrieve_applies_metadata_filter() -> None:
    retriever = FinanceRetriever(build_test_store())

    result = retriever.retrieve(
        query="revenue query",
        metadata_filter={"category": "profit"},
    )

    assert result.total_results == 1
    assert result.documents[0].document.id == "profit-doc"


def test_retrieve_applies_score_threshold() -> None:
    retriever = FinanceRetriever(build_test_store())

    result = retriever.retrieve(
        query="revenue query",
        score_threshold=0.99,
    )

    assert result.total_results == 1
    assert result.documents[0].document.id == "revenue-doc"


def test_retrieve_uses_config_defaults() -> None:
    config = RetrievalConfig(
        top_k=1,
        score_threshold=0.5,
    )

    retriever = FinanceRetriever(
        build_test_store(),
        config=config,
    )

    result = retriever.retrieve("revenue query")

    assert result.total_results == 1


def test_retrieve_documents_returns_only_documents() -> None:
    retriever = FinanceRetriever(build_test_store())

    documents = retriever.retrieve_documents(
        "revenue query",
        top_k=2,
    )

    assert len(documents) == 2
    assert all(
        isinstance(document, Document)
        for document in documents
    )


def test_retrieve_empty_store() -> None:
    store = InMemoryVectorStore(
        FixedEmbeddingService()
    )
    retriever = FinanceRetriever(store)

    result = retriever.retrieve("revenue query")

    assert result.total_results == 0
    assert result.context == ""
    assert result.has_results is False


@pytest.mark.parametrize("query", ["", " ", "   "])
def test_retrieve_rejects_empty_query(query: str) -> None:
    retriever = FinanceRetriever(build_test_store())

    with pytest.raises(ValueError, match="query cannot be empty"):
        retriever.retrieve(query)


def test_retrieve_rejects_non_string_query() -> None:
    retriever = FinanceRetriever(build_test_store())

    with pytest.raises(TypeError, match="query must be a string"):
        retriever.retrieve(100)  # type: ignore[arg-type]


def test_retrieve_rejects_invalid_metadata_filter() -> None:
    retriever = FinanceRetriever(build_test_store())

    with pytest.raises(
        TypeError,
        match="metadata_filter must be a mapping",
    ):
        retriever.retrieve(
            "revenue query",
            metadata_filter=["revenue"],  # type: ignore[arg-type]
        )


def test_retrieve_wraps_vector_store_failure() -> None:
    store = FailingVectorStore(
        FixedEmbeddingService()
    )
    retriever = FinanceRetriever(store)

    with pytest.raises(
        RetrieverError,
        match="Document retrieval failed",
    ):
        retriever.retrieve("revenue query")


def test_build_context_empty_documents() -> None:
    retriever = FinanceRetriever(build_test_store())

    assert retriever.build_context([]) == ""


def test_build_context_without_metadata() -> None:
    retriever = FinanceRetriever(
        build_test_store(),
        config=RetrievalConfig(
            include_metadata=False,
        ),
    )

    document = RetrievedDocument(
        document=Document(
            id="doc-1",
            text="Revenue policy",
            metadata={"category": "revenue"},
        ),
        score=0.8,
        rank=1,
    )

    context = retriever.build_context([document])

    assert "Metadata:" not in context
    assert "Revenue policy" in context


def test_build_context_without_scores() -> None:
    retriever = FinanceRetriever(
        build_test_store(),
        config=RetrievalConfig(
            include_scores=False,
        ),
    )

    document = RetrievedDocument(
        document=Document(
            id="doc-1",
            text="Revenue policy",
        ),
        score=0.8,
        rank=1,
    )

    context = retriever.build_context([document])

    assert "Similarity Score:" not in context


def test_build_context_respects_character_limit() -> None:
    retriever = FinanceRetriever(build_test_store())

    document = RetrievedDocument(
        document=Document(
            id="doc-1",
            text="A" * 500,
        ),
        score=0.8,
        rank=1,
    )

    context = retriever.build_context(
        [document],
        max_characters=100,
    )

    assert len(context) <= 100
    assert context.endswith("...")


def test_build_context_rejects_invalid_documents() -> None:
    retriever = FinanceRetriever(build_test_store())

    with pytest.raises(
        TypeError,
        match="RetrievedDocument",
    ):
        retriever.build_context(
            ["invalid"]  # type: ignore[list-item]
        )


def test_format_retrieval_context_helper() -> None:
    documents = [
        RetrievedDocument(
            document=Document(
                id="doc-1",
                text="Revenue policy",
                metadata={"category": "revenue"},
            ),
            score=0.9,
            rank=1,
        )
    ]

    context = format_retrieval_context(
        documents,
        include_metadata=True,
        include_scores=True,
    )

    assert "Source 1" in context
    assert "Document ID: doc-1" in context
    assert "Similarity Score: 0.900000" in context
    assert "category=revenue" in context


def test_create_retriever_helper() -> None:
    store = build_test_store()
    config = RetrievalConfig(top_k=2)

    retriever = create_retriever(
        vector_store=store,
        config=config,
    )

    assert isinstance(retriever, FinanceRetriever)
    assert retriever.vector_store is store
    assert retriever.config is config


def test_deduplicate_removes_duplicate_text() -> None:
    vectors = {
        "query": [1.0, 0.0, 0.0],
        "Duplicate finance policy": [1.0, 0.0, 0.0],
    }

    store = InMemoryVectorStore(
        FixedEmbeddingService(vectors=vectors)
    )

    store.add_documents(
        [
            {
                "id": "doc-1",
                "text": "Duplicate finance policy",
            },
            {
                "id": "doc-2",
                "text": "  duplicate   finance policy  ",
            },
        ]
    )

    retriever = FinanceRetriever(
        store,
        config=RetrievalConfig(deduplicate=True),
    )

    result = retriever.retrieve("query", top_k=2)

    assert result.total_results == 1
    assert result.documents[0].rank == 1


def test_deduplicate_can_be_disabled() -> None:
    vectors = {
        "query": [1.0, 0.0, 0.0],
        "Duplicate finance policy": [1.0, 0.0, 0.0],
    }

    store = InMemoryVectorStore(
        FixedEmbeddingService(vectors=vectors)
    )

    store.add_documents(
        [
            {
                "id": "doc-1",
                "text": "Duplicate finance policy",
            },
            {
                "id": "doc-2",
                "text": "  duplicate   finance policy  ",
            },
        ]
    )

    retriever = FinanceRetriever(
        store,
        config=RetrievalConfig(deduplicate=False),
    )

    result = retriever.retrieve("query", top_k=2)

    assert result.total_results == 2