"""
Tests for the in-memory RAG vector store.

These tests cover:

- Document validation
- SearchResult validation
- Single and batch document insertion
- Duplicate handling
- Atomic batch operations
- Document retrieval and deletion
- Upsert operations
- Similarity search
- Metadata filtering
- Score thresholds
- Vector-based search
- Helper functions
- Embedding error handling
"""

from __future__ import annotations

import math
from typing import Sequence

import pytest

from src.rag.embeddings import (
    BaseEmbeddingService,
    DeterministicEmbeddingService,
)
from src.rag.vector_store import (
    Document,
    DocumentNotFoundError,
    InMemoryVectorStore,
    SearchResult,
    VectorStoreError,
    build_vector_store,
    documents_from_texts,
    iter_document_batches,
)


class FixedEmbeddingService(BaseEmbeddingService):
    """
    Predictable embedding service for vector-store tests.

    Known text values are mapped to fixed vectors. Unknown text values use
    a default vector.
    """

    def __init__(
        self,
        vectors: dict[str, list[float]] | None = None,
        dimension: int = 3,
    ) -> None:
        self._dimension = dimension
        self.vectors = vectors or {}
        self.embed_text_calls: list[str] = []
        self.embed_texts_calls: list[list[str]] = []
        self.embed_query_calls: list[str] = []

    @property
    def dimension(self) -> int:
        """Return the configured embedding dimension."""

        return self._dimension

    def embed_text(self, text: str) -> list[float]:
        """Return a predictable vector for one text."""

        self.embed_text_calls.append(text)

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
        """Return predictable vectors for multiple texts."""

        copied_texts = list(texts)
        self.embed_texts_calls.append(copied_texts)

        return [
            list(
                self.vectors.get(
                    text,
                    [1.0] + [0.0] * (self._dimension - 1),
                )
            )
            for text in copied_texts
        ]

    def embed_query(self, query: str) -> list[float]:
        """Return a predictable query vector."""

        self.embed_query_calls.append(query)

        return list(
            self.vectors.get(
                query,
                [1.0] + [0.0] * (self._dimension - 1),
            )
        )


class FailingEmbeddingService(BaseEmbeddingService):
    """Embedding service that always fails."""

    @property
    def dimension(self) -> int:
        return 3

    def embed_text(self, text: str) -> list[float]:
        raise RuntimeError("Mock embedding failure")

    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        raise RuntimeError("Mock batch embedding failure")

    def embed_query(self, query: str) -> list[float]:
        raise RuntimeError("Mock query embedding failure")


class WrongDimensionEmbeddingService(BaseEmbeddingService):
    """Embedding service that returns vectors with an invalid dimension."""

    @property
    def dimension(self) -> int:
        return 3

    def embed_text(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, query: str) -> list[float]:
        return [1.0, 0.0]


class InvalidValueEmbeddingService(BaseEmbeddingService):
    """Embedding service that returns a non-numeric value."""

    @property
    def dimension(self) -> int:
        return 3

    def embed_text(self, text: str) -> list[float]:
        return [1.0, "invalid", 0.0]  # type: ignore[list-item]

    def embed_query(self, query: str) -> list[float]:
        return [1.0, "invalid", 0.0]  # type: ignore[list-item]


class IncorrectBatchCountEmbeddingService(BaseEmbeddingService):
    """Embedding service that returns too few vectors."""

    @property
    def dimension(self) -> int:
        return 3

    def embed_text(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        return [[1.0, 0.0, 0.0]]


def test_document_creation() -> None:
    """A valid document should retain its fields."""

    document = Document(
        id="finance-policy",
        text="Revenue recognition policy.",
        metadata={"category": "policy"},
    )

    assert document.id == "finance-policy"
    assert document.text == "Revenue recognition policy."
    assert document.metadata == {"category": "policy"}


def test_document_strips_id_and_text() -> None:
    """Document identifiers and text should be stripped."""

    document = Document(
        id="  budget-policy  ",
        text="  Budget approval policy.  ",
    )

    assert document.id == "budget-policy"
    assert document.text == "Budget approval policy."


@pytest.mark.parametrize("document_id", ["", " ", "   "])
def test_document_rejects_empty_id(
    document_id: str,
) -> None:
    """Document identifiers cannot be empty."""

    with pytest.raises(
        ValueError,
        match="document id cannot be empty",
    ):
        Document(
            id=document_id,
            text="Valid text",
        )


@pytest.mark.parametrize("text", ["", " ", "   "])
def test_document_rejects_empty_text(text: str) -> None:
    """Document text cannot be empty."""

    with pytest.raises(
        ValueError,
        match="document text cannot be empty",
    ):
        Document(
            id="doc-1",
            text=text,
        )


def test_document_rejects_non_dictionary_metadata() -> None:
    """Document metadata must be a dictionary."""

    with pytest.raises(
        TypeError,
        match="metadata must be a dictionary",
    ):
        Document(
            id="doc-1",
            text="Valid text",
            metadata=["policy"],  # type: ignore[arg-type]
        )


def test_document_deep_copies_metadata() -> None:
    """External metadata changes should not mutate a document."""

    metadata = {
        "category": "budget",
        "details": {"year": 2026},
    }

    document = Document(
        id="doc-1",
        text="Budget assumptions",
        metadata=metadata,
    )

    metadata["details"]["year"] = 2030

    assert document.metadata["details"]["year"] == 2026


def test_search_result_creation() -> None:
    """A valid search result should retain its values."""

    document = Document(
        id="doc-1",
        text="Revenue policy",
    )

    result = SearchResult(
        document=document,
        score=0.8,
        rank=1,
    )

    assert result.document == document
    assert result.score == 0.8
    assert result.rank == 1


@pytest.mark.parametrize("rank", [0, -1, -100])
def test_search_result_rejects_invalid_rank(rank: int) -> None:
    """Search result ranks must be positive."""

    document = Document(
        id="doc-1",
        text="Revenue policy",
    )

    with pytest.raises(
        ValueError,
        match="rank must be greater than zero",
    ):
        SearchResult(
            document=document,
            score=0.5,
            rank=rank,
        )


@pytest.mark.parametrize("score", [-1.1, 1.1, -5.0, 5.0])
def test_search_result_rejects_invalid_score(
    score: float,
) -> None:
    """Similarity scores must remain within cosine bounds."""

    document = Document(
        id="doc-1",
        text="Revenue policy",
    )

    with pytest.raises(
        ValueError,
        match="between -1.0 and 1.0",
    ):
        SearchResult(
            document=document,
            score=score,
            rank=1,
        )


def test_vector_store_uses_default_embedding_service() -> None:
    """Store should use deterministic embeddings by default."""

    store = InMemoryVectorStore()

    assert isinstance(
        store.embedding_service,
        DeterministicEmbeddingService,
    )


def test_vector_store_uses_supplied_embedding_service() -> None:
    """Store should retain a custom embedding provider."""

    embedding_service = FixedEmbeddingService()
    store = InMemoryVectorStore(embedding_service)

    assert store.embedding_service is embedding_service
    assert store.dimension == 3


def test_new_vector_store_is_empty() -> None:
    """A newly created vector store should contain no documents."""

    store = InMemoryVectorStore()

    assert len(store) == 0
    assert store.list_documents() == []


def test_add_document() -> None:
    """A document should be embedded and stored."""

    embedding_service = FixedEmbeddingService()
    store = InMemoryVectorStore(embedding_service)

    document = store.add_document(
        text="Revenue recognition policy.",
        metadata={"category": "policy"},
        document_id="revenue-policy",
    )

    assert document.id == "revenue-policy"
    assert document.text == "Revenue recognition policy."
    assert document.metadata == {"category": "policy"}
    assert len(store) == 1
    assert "revenue-policy" in store
    assert embedding_service.embed_text_calls == [
        "Revenue recognition policy."
    ]


def test_add_document_generates_id() -> None:
    """A UUID should be generated when no identifier is supplied."""

    store = InMemoryVectorStore(
        FixedEmbeddingService()
    )

    document = store.add_document(
        text="Forecast assumption document."
    )

    assert isinstance(document.id, str)
    assert document.id
    assert document.id in store


def test_add_document_strips_values() -> None:
    """Text and identifiers should be cleaned before storage."""

    store = InMemoryVectorStore(
        FixedEmbeddingService()
    )

    document = store.add_document(
        text="  Budget approval rules.  ",
        document_id="  budget-rules  ",
    )

    assert document.id == "budget-rules"
    assert document.text == "Budget approval rules."


@pytest.mark.parametrize("text", ["", " ", "   "])
def test_add_document_rejects_empty_text(text: str) -> None:
    """Empty documents should not be stored."""

    store = InMemoryVectorStore()

    with pytest.raises(ValueError, match="text cannot be empty"):
        store.add_document(text=text)

    assert len(store) == 0


def test_add_document_rejects_non_string_text() -> None:
    """Document text must be a string."""

    store = InMemoryVectorStore()

    with pytest.raises(TypeError, match="text must be a string"):
        store.add_document(text=100)  # type: ignore[arg-type]


def test_add_document_rejects_invalid_metadata() -> None:
    """Metadata must be a mapping."""

    store = InMemoryVectorStore()

    with pytest.raises(
        TypeError,
        match="metadata must be a mapping",
    ):
        store.add_document(
            text="Revenue policy",
            metadata=["policy"],  # type: ignore[arg-type]
        )


def test_add_document_rejects_duplicate_id() -> None:
    """Existing identifiers should not be overwritten by add."""

    store = InMemoryVectorStore(
        FixedEmbeddingService()
    )

    store.add_document(
        text="First policy",
        document_id="policy-1",
    )

    with pytest.raises(
        VectorStoreError,
        match="already exists",
    ):
        store.add_document(
            text="Second policy",
            document_id="policy-1",
        )

    assert len(store) == 1
    assert store.get_document("policy-1").text == "First policy"


def test_add_document_wraps_embedding_failure() -> None:
    """Embedding provider failures should become VectorStoreError."""

    store = InMemoryVectorStore(
        FailingEmbeddingService()
    )

    with pytest.raises(
        VectorStoreError,
        match="Unable to embed document",
    ):
        store.add_document(
            text="Revenue policy",
            document_id="doc-1",
        )

    assert len(store) == 0


def test_add_document_rejects_wrong_embedding_dimension() -> None:
    """Stored vectors must match the configured dimension."""

    store = InMemoryVectorStore(
        WrongDimensionEmbeddingService()
    )

    with pytest.raises(
        VectorStoreError,
        match="Embedding dimension mismatch",
    ):
        store.add_document(
            text="Revenue policy",
            document_id="doc-1",
        )


def test_add_document_rejects_non_numeric_embedding() -> None:
    """Embedding values must be numeric."""

    store = InMemoryVectorStore(
        InvalidValueEmbeddingService()
    )

    with pytest.raises(
        VectorStoreError,
        match="Embedding values must be numeric",
    ):
        store.add_document(
            text="Revenue policy",
            document_id="doc-1",
        )


def test_add_documents_with_document_objects() -> None:
    """Document objects should be accepted in batch insertion."""

    embedding_service = FixedEmbeddingService()
    store = InMemoryVectorStore(embedding_service)

    added_documents = store.add_documents(
        [
            Document(
                id="doc-1",
                text="Revenue policy",
                metadata={"category": "revenue"},
            ),
            Document(
                id="doc-2",
                text="Budget policy",
                metadata={"category": "budget"},
            ),
        ]
    )

    assert len(added_documents) == 2
    assert len(store) == 2
    assert embedding_service.embed_texts_calls == [
        ["Revenue policy", "Budget policy"]
    ]


def test_add_documents_with_strings() -> None:
    """Plain text strings should be accepted in batches."""

    store = InMemoryVectorStore(
        FixedEmbeddingService()
    )

    documents = store.add_documents(
        [
            "Revenue policy",
            "Budget policy",
        ]
    )

    assert len(documents) == 2
    assert len(store) == 2
    assert documents[0].text == "Revenue policy"
    assert documents[1].text == "Budget policy"
    assert documents[0].id != documents[1].id


def test_add_documents_with_mappings() -> None:
    """Document mappings should support id, text and metadata."""

    store = InMemoryVectorStore(
        FixedEmbeddingService()
    )

    documents = store.add_documents(
        [
            {
                "id": "revenue-doc",
                "text": "Revenue recognition policy",
                "metadata": {"category": "revenue"},
            },
            {
                "id": "budget-doc",
                "text": "Budget approval policy",
                "metadata": {"category": "budget"},
            },
        ]
    )

    assert len(documents) == 2
    assert store.get_document("revenue-doc").metadata == {
        "category": "revenue"
    }


def test_add_documents_accepts_empty_sequence() -> None:
    """An empty batch should perform no operation."""

    store = InMemoryVectorStore()

    result = store.add_documents([])

    assert result == []
    assert len(store) == 0


def test_add_documents_rejects_single_string() -> None:
    """One string must not be interpreted as a document sequence."""

    store = InMemoryVectorStore()

    with pytest.raises(
        TypeError,
        match="not a single string",
    ):
        store.add_documents("Revenue policy")  # type: ignore[arg-type]


def test_add_documents_rejects_non_sequence() -> None:
    """Batch input must be a sequence."""

    store = InMemoryVectorStore()

    with pytest.raises(
        TypeError,
        match="documents must be a sequence",
    ):
        store.add_documents(100)  # type: ignore[arg-type]


def test_add_documents_rejects_invalid_item() -> None:
    """Unsupported batch items should fail."""

    store = InMemoryVectorStore()

    with pytest.raises(
        TypeError,
        match="Document, string, or mapping",
    ):
        store.add_documents([100])  # type: ignore[list-item]


def test_add_documents_mapping_requires_text() -> None:
    """Document mappings must contain text."""

    store = InMemoryVectorStore()

    with pytest.raises(
        ValueError,
        match="must contain a 'text' field",
    ):
        store.add_documents(
            [{"id": "doc-1"}]
        )


def test_add_documents_rejects_duplicate_ids_in_batch() -> None:
    """Duplicate identifiers inside one batch should fail."""

    store = InMemoryVectorStore(
        FixedEmbeddingService()
    )

    with pytest.raises(
        VectorStoreError,
        match="duplicate document identifiers",
    ):
        store.add_documents(
            [
                {
                    "id": "duplicate",
                    "text": "First text",
                },
                {
                    "id": "duplicate",
                    "text": "Second text",
                },
            ]
        )

    assert len(store) == 0


def test_add_documents_rejects_existing_id() -> None:
    """A batch should fail if an identifier already exists."""

    store = InMemoryVectorStore(
        FixedEmbeddingService()
    )

    store.add_document(
        text="Existing document",
        document_id="existing",
    )

    with pytest.raises(
        VectorStoreError,
        match="already exist",
    ):
        store.add_documents(
            [
                {
                    "id": "new-document",
                    "text": "New text",
                },
                {
                    "id": "existing",
                    "text": "Replacement text",
                },
            ]
        )

    assert len(store) == 1
    assert "new-document" not in store


def test_add_documents_is_atomic_on_embedding_failure() -> None:
    """A failed batch should not partially insert documents."""

    store = InMemoryVectorStore(
        FailingEmbeddingService()
    )

    with pytest.raises(
        VectorStoreError,
        match="Unable to embed document batch",
    ):
        store.add_documents(
            [
                {
                    "id": "doc-1",
                    "text": "Revenue policy",
                },
                {
                    "id": "doc-2",
                    "text": "Budget policy",
                },
            ]
        )

    assert len(store) == 0


def test_add_documents_rejects_wrong_vector_count() -> None:
    """Batch vector count must equal document count."""

    store = InMemoryVectorStore(
        IncorrectBatchCountEmbeddingService()
    )

    with pytest.raises(
        VectorStoreError,
        match="unexpected number of vectors",
    ):
        store.add_documents(
            [
                {
                    "id": "doc-1",
                    "text": "Revenue policy",
                },
                {
                    "id": "doc-2",
                    "text": "Budget policy",
                },
            ]
        )

    assert len(store) == 0


def test_get_document() -> None:
    """Stored documents should be retrievable by identifier."""

    store = InMemoryVectorStore(
        FixedEmbeddingService()
    )

    store.add_document(
        text="Revenue policy",
        metadata={"category": "policy"},
        document_id="doc-1",
    )

    document = store.get_document("doc-1")

    assert document.id == "doc-1"
    assert document.text == "Revenue policy"
    assert document.metadata == {"category": "policy"}


def test_get_document_returns_independent_metadata() -> None:
    """Mutating returned metadata should not change stored metadata."""

    store = InMemoryVectorStore(
        FixedEmbeddingService()
    )

    store.add_document(
        text="Revenue policy",
        metadata={"details": {"year": 2026}},
        document_id="doc-1",
    )

    first_result = store.get_document("doc-1")
    first_result.metadata["details"]["year"] = 2030

    second_result = store.get_document("doc-1")

    assert second_result.metadata["details"]["year"] == 2026


def test_get_missing_document() -> None:
    """Missing identifiers should raise DocumentNotFoundError."""

    store = InMemoryVectorStore()

    with pytest.raises(
        DocumentNotFoundError,
        match="was not found",
    ):
        store.get_document("missing")


def test_get_embedding_returns_copy() -> None:
    """Returned embeddings should not expose internal vectors."""

    embedding_service = FixedEmbeddingService(
        vectors={"Revenue policy": [0.2, 0.4, 0.6]}
    )
    store = InMemoryVectorStore(embedding_service)

    store.add_document(
        text="Revenue policy",
        document_id="doc-1",
    )

    first_vector = store.get_embedding("doc-1")
    first_vector[0] = 100.0

    second_vector = store.get_embedding("doc-1")

    assert second_vector == [0.2, 0.4, 0.6]


def test_get_embedding_missing_document() -> None:
    """Missing embedding identifiers should fail."""

    store = InMemoryVectorStore()

    with pytest.raises(DocumentNotFoundError):
        store.get_embedding("missing")


def test_upsert_adds_new_document() -> None:
    """Upsert should insert a document that does not exist."""

    store = InMemoryVectorStore(
        FixedEmbeddingService()
    )

    document = store.upsert_document(
        text="Initial policy",
        document_id="policy",
    )

    assert document.text == "Initial policy"
    assert len(store) == 1


def test_upsert_replaces_existing_document() -> None:
    """Upsert should replace an existing document and vector."""

    embedding_service = FixedEmbeddingService(
        vectors={
            "Old policy": [1.0, 0.0, 0.0],
            "New policy": [0.0, 1.0, 0.0],
        }
    )
    store = InMemoryVectorStore(embedding_service)

    store.add_document(
        text="Old policy",
        document_id="policy",
    )

    store.upsert_document(
        text="New policy",
        metadata={"version": 2},
        document_id="policy",
    )

    document = store.get_document("policy")

    assert len(store) == 1
    assert document.text == "New policy"
    assert document.metadata == {"version": 2}
    assert store.get_embedding("policy") == [0.0, 1.0, 0.0]


def test_delete_document() -> None:
    """Deleting a document should return and remove it."""

    store = InMemoryVectorStore(
        FixedEmbeddingService()
    )

    store.add_document(
        text="Revenue policy",
        document_id="doc-1",
    )

    deleted_document = store.delete_document("doc-1")

    assert deleted_document.id == "doc-1"
    assert len(store) == 0
    assert "doc-1" not in store


def test_delete_missing_document() -> None:
    """Deleting a missing document should fail."""

    store = InMemoryVectorStore()

    with pytest.raises(
        DocumentNotFoundError,
        match="was not found",
    ):
        store.delete_document("missing")


def test_clear_store() -> None:
    """Clear should remove all documents and return their count."""

    store = InMemoryVectorStore(
        FixedEmbeddingService()
    )

    store.add_documents(
        [
            {"id": "doc-1", "text": "Revenue policy"},
            {"id": "doc-2", "text": "Budget policy"},
            {"id": "doc-3", "text": "Forecast policy"},
        ]
    )

    removed_count = store.clear()

    assert removed_count == 3
    assert len(store) == 0


def test_clear_empty_store() -> None:
    """Clearing an empty store should return zero."""

    store = InMemoryVectorStore()

    assert store.clear() == 0


def test_list_documents() -> None:
    """All stored documents should be returned."""

    store = InMemoryVectorStore(
        FixedEmbeddingService()
    )

    store.add_documents(
        [
            {"id": "doc-1", "text": "Revenue policy"},
            {"id": "doc-2", "text": "Budget policy"},
        ]
    )

    documents = store.list_documents()

    assert [document.id for document in documents] == [
        "doc-1",
        "doc-2",
    ]


def test_contains_rejects_non_string_without_error() -> None:
    """Non-string membership checks should return False."""

    store = InMemoryVectorStore()

    assert 100 not in store
    assert None not in store


def test_similarity_search_orders_by_score() -> None:
    """Search results should be ranked from highest score."""

    vectors = {
        "revenue query": [1.0, 0.0, 0.0],
        "Revenue policy": [1.0, 0.0, 0.0],
        "Profit policy": [0.8, 0.2, 0.0],
        "Budget policy": [0.0, 1.0, 0.0],
    }

    store = InMemoryVectorStore(
        FixedEmbeddingService(vectors=vectors)
    )

    store.add_documents(
        [
            {"id": "revenue", "text": "Revenue policy"},
            {"id": "profit", "text": "Profit policy"},
            {"id": "budget", "text": "Budget policy"},
        ]
    )

    results = store.similarity_search(
        query="revenue query",
        top_k=3,
    )

    assert [result.document.id for result in results] == [
        "revenue",
        "profit",
        "budget",
    ]
    assert [result.rank for result in results] == [1, 2, 3]
    assert results[0].score == pytest.approx(1.0)
    assert results[0].score >= results[1].score
    assert results[1].score >= results[2].score


def test_similarity_search_limits_top_k() -> None:
    """Search should return no more than top_k results."""

    store = InMemoryVectorStore(
        FixedEmbeddingService()
    )

    store.add_documents(
        [
            {"id": "doc-1", "text": "First document"},
            {"id": "doc-2", "text": "Second document"},
            {"id": "doc-3", "text": "Third document"},
        ]
    )

    results = store.similarity_search(
        query="query",
        top_k=2,
    )

    assert len(results) == 2


def test_similarity_search_empty_store() -> None:
    """Searching an empty store should return an empty list."""

    store = InMemoryVectorStore(
        FixedEmbeddingService()
    )

    results = store.similarity_search(
        query="revenue query"
    )

    assert results == []


def test_similarity_search_uses_query_embedding() -> None:
    """Search should call the embedding query interface."""

    embedding_service = FixedEmbeddingService()
    store = InMemoryVectorStore(embedding_service)

    store.similarity_search("Revenue performance")

    assert embedding_service.embed_query_calls == [
        "Revenue performance"
    ]


def test_similarity_search_with_score_threshold() -> None:
    """Results below the minimum score should be excluded."""

    vectors = {
        "query": [1.0, 0.0, 0.0],
        "Relevant": [1.0, 0.0, 0.0],
        "Partly relevant": [0.5, 0.5, 0.0],
        "Unrelated": [0.0, 1.0, 0.0],
    }

    store = InMemoryVectorStore(
        FixedEmbeddingService(vectors=vectors)
    )

    store.add_documents(
        [
            {"id": "relevant", "text": "Relevant"},
            {"id": "partial", "text": "Partly relevant"},
            {"id": "unrelated", "text": "Unrelated"},
        ]
    )

    results = store.similarity_search(
        query="query",
        top_k=5,
        score_threshold=0.8,
    )

    assert [result.document.id for result in results] == [
        "relevant"
    ]


def test_similarity_search_with_metadata_filter() -> None:
    """Only matching metadata records should be searched."""

    vectors = {
        "policy query": [1.0, 0.0, 0.0],
        "Revenue policy": [1.0, 0.0, 0.0],
        "Budget policy": [1.0, 0.0, 0.0],
    }

    store = InMemoryVectorStore(
        FixedEmbeddingService(vectors=vectors)
    )

    store.add_documents(
        [
            {
                "id": "revenue",
                "text": "Revenue policy",
                "metadata": {
                    "category": "revenue",
                    "year": 2026,
                },
            },
            {
                "id": "budget",
                "text": "Budget policy",
                "metadata": {
                    "category": "budget",
                    "year": 2026,
                },
            },
        ]
    )

    results = store.similarity_search(
        query="policy query",
        metadata_filter={"category": "budget"},
    )

    assert len(results) == 1
    assert results[0].document.id == "budget"


def test_metadata_filter_requires_all_fields() -> None:
    """Every metadata filter field should match."""

    store = InMemoryVectorStore(
        FixedEmbeddingService()
    )

    store.add_documents(
        [
            {
                "id": "doc-1",
                "text": "Budget 2026",
                "metadata": {
                    "category": "budget",
                    "year": 2026,
                },
            },
            {
                "id": "doc-2",
                "text": "Budget 2025",
                "metadata": {
                    "category": "budget",
                    "year": 2025,
                },
            },
        ]
    )

    results = store.similarity_search(
        query="budget",
        metadata_filter={
            "category": "budget",
            "year": 2026,
        },
    )

    assert [result.document.id for result in results] == [
        "doc-1"
    ]


def test_similarity_search_deterministic_tie_break() -> None:
    """Equal scores should be ordered by document identifier."""

    vectors = {
        "query": [1.0, 0.0, 0.0],
        "Alpha": [1.0, 0.0, 0.0],
        "Beta": [1.0, 0.0, 0.0],
    }

    store = InMemoryVectorStore(
        FixedEmbeddingService(vectors=vectors)
    )

    store.add_documents(
        [
            {"id": "z-document", "text": "Alpha"},
            {"id": "a-document", "text": "Beta"},
        ]
    )

    results = store.similarity_search(
        query="query",
        top_k=2,
    )

    assert [result.document.id for result in results] == [
        "a-document",
        "z-document",
    ]


def test_similarity_search_rejects_empty_query() -> None:
    """Search queries cannot be empty."""

    store = InMemoryVectorStore()

    with pytest.raises(ValueError, match="text cannot be empty"):
        store.similarity_search("   ")


@pytest.mark.parametrize("top_k", [0, -1, -10])
def test_similarity_search_rejects_invalid_top_k(
    top_k: int,
) -> None:
    """top_k must be greater than zero."""

    store = InMemoryVectorStore()

    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        store.similarity_search(
            query="revenue",
            top_k=top_k,
        )


@pytest.mark.parametrize(
    "top_k",
    [1.5, "5", True, None],
)
def test_similarity_search_rejects_non_integer_top_k(
    top_k: object,
) -> None:
    """top_k must be an integer."""

    store = InMemoryVectorStore()

    with pytest.raises(TypeError, match="top_k must be an integer"):
        store.similarity_search(
            query="revenue",
            top_k=top_k,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "threshold",
    [-1.1, 1.1, -5.0, 5.0],
)
def test_similarity_search_rejects_invalid_threshold(
    threshold: float,
) -> None:
    """Score thresholds must remain within cosine bounds."""

    store = InMemoryVectorStore()

    with pytest.raises(
        ValueError,
        match="between -1.0 and 1.0",
    ):
        store.similarity_search(
            query="revenue",
            score_threshold=threshold,
        )


@pytest.mark.parametrize("threshold", ["0.5", True, []])
def test_similarity_search_rejects_non_numeric_threshold(
    threshold: object,
) -> None:
    """Score threshold must be numeric."""

    store = InMemoryVectorStore()

    with pytest.raises(
        TypeError,
        match="score_threshold must be numeric",
    ):
        store.similarity_search(
            query="revenue",
            score_threshold=threshold,  # type: ignore[arg-type]
        )


def test_similarity_search_rejects_invalid_metadata_filter() -> None:
    """Metadata filters must be mappings."""

    store = InMemoryVectorStore()

    with pytest.raises(
        TypeError,
        match="metadata_filter must be a mapping",
    ):
        store.similarity_search(
            query="revenue",
            metadata_filter=["budget"],  # type: ignore[arg-type]
        )


def test_similarity_search_wraps_query_embedding_failure() -> None:
    """Query embedding errors should become VectorStoreError."""

    store = InMemoryVectorStore(
        FailingEmbeddingService()
    )

    with pytest.raises(
        VectorStoreError,
        match="Unable to embed search query",
    ):
        store.similarity_search("revenue")


def test_similarity_search_by_vector() -> None:
    """Precomputed vectors should support similarity search."""

    vectors = {
        "Revenue policy": [1.0, 0.0, 0.0],
        "Budget policy": [0.0, 1.0, 0.0],
    }

    store = InMemoryVectorStore(
        FixedEmbeddingService(vectors=vectors)
    )

    store.add_documents(
        [
            {"id": "revenue", "text": "Revenue policy"},
            {"id": "budget", "text": "Budget policy"},
        ]
    )

    results = store.similarity_search_by_vector(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=2,
    )

    assert [result.document.id for result in results] == [
        "revenue",
        "budget",
    ]


def test_similarity_search_by_vector_applies_metadata_filter() -> None:
    """Vector search should support metadata filtering."""

    store = InMemoryVectorStore(
        FixedEmbeddingService()
    )

    store.add_documents(
        [
            {
                "id": "doc-1",
                "text": "Revenue policy",
                "metadata": {"type": "policy"},
            },
            {
                "id": "doc-2",
                "text": "Revenue assumption",
                "metadata": {"type": "assumption"},
            },
        ]
    )

    results = store.similarity_search_by_vector(
        query_embedding=[1.0, 0.0, 0.0],
        metadata_filter={"type": "assumption"},
    )

    assert [result.document.id for result in results] == [
        "doc-2"
    ]


def test_similarity_search_by_vector_rejects_wrong_dimension() -> None:
    """Query vectors must match the vector-store dimension."""

    store = InMemoryVectorStore(
        FixedEmbeddingService(dimension=3)
    )

    with pytest.raises(
        VectorStoreError,
        match="Embedding dimension mismatch",
    ):
        store.similarity_search_by_vector(
            query_embedding=[1.0, 0.0],
        )


def test_build_vector_store_empty() -> None:
    """Helper should create an empty vector store."""

    embedding_service = FixedEmbeddingService()

    store = build_vector_store(
        embedding_service=embedding_service
    )

    assert isinstance(store, InMemoryVectorStore)
    assert store.embedding_service is embedding_service
    assert len(store) == 0


def test_build_vector_store_with_documents() -> None:
    """Helper should populate initial documents."""

    store = build_vector_store(
        documents=[
            {
                "id": "doc-1",
                "text": "Revenue policy",
            },
            {
                "id": "doc-2",
                "text": "Budget policy",
            },
        ],
        embedding_service=FixedEmbeddingService(),
    )

    assert len(store) == 2
    assert "doc-1" in store
    assert "doc-2" in store


def test_documents_from_texts() -> None:
    """Text helper should create Document objects."""

    documents = documents_from_texts(
        texts=[
            "Revenue policy",
            "Budget policy",
        ],
        metadatas=[
            {"category": "revenue"},
            {"category": "budget"},
        ],
        document_ids=[
            "revenue-doc",
            "budget-doc",
        ],
    )

    assert documents == [
        Document(
            id="revenue-doc",
            text="Revenue policy",
            metadata={"category": "revenue"},
        ),
        Document(
            id="budget-doc",
            text="Budget policy",
            metadata={"category": "budget"},
        ),
    ]


def test_documents_from_texts_generates_ids() -> None:
    """Document identifiers should be generated when omitted."""

    documents = documents_from_texts(
        texts=[
            "Revenue policy",
            "Budget policy",
        ]
    )

    assert len(documents) == 2
    assert documents[0].id
    assert documents[1].id
    assert documents[0].id != documents[1].id


def test_documents_from_texts_rejects_single_string() -> None:
    """The texts argument must be a sequence of strings."""

    with pytest.raises(
        TypeError,
        match="not a single string",
    ):
        documents_from_texts(
            texts="Revenue policy"  # type: ignore[arg-type]
        )


def test_documents_from_texts_rejects_metadata_length_mismatch() -> None:
    """Metadata count must match the text count."""

    with pytest.raises(
        ValueError,
        match="same length as texts",
    ):
        documents_from_texts(
            texts=["Revenue policy", "Budget policy"],
            metadatas=[{"category": "revenue"}],
        )


def test_documents_from_texts_rejects_id_length_mismatch() -> None:
    """Identifier count must match the text count."""

    with pytest.raises(
        ValueError,
        match="same length as texts",
    ):
        documents_from_texts(
            texts=["Revenue policy", "Budget policy"],
            document_ids=["revenue-doc"],
        )


def test_documents_from_texts_rejects_non_string_text() -> None:
    """Every text value must be a string."""

    with pytest.raises(TypeError, match="text must be a string"):
        documents_from_texts(
            texts=[
                "Revenue policy",
                100,  # type: ignore[list-item]
            ]
        )


def test_documents_from_texts_deep_copies_metadata() -> None:
    """Created documents should not share external metadata."""

    metadata = {
        "details": {
            "year": 2026,
        }
    }

    documents = documents_from_texts(
        texts=["Budget policy"],
        metadatas=[metadata],
        document_ids=["budget-doc"],
    )

    metadata["details"]["year"] = 2030

    assert documents[0].metadata["details"]["year"] == 2026


def test_iter_document_batches() -> None:
    """Documents should be yielded in fixed-size batches."""

    documents = [
        Document(
            id=f"doc-{index}",
            text=f"Document {index}",
        )
        for index in range(5)
    ]

    batches = list(
        iter_document_batches(
            documents=documents,
            batch_size=2,
        )
    )

    assert len(batches) == 3
    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert batches[0][0].id == "doc-0"
    assert batches[2][0].id == "doc-4"


def test_iter_document_batches_empty_collection() -> None:
    """An empty collection should produce no batches."""

    batches = list(
        iter_document_batches(
            documents=[],
            batch_size=2,
        )
    )

    assert batches == []


@pytest.mark.parametrize("batch_size", [0, -1, -10])
def test_iter_document_batches_rejects_invalid_size(
    batch_size: int,
) -> None:
    """Batch size must be positive."""

    with pytest.raises(
        ValueError,
        match="batch_size must be greater than zero",
    ):
        list(
            iter_document_batches(
                documents=[],
                batch_size=batch_size,
            )
        )


@pytest.mark.parametrize("batch_size", [1.5, "2", True])
def test_iter_document_batches_rejects_non_integer_size(
    batch_size: object,
) -> None:
    """Batch size must be an integer."""

    with pytest.raises(
        TypeError,
        match="batch_size must be an integer",
    ):
        list(
            iter_document_batches(
                documents=[],
                batch_size=batch_size,  # type: ignore[arg-type]
            )
        )


def test_real_deterministic_embedding_search() -> None:
    """Vector store should work with the real local embedding provider."""

    embedding_service = DeterministicEmbeddingService(
        dimension=256
    )
    store = InMemoryVectorStore(embedding_service)

    store.add_documents(
        [
            {
                "id": "revenue-volume",
                "text": (
                    "Revenue increased because actual volume "
                    "exceeded budget volume."
                ),
            },
            {
                "id": "cost-fuel",
                "text": (
                    "Operating cost increased because fuel "
                    "expense exceeded forecast."
                ),
            },
            {
                "id": "headcount-policy",
                "text": (
                    "New employees require management approval "
                    "before joining."
                ),
            },
        ]
    )

    results = store.similarity_search(
        query="What caused the revenue volume variance?",
        top_k=3,
    )

    assert len(results) == 3
    assert results[0].document.id == "revenue-volume"
    assert all(
        math.isfinite(result.score)
        for result in results
    )