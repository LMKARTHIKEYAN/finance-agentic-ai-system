"""
In-memory vector store for the Finance Agentic AI System.

This module stores embedded finance documents and retrieves the most relevant
documents using cosine similarity.

The vector store contains no finance calculation logic. It is a reusable RAG
infrastructure component that depends on the embedding service abstraction
defined in ``src.rag.embeddings``.
"""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Iterable, Mapping, Sequence

from src.rag.embeddings import (
    BaseEmbeddingService,
    DeterministicEmbeddingService,
    cosine_similarity,
)


class VectorStoreError(RuntimeError):
    """Raised when a vector-store operation fails."""


class DocumentNotFoundError(VectorStoreError):
    """Raised when a requested document does not exist."""


@dataclass(frozen=True)
class Document:
    """
    Represents one text document stored in the vector database.

    Attributes:
        id:
            Unique document identifier.
        text:
            Original document text.
        metadata:
            Optional searchable document metadata.
    """

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str):
            raise TypeError("document id must be a string.")

        if not isinstance(self.text, str):
            raise TypeError("document text must be a string.")

        document_id = self.id.strip()
        document_text = self.text.strip()

        if not document_id:
            raise ValueError("document id cannot be empty.")

        if not document_text:
            raise ValueError("document text cannot be empty.")

        if not isinstance(self.metadata, dict):
            raise TypeError("document metadata must be a dictionary.")

        object.__setattr__(self, "id", document_id)
        object.__setattr__(self, "text", document_text)
        object.__setattr__(
            self,
            "metadata",
            copy.deepcopy(self.metadata),
        )


@dataclass(frozen=True)
class SearchResult:
    """
    Represents one vector-search result.

    Attributes:
        document:
            Matching stored document.
        score:
            Cosine-similarity score between query and document.
        rank:
            One-based position in the search results.
    """

    document: Document
    score: float
    rank: int

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or not isinstance(
            self.rank,
            int,
        ):
            raise TypeError("rank must be an integer.")

        if self.rank <= 0:
            raise ValueError("rank must be greater than zero.")

        if isinstance(self.score, bool) or not isinstance(
            self.score,
            (int, float),
        ):
            raise TypeError("search score must be numeric.")

        if not -1.0 <= float(self.score) <= 1.0:
            raise ValueError(
                "search score must be between -1.0 and 1.0."
            )


@dataclass(frozen=True)
class StoredVector:
    """
    Internal vector-store record.

    Attributes:
        document:
            Original document.
        embedding:
            Vector representation of the document text.
    """

    document: Document
    embedding: tuple[float, ...]


class InMemoryVectorStore:
    """
    Thread-safe in-memory vector store.

    The store is suitable for:

    - Unit testing
    - Local development
    - Small finance policy and assumption collections
    - Initial RAG integration

    A persistent vector database can later implement the same public
    behaviour without changing the retriever or RAG agent.
    """

    def __init__(
        self,
        embedding_service: BaseEmbeddingService | None = None,
    ) -> None:
        self._embedding_service = (
            embedding_service or DeterministicEmbeddingService()
        )
        self._records: dict[str, StoredVector] = {}
        self._lock = RLock()

    @property
    def embedding_service(self) -> BaseEmbeddingService:
        """Return the configured embedding provider."""

        return self._embedding_service

    @property
    def dimension(self) -> int:
        """Return the vector dimension used by the store."""

        return self._embedding_service.dimension

    def __len__(self) -> int:
        """Return the number of stored documents."""

        with self._lock:
            return len(self._records)

    def __contains__(self, document_id: object) -> bool:
        """Return whether a document identifier exists."""

        if not isinstance(document_id, str):
            return False

        cleaned_id = document_id.strip()

        if not cleaned_id:
            return False

        with self._lock:
            return cleaned_id in self._records

    def add_document(
        self,
        text: str,
        metadata: Mapping[str, Any] | None = None,
        document_id: str | None = None,
    ) -> Document:
        """
        Add one document to the vector store.

        Args:
            text:
                Document content.
            metadata:
                Optional document metadata.
            document_id:
                Optional identifier. A UUID is generated when omitted.

        Returns:
            The created document.

        Raises:
            ValueError:
                If text or identifier is empty.
            VectorStoreError:
                If the identifier already exists or embedding fails.
        """

        validated_text = self._validate_text(text)
        resolved_id = self._resolve_document_id(document_id)
        resolved_metadata = self._validate_metadata(metadata)

        with self._lock:
            if resolved_id in self._records:
                raise VectorStoreError(
                    f"Document '{resolved_id}' already exists."
                )

        document = Document(
            id=resolved_id,
            text=validated_text,
            metadata=resolved_metadata,
        )

        try:
            embedding = self._embedding_service.embed_text(
                document.text
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Unable to embed document '{resolved_id}': {exc}"
            ) from exc

        self._validate_embedding(embedding)

        record = StoredVector(
            document=document,
            embedding=tuple(
                float(value) for value in embedding
            ),
        )

        with self._lock:
            if resolved_id in self._records:
                raise VectorStoreError(
                    f"Document '{resolved_id}' already exists."
                )

            self._records[resolved_id] = record

        return self._copy_document(document)

    def add_documents(
        self,
        documents: Sequence[
            Document | str | Mapping[str, Any]
        ],
    ) -> list[Document]:
        """
        Add multiple documents atomically.

        Supported item formats:

        - ``Document``
        - Plain text string
        - Mapping containing ``text`` and optional ``id`` and ``metadata``

        If validation or embedding fails, no document from the batch is added.
        """

        if isinstance(documents, (str, bytes)):
            raise TypeError(
                "documents must be a sequence, not a single string."
            )

        if not isinstance(documents, Sequence):
            raise TypeError("documents must be a sequence.")

        if not documents:
            return []

        prepared_documents = [
            self._coerce_document(item)
            for item in documents
        ]

        document_ids = [
            document.id
            for document in prepared_documents
        ]

        if len(document_ids) != len(set(document_ids)):
            raise VectorStoreError(
                "Batch contains duplicate document identifiers."
            )

        with self._lock:
            existing_ids = [
                document_id
                for document_id in document_ids
                if document_id in self._records
            ]

        if existing_ids:
            raise VectorStoreError(
                "The following documents already exist: "
                + ", ".join(sorted(existing_ids))
            )

        try:
            embeddings = self._embedding_service.embed_texts(
                [
                    document.text
                    for document in prepared_documents
                ]
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Unable to embed document batch: {exc}"
            ) from exc

        if len(embeddings) != len(prepared_documents):
            raise VectorStoreError(
                "Embedding service returned an unexpected number "
                "of vectors."
            )

        prepared_records: dict[str, StoredVector] = {}

        for document, embedding in zip(
            prepared_documents,
            embeddings,
        ):
            self._validate_embedding(embedding)

            prepared_records[document.id] = StoredVector(
                document=document,
                embedding=tuple(
                    float(value) for value in embedding
                ),
            )

        with self._lock:
            newly_existing_ids = [
                document_id
                for document_id in document_ids
                if document_id in self._records
            ]

            if newly_existing_ids:
                raise VectorStoreError(
                    "The following documents already exist: "
                    + ", ".join(sorted(newly_existing_ids))
                )

            self._records.update(prepared_records)

        return [
            self._copy_document(document)
            for document in prepared_documents
        ]

    def upsert_document(
        self,
        text: str,
        metadata: Mapping[str, Any] | None = None,
        document_id: str | None = None,
    ) -> Document:
        """
        Insert a new document or replace an existing document.

        A generated identifier is used if ``document_id`` is omitted.
        """

        validated_text = self._validate_text(text)
        resolved_id = self._resolve_document_id(document_id)
        resolved_metadata = self._validate_metadata(metadata)

        document = Document(
            id=resolved_id,
            text=validated_text,
            metadata=resolved_metadata,
        )

        try:
            embedding = self._embedding_service.embed_text(
                document.text
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Unable to embed document '{resolved_id}': {exc}"
            ) from exc

        self._validate_embedding(embedding)

        record = StoredVector(
            document=document,
            embedding=tuple(
                float(value) for value in embedding
            ),
        )

        with self._lock:
            self._records[resolved_id] = record

        return self._copy_document(document)

    def get_document(self, document_id: str) -> Document:
        """
        Return one stored document.

        Raises:
            DocumentNotFoundError:
                If the document identifier does not exist.
        """

        resolved_id = self._validate_document_id(document_id)

        with self._lock:
            record = self._records.get(resolved_id)

        if record is None:
            raise DocumentNotFoundError(
                f"Document '{resolved_id}' was not found."
            )

        return self._copy_document(record.document)

    def get_embedding(
        self,
        document_id: str,
    ) -> list[float]:
        """
        Return a copy of a stored document vector.

        This method is mainly useful for diagnostics and testing.
        """

        resolved_id = self._validate_document_id(document_id)

        with self._lock:
            record = self._records.get(resolved_id)

        if record is None:
            raise DocumentNotFoundError(
                f"Document '{resolved_id}' was not found."
            )

        return list(record.embedding)

    def list_documents(self) -> list[Document]:
        """Return copies of all stored documents."""

        with self._lock:
            documents = [
                record.document
                for record in self._records.values()
            ]

        return [
            self._copy_document(document)
            for document in documents
        ]

    def delete_document(
        self,
        document_id: str,
    ) -> Document:
        """
        Remove and return one document.

        Raises:
            DocumentNotFoundError:
                If the document identifier does not exist.
        """

        resolved_id = self._validate_document_id(document_id)

        with self._lock:
            record = self._records.pop(resolved_id, None)

        if record is None:
            raise DocumentNotFoundError(
                f"Document '{resolved_id}' was not found."
            )

        return self._copy_document(record.document)

    def clear(self) -> int:
        """
        Remove all documents.

        Returns:
            Number of removed documents.
        """

        with self._lock:
            removed_count = len(self._records)
            self._records.clear()

        return removed_count

    def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float | None = None,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Find documents most similar to a query.

        Args:
            query:
                Search query text.
            top_k:
                Maximum number of results.
            score_threshold:
                Optional minimum cosine-similarity score.
            metadata_filter:
                Optional exact-match metadata filter.

        Returns:
            Ranked search results ordered from highest to lowest score.
        """

        validated_query = self._validate_text(query)
        self._validate_top_k(top_k)
        self._validate_score_threshold(score_threshold)

        resolved_filter = self._validate_metadata_filter(
            metadata_filter
        )

        try:
            query_embedding = (
                self._embedding_service.embed_query(
                    validated_query
                )
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Unable to embed search query: {exc}"
            ) from exc

        self._validate_embedding(query_embedding)

        return self._search_records(
            query_embedding=query_embedding,
            top_k=top_k,
            score_threshold=score_threshold,
            metadata_filter=resolved_filter,
        )

    def similarity_search_by_vector(
        self,
        query_embedding: Sequence[float],
        top_k: int = 5,
        score_threshold: float | None = None,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Find documents using an already-generated query vector.
        """

        self._validate_embedding(query_embedding)
        self._validate_top_k(top_k)
        self._validate_score_threshold(score_threshold)

        resolved_filter = self._validate_metadata_filter(
            metadata_filter
        )

        return self._search_records(
            query_embedding=query_embedding,
            top_k=top_k,
            score_threshold=score_threshold,
            metadata_filter=resolved_filter,
        )

    def _search_records(
        self,
        query_embedding: Sequence[float],
        top_k: int,
        score_threshold: float | None,
        metadata_filter: Mapping[str, Any],
    ) -> list[SearchResult]:
        """Search stored records using a query vector."""

        with self._lock:
            records = list(self._records.values())

        scored_documents: list[tuple[Document, float]] = []

        for record in records:
            if not self._metadata_matches(
                record.document.metadata,
                metadata_filter,
            ):
                continue

            score = cosine_similarity(
                query_embedding,
                record.embedding,
            )

            if (
                score_threshold is not None
                and score < float(score_threshold)
            ):
                continue

            scored_documents.append(
                (
                    record.document,
                    float(score),
                )
            )

        scored_documents.sort(
            key=lambda item: (
                -item[1],
                item[0].id,
            )
        )

        selected_documents = scored_documents[:top_k]

        return [
            SearchResult(
                document=self._copy_document(document),
                score=score,
                rank=index,
            )
            for index, (document, score) in enumerate(
                selected_documents,
                start=1,
            )
        ]

    @staticmethod
    def _copy_document(
        document: Document,
    ) -> Document:
        """
        Return an independent copy of a document.

        A deep copy of metadata prevents callers from modifying the
        vector store's internal document data.
        """

        return Document(
            id=document.id,
            text=document.text,
            metadata=copy.deepcopy(document.metadata),
        )

    def _coerce_document(
        self,
        item: Document | str | Mapping[str, Any],
    ) -> Document:
        """Convert supported document inputs into a Document."""

        if isinstance(item, Document):
            return self._copy_document(item)

        if isinstance(item, str):
            return Document(
                id=self._generate_document_id(),
                text=self._validate_text(item),
                metadata={},
            )

        if isinstance(item, Mapping):
            if "text" not in item:
                raise ValueError(
                    "document mapping must contain a 'text' field."
                )

            text = item["text"]
            document_id = item.get("id")
            metadata = item.get("metadata", {})

            validated_text = self._validate_text(text)
            resolved_id = self._resolve_document_id(
                document_id
            )
            resolved_metadata = self._validate_metadata(
                metadata
            )

            return Document(
                id=resolved_id,
                text=validated_text,
                metadata=resolved_metadata,
            )

        raise TypeError(
            "each document must be a Document, string, or mapping."
        )

    def _validate_embedding(
        self,
        embedding: Sequence[float],
    ) -> None:
        """Validate vector length and numeric values."""

        if isinstance(embedding, (str, bytes)):
            raise VectorStoreError(
                "Embedding must be a numeric sequence."
            )

        if not isinstance(embedding, Sequence):
            raise VectorStoreError(
                "Embedding must be a numeric sequence."
            )

        if len(embedding) == 0:
            raise VectorStoreError(
                "Embedding vector cannot be empty."
            )

        expected_dimension = self.dimension

        if len(embedding) != expected_dimension:
            raise VectorStoreError(
                "Embedding dimension mismatch. "
                f"Expected {expected_dimension}, "
                f"received {len(embedding)}."
            )

        for value in embedding:
            if isinstance(value, bool) or not isinstance(
                value,
                (int, float),
            ):
                raise VectorStoreError(
                    "Embedding values must be numeric."
                )

    @staticmethod
    def _validate_text(text: Any) -> str:
        """Validate document or query text."""

        if not isinstance(text, str):
            raise TypeError(
                f"text must be a string, "
                f"received {type(text).__name__}."
            )

        cleaned_text = text.strip()

        if not cleaned_text:
            raise ValueError("text cannot be empty.")

        return cleaned_text

    @staticmethod
    def _validate_document_id(
        document_id: Any,
    ) -> str:
        """Validate a document identifier."""

        if not isinstance(document_id, str):
            raise TypeError(
                "document id must be a string."
            )

        cleaned_id = document_id.strip()

        if not cleaned_id:
            raise ValueError(
                "document id cannot be empty."
            )

        return cleaned_id

    def _resolve_document_id(
        self,
        document_id: Any,
    ) -> str:
        """Return a validated or generated identifier."""

        if document_id is None:
            return self._generate_document_id()

        return self._validate_document_id(document_id)

    @staticmethod
    def _generate_document_id() -> str:
        """Generate a unique document identifier."""

        return str(uuid.uuid4())

    @staticmethod
    def _validate_metadata(
        metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Validate and copy document metadata."""

        if metadata is None:
            return {}

        if not isinstance(metadata, Mapping):
            raise TypeError(
                "metadata must be a mapping."
            )

        return copy.deepcopy(dict(metadata))

    @staticmethod
    def _validate_metadata_filter(
        metadata_filter: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Validate and copy metadata search filters."""

        if metadata_filter is None:
            return {}

        if not isinstance(metadata_filter, Mapping):
            raise TypeError(
                "metadata_filter must be a mapping."
            )

        return copy.deepcopy(dict(metadata_filter))

    @staticmethod
    def _metadata_matches(
        document_metadata: Mapping[str, Any],
        metadata_filter: Mapping[str, Any],
    ) -> bool:
        """Return whether metadata matches all filter fields."""

        return all(
            key in document_metadata
            and document_metadata[key] == expected_value
            for key, expected_value in metadata_filter.items()
        )

    @staticmethod
    def _validate_top_k(top_k: int) -> None:
        """Validate result-count configuration."""

        if isinstance(top_k, bool) or not isinstance(
            top_k,
            int,
        ):
            raise TypeError(
                "top_k must be an integer."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

    @staticmethod
    def _validate_score_threshold(
        score_threshold: float | None,
    ) -> None:
        """Validate cosine-similarity threshold."""

        if score_threshold is None:
            return

        if isinstance(
            score_threshold,
            bool,
        ) or not isinstance(
            score_threshold,
            (int, float),
        ):
            raise TypeError(
                "score_threshold must be numeric."
            )

        if not -1.0 <= float(score_threshold) <= 1.0:
            raise ValueError(
                "score_threshold must be between "
                "-1.0 and 1.0."
            )


def build_vector_store(
    documents: Sequence[
        Document | str | Mapping[str, Any]
    ]
    | None = None,
    embedding_service: BaseEmbeddingService | None = None,
) -> InMemoryVectorStore:
    """
    Create an in-memory vector store and optionally populate it.

    Args:
        documents:
            Optional initial document collection.
        embedding_service:
            Optional embedding provider.

    Returns:
        Configured in-memory vector store.
    """

    store = InMemoryVectorStore(
        embedding_service=embedding_service
    )

    if documents:
        store.add_documents(documents)

    return store


def documents_from_texts(
    texts: Sequence[str],
    metadatas: Sequence[
        Mapping[str, Any] | None
    ]
    | None = None,
    document_ids: Sequence[str | None] | None = None,
) -> list[Document]:
    """
    Convert text collections into Document objects.

    Metadata and identifier collections must have the same length as texts
    when supplied.
    """

    if isinstance(texts, (str, bytes)):
        raise TypeError(
            "texts must be a sequence, not a single string."
        )

    if not isinstance(texts, Sequence):
        raise TypeError(
            "texts must be a sequence."
        )

    text_count = len(texts)

    if (
        metadatas is not None
        and len(metadatas) != text_count
    ):
        raise ValueError(
            "metadatas must have the same length as texts."
        )

    if (
        document_ids is not None
        and len(document_ids) != text_count
    ):
        raise ValueError(
            "document_ids must have the same length as texts."
        )

    documents: list[Document] = []

    for index, text in enumerate(texts):
        metadata = (
            metadatas[index]
            if metadatas is not None
            else None
        )

        document_id = (
            document_ids[index]
            if document_ids is not None
            else None
        )

        if not isinstance(text, str):
            raise TypeError(
                f"text must be a string, "
                f"received {type(text).__name__}."
            )

        cleaned_text = text.strip()

        if not cleaned_text:
            raise ValueError(
                "document text cannot be empty."
            )

        if metadata is not None and not isinstance(
            metadata,
            Mapping,
        ):
            raise TypeError(
                "metadata must be a mapping."
            )

        if document_id is None:
            resolved_id = str(uuid.uuid4())
        else:
            if not isinstance(document_id, str):
                raise TypeError(
                    "document id must be a string."
                )

            resolved_id = document_id.strip()

            if not resolved_id:
                raise ValueError(
                    "document id cannot be empty."
                )

        documents.append(
            Document(
                id=resolved_id,
                text=cleaned_text,
                metadata=(
                    copy.deepcopy(dict(metadata))
                    if metadata is not None
                    else {}
                ),
            )
        )

    return documents


def iter_document_batches(
    documents: Sequence[Document],
    batch_size: int,
) -> Iterable[list[Document]]:
    """Yield fixed-size document batches."""

    if isinstance(batch_size, bool) or not isinstance(
        batch_size,
        int,
    ):
        raise TypeError(
            "batch_size must be an integer."
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    for start_index in range(
        0,
        len(documents),
        batch_size,
    ):
        yield list(
            documents[
                start_index : start_index + batch_size
            ]
        )