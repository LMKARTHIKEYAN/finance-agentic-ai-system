"""
Document retriever for the Finance Agentic AI System.

This module provides a typed retrieval layer over the vector store. It is
responsible for:

- Validating retrieval requests
- Applying similarity thresholds
- Applying metadata filters
- Returning structured retrieval results
- Building context text for the RAG agent
- Preventing duplicate retrieved documents

The retriever contains no finance calculation logic.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.rag.vector_store import (
    Document,
    InMemoryVectorStore,
    SearchResult,
    VectorStoreError,
)


DEFAULT_TOP_K = 5
DEFAULT_MAX_CONTEXT_CHARACTERS = 12_000


class RetrieverError(RuntimeError):
    """Raised when document retrieval fails."""


@dataclass(frozen=True)
class RetrievalConfig:
    """
    Configuration used by the document retriever.

    Attributes:
        top_k:
            Maximum number of documents returned.
        score_threshold:
            Optional minimum cosine-similarity score.
        max_context_characters:
            Maximum number of characters allowed in generated context.
        include_metadata:
            Whether document metadata should be included in context.
        include_scores:
            Whether similarity scores should be included in context.
        deduplicate:
            Whether duplicate document text should be removed.
    """

    top_k: int = DEFAULT_TOP_K
    score_threshold: float | None = None
    max_context_characters: int = DEFAULT_MAX_CONTEXT_CHARACTERS
    include_metadata: bool = True
    include_scores: bool = True
    deduplicate: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.top_k, bool) or not isinstance(
            self.top_k,
            int,
        ):
            raise TypeError("top_k must be an integer.")

        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if self.score_threshold is not None:
            if isinstance(
                self.score_threshold,
                bool,
            ) or not isinstance(
                self.score_threshold,
                (int, float),
            ):
                raise TypeError(
                    "score_threshold must be numeric."
                )

            if not -1.0 <= float(self.score_threshold) <= 1.0:
                raise ValueError(
                    "score_threshold must be between "
                    "-1.0 and 1.0."
                )

        if isinstance(
            self.max_context_characters,
            bool,
        ) or not isinstance(
            self.max_context_characters,
            int,
        ):
            raise TypeError(
                "max_context_characters must be an integer."
            )

        if self.max_context_characters <= 0:
            raise ValueError(
                "max_context_characters must be greater than zero."
            )

        if not isinstance(self.include_metadata, bool):
            raise TypeError(
                "include_metadata must be a boolean."
            )

        if not isinstance(self.include_scores, bool):
            raise TypeError(
                "include_scores must be a boolean."
            )

        if not isinstance(self.deduplicate, bool):
            raise TypeError(
                "deduplicate must be a boolean."
            )


@dataclass(frozen=True)
class RetrievedDocument:
    """
    One document returned by the retriever.

    Attributes:
        document:
            Retrieved source document.
        score:
            Similarity score returned by the vector store.
        rank:
            One-based rank.
    """

    document: Document
    score: float
    rank: int

    def __post_init__(self) -> None:
        if not isinstance(self.document, Document):
            raise TypeError(
                "document must be a Document instance."
            )

        if isinstance(self.score, bool) or not isinstance(
            self.score,
            (int, float),
        ):
            raise TypeError("score must be numeric.")

        if not -1.0 <= float(self.score) <= 1.0:
            raise ValueError(
                "score must be between -1.0 and 1.0."
            )

        if isinstance(self.rank, bool) or not isinstance(
            self.rank,
            int,
        ):
            raise TypeError("rank must be an integer.")

        if self.rank <= 0:
            raise ValueError(
                "rank must be greater than zero."
            )

        object.__setattr__(
            self,
            "document",
            Document(
                id=self.document.id,
                text=self.document.text,
                metadata=copy.deepcopy(
                    self.document.metadata
                ),
            ),
        )


@dataclass(frozen=True)
class RetrievalResult:
    """
    Complete result of one retrieval operation.

    Attributes:
        query:
            Original validated query.
        documents:
            Ranked retrieved documents.
        context:
            Formatted context suitable for a RAG prompt.
        metadata_filter:
            Metadata filter used in the search.
        total_results:
            Number of documents returned.
    """

    query: str
    documents: tuple[RetrievedDocument, ...]
    context: str
    metadata_filter: dict[str, Any] = field(
        default_factory=dict
    )
    total_results: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.query, str):
            raise TypeError("query must be a string.")

        cleaned_query = self.query.strip()

        if not cleaned_query:
            raise ValueError("query cannot be empty.")

        if not isinstance(self.documents, tuple):
            raise TypeError(
                "documents must be a tuple."
            )

        if not all(
            isinstance(item, RetrievedDocument)
            for item in self.documents
        ):
            raise TypeError(
                "documents must contain RetrievedDocument values."
            )

        if not isinstance(self.context, str):
            raise TypeError("context must be a string.")

        if not isinstance(self.metadata_filter, dict):
            raise TypeError(
                "metadata_filter must be a dictionary."
            )

        if isinstance(self.total_results, bool) or not isinstance(
            self.total_results,
            int,
        ):
            raise TypeError(
                "total_results must be an integer."
            )

        if self.total_results < 0:
            raise ValueError(
                "total_results cannot be negative."
            )

        if self.total_results != len(self.documents):
            raise ValueError(
                "total_results must match the number of documents."
            )

        object.__setattr__(self, "query", cleaned_query)
        object.__setattr__(
            self,
            "metadata_filter",
            copy.deepcopy(self.metadata_filter),
        )

    @property
    def has_results(self) -> bool:
        """Return whether at least one document was retrieved."""

        return self.total_results > 0

    @property
    def source_documents(self) -> list[Document]:
        """Return independent copies of source documents."""

        return [
            Document(
                id=item.document.id,
                text=item.document.text,
                metadata=copy.deepcopy(
                    item.document.metadata
                ),
            )
            for item in self.documents
        ]


class FinanceRetriever:
    """
    Retrieve relevant finance documents from a vector store.

    The retriever wraps the vector store and converts raw similarity-search
    results into RAG-ready structured output.
    """

    def __init__(
        self,
        vector_store: InMemoryVectorStore,
        config: RetrievalConfig | None = None,
    ) -> None:
        if not isinstance(
            vector_store,
            InMemoryVectorStore,
        ):
            raise TypeError(
                "vector_store must be an "
                "InMemoryVectorStore instance."
            )

        self._vector_store = vector_store
        self._config = config or RetrievalConfig()

    @property
    def vector_store(self) -> InMemoryVectorStore:
        """Return the configured vector store."""

        return self._vector_store

    @property
    def config(self) -> RetrievalConfig:
        """Return the retrieval configuration."""

        return self._config

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        score_threshold: float | None = None,
        metadata_filter: Mapping[str, Any] | None = None,
        max_context_characters: int | None = None,
    ) -> RetrievalResult:
        """
        Retrieve relevant documents and build RAG context.

        Args:
            query:
                User question or retrieval query.
            top_k:
                Optional per-request override.
            score_threshold:
                Optional per-request score threshold.
            metadata_filter:
                Optional exact-match metadata filter.
            max_context_characters:
                Optional per-request context-size limit.

        Returns:
            Structured retrieval result.
        """

        validated_query = self._validate_query(query)

        resolved_top_k = (
            self._config.top_k
            if top_k is None
            else self._validate_top_k(top_k)
        )

        resolved_threshold = (
            self._config.score_threshold
            if score_threshold is None
            else self._validate_score_threshold(
                score_threshold
            )
        )

        resolved_filter = self._validate_metadata_filter(
            metadata_filter
        )

        resolved_context_limit = (
            self._config.max_context_characters
            if max_context_characters is None
            else self._validate_context_limit(
                max_context_characters
            )
        )

        try:
            search_results = (
                self._vector_store.similarity_search(
                    query=validated_query,
                    top_k=resolved_top_k,
                    score_threshold=resolved_threshold,
                    metadata_filter=resolved_filter,
                )
            )
        except VectorStoreError as exc:
            raise RetrieverError(
                f"Document retrieval failed: {exc}"
            ) from exc
        except Exception as exc:
            raise RetrieverError(
                f"Unexpected retrieval failure: {exc}"
            ) from exc

        retrieved_documents = self._convert_results(
            search_results
        )

        if self._config.deduplicate:
            retrieved_documents = self._deduplicate_documents(
                retrieved_documents
            )

        reranked_documents = self._rerank_documents(
            retrieved_documents
        )

        context = self.build_context(
            reranked_documents,
            max_characters=resolved_context_limit,
        )

        return RetrievalResult(
            query=validated_query,
            documents=tuple(reranked_documents),
            context=context,
            metadata_filter=resolved_filter,
            total_results=len(reranked_documents),
        )

    def retrieve_documents(
        self,
        query: str,
        *,
        top_k: int | None = None,
        score_threshold: float | None = None,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> list[Document]:
        """
        Retrieve only the source documents.

        This is a convenience method for callers that do not require scores
        or formatted prompt context.
        """

        result = self.retrieve(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
            metadata_filter=metadata_filter,
        )

        return result.source_documents

    def build_context(
        self,
        documents: Sequence[RetrievedDocument],
        *,
        max_characters: int | None = None,
    ) -> str:
        """
        Build prompt-ready context from retrieved documents.

        Context is truncated by whole document blocks where possible. If the
        first document alone exceeds the limit, that block is safely truncated.
        """

        if isinstance(documents, (str, bytes)):
            raise TypeError(
                "documents must be a sequence of "
                "RetrievedDocument values."
            )

        if not isinstance(documents, Sequence):
            raise TypeError(
                "documents must be a sequence."
            )

        if not all(
            isinstance(item, RetrievedDocument)
            for item in documents
        ):
            raise TypeError(
                "documents must contain RetrievedDocument values."
            )

        context_limit = (
            self._config.max_context_characters
            if max_characters is None
            else self._validate_context_limit(
                max_characters
            )
        )

        if not documents:
            return ""

        context_blocks: list[str] = []
        current_length = 0

        for item in documents:
            block = self._format_context_block(item)

            separator_length = 2 if context_blocks else 0
            proposed_length = (
                current_length
                + separator_length
                + len(block)
            )

            if proposed_length <= context_limit:
                context_blocks.append(block)
                current_length = proposed_length
                continue

            if not context_blocks:
                context_blocks.append(
                    self._truncate_text(
                        block,
                        context_limit,
                    )
                )

            break

        return "\n\n".join(context_blocks)

    def _convert_results(
        self,
        search_results: Sequence[SearchResult],
    ) -> list[RetrievedDocument]:
        """Convert vector-store results into retriever results."""

        converted_results: list[RetrievedDocument] = []

        for result in search_results:
            converted_results.append(
                RetrievedDocument(
                    document=Document(
                        id=result.document.id,
                        text=result.document.text,
                        metadata=copy.deepcopy(
                            result.document.metadata
                        ),
                    ),
                    score=float(result.score),
                    rank=result.rank,
                )
            )

        return converted_results

    @staticmethod
    def _deduplicate_documents(
        documents: Sequence[RetrievedDocument],
    ) -> list[RetrievedDocument]:
        """
        Remove duplicate documents by normalized text.

        The highest-ranked occurrence is retained.
        """

        seen_texts: set[str] = set()
        unique_documents: list[RetrievedDocument] = []

        for item in documents:
            normalized_text = " ".join(
                item.document.text.lower().split()
            )

            if normalized_text in seen_texts:
                continue

            seen_texts.add(normalized_text)
            unique_documents.append(item)

        return unique_documents

    @staticmethod
    def _rerank_documents(
        documents: Sequence[RetrievedDocument],
    ) -> list[RetrievedDocument]:
        """Assign continuous one-based ranks after filtering."""

        return [
            RetrievedDocument(
                document=item.document,
                score=item.score,
                rank=index,
            )
            for index, item in enumerate(
                documents,
                start=1,
            )
        ]

    def _format_context_block(
        self,
        item: RetrievedDocument,
    ) -> str:
        """Format one retrieved document for a prompt."""

        header_parts = [
            f"Source {item.rank}",
            f"Document ID: {item.document.id}",
        ]

        if self._config.include_scores:
            header_parts.append(
                f"Similarity Score: {item.score:.6f}"
            )

        if (
            self._config.include_metadata
            and item.document.metadata
        ):
            metadata_text = self._format_metadata(
                item.document.metadata
            )
            header_parts.append(
                f"Metadata: {metadata_text}"
            )

        header = "\n".join(header_parts)

        return (
            f"{header}\n"
            f"Content:\n"
            f"{item.document.text}"
        )

    @staticmethod
    def _format_metadata(
        metadata: Mapping[str, Any],
    ) -> str:
        """Create deterministic metadata text."""

        sorted_items = sorted(
            metadata.items(),
            key=lambda item: str(item[0]),
        )

        return ", ".join(
            f"{key}={value}"
            for key, value in sorted_items
        )

    @staticmethod
    def _truncate_text(
        text: str,
        maximum_characters: int,
    ) -> str:
        """Truncate text without exceeding the character limit."""

        if len(text) <= maximum_characters:
            return text

        if maximum_characters <= 3:
            return text[:maximum_characters]

        return text[: maximum_characters - 3].rstrip() + "..."

    @staticmethod
    def _validate_query(query: Any) -> str:
        """Validate retrieval query."""

        if not isinstance(query, str):
            raise TypeError(
                f"query must be a string, "
                f"received {type(query).__name__}."
            )

        cleaned_query = query.strip()

        if not cleaned_query:
            raise ValueError("query cannot be empty.")

        return cleaned_query

    @staticmethod
    def _validate_top_k(top_k: Any) -> int:
        """Validate top-k override."""

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

        return top_k

    @staticmethod
    def _validate_score_threshold(
        score_threshold: Any,
    ) -> float:
        """Validate score-threshold override."""

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

        resolved_threshold = float(score_threshold)

        if not -1.0 <= resolved_threshold <= 1.0:
            raise ValueError(
                "score_threshold must be between "
                "-1.0 and 1.0."
            )

        return resolved_threshold

    @staticmethod
    def _validate_context_limit(
        maximum_characters: Any,
    ) -> int:
        """Validate context-size limit."""

        if isinstance(
            maximum_characters,
            bool,
        ) or not isinstance(
            maximum_characters,
            int,
        ):
            raise TypeError(
                "max_context_characters must be an integer."
            )

        if maximum_characters <= 0:
            raise ValueError(
                "max_context_characters must be greater than zero."
            )

        return maximum_characters

    @staticmethod
    def _validate_metadata_filter(
        metadata_filter: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Validate and copy metadata filters."""

        if metadata_filter is None:
            return {}

        if not isinstance(metadata_filter, Mapping):
            raise TypeError(
                "metadata_filter must be a mapping."
            )

        return copy.deepcopy(
            dict(metadata_filter)
        )


def create_retriever(
    vector_store: InMemoryVectorStore,
    config: RetrievalConfig | None = None,
) -> FinanceRetriever:
    """
    Create a finance document retriever.

    Args:
        vector_store:
            Populated vector store.
        config:
            Optional retrieval configuration.

    Returns:
        Configured FinanceRetriever.
    """

    return FinanceRetriever(
        vector_store=vector_store,
        config=config,
    )


def format_retrieval_context(
    documents: Sequence[RetrievedDocument],
    *,
    max_characters: int = DEFAULT_MAX_CONTEXT_CHARACTERS,
    include_metadata: bool = True,
    include_scores: bool = True,
) -> str:
    """
    Format retrieved documents without manually creating a retriever.

    A temporary empty vector store is used only to reuse the context-formatting
    rules. No embedding or search operation is performed.
    """

    config = RetrievalConfig(
        max_context_characters=max_characters,
        include_metadata=include_metadata,
        include_scores=include_scores,
    )

    retriever = FinanceRetriever(
        vector_store=InMemoryVectorStore(),
        config=config,
    )

    return retriever.build_context(
        documents,
        max_characters=max_characters,
    )