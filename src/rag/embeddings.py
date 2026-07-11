"""
Embedding services for the Finance Agentic AI System.

This module provides a common interface for converting text into numerical
vectors used by the RAG pipeline.

The default implementation is deterministic and works locally without an
API key. An optional OpenAI implementation can be enabled later without
changing the vector store or retriever interfaces.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


DEFAULT_LOCAL_DIMENSION = 384
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+(?:[.%/-][A-Za-z0-9_]+)*")


class EmbeddingError(RuntimeError):
    """Raised when text embedding generation fails."""


@dataclass(frozen=True)
class EmbeddingConfig:
    """
    Configuration for an embedding provider.

    Attributes:
        provider:
            Embedding provider name. Supported values are ``local`` and
            ``openai``.
        model:
            Provider-specific embedding model name.
        dimension:
            Expected embedding vector dimension.
        normalize:
            Whether generated vectors should be L2-normalized.
        api_key:
            Optional OpenAI API key. Environment-based configuration can also
            be used when this value is not supplied.
        batch_size:
            Maximum number of documents sent in one OpenAI request.
    """

    provider: str = "local"
    model: str | None = None
    dimension: int = DEFAULT_LOCAL_DIMENSION
    normalize: bool = True
    api_key: str | None = None
    batch_size: int = 100

    def __post_init__(self) -> None:
        provider = self.provider.strip().lower()

        if provider not in {"local", "openai"}:
            raise ValueError(
                "provider must be either 'local' or 'openai'."
            )

        if self.dimension <= 0:
            raise ValueError("dimension must be greater than zero.")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        object.__setattr__(self, "provider", provider)


class BaseEmbeddingService(ABC):
    """
    Abstract interface implemented by all embedding providers.

    Vector stores and retrievers should depend on this interface rather than
    directly depending on OpenAI or another external provider.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the size of every generated embedding vector."""

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Create an embedding for one text value."""

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """
        Create embeddings for multiple text values.

        Implementations may override this method to support provider-level
        batching.
        """

        validated_texts = _validate_text_collection(texts)
        return [self.embed_text(text) for text in validated_texts]

    def embed_query(self, query: str) -> list[float]:
        """
        Create an embedding for a search query.

        This alias follows common vector-store terminology.
        """

        return self.embed_text(query)

    def embed_documents(
        self,
        documents: Sequence[str],
    ) -> list[list[float]]:
        """
        Create embeddings for document text.

        This alias makes the service compatible with common RAG interfaces.
        """

        return self.embed_texts(documents)


class DeterministicEmbeddingService(BaseEmbeddingService):
    """
    Generate stable local embeddings without external dependencies.

    The implementation uses feature hashing over normalized tokens and token
    bigrams. It is intended for:

    - Unit tests
    - Local development
    - Offline execution
    - Deterministic RAG pipeline validation

    It is not intended to replace a production semantic embedding model.
    """

    def __init__(
        self,
        dimension: int = DEFAULT_LOCAL_DIMENSION,
        normalize: bool = True,
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be greater than zero.")

        self._dimension = dimension
        self._normalize = normalize

    @property
    def dimension(self) -> int:
        """Return the configured vector dimension."""

        return self._dimension

    def embed_text(self, text: str) -> list[float]:
        """
        Convert text into a deterministic numeric vector.

        The same text and configuration always produce the same vector.
        Empty or whitespace-only text produces a zero vector.
        """

        validated_text = _validate_text(text)
        tokens = self._tokenize(validated_text)

        vector = [0.0] * self._dimension

        if not tokens:
            return vector

        features = list(tokens)
        features.extend(
            f"{tokens[index]}::{tokens[index + 1]}"
            for index in range(len(tokens) - 1)
        )

        for feature in features:
            index, sign = self._feature_position(feature)
            vector[index] += sign

        if self._normalize:
            vector = _l2_normalize(vector)

        return vector

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Create deterministic embeddings for multiple texts."""

        validated_texts = _validate_text_collection(texts)
        return [self.embed_text(text) for text in validated_texts]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Normalize and tokenize text."""

        return [
            match.group(0).lower()
            for match in _TOKEN_PATTERN.finditer(text)
        ]

    def _feature_position(self, feature: str) -> tuple[int, float]:
        """
        Map a feature to a vector position and deterministic sign.

        A signed hash reduces systematic collision bias.
        """

        digest = hashlib.blake2b(
            feature.encode("utf-8"),
            digest_size=16,
        ).digest()

        index = int.from_bytes(digest[:8], byteorder="big") % self._dimension
        sign = 1.0 if digest[8] % 2 == 0 else -1.0

        return index, sign


class OpenAIEmbeddingService(BaseEmbeddingService):
    """
    Generate embeddings through the OpenAI embeddings API.

    The OpenAI package is imported only when this provider is instantiated.
    Therefore, local development and tests do not require the package or an
    API key.
    """

    def __init__(
        self,
        model: str = DEFAULT_OPENAI_MODEL,
        dimension: int | None = None,
        api_key: str | None = None,
        batch_size: int = 100,
        client: Any | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model cannot be empty.")

        if dimension is not None and dimension <= 0:
            raise ValueError("dimension must be greater than zero.")

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        self._model = model.strip()
        self._dimension = dimension
        self._batch_size = batch_size
        self._client = client or self._create_client(api_key)

    @property
    def dimension(self) -> int:
        """
        Return the configured embedding dimension.

        OpenAI dimensions must be configured or discovered after the first
        successful embedding response.
        """

        if self._dimension is None:
            raise EmbeddingError(
                "Embedding dimension is not known yet. Configure a dimension "
                "or generate at least one embedding first."
            )

        return self._dimension

    def embed_text(self, text: str) -> list[float]:
        """Create an OpenAI embedding for one text value."""

        embeddings = self.embed_texts([text])
        return embeddings[0]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Create OpenAI embeddings using batched API requests."""

        validated_texts = _validate_text_collection(texts)

        if not validated_texts:
            return []

        all_embeddings: list[list[float]] = []

        try:
            for batch in _batched(validated_texts, self._batch_size):
                request_arguments: dict[str, Any] = {
                    "model": self._model,
                    "input": batch,
                }

                if self._dimension is not None:
                    request_arguments["dimensions"] = self._dimension

                response = self._client.embeddings.create(
                    **request_arguments
                )

                ordered_data = sorted(
                    response.data,
                    key=lambda item: item.index,
                )

                batch_embeddings = [
                    [float(value) for value in item.embedding]
                    for item in ordered_data
                ]

                if len(batch_embeddings) != len(batch):
                    raise EmbeddingError(
                        "OpenAI returned an unexpected number of embeddings."
                    )

                self._validate_and_set_dimension(batch_embeddings)
                all_embeddings.extend(batch_embeddings)

        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(
                f"OpenAI embedding generation failed: {exc}"
            ) from exc

        return all_embeddings

    def _validate_and_set_dimension(
        self,
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Ensure every response vector has one consistent dimension."""

        if not embeddings:
            return

        response_dimension = len(embeddings[0])

        if response_dimension == 0:
            raise EmbeddingError("OpenAI returned an empty embedding vector.")

        for embedding in embeddings:
            if len(embedding) != response_dimension:
                raise EmbeddingError(
                    "OpenAI returned inconsistent embedding dimensions."
                )

        if self._dimension is None:
            self._dimension = response_dimension
        elif response_dimension != self._dimension:
            raise EmbeddingError(
                "OpenAI embedding dimension does not match the configured "
                f"dimension. Expected {self._dimension}, received "
                f"{response_dimension}."
            )

    @staticmethod
    def _create_client(api_key: str | None) -> Any:
        """Create an OpenAI client only when the provider is used."""

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise EmbeddingError(
                "The OpenAI package is not installed. Install it before "
                "using the OpenAI embedding provider."
            ) from exc

        try:
            return OpenAI(api_key=api_key) if api_key else OpenAI()
        except Exception as exc:
            raise EmbeddingError(
                f"Unable to initialize the OpenAI client: {exc}"
            ) from exc


def create_embedding_service(
    config: EmbeddingConfig | None = None,
) -> BaseEmbeddingService:
    """
    Create an embedding service from configuration.

    The local deterministic provider is the default so that the application
    remains usable without an API key.
    """

    resolved_config = config or EmbeddingConfig()

    if resolved_config.provider == "local":
        return DeterministicEmbeddingService(
            dimension=resolved_config.dimension,
            normalize=resolved_config.normalize,
        )

    return OpenAIEmbeddingService(
        model=resolved_config.model or DEFAULT_OPENAI_MODEL,
        dimension=resolved_config.dimension,
        api_key=resolved_config.api_key,
        batch_size=resolved_config.batch_size,
    )


def cosine_similarity(
    first_vector: Sequence[float],
    second_vector: Sequence[float],
) -> float:
    """
    Calculate cosine similarity between two vectors.

    Returns:
        A value between -1.0 and 1.0. If either vector is a zero vector,
        returns 0.0.
    """

    if not first_vector or not second_vector:
        raise ValueError("vectors cannot be empty.")

    if len(first_vector) != len(second_vector):
        raise ValueError(
            "vectors must have the same dimension."
        )

    first_norm = math.sqrt(
        sum(float(value) ** 2 for value in first_vector)
    )
    second_norm = math.sqrt(
        sum(float(value) ** 2 for value in second_vector)
    )

    if first_norm == 0.0 or second_norm == 0.0:
        return 0.0

    dot_product = sum(
        float(first) * float(second)
        for first, second in zip(first_vector, second_vector)
    )

    similarity = dot_product / (first_norm * second_norm)

    return max(-1.0, min(1.0, similarity))


def _validate_text(text: str) -> str:
    """Validate one embedding input."""

    if not isinstance(text, str):
        raise TypeError(
            f"text must be a string, received {type(text).__name__}."
        )

    return text.strip()


def _validate_text_collection(texts: Sequence[str]) -> list[str]:
    """Validate and copy a sequence of embedding inputs."""

    if isinstance(texts, (str, bytes)):
        raise TypeError(
            "texts must be a sequence of strings, not a single string."
        )

    if not isinstance(texts, Sequence):
        raise TypeError("texts must be a sequence of strings.")

    return [_validate_text(text) for text in texts]


def _l2_normalize(vector: Sequence[float]) -> list[float]:
    """Return an L2-normalized copy of a vector."""

    magnitude = math.sqrt(
        sum(float(value) ** 2 for value in vector)
    )

    if magnitude == 0.0:
        return [0.0 for _ in vector]

    return [float(value) / magnitude for value in vector]


def _batched(
    values: Sequence[str],
    batch_size: int,
) -> Iterable[list[str]]:
    """Yield fixed-size batches from a sequence."""

    for start_index in range(0, len(values), batch_size):
        yield list(values[start_index : start_index + batch_size])