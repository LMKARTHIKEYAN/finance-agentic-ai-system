"""
Tests for the RAG embedding services.

These tests verify the deterministic local embedding provider, embedding
configuration, cosine similarity helper, factory function, and mocked OpenAI
embedding provider.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from src.rag.embeddings import (
    DEFAULT_LOCAL_DIMENSION,
    DEFAULT_OPENAI_MODEL,
    BaseEmbeddingService,
    DeterministicEmbeddingService,
    EmbeddingConfig,
    EmbeddingError,
    OpenAIEmbeddingService,
    cosine_similarity,
    create_embedding_service,
)


class FakeEmbeddingsAPI:
    """Mock OpenAI embeddings API."""

    def __init__(
        self,
        vectors: list[list[float]] | None = None,
    ) -> None:
        self.vectors = vectors or []
        self.calls: list[dict[str, object]] = []

    def create(
        self,
        *,
        model: str,
        input: list[str],
        dimensions: int | None = None,
    ) -> SimpleNamespace:
        """Return a fake OpenAI embedding response."""

        call: dict[str, object] = {
            "model": model,
            "input": input,
        }

        if dimensions is not None:
            call["dimensions"] = dimensions

        self.calls.append(call)

        if self.vectors:
            selected_vectors = self.vectors[: len(input)]
        else:
            vector_dimension = dimensions or 3
            selected_vectors = [
                [float(index + 1)] * vector_dimension
                for index, _ in enumerate(input)
            ]

        response_data = [
            SimpleNamespace(
                index=index,
                embedding=vector,
            )
            for index, vector in enumerate(selected_vectors)
        ]

        return SimpleNamespace(data=response_data)


class FakeOpenAIClient:
    """Mock OpenAI client containing an embeddings API."""

    def __init__(
        self,
        vectors: list[list[float]] | None = None,
    ) -> None:
        self.embeddings = FakeEmbeddingsAPI(vectors=vectors)


class FailingEmbeddingsAPI:
    """Mock OpenAI API that raises an exception."""

    def create(self, **_: object) -> None:
        raise RuntimeError("Mock API failure")


class FailingOpenAIClient:
    """Mock OpenAI client that always fails."""

    def __init__(self) -> None:
        self.embeddings = FailingEmbeddingsAPI()


def vector_magnitude(vector: list[float]) -> float:
    """Calculate the Euclidean magnitude of a vector."""

    return math.sqrt(sum(value**2 for value in vector))


def test_default_embedding_config() -> None:
    """Default configuration should use local deterministic embeddings."""

    config = EmbeddingConfig()

    assert config.provider == "local"
    assert config.model is None
    assert config.dimension == DEFAULT_LOCAL_DIMENSION
    assert config.normalize is True
    assert config.api_key is None
    assert config.batch_size == 100


@pytest.mark.parametrize(
    ("provider", "expected_provider"),
    [
        ("local", "local"),
        ("LOCAL", "local"),
        (" openai ", "openai"),
        ("OpenAI", "openai"),
    ],
)
def test_embedding_config_normalizes_provider(
    provider: str,
    expected_provider: str,
) -> None:
    """Provider names should be normalized."""

    config = EmbeddingConfig(provider=provider)

    assert config.provider == expected_provider


def test_embedding_config_rejects_unknown_provider() -> None:
    """Unsupported embedding providers should fail."""

    with pytest.raises(
        ValueError,
        match="provider must be either",
    ):
        EmbeddingConfig(provider="unknown")


@pytest.mark.parametrize("dimension", [0, -1, -100])
def test_embedding_config_rejects_invalid_dimension(
    dimension: int,
) -> None:
    """Embedding dimensions must be positive."""

    with pytest.raises(
        ValueError,
        match="dimension must be greater than zero",
    ):
        EmbeddingConfig(dimension=dimension)


@pytest.mark.parametrize("batch_size", [0, -1, -50])
def test_embedding_config_rejects_invalid_batch_size(
    batch_size: int,
) -> None:
    """Batch sizes must be positive."""

    with pytest.raises(
        ValueError,
        match="batch_size must be greater than zero",
    ):
        EmbeddingConfig(batch_size=batch_size)


def test_local_embedding_service_is_base_service() -> None:
    """Local provider should implement the shared embedding interface."""

    service = DeterministicEmbeddingService()

    assert isinstance(service, BaseEmbeddingService)


def test_local_embedding_dimension() -> None:
    """Local embeddings should use the configured dimension."""

    service = DeterministicEmbeddingService(dimension=64)

    assert service.dimension == 64


@pytest.mark.parametrize("dimension", [0, -1, -384])
def test_local_embedding_rejects_invalid_dimension(
    dimension: int,
) -> None:
    """Local provider dimensions must be positive."""

    with pytest.raises(
        ValueError,
        match="dimension must be greater than zero",
    ):
        DeterministicEmbeddingService(dimension=dimension)


def test_local_embedding_has_correct_length() -> None:
    """One generated vector should have the configured length."""

    service = DeterministicEmbeddingService(dimension=32)

    vector = service.embed_text("Revenue increased above budget.")

    assert len(vector) == 32
    assert all(isinstance(value, float) for value in vector)


def test_local_embedding_is_deterministic() -> None:
    """The same text should always produce the same vector."""

    service = DeterministicEmbeddingService(dimension=64)

    first_vector = service.embed_text(
        "Actual revenue exceeded budget."
    )
    second_vector = service.embed_text(
        "Actual revenue exceeded budget."
    )

    assert first_vector == second_vector


def test_local_embedding_changes_for_different_text() -> None:
    """Different text should generally produce different vectors."""

    service = DeterministicEmbeddingService(dimension=128)

    revenue_vector = service.embed_text(
        "Revenue increased because of higher volume."
    )
    cost_vector = service.embed_text(
        "Operating cost increased because of fuel expense."
    )

    assert revenue_vector != cost_vector


def test_local_embedding_is_case_insensitive() -> None:
    """Token normalization should make casing irrelevant."""

    service = DeterministicEmbeddingService(dimension=64)

    lower_vector = service.embed_text("revenue variance")
    upper_vector = service.embed_text("REVENUE VARIANCE")

    assert lower_vector == upper_vector


def test_local_embedding_strips_surrounding_whitespace() -> None:
    """Leading and trailing whitespace should not affect embeddings."""

    service = DeterministicEmbeddingService(dimension=64)

    normal_vector = service.embed_text("budget variance")
    padded_vector = service.embed_text("   budget variance   ")

    assert normal_vector == padded_vector


def test_local_embedding_empty_text_returns_zero_vector() -> None:
    """Empty text should return a zero vector."""

    service = DeterministicEmbeddingService(dimension=16)

    vector = service.embed_text("")

    assert vector == [0.0] * 16


def test_local_embedding_whitespace_returns_zero_vector() -> None:
    """Whitespace-only input should return a zero vector."""

    service = DeterministicEmbeddingService(dimension=16)

    vector = service.embed_text("     ")

    assert vector == [0.0] * 16


def test_local_embedding_is_normalized_by_default() -> None:
    """Non-empty local vectors should have magnitude one."""

    service = DeterministicEmbeddingService(dimension=128)

    vector = service.embed_text(
        "Gross profit margin improved due to lower cost."
    )

    assert vector_magnitude(vector) == pytest.approx(1.0)


def test_local_embedding_can_disable_normalization() -> None:
    """Normalization should be optional."""

    service = DeterministicEmbeddingService(
        dimension=128,
        normalize=False,
    )

    vector = service.embed_text(
        "Gross profit margin improved due to lower cost."
    )

    assert vector_magnitude(vector) != pytest.approx(1.0)


def test_local_embed_texts_returns_multiple_vectors() -> None:
    """Batch embedding should return one vector per text."""

    service = DeterministicEmbeddingService(dimension=24)

    vectors = service.embed_texts(
        [
            "Revenue increased.",
            "Cost decreased.",
            "Profit improved.",
        ]
    )

    assert len(vectors) == 3
    assert all(len(vector) == 24 for vector in vectors)


def test_local_embed_texts_accepts_empty_sequence() -> None:
    """An empty document collection should return an empty list."""

    service = DeterministicEmbeddingService()

    assert service.embed_texts([]) == []


def test_embed_query_matches_embed_text() -> None:
    """Query embedding alias should behave like single-text embedding."""

    service = DeterministicEmbeddingService(dimension=32)

    query_vector = service.embed_query("Show KPI performance")
    text_vector = service.embed_text("Show KPI performance")

    assert query_vector == text_vector


def test_embed_documents_matches_embed_texts() -> None:
    """Document embedding alias should behave like batch embedding."""

    service = DeterministicEmbeddingService(dimension=32)
    documents = [
        "Budget assumptions",
        "Forecast methodology",
    ]

    assert service.embed_documents(documents) == service.embed_texts(
        documents
    )


@pytest.mark.parametrize(
    "invalid_text",
    [
        None,
        100,
        12.5,
        ["revenue"],
        {"text": "revenue"},
    ],
)
def test_embed_text_rejects_non_string_values(
    invalid_text: object,
) -> None:
    """Single embedding input must be a string."""

    service = DeterministicEmbeddingService()

    with pytest.raises(TypeError, match="text must be a string"):
        service.embed_text(invalid_text)  # type: ignore[arg-type]


def test_embed_texts_rejects_single_string() -> None:
    """Batch embedding must not treat one string as a sequence."""

    service = DeterministicEmbeddingService()

    with pytest.raises(
        TypeError,
        match="not a single string",
    ):
        service.embed_texts("revenue")  # type: ignore[arg-type]


def test_embed_texts_rejects_non_sequence() -> None:
    """Batch embedding input must be a sequence."""

    service = DeterministicEmbeddingService()

    with pytest.raises(
        TypeError,
        match="texts must be a sequence",
    ):
        service.embed_texts(100)  # type: ignore[arg-type]


def test_embed_texts_rejects_invalid_member() -> None:
    """Every item in a document collection must be a string."""

    service = DeterministicEmbeddingService()

    with pytest.raises(TypeError, match="text must be a string"):
        service.embed_texts(
            ["valid text", 100]  # type: ignore[list-item]
        )


def test_create_embedding_service_defaults_to_local() -> None:
    """Factory should create the deterministic provider by default."""

    service = create_embedding_service()

    assert isinstance(service, DeterministicEmbeddingService)
    assert service.dimension == DEFAULT_LOCAL_DIMENSION


def test_create_embedding_service_uses_local_config() -> None:
    """Factory should apply local provider configuration."""

    text = (
        "Revenue increased because actual volume exceeded budget."
    )

    unnormalized_service = create_embedding_service(
        EmbeddingConfig(
            provider="local",
            dimension=72,
            normalize=False,
        )
    )

    normalized_service = create_embedding_service(
        EmbeddingConfig(
            provider="local",
            dimension=72,
            normalize=True,
        )
    )

    unnormalized_vector = unnormalized_service.embed_text(text)
    normalized_vector = normalized_service.embed_text(text)

    assert isinstance(
        unnormalized_service,
        DeterministicEmbeddingService,
    )
    assert unnormalized_service.dimension == 72
    assert len(unnormalized_vector) == 72

    assert vector_magnitude(normalized_vector) == pytest.approx(1.0)
    assert vector_magnitude(unnormalized_vector) != pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors() -> None:
    """Orthogonal vectors should have zero similarity."""

    similarity = cosine_similarity(
        [1.0, 0.0],
        [0.0, 1.0],
    )

    assert similarity == pytest.approx(0.0)


def test_cosine_similarity_zero_vector_returns_zero() -> None:
    """Similarity involving a zero vector should return zero."""

    similarity = cosine_similarity(
        [0.0, 0.0],
        [1.0, 1.0],
    )

    assert similarity == 0.0


def test_cosine_similarity_rejects_empty_vector() -> None:
    """Empty vectors should not be accepted."""

    with pytest.raises(ValueError, match="vectors cannot be empty"):
        cosine_similarity([], [])


def test_cosine_similarity_rejects_different_dimensions() -> None:
    """Both vectors must have the same length."""

    with pytest.raises(
        ValueError,
        match="same dimension",
    ):
        cosine_similarity(
            [1.0, 2.0],
            [1.0, 2.0, 3.0],
        )


def test_related_finance_text_has_positive_similarity() -> None:
    """Finance text sharing terms should have positive similarity."""

    service = DeterministicEmbeddingService(dimension=256)

    first_vector = service.embed_text(
        "Revenue increased because actual volume exceeded budget."
    )
    second_vector = service.embed_text(
        "Revenue variance was driven by higher actual volume."
    )

    similarity = cosine_similarity(
        first_vector,
        second_vector,
    )

    assert similarity > 0.0


def test_openai_service_is_base_service() -> None:
    """OpenAI provider should implement the common interface."""

    client = FakeOpenAIClient()

    service = OpenAIEmbeddingService(
        model=DEFAULT_OPENAI_MODEL,
        dimension=3,
        client=client,
    )

    assert isinstance(service, BaseEmbeddingService)


def test_openai_service_returns_single_embedding() -> None:
    """OpenAI provider should return one vector for one text."""

    client = FakeOpenAIClient(
        vectors=[[0.1, 0.2, 0.3]]
    )

    service = OpenAIEmbeddingService(
        model="test-model",
        dimension=3,
        client=client,
    )

    vector = service.embed_text("Revenue variance")

    assert vector == [0.1, 0.2, 0.3]
    assert service.dimension == 3


def test_openai_service_returns_multiple_embeddings() -> None:
    """OpenAI provider should preserve response order."""

    client = FakeOpenAIClient(
        vectors=[
            [0.1, 0.2],
            [0.3, 0.4],
        ]
    )

    service = OpenAIEmbeddingService(
        model="test-model",
        dimension=2,
        client=client,
    )

    vectors = service.embed_texts(
        [
            "Revenue analysis",
            "Cost analysis",
        ]
    )

    assert vectors == [
        [0.1, 0.2],
        [0.3, 0.4],
    ]


def test_openai_service_sends_model_and_dimension() -> None:
    """OpenAI requests should include configured values."""

    client = FakeOpenAIClient(
        vectors=[[0.1, 0.2, 0.3]]
    )

    service = OpenAIEmbeddingService(
        model="finance-embedding-model",
        dimension=3,
        client=client,
    )

    service.embed_text("Forecast assumptions")

    assert client.embeddings.calls == [
        {
            "model": "finance-embedding-model",
            "input": ["Forecast assumptions"],
            "dimensions": 3,
        }
    ]


def test_openai_service_uses_batches() -> None:
    """Large collections should be divided into configured batches."""

    client = FakeOpenAIClient()

    service = OpenAIEmbeddingService(
        model="test-model",
        dimension=3,
        batch_size=2,
        client=client,
    )

    service.embed_texts(
        [
            "Document one",
            "Document two",
            "Document three",
            "Document four",
            "Document five",
        ]
    )

    assert len(client.embeddings.calls) == 3

    assert client.embeddings.calls[0]["input"] == [
        "Document one",
        "Document two",
    ]
    assert client.embeddings.calls[1]["input"] == [
        "Document three",
        "Document four",
    ]
    assert client.embeddings.calls[2]["input"] == [
        "Document five",
    ]


def test_openai_service_discovers_dimension() -> None:
    """Provider should detect vector size when none is configured."""

    client = FakeOpenAIClient(
        vectors=[[0.1, 0.2, 0.3, 0.4]]
    )

    service = OpenAIEmbeddingService(
        model="test-model",
        dimension=None,
        client=client,
    )

    vector = service.embed_text("KPI performance")

    assert vector == [0.1, 0.2, 0.3, 0.4]
    assert service.dimension == 4


def test_openai_dimension_unknown_before_first_call() -> None:
    """Unknown OpenAI dimensions should raise before generation."""

    client = FakeOpenAIClient()

    service = OpenAIEmbeddingService(
        model="test-model",
        dimension=None,
        client=client,
    )

    with pytest.raises(
        EmbeddingError,
        match="dimension is not known yet",
    ):
        _ = service.dimension


def test_openai_service_accepts_empty_sequence() -> None:
    """No API request should be made for an empty collection."""

    client = FakeOpenAIClient()

    service = OpenAIEmbeddingService(
        model="test-model",
        dimension=3,
        client=client,
    )

    result = service.embed_texts([])

    assert result == []
    assert client.embeddings.calls == []


@pytest.mark.parametrize("batch_size", [0, -1, -10])
def test_openai_service_rejects_invalid_batch_size(
    batch_size: int,
) -> None:
    """OpenAI batch size must be positive."""

    with pytest.raises(
        ValueError,
        match="batch_size must be greater than zero",
    ):
        OpenAIEmbeddingService(
            batch_size=batch_size,
            client=FakeOpenAIClient(),
        )


def test_openai_service_rejects_empty_model() -> None:
    """OpenAI model name cannot be blank."""

    with pytest.raises(ValueError, match="model cannot be empty"):
        OpenAIEmbeddingService(
            model="   ",
            client=FakeOpenAIClient(),
        )


def test_openai_service_rejects_invalid_dimension() -> None:
    """Configured OpenAI dimension must be positive."""

    with pytest.raises(
        ValueError,
        match="dimension must be greater than zero",
    ):
        OpenAIEmbeddingService(
            dimension=0,
            client=FakeOpenAIClient(),
        )


def test_openai_service_detects_dimension_mismatch() -> None:
    """Unexpected vector dimensions should raise an embedding error."""

    client = FakeOpenAIClient(
        vectors=[[0.1, 0.2]]
    )

    service = OpenAIEmbeddingService(
        model="test-model",
        dimension=3,
        client=client,
    )

    with pytest.raises(
        EmbeddingError,
        match="does not match the configured dimension",
    ):
        service.embed_text("Revenue")


def test_openai_service_detects_empty_embedding() -> None:
    """An empty OpenAI vector should raise an error."""

    client = FakeOpenAIClient(vectors=[[]])

    service = OpenAIEmbeddingService(
        model="test-model",
        dimension=None,
        client=client,
    )

    with pytest.raises(
        EmbeddingError,
        match="empty embedding vector",
    ):
        service.embed_text("Revenue")


def test_openai_service_wraps_api_failure() -> None:
    """Provider API errors should become EmbeddingError."""

    service = OpenAIEmbeddingService(
        model="test-model",
        dimension=3,
        client=FailingOpenAIClient(),
    )

    with pytest.raises(
        EmbeddingError,
        match="OpenAI embedding generation failed",
    ):
        service.embed_text("Revenue variance")