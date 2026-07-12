"""
Tests for the PostgreSQL + pgvector vector store.

The unit tests validate:

* PGVectorConfig
* Optional dependency handling
* Store construction and properties
* Text, metadata, ID, embedding, and search validation
* Document coercion
* Independent document copies
* PostgreSQL row conversion

The integration tests validate actual PostgreSQL behaviour only when the
environment variable ``TEST_PGVECTOR_DSN`` is configured.

Example integration-test DSN:

    postgresql://postgres:password@localhost:5432/finance_agentic_ai

The PostgreSQL database must have permission to create the ``vector``
extension, schema, table, and test index when those options are enabled.
"""

from __future__ import annotations

import copy
import os
import uuid
from collections.abc import Mapping
from typing import Any

import pytest

from collections.abc import Generator

from src.rag.embeddings import (
    BaseEmbeddingService,
    DeterministicEmbeddingService,
)
from src.rag.pgvector_store import (
    PGVectorConfig,
    PGVectorConfigurationError,
    PGVectorDependencyError,
    PGVectorStore,
)
from src.rag.vector_store import (
    Document,
    DocumentNotFoundError,
    SearchResult,
    VectorStoreError,
)


TEST_DSN_ENVIRONMENT_VARIABLE = "TEST_PGVECTOR_DSN"


class FakeVector:
    """Minimal replacement for pgvector.Vector in unit tests."""

    def __init__(self, values: list[float]) -> None:
        self.values = list(values)

    def __iter__(self):
        return iter(self.values)

    def to_list(self) -> list[float]:
        """Return a normal Python list."""

        return list(self.values)


class FakePsycopg:
    """Placeholder Psycopg module used for non-database unit tests."""

    @staticmethod
    def connect(*args: Any, **kwargs: Any) -> None:
        """Fail when a unit test unexpectedly performs database access."""

        raise AssertionError(
            "Unit test unexpectedly attempted a database connection."
        )


class FakeSQLModule:
    """
    Placeholder SQL module.

    Pure validation tests do not compose or execute SQL, so these methods
    deliberately fail if unexpected database behaviour occurs.
    """

    @staticmethod
    def SQL(value: str) -> str:
        """Return the supplied SQL text for simple property-level tests."""

        return value

    @staticmethod
    def Identifier(*parts: str) -> tuple[str, ...]:
        """Represent an SQL identifier without a real Psycopg dependency."""

        return tuple(parts)


def fake_register_vector(connection: Any) -> None:
    """Placeholder pgvector registration function."""

    del connection


@pytest.fixture
def patched_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replace optional PostgreSQL imports with lightweight test doubles."""

    monkeypatch.setattr(
        PGVectorStore,
        "_load_dependencies",
        staticmethod(
            lambda: (
                FakePsycopg,
                FakeSQLModule,
                FakeVector,
                fake_register_vector,
            )
        ),
    )


@pytest.fixture
def config() -> PGVectorConfig:
    """Return a valid test configuration."""

    return PGVectorConfig(
        dsn=(
            "postgresql://postgres:password@localhost:5432/"
            "finance_agentic_ai"
        ),
        schema_name="rag_test",
        table_name="document_embeddings",
        create_extension=False,
        create_schema=False,
        create_table=False,
        create_hnsw_index=False,
        connect_timeout=5,
    )


@pytest.fixture
def store(
    patched_dependencies: None,
    config: PGVectorConfig,
) -> PGVectorStore:
    """Create a pgvector store without initializing a real database."""

    del patched_dependencies

    return PGVectorStore(
        config=config,
        embedding_service=DeterministicEmbeddingService(
            dimension=16,
        ),
        initialize=False,
    )


def test_pgvector_config_creates_valid_instance() -> None:
    """Configuration should preserve valid PostgreSQL settings."""

    config = PGVectorConfig(
        dsn="  postgresql://localhost/finance_ai  ",
        schema_name="rag",
        table_name="document_embeddings",
        create_extension=True,
        create_schema=True,
        create_table=True,
        create_hnsw_index=False,
        connect_timeout=12,
    )

    assert config.dsn == "postgresql://localhost/finance_ai"
    assert config.schema_name == "rag"
    assert config.table_name == "document_embeddings"
    assert config.create_extension is True
    assert config.create_schema is True
    assert config.create_table is True
    assert config.create_hnsw_index is False
    assert config.connect_timeout == 12


def test_pgvector_config_uses_defaults() -> None:
    """Configuration should expose the documented defaults."""

    config = PGVectorConfig(
        dsn="postgresql://localhost/finance_ai"
    )

    assert config.schema_name == "rag"
    assert config.table_name == "document_embeddings"
    assert config.create_extension is True
    assert config.create_schema is True
    assert config.create_table is True
    assert config.create_hnsw_index is False
    assert config.connect_timeout == 10


def test_pgvector_config_rejects_non_string_dsn() -> None:
    """DSN must be a string."""

    with pytest.raises(
        TypeError,
        match="dsn must be a string",
    ):
        PGVectorConfig(
            dsn=123,  # type: ignore[arg-type]
        )


def test_pgvector_config_rejects_empty_dsn() -> None:
    """DSN cannot contain only whitespace."""

    with pytest.raises(
        ValueError,
        match="dsn cannot be empty",
    ):
        PGVectorConfig(
            dsn="   ",
        )


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("schema_name", ""),
        ("schema_name", "123rag"),
        ("schema_name", "rag-schema"),
        ("schema_name", "rag schema"),
        ("table_name", ""),
        ("table_name", "123documents"),
        ("table_name", "document-embeddings"),
        ("table_name", "document embeddings"),
    ],
)
def test_pgvector_config_rejects_invalid_identifiers(
    field_name: str,
    field_value: str,
) -> None:
    """Schema and table names must be safe PostgreSQL identifiers."""

    arguments: dict[str, Any] = {
        "dsn": "postgresql://localhost/finance_ai",
        field_name: field_value,
    }

    with pytest.raises(ValueError):
        PGVectorConfig(**arguments)


@pytest.mark.parametrize(
    "identifier",
    [
        "rag",
        "_rag",
        "rag_2026",
        "document_embeddings",
        "DOCUMENTS",
    ],
)
def test_pgvector_config_accepts_valid_identifiers(
    identifier: str,
) -> None:
    """Normal PostgreSQL-style identifiers should be accepted."""

    config = PGVectorConfig(
        dsn="postgresql://localhost/finance_ai",
        schema_name=identifier,
        table_name=identifier,
    )

    assert config.schema_name == identifier
    assert config.table_name == identifier


@pytest.mark.parametrize(
    "field_name",
    [
        "create_extension",
        "create_schema",
        "create_table",
        "create_hnsw_index",
    ],
)
def test_pgvector_config_rejects_non_boolean_flags(
    field_name: str,
) -> None:
    """DDL configuration flags must be booleans."""

    arguments: dict[str, Any] = {
        "dsn": "postgresql://localhost/finance_ai",
        field_name: "yes",
    }

    with pytest.raises(
        TypeError,
        match=f"{field_name} must be a boolean",
    ):
        PGVectorConfig(**arguments)


@pytest.mark.parametrize(
    "connect_timeout",
    [
        True,
        1.5,
        "10",
        None,
    ],
)
def test_pgvector_config_rejects_invalid_timeout_type(
    connect_timeout: Any,
) -> None:
    """Connection timeout must be an integer."""

    with pytest.raises(
        TypeError,
        match="connect_timeout must be an integer",
    ):
        PGVectorConfig(
            dsn="postgresql://localhost/finance_ai",
            connect_timeout=connect_timeout,
        )


@pytest.mark.parametrize(
    "connect_timeout",
    [
        0,
        -1,
        -10,
    ],
)
def test_pgvector_config_rejects_non_positive_timeout(
    connect_timeout: int,
) -> None:
    """Connection timeout must be positive."""

    with pytest.raises(
        ValueError,
        match="connect_timeout must be greater than zero",
    ):
        PGVectorConfig(
            dsn="postgresql://localhost/finance_ai",
            connect_timeout=connect_timeout,
        )


def test_store_rejects_invalid_config(
    patched_dependencies: None,
) -> None:
    """Store construction should require PGVectorConfig."""

    del patched_dependencies

    with pytest.raises(
        TypeError,
        match="config must be a PGVectorConfig",
    ):
        PGVectorStore(
            config="invalid",  # type: ignore[arg-type]
            initialize=False,
        )


def test_store_rejects_invalid_embedding_service(
    patched_dependencies: None,
    config: PGVectorConfig,
) -> None:
    """Store should require the existing embedding abstraction."""

    del patched_dependencies

    with pytest.raises(
        TypeError,
        match="embedding_service must implement",
    ):
        PGVectorStore(
            config=config,
            embedding_service="invalid",  # type: ignore[arg-type]
            initialize=False,
        )


def test_store_rejects_invalid_initialize_flag(
    patched_dependencies: None,
    config: PGVectorConfig,
) -> None:
    """Initialization control must be boolean."""

    del patched_dependencies

    with pytest.raises(
        TypeError,
        match="initialize must be a boolean",
    ):
        PGVectorStore(
            config=config,
            initialize="yes",  # type: ignore[arg-type]
        )


def test_store_uses_default_embedding_service(
    patched_dependencies: None,
    config: PGVectorConfig,
) -> None:
    """The local deterministic embedding service should be the default."""

    del patched_dependencies

    store = PGVectorStore(
        config=config,
        initialize=False,
    )

    assert isinstance(
        store.embedding_service,
        DeterministicEmbeddingService,
    )


def test_store_preserves_supplied_embedding_service(
    patched_dependencies: None,
    config: PGVectorConfig,
) -> None:
    """A caller-provided embedding service should not be replaced."""

    del patched_dependencies

    embedding_service = DeterministicEmbeddingService(
        dimension=24,
    )

    store = PGVectorStore(
        config=config,
        embedding_service=embedding_service,
        initialize=False,
    )

    assert store.embedding_service is embedding_service
    assert store.dimension == 24


def test_store_exposes_configuration_and_table_name(
    store: PGVectorStore,
    config: PGVectorConfig,
) -> None:
    """Store properties should preserve database configuration."""

    assert store.config is config
    assert store.dimension == 16
    assert store.qualified_table_name == (
        "rag_test.document_embeddings"
    )


def test_store_raises_clear_dependency_error(
    monkeypatch: pytest.MonkeyPatch,
    config: PGVectorConfig,
) -> None:
    """Missing optional packages should produce a focused exception."""

    def raise_dependency_error() -> tuple[Any, Any, Any, Any]:
        raise PGVectorDependencyError(
            "PostgreSQL dependencies are unavailable."
        )

    monkeypatch.setattr(
        PGVectorStore,
        "_load_dependencies",
        staticmethod(raise_dependency_error),
    )

    with pytest.raises(
        PGVectorDependencyError,
        match="dependencies are unavailable",
    ):
        PGVectorStore(
            config=config,
            initialize=False,
        )


def test_validate_text_trims_valid_text(
    store: PGVectorStore,
) -> None:
    """Document and query text should be normalized."""

    assert store._validate_text(
        "  Finance policy  "
    ) == "Finance policy"


def test_validate_text_rejects_non_string(
    store: PGVectorStore,
) -> None:
    """Text must be supplied as a string."""

    with pytest.raises(
        TypeError,
        match="text must be a string",
    ):
        store._validate_text(
            None  # type: ignore[arg-type]
        )


def test_validate_text_rejects_empty_value(
    store: PGVectorStore,
) -> None:
    """Whitespace-only text should not be stored."""

    with pytest.raises(
        ValueError,
        match="text cannot be empty",
    ):
        store._validate_text("   ")


def test_validate_metadata_returns_independent_copy(
    store: PGVectorStore,
) -> None:
    """Metadata should be deep copied before persistence."""

    metadata = {
        "financial_year": "2026",
        "access": {
            "business_units": ["Chennai"],
        },
    }

    result = store._validate_metadata(metadata)

    assert result == metadata
    assert result is not metadata
    assert result["access"] is not metadata["access"]


def test_validate_metadata_accepts_none(
    store: PGVectorStore,
) -> None:
    """Missing metadata should become an empty dictionary."""

    assert store._validate_metadata(None) == {}


def test_validate_metadata_rejects_non_mapping(
    store: PGVectorStore,
) -> None:
    """Metadata must implement Mapping."""

    with pytest.raises(
        TypeError,
        match="metadata must be a mapping",
    ):
        store._validate_metadata(
            ["finance"]  # type: ignore[arg-type]
        )


def test_validate_metadata_rejects_non_json_value(
    store: PGVectorStore,
) -> None:
    """PostgreSQL JSONB metadata must be JSON serializable."""

    with pytest.raises(
        ValueError,
        match="JSON-serializable",
    ):
        store._validate_metadata(
            {
                "invalid": object(),
            }
        )


def test_validate_metadata_filter_returns_copy(
    store: PGVectorStore,
) -> None:
    """Search filters should be validated and independently copied."""

    metadata_filter = {
        "document_category": "budget",
        "financial_year": "2026",
    }

    result = store._validate_metadata_filter(
        metadata_filter
    )

    assert result == metadata_filter
    assert result is not metadata_filter


def test_validate_document_id_trims_value(
    store: PGVectorStore,
) -> None:
    """Document identifiers should be normalized."""

    assert store._validate_document_id(
        "  chunk_123  "
    ) == "chunk_123"


def test_validate_document_id_rejects_invalid_type(
    store: PGVectorStore,
) -> None:
    """Document identifiers must be strings."""

    with pytest.raises(
        TypeError,
        match="document id must be a string",
    ):
        store._validate_document_id(
            123  # type: ignore[arg-type]
        )


def test_validate_document_id_rejects_empty_value(
    store: PGVectorStore,
) -> None:
    """Document identifiers cannot be empty."""

    with pytest.raises(
        ValueError,
        match="document id cannot be empty",
    ):
        store._validate_document_id("   ")


def test_resolve_document_id_preserves_supplied_id(
    store: PGVectorStore,
) -> None:
    """A supplied identifier should be validated and preserved."""

    assert store._resolve_document_id(
        "  finance-policy  "
    ) == "finance-policy"


def test_resolve_document_id_generates_uuid(
    store: PGVectorStore,
) -> None:
    """A missing identifier should produce a valid UUID string."""

    document_id = store._resolve_document_id(None)

    assert str(uuid.UUID(document_id)) == document_id


def test_validate_embedding_accepts_correct_dimension(
    store: PGVectorStore,
) -> None:
    """Embedding vectors matching the service dimension should pass."""

    store._validate_embedding(
        [0.0] * 16
    )


def test_validate_embedding_rejects_dimension_mismatch(
    store: PGVectorStore,
) -> None:
    """Embedding length must match the configured provider."""

    with pytest.raises(
        VectorStoreError,
        match="Embedding dimension mismatch",
    ):
        store._validate_embedding(
            [0.0] * 8
        )


@pytest.mark.parametrize(
    "embedding",
    [
        "invalid",
        b"invalid",
        None,
        123,
    ],
)
def test_validate_embedding_rejects_invalid_sequence(
    store: PGVectorStore,
    embedding: Any,
) -> None:
    """Embeddings must be numeric sequences."""

    with pytest.raises(TypeError):
        store._validate_embedding(embedding)


def test_validate_embedding_rejects_non_numeric_values(
    store: PGVectorStore,
) -> None:
    """Every embedding element must be numeric."""

    embedding: list[Any] = [0.0] * 16
    embedding[4] = "invalid"

    with pytest.raises(
        TypeError,
        match="embedding values must be numeric",
    ):
        store._validate_embedding(embedding)


@pytest.mark.parametrize(
    "top_k",
    [
        True,
        1.5,
        "5",
        None,
    ],
)
def test_validate_top_k_rejects_invalid_type(
    store: PGVectorStore,
    top_k: Any,
) -> None:
    """Search result limit must be an integer."""

    with pytest.raises(
        TypeError,
        match="top_k must be an integer",
    ):
        store._validate_top_k(top_k)


@pytest.mark.parametrize(
    "top_k",
    [
        0,
        -1,
        -10,
    ],
)
def test_validate_top_k_rejects_non_positive_value(
    store: PGVectorStore,
    top_k: int,
) -> None:
    """Search result limit must be positive."""

    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        store._validate_top_k(top_k)


@pytest.mark.parametrize(
    "threshold",
    [
        -1.0,
        -0.5,
        0,
        0.5,
        1.0,
        None,
    ],
)
def test_validate_score_threshold_accepts_valid_values(
    store: PGVectorStore,
    threshold: float | None,
) -> None:
    """Cosine threshold should support the complete valid range."""

    store._validate_score_threshold(threshold)


@pytest.mark.parametrize(
    "threshold",
    [
        -1.1,
        1.1,
        -10,
        10,
    ],
)
def test_validate_score_threshold_rejects_out_of_range(
    store: PGVectorStore,
    threshold: float,
) -> None:
    """Cosine threshold must remain between negative and positive one."""

    with pytest.raises(
        ValueError,
        match="between -1.0 and 1.0",
    ):
        store._validate_score_threshold(threshold)


def test_coerce_document_copies_document(
    store: PGVectorStore,
) -> None:
    """Existing Document values should be returned as independent copies."""

    original = Document(
        id="chunk_1",
        text="Budget assumption",
        metadata={
            "financial_year": "2026",
        },
    )

    result = store._coerce_document(original)

    assert result == original
    assert result is not original
    assert result.metadata is not original.metadata


def test_coerce_document_accepts_string(
    store: PGVectorStore,
) -> None:
    """A plain string should become a Document with a generated ID."""

    result = store._coerce_document(
        "  Forecast methodology  "
    )

    assert result.text == "Forecast methodology"
    assert result.metadata == {}
    assert str(uuid.UUID(result.id)) == result.id


def test_coerce_document_accepts_mapping(
    store: PGVectorStore,
) -> None:
    """Document mappings should preserve ID, text, and metadata."""

    result = store._coerce_document(
        {
            "id": "chunk_2",
            "text": "  Finance policy  ",
            "metadata": {
                "document_category": "policy",
            },
        }
    )

    assert result == Document(
        id="chunk_2",
        text="Finance policy",
        metadata={
            "document_category": "policy",
        },
    )


def test_coerce_document_generates_id_for_mapping(
    store: PGVectorStore,
) -> None:
    """A mapping without an ID should receive a UUID."""

    result = store._coerce_document(
        {
            "text": "Finance policy",
        }
    )

    assert result.text == "Finance policy"
    assert str(uuid.UUID(result.id)) == result.id


def test_coerce_document_rejects_mapping_without_text(
    store: PGVectorStore,
) -> None:
    """Document mappings must contain text."""

    with pytest.raises(
        ValueError,
        match="must contain 'text'",
    ):
        store._coerce_document(
            {
                "id": "chunk_1",
            }
        )


def test_coerce_document_rejects_unsupported_type(
    store: PGVectorStore,
) -> None:
    """Unsupported batch values should fail before database access."""

    with pytest.raises(
        TypeError,
        match="Document, string, or mapping",
    ):
        store._coerce_document(
            123  # type: ignore[arg-type]
        )


def test_copy_document_creates_deep_copy(
    store: PGVectorStore,
) -> None:
    """Copied documents must not share nested metadata."""

    original = Document(
        id="chunk_1",
        text="Finance policy",
        metadata={
            "permissions": {
                "teams": ["finance"],
            },
        },
    )

    copied = store._copy_document(original)

    assert copied == original
    assert copied is not original
    assert copied.metadata is not original.metadata
    assert (
        copied.metadata["permissions"]
        is not original.metadata["permissions"]
    )


def test_row_to_document_accepts_dictionary_metadata() -> None:
    """PostgreSQL rows with decoded JSONB should become Documents."""

    document = PGVectorStore._row_to_document(
        (
            "chunk_1",
            "Finance policy",
            {
                "document_category": "policy",
            },
        )
    )

    assert document == Document(
        id="chunk_1",
        text="Finance policy",
        metadata={
            "document_category": "policy",
        },
    )


def test_row_to_document_accepts_json_string_metadata() -> None:
    """String JSON metadata should be decoded when required."""

    document = PGVectorStore._row_to_document(
        (
            "chunk_1",
            "Budget assumptions",
            '{"financial_year": "2026"}',
        )
    )

    assert document.metadata == {
        "financial_year": "2026",
    }


def test_row_to_document_handles_null_metadata() -> None:
    """Null PostgreSQL metadata should become an empty dictionary."""

    document = PGVectorStore._row_to_document(
        (
            "chunk_1",
            "Forecast methodology",
            None,
        )
    )

    assert document.metadata == {}


# ---------------------------------------------------------------------------
# Optional PostgreSQL integration tests
# ---------------------------------------------------------------------------


def get_integration_dsn() -> str | None:
    """Return the configured integration-test DSN."""

    value = os.getenv(
        TEST_DSN_ENVIRONMENT_VARIABLE
    )

    if value is None:
        return None

    cleaned_value = value.strip()
    return cleaned_value or None


@pytest.fixture
def integration_store() -> Generator[
    PGVectorStore,
    None,
    None,
]:
    
    """
    Create an isolated real PostgreSQL vector store.

    This fixture is skipped unless TEST_PGVECTOR_DSN is available.
    """

    dsn = get_integration_dsn()

    if dsn is None:
        pytest.skip(
            f"{TEST_DSN_ENVIRONMENT_VARIABLE} is not configured."
        )

    unique_suffix = uuid.uuid4().hex[:10]

    config = PGVectorConfig(
        dsn=dsn,
        schema_name="rag_test",
        table_name=(
            f"document_embeddings_{unique_suffix}"
        ),
        create_extension=True,
        create_schema=True,
        create_table=True,
        create_hnsw_index=False,
        connect_timeout=10,
    )

    try:
        store = PGVectorStore(
            config=config,
            embedding_service=(
                DeterministicEmbeddingService(
                    dimension=32,
                )
            ),
            initialize=True,
        )
    except PGVectorDependencyError:
        pytest.skip(
            "psycopg and pgvector packages are not installed."
        )

    yield store

    try:
        store.clear()

        with store._connection() as connection:
            drop_query = store._sql.SQL(
                "DROP TABLE IF EXISTS {}"
            ).format(
                store._table_identifier()
            )
            connection.execute(drop_query)
            connection.commit()
    except Exception:
        # Cleanup failure should not hide the actual test result.
        pass


@pytest.mark.integration
def test_integration_add_and_get_document(
    integration_store: PGVectorStore,
) -> None:
    """A document should persist and be retrievable from PostgreSQL."""

    created = integration_store.add_document(
        text="Budget assumptions for financial year 2026.",
        metadata={
            "document_category": "budget",
            "financial_year": "2026",
        },
        document_id="budget_chunk_1",
    )

    retrieved = integration_store.get_document(
        "budget_chunk_1"
    )

    assert retrieved == created
    assert len(integration_store) == 1
    assert "budget_chunk_1" in integration_store


@pytest.mark.integration
def test_integration_add_documents_is_atomic(
    integration_store: PGVectorStore,
) -> None:
    """A valid document batch should be committed together."""

    documents = [
        Document(
            id="policy_chunk_1",
            text="Finance policy approval levels.",
            metadata={
                "document_category": "policy",
            },
        ),
        Document(
            id="forecast_chunk_1",
            text="Forecast methodology and assumptions.",
            metadata={
                "document_category": "forecast",
            },
        ),
    ]

    result = integration_store.add_documents(
        documents
    )

    assert result == documents
    assert len(integration_store) == 2


@pytest.mark.integration
def test_integration_rejects_duplicate_document(
    integration_store: PGVectorStore,
) -> None:
    """Duplicate IDs should not overwrite existing documents."""

    integration_store.add_document(
        text="Original policy.",
        document_id="policy_chunk_1",
    )

    with pytest.raises(
        VectorStoreError,
        match="already exists",
    ):
        integration_store.add_document(
            text="Replacement policy.",
            document_id="policy_chunk_1",
        )

    assert integration_store.get_document(
        "policy_chunk_1"
    ).text == "Original policy."


@pytest.mark.integration
def test_integration_upsert_document(
    integration_store: PGVectorStore,
) -> None:
    """Upsert should insert and later replace one document."""

    integration_store.upsert_document(
        text="Initial forecast.",
        document_id="forecast_chunk_1",
    )

    updated = integration_store.upsert_document(
        text="Updated forecast.",
        metadata={
            "version": "2",
        },
        document_id="forecast_chunk_1",
    )

    assert updated.text == "Updated forecast."

    retrieved = integration_store.get_document(
        "forecast_chunk_1"
    )

    assert retrieved.text == "Updated forecast."
    assert retrieved.metadata == {
        "version": "2",
    }
    assert len(integration_store) == 1


@pytest.mark.integration
def test_integration_similarity_search(
    integration_store: PGVectorStore,
) -> None:
    """Cosine search should rank related finance text."""

    integration_store.add_documents(
        [
            Document(
                id="budget",
                text=(
                    "Budget assumptions include revenue growth "
                    "and cost controls."
                ),
                metadata={
                    "document_category": "budget",
                },
            ),
            Document(
                id="forecast",
                text=(
                    "Forecast methodology uses demand and "
                    "pricing drivers."
                ),
                metadata={
                    "document_category": "forecast",
                },
            ),
            Document(
                id="policy",
                text=(
                    "Finance policy defines approval authority."
                ),
                metadata={
                    "document_category": "policy",
                },
            ),
        ]
    )

    results = integration_store.similarity_search(
        query="revenue budget growth assumptions",
        top_k=2,
    )

    assert len(results) == 2
    assert all(
        isinstance(result, SearchResult)
        for result in results
    )
    assert results[0].document.id == "budget"
    assert results[0].rank == 1
    assert results[1].rank == 2
    assert results[0].score >= results[1].score


@pytest.mark.integration
def test_integration_similarity_search_with_metadata_filter(
    integration_store: PGVectorStore,
) -> None:
    """JSONB metadata filters should limit eligible documents."""

    integration_store.add_documents(
        [
            Document(
                id="budget_2025",
                text="Revenue budget assumptions.",
                metadata={
                    "document_category": "budget",
                    "financial_year": "2025",
                },
            ),
            Document(
                id="budget_2026",
                text="Revenue budget assumptions.",
                metadata={
                    "document_category": "budget",
                    "financial_year": "2026",
                },
            ),
        ]
    )

    results = integration_store.similarity_search(
        query="revenue budget",
        top_k=5,
        metadata_filter={
            "financial_year": "2026",
        },
    )

    assert len(results) == 1
    assert results[0].document.id == "budget_2026"


@pytest.mark.integration
def test_integration_get_embedding(
    integration_store: PGVectorStore,
) -> None:
    """Stored vectors should retain the configured dimension."""

    integration_store.add_document(
        text="Finance policy.",
        document_id="policy_chunk_1",
    )

    embedding = integration_store.get_embedding(
        "policy_chunk_1"
    )

    assert isinstance(embedding, list)
    assert len(embedding) == 32
    assert all(
        isinstance(value, float)
        for value in embedding
    )


@pytest.mark.integration
def test_integration_list_and_delete_documents(
    integration_store: PGVectorStore,
) -> None:
    """Documents should be listable and removable."""

    integration_store.add_documents(
        [
            Document(
                id="b",
                text="Budget",
            ),
            Document(
                id="a",
                text="Actuals",
            ),
        ]
    )

    listed = integration_store.list_documents()

    assert tuple(
        document.id for document in listed
    ) == ("a", "b")

    deleted = integration_store.delete_document("a")

    assert deleted.id == "a"
    assert len(integration_store) == 1
    assert "a" not in integration_store


@pytest.mark.integration
def test_integration_missing_document_errors(
    integration_store: PGVectorStore,
) -> None:
    """Missing document operations should use existing store exceptions."""

    with pytest.raises(
        DocumentNotFoundError,
    ):
        integration_store.get_document(
            "missing"
        )

    with pytest.raises(
        DocumentNotFoundError,
    ):
        integration_store.get_embedding(
            "missing"
        )

    with pytest.raises(
        DocumentNotFoundError,
    ):
        integration_store.delete_document(
            "missing"
        )


@pytest.mark.integration
def test_integration_clear_returns_deleted_count(
    integration_store: PGVectorStore,
) -> None:
    """Clearing the table should report the removed document count."""

    integration_store.add_documents(
        [
            Document(
                id="one",
                text="Document one",
            ),
            Document(
                id="two",
                text="Document two",
            ),
        ]
    )

    removed_count = integration_store.clear()

    assert removed_count == 2
    assert len(integration_store) == 0