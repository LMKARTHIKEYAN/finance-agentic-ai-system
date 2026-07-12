"""
Persistent PostgreSQL + pgvector store for the Finance Agentic AI System.

This module provides a persistent alternative to InMemoryVectorStore while
preserving the existing vector-store public behaviour.

Responsibilities:

* Persist documents, metadata, and embeddings in PostgreSQL
* Use pgvector for cosine-similarity search
* Support exact-match JSON metadata filters
* Support atomic batch insertion
* Support document retrieval, update, deletion, and clearing
* Delegate embedding generation to BaseEmbeddingService

This module contains no:

* PDF loading
* Text chunking
* Finance business logic
* LLM logic
* Retrieval prompt logic
* FastAPI or Streamlit logic

The PostgreSQL and pgvector packages are imported only when PGVectorStore is
instantiated. Existing local tests can therefore continue using the in-memory
store without requiring a database.
"""

from __future__ import annotations

import copy
import json
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

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
)


class PGVectorStoreError(VectorStoreError):
    """Raised when a PostgreSQL vector-store operation fails."""


class PGVectorDependencyError(PGVectorStoreError):
    """Raised when required PostgreSQL packages are unavailable."""


class PGVectorConnectionError(PGVectorStoreError):
    """Raised when PostgreSQL cannot be reached or initialized."""


class PGVectorConfigurationError(PGVectorStoreError):
    """Raised when the pgvector store configuration is invalid."""


@dataclass(frozen=True)
class PGVectorConfig:
    """
    Configuration for PostgreSQL + pgvector persistence.

    Attributes:
        dsn:
            PostgreSQL connection string.

            Example:

            ``postgresql://postgres:password@localhost:5432/finance_ai``

        schema_name:
            PostgreSQL schema containing the vector table.

        table_name:
            Table used to store documents and embeddings.

        create_extension:
            Whether the service should run
            ``CREATE EXTENSION IF NOT EXISTS vector``.

        create_schema:
            Whether the configured schema should be created automatically.

        create_table:
            Whether the vector table should be created automatically.

        create_hnsw_index:
            Whether to create an HNSW cosine-similarity index.

        connect_timeout:
            PostgreSQL connection timeout in seconds.
    """

    dsn: str
    schema_name: str = "rag"
    table_name: str = "document_embeddings"
    create_extension: bool = True
    create_schema: bool = True
    create_table: bool = True
    create_hnsw_index: bool = False
    connect_timeout: int = 10

    def __post_init__(self) -> None:
        if not isinstance(self.dsn, str):
            raise TypeError("dsn must be a string.")

        cleaned_dsn = self.dsn.strip()

        if not cleaned_dsn:
            raise ValueError("dsn cannot be empty.")

        cleaned_schema = self._validate_identifier(
            self.schema_name,
            "schema_name",
        )
        cleaned_table = self._validate_identifier(
            self.table_name,
            "table_name",
        )

        boolean_fields = {
            "create_extension": self.create_extension,
            "create_schema": self.create_schema,
            "create_table": self.create_table,
            "create_hnsw_index": self.create_hnsw_index,
        }

        for field_name, value in boolean_fields.items():
            if not isinstance(value, bool):
                raise TypeError(
                    f"{field_name} must be a boolean."
                )

        if (
            isinstance(self.connect_timeout, bool)
            or not isinstance(self.connect_timeout, int)
        ):
            raise TypeError(
                "connect_timeout must be an integer."
            )

        if self.connect_timeout <= 0:
            raise ValueError(
                "connect_timeout must be greater than zero."
            )

        object.__setattr__(self, "dsn", cleaned_dsn)
        object.__setattr__(
            self,
            "schema_name",
            cleaned_schema,
        )
        object.__setattr__(
            self,
            "table_name",
            cleaned_table,
        )

    @staticmethod
    def _validate_identifier(
        value: str,
        field_name: str,
    ) -> str:
        """Validate a PostgreSQL schema or table identifier."""

        if not isinstance(value, str):
            raise TypeError(
                f"{field_name} must be a string."
            )

        cleaned_value = value.strip()

        if not cleaned_value:
            raise ValueError(
                f"{field_name} cannot be empty."
            )

        if not (
            cleaned_value[0].isalpha()
            or cleaned_value[0] == "_"
        ):
            raise ValueError(
                f"{field_name} must start with a letter "
                "or underscore."
            )

        if not all(
            character.isalnum() or character == "_"
            for character in cleaned_value
        ):
            raise ValueError(
                f"{field_name} may contain only letters, "
                "numbers, and underscores."
            )

        return cleaned_value


class PGVectorStore(InMemoryVectorStore):
    """
    Persistent PostgreSQL vector store using pgvector.

    The class subclasses ``InMemoryVectorStore`` only to remain compatible
    with the current ingestion service's existing type validation. It does not
    use the in-memory record dictionary; every record is stored in PostgreSQL.

    All public methods used by ingestion and retrieval are overridden.

    Args:
        config:
            PostgreSQL connection and initialization configuration.

        embedding_service:
            Embedding provider. The deterministic local embedding service is
            used when omitted.

        initialize:
            Whether to initialize the pgvector extension, schema, table, and
            optional index during construction.
    """

    def __init__(
        self,
        config: PGVectorConfig,
        embedding_service: BaseEmbeddingService | None = None,
        initialize: bool = True,
    ) -> None:
        if not isinstance(config, PGVectorConfig):
            raise TypeError(
                "config must be a PGVectorConfig."
            )

        if (
            embedding_service is not None
            and not isinstance(
                embedding_service,
                BaseEmbeddingService,
            )
        ):
            raise TypeError(
                "embedding_service must implement "
                "BaseEmbeddingService."
            )

        if not isinstance(initialize, bool):
            raise TypeError(
                "initialize must be a boolean."
            )

        # Do not call InMemoryVectorStore.__init__ because this store does not
        # maintain in-memory records.
        self._config = config
        self._embedding_service = (
            embedding_service
            if embedding_service is not None
            else DeterministicEmbeddingService()
        )

        (
            self._psycopg,
            self._sql,
            self._Vector,
            self._register_vector,
        ) = self._load_dependencies()

        if initialize:
            self.initialize()

    @property
    def config(self) -> PGVectorConfig:
        """Return the PostgreSQL vector-store configuration."""

        return self._config

    @property
    def embedding_service(self) -> BaseEmbeddingService:
        """Return the configured embedding service."""

        return self._embedding_service

    @property
    def dimension(self) -> int:
        """Return the embedding-vector dimension."""

        return self._embedding_service.dimension

    @property
    def qualified_table_name(self) -> str:
        """Return the configured schema-qualified table name."""

        return (
            f"{self._config.schema_name}."
            f"{self._config.table_name}"
        )

    @staticmethod
    def _load_dependencies() -> tuple[
        Any,
        Any,
        Any,
        Any,
    ]:
        """Load optional PostgreSQL dependencies."""

        try:
            import psycopg
            from pgvector import Vector
            from pgvector.psycopg import register_vector
            from psycopg import sql
        except ImportError as exc:
            raise PGVectorDependencyError(
                "PostgreSQL vector storage requires "
                "'psycopg[binary]' and 'pgvector'. Install them "
                "before creating PGVectorStore."
            ) from exc

        return (
            psycopg,
            sql,
            Vector,
            register_vector,
        )

    def initialize(self) -> None:
        """
        Initialize the pgvector extension and storage table.

        This operation is idempotent when automatic creation is enabled.
        """

        try:
            with self._raw_connection() as connection:
                if self._config.create_extension:
                    connection.execute(
                        "CREATE EXTENSION IF NOT EXISTS vector"
                    )

                if self._config.create_schema:
                    connection.execute(
                        self._sql.SQL(
                            "CREATE SCHEMA IF NOT EXISTS {}"
                        ).format(
                            self._sql.Identifier(
                                self._config.schema_name
                            )
                        )
                    )

                if self._config.create_table:
                    self._create_table(connection)

                connection.commit()

                self._register_vector(connection)

                if self._config.create_hnsw_index:
                    self._create_hnsw_index(connection)
                    connection.commit()

                self._validate_existing_dimension(
                    connection
                )
        except PGVectorStoreError:
            raise
        except Exception as exc:
            raise PGVectorConnectionError(
                "Unable to initialize PostgreSQL pgvector "
                f"storage: {exc}"
            ) from exc

    def _create_table(self, connection: Any) -> None:
        """Create the persistent document table."""

        table_identifier = self._table_identifier()

        query = self._sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {} (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                embedding vector({}) NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        ).format(
            table_identifier,
            self._sql.SQL(str(self.dimension)),
        )

        connection.execute(query)

    def _create_hnsw_index(
        self,
        connection: Any,
    ) -> None:
        """Create an HNSW index for cosine-distance search."""

        index_name = (
            f"{self._config.table_name}_embedding_hnsw_idx"
        )

        query = self._sql.SQL(
            """
            CREATE INDEX IF NOT EXISTS {}
            ON {}
            USING hnsw (embedding vector_cosine_ops)
            """
        ).format(
            self._sql.Identifier(index_name),
            self._table_identifier(),
        )

        connection.execute(query)

    def _validate_existing_dimension(
        self,
        connection: Any,
    ) -> None:
        """
        Ensure the existing vector column uses the configured dimension.
        """

        query = """
            SELECT format_type(attribute.atttypid, attribute.atttypmod)
            FROM pg_attribute AS attribute
            JOIN pg_class AS relation
              ON relation.oid = attribute.attrelid
            JOIN pg_namespace AS namespace
              ON namespace.oid = relation.relnamespace
            WHERE namespace.nspname = %s
              AND relation.relname = %s
              AND attribute.attname = 'embedding'
              AND attribute.attnum > 0
              AND NOT attribute.attisdropped
        """

        row = connection.execute(
            query,
            (
                self._config.schema_name,
                self._config.table_name,
            ),
        ).fetchone()

        if row is None:
            if self._config.create_table:
                raise PGVectorConfigurationError(
                    "The vector table was not created."
                )

            return

        expected_type = f"vector({self.dimension})"
        actual_type = str(row[0]).lower()

        if actual_type != expected_type:
            raise PGVectorConfigurationError(
                "Existing embedding column dimension does not "
                f"match the embedding service. Expected "
                f"'{expected_type}', found '{actual_type}'."
            )

    @contextmanager
    def _raw_connection(self) -> Iterator[Any]:
        """Open an unregistered Psycopg connection."""

        connection = None

        try:
            connection = self._psycopg.connect(
                self._config.dsn,
                connect_timeout=(
                    self._config.connect_timeout
                ),
            )
            yield connection
        finally:
            if connection is not None:
                connection.close()

    @contextmanager
    def _connection(self) -> Iterator[Any]:
        """Open a connection with pgvector types registered."""

        try:
            with self._raw_connection() as connection:
                self._register_vector(connection)

                try:
                    yield connection
                except Exception:
                    connection.rollback()
                    raise
        except PGVectorStoreError:
            raise
        except Exception as exc:
            raise PGVectorConnectionError(
                f"PostgreSQL operation failed: {exc}"
            ) from exc

    def _table_identifier(self) -> Any:
        """Return a safely quoted schema-qualified table."""

        return self._sql.Identifier(
            self._config.schema_name,
            self._config.table_name,
        )

    def __len__(self) -> int:
        """Return the number of persisted documents."""

        query = self._sql.SQL(
            "SELECT COUNT(*) FROM {}"
        ).format(self._table_identifier())

        with self._connection() as connection:
            row = connection.execute(query).fetchone()

        return int(row[0]) if row is not None else 0

    def __contains__(
        self,
        document_id: object,
    ) -> bool:
        """Return whether a document identifier exists."""

        if not isinstance(document_id, str):
            return False

        cleaned_id = document_id.strip()

        if not cleaned_id:
            return False

        query = self._sql.SQL(
            "SELECT 1 FROM {} WHERE id = %s LIMIT 1"
        ).format(self._table_identifier())

        with self._connection() as connection:
            row = connection.execute(
                query,
                (cleaned_id,),
            ).fetchone()

        return row is not None

    def add_document(
        self,
        text: str,
        metadata: Mapping[str, Any] | None = None,
        document_id: str | None = None,
    ) -> Document:
        """Embed and persist one new document."""

        validated_text = self._validate_text(text)
        resolved_id = self._resolve_document_id(
            document_id
        )
        resolved_metadata = self._validate_metadata(
            metadata
        )

        document = Document(
            id=resolved_id,
            text=validated_text,
            metadata=resolved_metadata,
        )

        if resolved_id in self:
            raise VectorStoreError(
                f"Document '{resolved_id}' already exists."
            )

        try:
            embedding = self._embedding_service.embed_text(
                document.text
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Unable to embed document "
                f"'{resolved_id}': {exc}"
            ) from exc

        self._validate_embedding(embedding)
        self._insert_documents(
            [(document, embedding)]
        )

        return self._copy_document(document)

    def add_documents(
        self,
        documents: Sequence[
            Document | str | Mapping[str, Any]
        ],
    ) -> list[Document]:
        """
        Embed and persist multiple documents atomically.

        If validation, embedding, or insertion fails, no document from the
        batch is committed.
        """

        if isinstance(documents, (str, bytes)):
            raise TypeError(
                "documents must be a sequence, "
                "not a single string."
            )

        if not isinstance(documents, Sequence):
            raise TypeError(
                "documents must be a sequence."
            )

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

        existing_ids = self._find_existing_ids(
            document_ids
        )

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
                "Embedding service returned an unexpected "
                "number of vectors."
            )

        records: list[
            tuple[Document, Sequence[float]]
        ] = []

        for document, embedding in zip(
            prepared_documents,
            embeddings,
        ):
            self._validate_embedding(embedding)
            records.append((document, embedding))

        self._insert_documents(records)

        return [
            self._copy_document(document)
            for document in prepared_documents
        ]

    def _insert_documents(
        self,
        records: Sequence[
            tuple[Document, Sequence[float]]
        ],
    ) -> None:
        """Insert a collection of document-vector records atomically."""

        query = self._sql.SQL(
            """
            INSERT INTO {} (
                id,
                text,
                metadata,
                embedding
            )
            VALUES (
                %s,
                %s,
                %s::jsonb,
                %s
            )
            """
        ).format(self._table_identifier())

        values = [
            (
                document.id,
                document.text,
                json.dumps(document.metadata),
                self._Vector(
                    [
                        float(value)
                        for value in embedding
                    ]
                ),
            )
            for document, embedding in records
        ]

        try:
            with self._connection() as connection:
                connection.executemany(
                    query,
                    values,
                )
                connection.commit()
        except Exception as exc:
            message = str(exc)

            if (
                "duplicate key" in message.lower()
                or "unique constraint" in message.lower()
            ):
                raise VectorStoreError(
                    "One or more document identifiers "
                    "already exist."
                ) from exc

            raise PGVectorStoreError(
                f"Unable to persist document batch: {exc}"
            ) from exc

    def upsert_document(
        self,
        text: str,
        metadata: Mapping[str, Any] | None = None,
        document_id: str | None = None,
    ) -> Document:
        """Insert or replace one persisted document."""

        validated_text = self._validate_text(text)
        resolved_id = self._resolve_document_id(
            document_id
        )
        resolved_metadata = self._validate_metadata(
            metadata
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
                f"Unable to embed document "
                f"'{resolved_id}': {exc}"
            ) from exc

        self._validate_embedding(embedding)

        query = self._sql.SQL(
            """
            INSERT INTO {} (
                id,
                text,
                metadata,
                embedding
            )
            VALUES (
                %s,
                %s,
                %s::jsonb,
                %s
            )
            ON CONFLICT (id)
            DO UPDATE SET
                text = EXCLUDED.text,
                metadata = EXCLUDED.metadata,
                embedding = EXCLUDED.embedding,
                updated_at = NOW()
            """
        ).format(self._table_identifier())

        with self._connection() as connection:
            connection.execute(
                query,
                (
                    document.id,
                    document.text,
                    json.dumps(document.metadata),
                    self._Vector(
                        [
                            float(value)
                            for value in embedding
                        ]
                    ),
                ),
            )
            connection.commit()

        return self._copy_document(document)

    def get_document(
        self,
        document_id: str,
    ) -> Document:
        """Return one persisted document."""

        resolved_id = self._validate_document_id(
            document_id
        )

        query = self._sql.SQL(
            """
            SELECT id, text, metadata
            FROM {}
            WHERE id = %s
            """
        ).format(self._table_identifier())

        with self._connection() as connection:
            row = connection.execute(
                query,
                (resolved_id,),
            ).fetchone()

        if row is None:
            raise DocumentNotFoundError(
                f"Document '{resolved_id}' was not found."
            )

        return self._row_to_document(row)

    def get_embedding(
        self,
        document_id: str,
    ) -> list[float]:
        """Return the stored vector for one document."""

        resolved_id = self._validate_document_id(
            document_id
        )

        query = self._sql.SQL(
            """
            SELECT embedding
            FROM {}
            WHERE id = %s
            """
        ).format(self._table_identifier())

        with self._connection() as connection:
            row = connection.execute(
                query,
                (resolved_id,),
            ).fetchone()

        if row is None:
            raise DocumentNotFoundError(
                f"Document '{resolved_id}' was not found."
            )

        embedding = row[0]

        if hasattr(embedding, "to_list"):
            return [
                float(value)
                for value in embedding.to_list()
            ]

        return [
            float(value)
            for value in embedding
        ]

    def list_documents(self) -> list[Document]:
        """Return all persisted documents ordered by identifier."""

        query = self._sql.SQL(
            """
            SELECT id, text, metadata
            FROM {}
            ORDER BY id
            """
        ).format(self._table_identifier())

        with self._connection() as connection:
            rows = connection.execute(query).fetchall()

        return [
            self._row_to_document(row)
            for row in rows
        ]

    def delete_document(
        self,
        document_id: str,
    ) -> Document:
        """Delete and return one persisted document."""

        resolved_id = self._validate_document_id(
            document_id
        )

        query = self._sql.SQL(
            """
            DELETE FROM {}
            WHERE id = %s
            RETURNING id, text, metadata
            """
        ).format(self._table_identifier())

        with self._connection() as connection:
            row = connection.execute(
                query,
                (resolved_id,),
            ).fetchone()
            connection.commit()

        if row is None:
            raise DocumentNotFoundError(
                f"Document '{resolved_id}' was not found."
            )

        return self._row_to_document(row)

    def clear(self) -> int:
        """Delete all documents and return the removed count."""

        query = self._sql.SQL(
            """
            WITH deleted AS (
                DELETE FROM {}
                RETURNING id
            )
            SELECT COUNT(*) FROM deleted
            """
        ).format(self._table_identifier())

        with self._connection() as connection:
            row = connection.execute(query).fetchone()
            connection.commit()

        return int(row[0]) if row is not None else 0

    def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float | None = None,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Embed a query and perform cosine-similarity search."""

        validated_query = self._validate_text(query)
        self._validate_top_k(top_k)
        self._validate_score_threshold(
            score_threshold
        )
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

        return self.similarity_search_by_vector(
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
        """Search using an already-generated query vector."""

        self._validate_embedding(query_embedding)
        self._validate_top_k(top_k)
        self._validate_score_threshold(
            score_threshold
        )
        resolved_filter = self._validate_metadata_filter(
            metadata_filter
        )

        query_vector = self._Vector(
            [
                float(value)
                for value in query_embedding
            ]
        )

        conditions: list[Any] = []
        parameters: list[Any] = []

        if resolved_filter:
            conditions.append(
                self._sql.SQL("metadata @> %s::jsonb")
            )
            parameters.append(
                json.dumps(resolved_filter)
            )

        if score_threshold is not None:
            conditions.append(
                self._sql.SQL(
                    "(1.0 - (embedding <=> %s)) >= %s"
                )
            )
            parameters.extend(
                [
                    query_vector,
                    float(score_threshold),
                ]
            )

        where_clause = self._sql.SQL("")

        if conditions:
            where_clause = (
                self._sql.SQL("WHERE ")
                + self._sql.SQL(" AND ").join(
                    conditions
                )
            )

        search_query = self._sql.SQL(
            """
            SELECT
                id,
                text,
                metadata,
                1.0 - (embedding <=> %s) AS score
            FROM {}
            {}
            ORDER BY embedding <=> %s, id
            LIMIT %s
            """
        ).format(
            self._table_identifier(),
            where_clause,
        )

        final_parameters: list[Any] = [
            query_vector,
            *parameters,
            query_vector,
            top_k,
        ]

        with self._connection() as connection:
            rows = connection.execute(
                search_query,
                final_parameters,
            ).fetchall()

        return [
            SearchResult(
                document=self._row_to_document(
                    row[:3]
                ),
                score=max(
                    -1.0,
                    min(1.0, float(row[3])),
                ),
                rank=rank,
            )
            for rank, row in enumerate(
                rows,
                start=1,
            )
        ]

    def _find_existing_ids(
        self,
        document_ids: Sequence[str],
    ) -> list[str]:
        """Return identifiers already stored in PostgreSQL."""

        if not document_ids:
            return []

        query = self._sql.SQL(
            """
            SELECT id
            FROM {}
            WHERE id = ANY(%s)
            ORDER BY id
            """
        ).format(self._table_identifier())

        with self._connection() as connection:
            rows = connection.execute(
                query,
                (list(document_ids),),
            ).fetchall()

        return [
            str(row[0])
            for row in rows
        ]

    @staticmethod
    def _row_to_document(
        row: Sequence[Any],
    ) -> Document:
        """Convert a PostgreSQL row into a Document."""

        metadata = row[2]

        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        if metadata is None:
            metadata = {}

        return Document(
            id=str(row[0]),
            text=str(row[1]),
            metadata=copy.deepcopy(
                dict(metadata)
            ),
        )

    @staticmethod
    def _copy_document(
        document: Document,
    ) -> Document:
        """Return an independent document copy."""

        return Document(
            id=document.id,
            text=document.text,
            metadata=copy.deepcopy(
                document.metadata
            ),
        )

    @staticmethod
    def _validate_text(text: str) -> str:
        """Validate document or query text."""

        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        cleaned_text = text.strip()

        if not cleaned_text:
            raise ValueError("text cannot be empty.")

        return cleaned_text

    @staticmethod
    def _validate_metadata(
        metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Validate JSON-compatible metadata."""

        if metadata is None:
            return {}

        if not isinstance(metadata, Mapping):
            raise TypeError(
                "metadata must be a mapping."
            )

        copied_metadata = copy.deepcopy(
            dict(metadata)
        )

        try:
            json.dumps(copied_metadata)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "metadata must contain JSON-serializable values."
            ) from exc

        return copied_metadata

    @staticmethod
    def _validate_metadata_filter(
        metadata_filter: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Validate exact-match JSON metadata filters."""

        if metadata_filter is None:
            return {}

        if not isinstance(metadata_filter, Mapping):
            raise TypeError(
                "metadata_filter must be a mapping."
            )

        resolved_filter = copy.deepcopy(
            dict(metadata_filter)
        )

        try:
            json.dumps(resolved_filter)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "metadata_filter must contain "
                "JSON-serializable values."
            ) from exc

        return resolved_filter

    @staticmethod
    def _resolve_document_id(
        document_id: str | None,
    ) -> str:
        """Validate or generate a document identifier."""

        if document_id is None:
            return str(uuid.uuid4())

        return PGVectorStore._validate_document_id(
            document_id
        )

    @staticmethod
    def _validate_document_id(
        document_id: str,
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

    def _validate_embedding(
        self,
        embedding: Sequence[float],
    ) -> None:
        """Validate embedding type, values, and dimension."""

        if isinstance(embedding, (str, bytes)):
            raise TypeError(
                "embedding must be a numeric sequence."
            )

        if not isinstance(embedding, Sequence):
            raise TypeError(
                "embedding must be a sequence."
            )

        if len(embedding) != self.dimension:
            raise VectorStoreError(
                "Embedding dimension mismatch. Expected "
                f"{self.dimension}, received {len(embedding)}."
            )

        for value in embedding:
            if isinstance(value, bool) or not isinstance(
                value,
                (int, float),
            ):
                raise TypeError(
                    "embedding values must be numeric."
                )

    @staticmethod
    def _validate_top_k(top_k: int) -> None:
        """Validate search-result limit."""

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

    def _coerce_document(
        self,
        item: Document | str | Mapping[str, Any],
    ) -> Document:
        """Convert supported inputs into a Document."""

        if isinstance(item, Document):
            return self._copy_document(item)

        if isinstance(item, str):
            return Document(
                id=str(uuid.uuid4()),
                text=self._validate_text(item),
                metadata={},
            )

        if isinstance(item, Mapping):
            if "text" not in item:
                raise ValueError(
                    "Document mapping must contain 'text'."
                )

            text = self._validate_text(
                item["text"]
            )

            document_id_value = item.get("id")

            if document_id_value is None:
                document_id = str(uuid.uuid4())
            else:
                document_id = (
                    self._validate_document_id(
                        document_id_value
                    )
                )

            metadata = self._validate_metadata(
                item.get("metadata")
            )

            return Document(
                id=document_id,
                text=text,
                metadata=metadata,
            )

        raise TypeError(
            "Each document must be a Document, string, "
            "or mapping."
        )