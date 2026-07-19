
"""
Persistent long-term memory for the Finance Agentic AI System.

This module stores historical finance information in SQLite so that memory
survives application restarts.

Long-term memory can retain information such as:

- Historical user questions
- Finance agent outputs
- Variance-analysis results
- Management reports
- P&L commentary
- Forecast assumptions
- Workflow outcomes
- User and company preferences

Session memory is temporary and exists only during the current application
runtime. Long-term memory is persistent and remains available after the
application process stops.

The implementation uses only Python's standard library and SQLite. It can
later be replaced by PostgreSQL, pgvector, Snowflake, or another enterprise
database without changing the high-level memory workflow.
"""

from __future__ import annotations

import json
import pickle
import sqlite3
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Generator
from uuid import uuid4


def _utc_now() -> datetime:
    """
    Return the current timezone-aware UTC datetime.

    Returns:
        Current UTC datetime.
    """

    return datetime.now(timezone.utc)


def _datetime_to_string(value: datetime) -> str:
    """
    Convert a datetime value to an ISO-formatted UTC string.

    Args:
        value: Datetime value to convert.

    Returns:
        ISO-formatted datetime string.

    Raises:
        ValueError: If the datetime is timezone-naive.
    """

    if value.tzinfo is None:
        raise ValueError("datetime value must be timezone-aware")

    return value.astimezone(timezone.utc).isoformat()


def _datetime_from_string(value: str) -> datetime:
    """
    Convert an ISO-formatted datetime string to a datetime object.

    Args:
        value: ISO-formatted datetime string.

    Returns:
        Timezone-aware datetime value.
    """

    parsed_value = datetime.fromisoformat(value)

    if parsed_value.tzinfo is None:
        parsed_value = parsed_value.replace(tzinfo=timezone.utc)

    return parsed_value.astimezone(timezone.utc)


@dataclass
class LongTermMemoryEntry:
    """
    Represents one persistent long-term memory entry.

    Attributes:
        memory_id:
            Unique identifier of the stored memory.
        namespace:
            Logical grouping such as user, company, report, or workflow.
        key:
            Business-friendly memory key.
        value:
            Stored Python value.
        created_at:
            Time when the memory was first created.
        updated_at:
            Time when the memory was most recently updated.
        metadata:
            Optional searchable descriptive information.
        tags:
            Optional labels used for filtering memories.
    """

    memory_id: str
    namespace: str
    key: str
    value: Any
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the memory entry to a dictionary.

        Returns:
            Deep-copied dictionary representation of the entry.
        """

        return {
            "memory_id": self.memory_id,
            "namespace": self.namespace,
            "key": self.key,
            "value": deepcopy(self.value),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": deepcopy(self.metadata),
            "tags": deepcopy(self.tags),
        }


@dataclass
class LongTermMemorySummary:
    """
    Represents a lightweight memory summary without the stored value.

    Attributes:
        memory_id:
            Unique memory identifier.
        namespace:
            Logical grouping of the memory.
        key:
            Business-friendly memory key.
        created_at:
            Memory creation time.
        updated_at:
            Most recent update time.
        metadata:
            Descriptive memory metadata.
        tags:
            Searchable memory labels.
    """

    memory_id: str
    namespace: str
    key: str
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the summary to a dictionary.

        Returns:
            Deep-copied dictionary representation.
        """

        return {
            "memory_id": self.memory_id,
            "namespace": self.namespace,
            "key": self.key,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": deepcopy(self.metadata),
            "tags": deepcopy(self.tags),
        }


class LongTermMemory:
    """
    Thread-safe SQLite-backed persistent memory manager.

    The manager stores historical Finance Agentic AI information in a local
    SQLite database.

    Example:
        memory = LongTermMemory(
            database_path="data/finance_memory.db"
        )

        memory_id = memory.save(
            namespace="reports",
            key="chennai-2026-06",
            value={
                "status": "REVIEW",
                "revenue_variance": -200_000,
            },
            metadata={
                "branch": "Chennai",
                "period": "2026-06",
            },
            tags=["monthly", "variance", "management-report"],
        )

        report = memory.get(memory_id)
    """

    _TABLE_NAME = "long_term_memory"

    def __init__(
        self,
        database_path: str | Path = "data/long_term_memory.db",
        timeout_seconds: float = 30.0,
    ) -> None:
        """
        Initialize persistent long-term memory.

        Args:
            database_path:
                SQLite database file path. Use ":memory:" for a temporary
                in-memory SQLite database during isolated tests.
            timeout_seconds:
                Number of seconds SQLite should wait when the database is
                locked before raising an error.

        Raises:
            ValueError:
                If database_path is empty or timeout_seconds is not positive.
        """

        self._database_path = self._validate_database_path(
            database_path
        )

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        self._timeout_seconds = float(timeout_seconds)
        self._lock = RLock()

        self._shared_connection: sqlite3.Connection | None = None

        if self._database_path == ":memory:":
            self._shared_connection = self._create_connection()

        self._prepare_database_directory()
        self._initialize_database()

    @property
    def database_path(self) -> str:
        """
        Return the configured SQLite database path.

        Returns:
            Database path string.
        """

        return self._database_path

    @property
    def count(self) -> int:
        """
        Return the total number of stored memories.

        Returns:
            Number of long-term memory records.
        """

        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                f"""
                SELECT COUNT(*)
                FROM {self._TABLE_NAME}
                """
            )
            result = cursor.fetchone()

            return int(result[0]) if result else 0

    def save(
        self,
        namespace: str,
        key: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | tuple[str, ...] | set[str] | None = None,
        memory_id: str | None = None,
        overwrite: bool = False,
    ) -> str:
        """
        Store a new long-term memory.

        Args:
            namespace:
                Logical grouping such as reports, users, companies, or
                workflows.
            key:
                Business-friendly memory key.
            value:
                Python value to persist.
            metadata:
                Optional JSON-compatible metadata.
            tags:
                Optional labels for filtering and search.
            memory_id:
                Optional custom memory identifier. A UUID is generated when
                this value is omitted.
            overwrite:
                Whether an existing namespace-and-key memory should be
                replaced.

        Returns:
            Identifier of the stored memory.

        Raises:
            ValueError:
                If namespace, key, memory_id, metadata, or tags are invalid.
            KeyError:
                If the memory already exists and overwrite is False.
            TypeError:
                If metadata is not JSON serializable or the value cannot be
                serialized.
        """

        resolved_namespace = self._validate_required_string(
            namespace,
            "namespace",
        )
        resolved_key = self._validate_required_string(
            key,
            "key",
        )
        resolved_memory_id = (
            self._validate_required_string(memory_id, "memory_id")
            if memory_id is not None
            else str(uuid4())
        )
        resolved_metadata = self._validate_metadata(metadata)
        resolved_tags = self._validate_tags(tags)

        serialized_value = self._serialize_value(value)
        serialized_metadata = self._serialize_json(
            resolved_metadata,
            "metadata",
        )
        serialized_tags = self._serialize_json(
            resolved_tags,
            "tags",
        )

        now = _datetime_to_string(_utc_now())

        with self._lock, self._connection() as connection:
            existing = connection.execute(
                f"""
                SELECT memory_id
                FROM {self._TABLE_NAME}
                WHERE namespace = ? AND key = ?
                """,
                (resolved_namespace, resolved_key),
            ).fetchone()

            if existing is not None:
                if not overwrite:
                    raise KeyError(
                        "memory already exists for "
                        f"namespace={resolved_namespace!r}, "
                        f"key={resolved_key!r}"
                    )

                existing_memory_id = str(existing["memory_id"])

                connection.execute(
                    f"""
                    UPDATE {self._TABLE_NAME}
                    SET value_blob = ?,
                        metadata_json = ?,
                        tags_json = ?,
                        updated_at = ?
                    WHERE memory_id = ?
                    """,
                    (
                        serialized_value,
                        serialized_metadata,
                        serialized_tags,
                        now,
                        existing_memory_id,
                    ),
                )
                connection.commit()

                return existing_memory_id

            memory_id_exists = connection.execute(
                f"""
                SELECT 1
                FROM {self._TABLE_NAME}
                WHERE memory_id = ?
                """,
                (resolved_memory_id,),
            ).fetchone()

            if memory_id_exists is not None:
                raise KeyError(
                    f"memory_id already exists: {resolved_memory_id}"
                )

            connection.execute(
                f"""
                INSERT INTO {self._TABLE_NAME} (
                    memory_id,
                    namespace,
                    key,
                    value_blob,
                    metadata_json,
                    tags_json,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    resolved_memory_id,
                    resolved_namespace,
                    resolved_key,
                    serialized_value,
                    serialized_metadata,
                    serialized_tags,
                    now,
                    now,
                ),
            )
            connection.commit()

            return resolved_memory_id

    def upsert(
        self,
        namespace: str,
        key: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> str:
        """
        Create a memory or update it when namespace and key already exist.

        Args:
            namespace:
                Logical memory grouping.
            key:
                Business-friendly memory key.
            value:
                Python value to persist.
            metadata:
                Optional JSON-compatible metadata.
            tags:
                Optional searchable labels.

        Returns:
            Identifier of the created or updated memory.
        """

        return self.save(
            namespace=namespace,
            key=key,
            value=value,
            metadata=metadata,
            tags=tags,
            overwrite=True,
        )

    def get(self, memory_id: str) -> LongTermMemoryEntry:
        """
        Retrieve a memory using its unique identifier.

        Args:
            memory_id: Unique memory identifier.

        Returns:
            Complete long-term memory entry.

        Raises:
            KeyError:
                If no matching memory exists.
        """

        resolved_memory_id = self._validate_required_string(
            memory_id,
            "memory_id",
        )

        with self._lock, self._connection() as connection:
            row = connection.execute(
                f"""
                SELECT *
                FROM {self._TABLE_NAME}
                WHERE memory_id = ?
                """,
                (resolved_memory_id,),
            ).fetchone()

        if row is None:
            raise KeyError(
                f"long-term memory not found: {resolved_memory_id}"
            )

        return self._row_to_entry(row)

    def get_by_key(
        self,
        namespace: str,
        key: str,
    ) -> LongTermMemoryEntry:
        """
        Retrieve a memory using namespace and business key.

        Args:
            namespace: Logical memory grouping.
            key: Business-friendly memory key.

        Returns:
            Complete long-term memory entry.

        Raises:
            KeyError:
                If no matching memory exists.
        """

        resolved_namespace = self._validate_required_string(
            namespace,
            "namespace",
        )
        resolved_key = self._validate_required_string(
            key,
            "key",
        )

        with self._lock, self._connection() as connection:
            row = connection.execute(
                f"""
                SELECT *
                FROM {self._TABLE_NAME}
                WHERE namespace = ? AND key = ?
                """,
                (resolved_namespace, resolved_key),
            ).fetchone()

        if row is None:
            raise KeyError(
                "long-term memory not found for "
                f"namespace={resolved_namespace!r}, "
                f"key={resolved_key!r}"
            )

        return self._row_to_entry(row)

    def exists(self, memory_id: str) -> bool:
        """
        Check whether a memory identifier exists.

        Args:
            memory_id: Unique memory identifier.

        Returns:
            True when the memory exists.
        """

        resolved_memory_id = self._validate_required_string(
            memory_id,
            "memory_id",
        )

        with self._lock, self._connection() as connection:
            result = connection.execute(
                f"""
                SELECT 1
                FROM {self._TABLE_NAME}
                WHERE memory_id = ?
                """,
                (resolved_memory_id,),
            ).fetchone()

            return result is not None

    def key_exists(
        self,
        namespace: str,
        key: str,
    ) -> bool:
        """
        Check whether a namespace-and-key combination exists.

        Args:
            namespace: Logical memory grouping.
            key: Business-friendly memory key.

        Returns:
            True when a matching memory exists.
        """

        resolved_namespace = self._validate_required_string(
            namespace,
            "namespace",
        )
        resolved_key = self._validate_required_string(
            key,
            "key",
        )

        with self._lock, self._connection() as connection:
            result = connection.execute(
                f"""
                SELECT 1
                FROM {self._TABLE_NAME}
                WHERE namespace = ? AND key = ?
                """,
                (resolved_namespace, resolved_key),
            ).fetchone()

            return result is not None

    def update(
        self,
        memory_id: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | tuple[str, ...] | set[str] | None = None,
        preserve_metadata: bool = True,
        preserve_tags: bool = True,
    ) -> LongTermMemoryEntry:
        """
        Update an existing long-term memory.

        Args:
            memory_id:
                Unique memory identifier.
            value:
                New value to store.
            metadata:
                Optional replacement metadata.
            tags:
                Optional replacement tags.
            preserve_metadata:
                Preserve current metadata when metadata is omitted.
            preserve_tags:
                Preserve current tags when tags are omitted.

        Returns:
            Updated long-term memory entry.

        Raises:
            KeyError:
                If the memory does not exist.
        """

        resolved_memory_id = self._validate_required_string(
            memory_id,
            "memory_id",
        )
        current_entry = self.get(resolved_memory_id)

        if metadata is None and preserve_metadata:
            resolved_metadata = current_entry.metadata
        else:
            resolved_metadata = self._validate_metadata(metadata)

        if tags is None and preserve_tags:
            resolved_tags = current_entry.tags
        else:
            resolved_tags = self._validate_tags(tags)

        serialized_value = self._serialize_value(value)
        serialized_metadata = self._serialize_json(
            resolved_metadata,
            "metadata",
        )
        serialized_tags = self._serialize_json(
            resolved_tags,
            "tags",
        )
        updated_at = _datetime_to_string(_utc_now())

        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                f"""
                UPDATE {self._TABLE_NAME}
                SET value_blob = ?,
                    metadata_json = ?,
                    tags_json = ?,
                    updated_at = ?
                WHERE memory_id = ?
                """,
                (
                    serialized_value,
                    serialized_metadata,
                    serialized_tags,
                    updated_at,
                    resolved_memory_id,
                ),
            )

            if cursor.rowcount == 0:
                raise KeyError(
                    "long-term memory not found: "
                    f"{resolved_memory_id}"
                )

            connection.commit()

        return self.get(resolved_memory_id)

    def update_metadata(
        self,
        memory_id: str,
        metadata: dict[str, Any],
        merge: bool = True,
    ) -> dict[str, Any]:
        """
        Update only the metadata of an existing memory.

        Args:
            memory_id: Unique memory identifier.
            metadata: Metadata values to apply.
            merge:
                Merge metadata with existing values when True. Replace all
                metadata when False.

        Returns:
            Updated metadata dictionary.
        """

        resolved_metadata = self._validate_metadata(metadata)
        current_entry = self.get(memory_id)

        if merge:
            updated_metadata = deepcopy(current_entry.metadata)
            updated_metadata.update(resolved_metadata)
        else:
            updated_metadata = resolved_metadata

        updated_entry = self.update(
            memory_id=memory_id,
            value=current_entry.value,
            metadata=updated_metadata,
            tags=current_entry.tags,
        )

        return deepcopy(updated_entry.metadata)

    def rename(
        self,
        memory_id: str,
        namespace: str | None = None,
        key: str | None = None,
    ) -> LongTermMemoryEntry:
        """
        Change the namespace or key of an existing memory.

        Args:
            memory_id: Unique memory identifier.
            namespace: Optional new namespace.
            key: Optional new business key.

        Returns:
            Updated long-term memory entry.

        Raises:
            ValueError:
                If neither namespace nor key is provided.
            KeyError:
                If the memory does not exist.
        """

        if namespace is None and key is None:
            raise ValueError(
                "namespace or key must be provided"
            )

        current_entry = self.get(memory_id)

        resolved_namespace = (
            self._validate_required_string(
                namespace,
                "namespace",
            )
            if namespace is not None
            else current_entry.namespace
        )
        resolved_key = (
            self._validate_required_string(key, "key")
            if key is not None
            else current_entry.key
        )

        with self._lock, self._connection() as connection:
            duplicate = connection.execute(
                f"""
                SELECT memory_id
                FROM {self._TABLE_NAME}
                WHERE namespace = ? AND key = ?
                  AND memory_id != ?
                """,
                (
                    resolved_namespace,
                    resolved_key,
                    current_entry.memory_id,
                ),
            ).fetchone()

            if duplicate is not None:
                raise KeyError(
                    "memory already exists for "
                    f"namespace={resolved_namespace!r}, "
                    f"key={resolved_key!r}"
                )

            connection.execute(
                f"""
                UPDATE {self._TABLE_NAME}
                SET namespace = ?,
                    key = ?,
                    updated_at = ?
                WHERE memory_id = ?
                """,
                (
                    resolved_namespace,
                    resolved_key,
                    _datetime_to_string(_utc_now()),
                    current_entry.memory_id,
                ),
            )
            connection.commit()

        return self.get(current_entry.memory_id)

    def delete(self, memory_id: str) -> bool:
        """
        Delete a memory using its identifier.

        Args:
            memory_id: Unique memory identifier.

        Returns:
            True if a memory was deleted, otherwise False.
        """

        resolved_memory_id = self._validate_required_string(
            memory_id,
            "memory_id",
        )

        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                f"""
                DELETE FROM {self._TABLE_NAME}
                WHERE memory_id = ?
                """,
                (resolved_memory_id,),
            )
            connection.commit()

            return cursor.rowcount > 0

    def delete_by_key(
        self,
        namespace: str,
        key: str,
    ) -> bool:
        """
        Delete a memory using namespace and key.

        Args:
            namespace: Logical memory grouping.
            key: Business-friendly memory key.

        Returns:
            True if a memory was deleted, otherwise False.
        """

        resolved_namespace = self._validate_required_string(
            namespace,
            "namespace",
        )
        resolved_key = self._validate_required_string(
            key,
            "key",
        )

        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                f"""
                DELETE FROM {self._TABLE_NAME}
                WHERE namespace = ? AND key = ?
                """,
                (resolved_namespace, resolved_key),
            )
            connection.commit()

            return cursor.rowcount > 0

    def list_memories(
        self,
        namespace: str | None = None,
        tag: str | None = None,
        limit: int | None = None,
        offset: int = 0,
        newest_first: bool = True,
    ) -> list[LongTermMemorySummary]:
        """
        List memory summaries with optional filtering.

        Args:
            namespace:
                Optional namespace filter.
            tag:
                Optional exact tag filter.
            limit:
                Optional maximum number of results.
            offset:
                Number of records to skip.
            newest_first:
                Sort by newest update first when True.

        Returns:
            List of lightweight memory summaries.
        """

        resolved_namespace = (
            self._validate_required_string(
                namespace,
                "namespace",
            )
            if namespace is not None
            else None
        )
        resolved_tag = (
            self._validate_required_string(tag, "tag")
            if tag is not None
            else None
        )
        resolved_limit = self._validate_limit(limit)
        resolved_offset = self._validate_offset(offset)

        clauses: list[str] = []
        parameters: list[Any] = []

        if resolved_namespace is not None:
            clauses.append("namespace = ?")
            parameters.append(resolved_namespace)

        if resolved_tag is not None:
            clauses.append(
                "EXISTS ("
                "SELECT 1 FROM json_each(tags_json) "
                "WHERE json_each.value = ?"
                ")"
            )
            parameters.append(resolved_tag)

        where_clause = (
            f"WHERE {' AND '.join(clauses)}"
            if clauses
            else ""
        )
        order_direction = "DESC" if newest_first else "ASC"

        query = f"""
            SELECT
                memory_id,
                namespace,
                key,
                metadata_json,
                tags_json,
                created_at,
                updated_at
            FROM {self._TABLE_NAME}
            {where_clause}
            ORDER BY updated_at {order_direction}, memory_id ASC
        """

        if resolved_limit is not None:
            query += " LIMIT ? OFFSET ?"
            parameters.extend(
                [resolved_limit, resolved_offset]
            )
        elif resolved_offset:
            query += " LIMIT -1 OFFSET ?"
            parameters.append(resolved_offset)

        with self._lock, self._connection() as connection:
            rows = connection.execute(
                query,
                tuple(parameters),
            ).fetchall()

        return [
            self._row_to_summary(row)
            for row in rows
        ]

    def search(
        self,
        query: str,
        namespace: str | None = None,
        limit: int = 50,
    ) -> list[LongTermMemorySummary]:
        """
        Search memory keys, namespaces, metadata, and tags.

        This is a lightweight SQLite text search. Semantic similarity search
        will be handled later by pgvector or the project's RAG vector store.

        Args:
            query:
                Case-insensitive search text.
            namespace:
                Optional namespace filter.
            limit:
                Maximum number of results.

        Returns:
            Matching memory summaries ordered by most recent update.
        """

        resolved_query = self._validate_required_string(
            query,
            "query",
        )
        resolved_namespace = (
            self._validate_required_string(
                namespace,
                "namespace",
            )
            if namespace is not None
            else None
        )
        resolved_limit = self._validate_limit(limit)

        if resolved_limit is None:
            raise ValueError("limit must be provided")

        search_pattern = f"%{resolved_query.lower()}%"

        clauses = [
            """
            (
                LOWER(namespace) LIKE ?
                OR LOWER(key) LIKE ?
                OR LOWER(metadata_json) LIKE ?
                OR LOWER(tags_json) LIKE ?
            )
            """
        ]
        parameters: list[Any] = [
            search_pattern,
            search_pattern,
            search_pattern,
            search_pattern,
        ]

        if resolved_namespace is not None:
            clauses.append("namespace = ?")
            parameters.append(resolved_namespace)

        parameters.append(resolved_limit)

        with self._lock, self._connection() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    memory_id,
                    namespace,
                    key,
                    metadata_json,
                    tags_json,
                    created_at,
                    updated_at
                FROM {self._TABLE_NAME}
                WHERE {' AND '.join(clauses)}
                ORDER BY updated_at DESC, memory_id ASC
                LIMIT ?
                """,
                tuple(parameters),
            ).fetchall()

        return [
            self._row_to_summary(row)
            for row in rows
        ]

    def get_latest(
        self,
        namespace: str,
        tag: str | None = None,
    ) -> LongTermMemoryEntry | None:
        """
        Retrieve the most recently updated memory in a namespace.

        Args:
            namespace: Logical memory grouping.
            tag: Optional exact tag filter.

        Returns:
            Latest memory entry, or None when no memory matches.
        """

        summaries = self.list_memories(
            namespace=namespace,
            tag=tag,
            limit=1,
            newest_first=True,
        )

        if not summaries:
            return None

        return self.get(summaries[0].memory_id)

    def count_by_namespace(
        self,
        namespace: str,
    ) -> int:
        """
        Count memories within a namespace.

        Args:
            namespace: Logical memory grouping.

        Returns:
            Number of matching memories.
        """

        resolved_namespace = self._validate_required_string(
            namespace,
            "namespace",
        )

        with self._lock, self._connection() as connection:
            row = connection.execute(
                f"""
                SELECT COUNT(*)
                FROM {self._TABLE_NAME}
                WHERE namespace = ?
                """,
                (resolved_namespace,),
            ).fetchone()

            return int(row[0]) if row else 0

    def clear_namespace(self, namespace: str) -> int:
        """
        Delete every memory in one namespace.

        Args:
            namespace: Namespace to clear.

        Returns:
            Number of deleted memories.
        """

        resolved_namespace = self._validate_required_string(
            namespace,
            "namespace",
        )

        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                f"""
                DELETE FROM {self._TABLE_NAME}
                WHERE namespace = ?
                """,
                (resolved_namespace,),
            )
            connection.commit()

            return cursor.rowcount

    def clear_all(self) -> int:
        """
        Delete every stored long-term memory.

        Returns:
            Number of deleted memories.
        """

        with self._lock, self._connection() as connection:
            current_count = connection.execute(
                f"""
                SELECT COUNT(*)
                FROM {self._TABLE_NAME}
                """
            ).fetchone()

            removed_count = (
                int(current_count[0])
                if current_count
                else 0
            )

            connection.execute(
                f"""
                DELETE FROM {self._TABLE_NAME}
                """
            )
            connection.commit()

            return removed_count

    def close(self) -> None:
        """
        Close the shared SQLite connection when one is active.

        A shared connection is used only when database_path is ":memory:".
        File-backed operations create and close connections automatically.
        """

        with self._lock:
            if self._shared_connection is not None:
                self._shared_connection.close()
                self._shared_connection = None

    def __enter__(self) -> LongTermMemory:
        """
        Enter a context-managed memory instance.

        Returns:
            Current LongTermMemory instance.
        """

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> None:
        """
        Close resources when leaving the context manager.
        """

        self.close()

    def _initialize_database(self) -> None:
        """
        Create the long-term memory table and indexes.
        """

        with self._lock, self._connection() as connection:
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._TABLE_NAME} (
                    memory_id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value_blob BLOB NOT NULL,
                    metadata_json TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(namespace, key)
                )
                """
            )

            connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS
                    idx_long_term_memory_namespace
                ON {self._TABLE_NAME}(namespace)
                """
            )

            connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS
                    idx_long_term_memory_updated_at
                ON {self._TABLE_NAME}(updated_at)
                """
            )

            connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS
                    idx_long_term_memory_namespace_updated
                ON {self._TABLE_NAME}(namespace, updated_at)
                """
            )

            connection.commit()

    def _prepare_database_directory(self) -> None:
        """
        Create the parent directory for a file-backed database.
        """

        if self._database_path == ":memory:":
            return

        database_file = Path(self._database_path)
        database_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _create_connection(self) -> sqlite3.Connection:
        """
        Create and configure a SQLite connection.

        Returns:
            Configured SQLite connection.
        """

        connection = sqlite3.connect(
            self._database_path,
            timeout=self._timeout_seconds,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row

        connection.execute("PRAGMA foreign_keys = ON")

        if self._database_path != ":memory:":
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")

        return connection

    @contextmanager
    def _connection(
        self,
    ) -> Generator[sqlite3.Connection, None, None]:
        """
        Yield a configured SQLite connection.

        For an in-memory database, the same connection must be reused because
        each separate SQLite in-memory connection creates a separate database.
        """

        if self._database_path == ":memory:":
            if self._shared_connection is None:
                self._shared_connection = self._create_connection()

            yield self._shared_connection
            return

        connection = self._create_connection()

        try:
            yield connection
        finally:
            connection.close()

    @classmethod
    def _row_to_entry(
        cls,
        row: sqlite3.Row,
    ) -> LongTermMemoryEntry:
        """
        Convert a SQLite row to a complete memory entry.

        Args:
            row: SQLite result row.

        Returns:
            LongTermMemoryEntry instance.
        """

        return LongTermMemoryEntry(
            memory_id=str(row["memory_id"]),
            namespace=str(row["namespace"]),
            key=str(row["key"]),
            value=cls._deserialize_value(
                row["value_blob"]
            ),
            created_at=_datetime_from_string(
                str(row["created_at"])
            ),
            updated_at=_datetime_from_string(
                str(row["updated_at"])
            ),
            metadata=cls._deserialize_json_object(
                str(row["metadata_json"])
            ),
            tags=cls._deserialize_json_list(
                str(row["tags_json"])
            ),
        )

    @classmethod
    def _row_to_summary(
        cls,
        row: sqlite3.Row,
    ) -> LongTermMemorySummary:
        """
        Convert a SQLite row to a lightweight memory summary.

        Args:
            row: SQLite result row.

        Returns:
            LongTermMemorySummary instance.
        """

        return LongTermMemorySummary(
            memory_id=str(row["memory_id"]),
            namespace=str(row["namespace"]),
            key=str(row["key"]),
            created_at=_datetime_from_string(
                str(row["created_at"])
            ),
            updated_at=_datetime_from_string(
                str(row["updated_at"])
            ),
            metadata=cls._deserialize_json_object(
                str(row["metadata_json"])
            ),
            tags=cls._deserialize_json_list(
                str(row["tags_json"])
            ),
        )

    @staticmethod
    def _serialize_value(value: Any) -> bytes:
        """
        Serialize an arbitrary Python value using pickle.

        Pickle is used because finance agents may return dataclasses, custom
        result objects, pandas-compatible structures, or other Python values.

        Only data created by this trusted application should be loaded from
        this database. Pickle data from untrusted external sources must never
        be deserialized.

        Args:
            value: Python value to serialize.

        Returns:
            Serialized bytes.

        Raises:
            TypeError:
                If the value cannot be serialized.
        """

        try:
            return pickle.dumps(
                deepcopy(value),
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        except (
            pickle.PickleError,
            TypeError,
            AttributeError,
            RecursionError,
        ) as exc:
            raise TypeError(
                "value could not be serialized"
            ) from exc

    @staticmethod
    def _deserialize_value(value_blob: bytes) -> Any:
        """
        Deserialize a trusted Python value.

        Args:
            value_blob: Pickled value from the trusted application database.

        Returns:
            Deep-copied Python value.
        """

        try:
            value = pickle.loads(value_blob)
        except (
            pickle.PickleError,
            EOFError,
            AttributeError,
            ImportError,
            IndexError,
        ) as exc:
            raise ValueError(
                "stored memory value could not be deserialized"
            ) from exc

        return deepcopy(value)

    @staticmethod
    def _serialize_json(
        value: Any,
        field_name: str,
    ) -> str:
        """
        Serialize a JSON-compatible value.

        Args:
            value: Value to serialize.
            field_name: Field name used in validation errors.

        Returns:
            JSON text.

        Raises:
            TypeError:
                If the value is not JSON serializable.
        """

        try:
            return json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
            )
        except (
            TypeError,
            ValueError,
            OverflowError,
        ) as exc:
            raise TypeError(
                f"{field_name} must be JSON serializable"
            ) from exc

    @staticmethod
    def _deserialize_json_object(
        value: str,
    ) -> dict[str, Any]:
        """
        Deserialize a JSON object.

        Args:
            value: JSON text.

        Returns:
            Dictionary value.
        """

        parsed_value = json.loads(value)

        if not isinstance(parsed_value, dict):
            raise ValueError(
                "stored metadata is not a JSON object"
            )

        return parsed_value

    @staticmethod
    def _deserialize_json_list(
        value: str,
    ) -> list[str]:
        """
        Deserialize a JSON tag list.

        Args:
            value: JSON text.

        Returns:
            String tag list.
        """

        parsed_value = json.loads(value)

        if not isinstance(parsed_value, list):
            raise ValueError(
                "stored tags are not a JSON list"
            )

        if not all(
            isinstance(item, str)
            for item in parsed_value
        ):
            raise ValueError(
                "stored tags contain invalid values"
            )

        return parsed_value

    @staticmethod
    def _validate_database_path(
        database_path: str | Path,
    ) -> str:
        """
        Validate and normalize a database path.

        Args:
            database_path: SQLite path.

        Returns:
            Normalized path string.
        """

        if isinstance(database_path, Path):
            database_path = str(database_path)

        if not isinstance(database_path, str):
            raise ValueError(
                "database_path must be a non-empty string or Path"
            )

        normalized_path = database_path.strip()

        if not normalized_path:
            raise ValueError(
                "database_path must be a non-empty string or Path"
            )

        return normalized_path

    @staticmethod
    def _validate_required_string(
        value: Any,
        field_name: str,
    ) -> str:
        """
        Validate and normalize a required string.

        Args:
            value: Value to validate.
            field_name: Field name used in error messages.

        Returns:
            Normalized non-empty string.
        """

        if not isinstance(value, str):
            raise ValueError(
                f"{field_name} must be a non-empty string"
            )

        normalized_value = value.strip()

        if not normalized_value:
            raise ValueError(
                f"{field_name} must be a non-empty string"
            )

        return normalized_value

    @classmethod
    def _validate_metadata(
        cls,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """
        Validate memory metadata.

        Args:
            metadata: Metadata dictionary or None.

        Returns:
            Deep-copied metadata dictionary.
        """

        if metadata is None:
            return {}

        if not isinstance(metadata, dict):
            raise TypeError("metadata must be a dictionary")

        resolved_metadata = deepcopy(metadata)

        cls._serialize_json(
            resolved_metadata,
            "metadata",
        )

        return resolved_metadata

    @classmethod
    def _validate_tags(
        cls,
        tags: (
            list[str]
            | tuple[str, ...]
            | set[str]
            | None
        ),
    ) -> list[str]:
        """
        Validate, normalize, deduplicate, and sort memory tags.

        Args:
            tags: Collection of string tags or None.

        Returns:
            Sorted unique tag list.
        """

        if tags is None:
            return []

        if not isinstance(tags, (list, tuple, set)):
            raise TypeError(
                "tags must be a list, tuple, or set of strings"
            )

        normalized_tags: set[str] = set()

        for tag in tags:
            normalized_tag = cls._validate_required_string(
                tag,
                "tag",
            )
            normalized_tags.add(normalized_tag)

        return sorted(normalized_tags)

    @staticmethod
    def _validate_limit(
        limit: int | None,
    ) -> int | None:
        """
        Validate an optional result limit.

        Args:
            limit: Optional maximum result count.

        Returns:
            Validated limit.
        """

        if limit is None:
            return None

        if isinstance(limit, bool) or not isinstance(limit, int):
            raise ValueError(
                "limit must be a positive integer or None"
            )

        if limit <= 0:
            raise ValueError(
                "limit must be a positive integer or None"
            )

        return limit

    @staticmethod
    def _validate_offset(offset: int) -> int:
        """
        Validate a result offset.

        Args:
            offset: Number of records to skip.

        Returns:
            Validated offset.
        """

        if isinstance(offset, bool) or not isinstance(offset, int):
            raise ValueError(
                "offset must be a non-negative integer"
            )

        if offset < 0:
            raise ValueError(
                "offset must be a non-negative integer"
            )

        return offset
