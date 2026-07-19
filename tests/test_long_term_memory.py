
"""
Tests for the Finance Agentic AI long-term memory module.

These tests verify:

- SQLite database initialization
- Persistent storage across application instances
- Memory creation and retrieval
- Namespace and key retrieval
- Arbitrary Python object storage
- Metadata and tag handling
- Updates and upserts
- Renaming
- Searching and filtering
- Pagination and ordering
- Memory deletion
- Namespace cleanup
- Context-manager behaviour
- Validation and error handling
- Deep-copy protection
- Realistic FP&A memory workflows
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from collections.abc import Generator
from typing import Any

import pytest

from src.memory.long_term_memory import (
    LongTermMemory,
    LongTermMemoryEntry,
    LongTermMemorySummary,
    _datetime_from_string,
    _datetime_to_string,
)


@dataclass
class SampleFinanceResult:
    """
    Sample serializable finance-agent result used in tests.
    """

    status: str
    actual_revenue: float
    budget_revenue: float
    revenue_variance: float


@pytest.fixture
def memory(
    tmp_path: Path,
) -> Generator[LongTermMemory, None, None]:
    """
    Create an isolated file-backed long-term memory instance.
    """

    database_path = tmp_path / "long_term_memory.db"
    instance = LongTermMemory(database_path=database_path)

    yield instance

    instance.close()


@pytest.fixture
def in_memory_database(
) -> Generator[LongTermMemory, None, None]:
    """
    Create an isolated SQLite in-memory database.
    """

    instance = LongTermMemory(database_path=":memory:")

    yield instance

    instance.close()


def save_sample_memory(
    memory: LongTermMemory,
    *,
    namespace: str = "reports",
    key: str = "chennai-2026-06",
    value: Any = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> str:
    """
    Store a reusable sample memory for tests.
    """

    resolved_value = (
        value
        if value is not None
        else {
            "status": "REVIEW",
            "revenue_variance": -200_000,
        }
    )

    resolved_metadata = (
        metadata
        if metadata is not None
        else {
            "branch": "Chennai",
            "period": "2026-06",
        }
    )

    resolved_tags = (
        tags
        if tags is not None
        else [
            "monthly",
            "management-report",
            "variance",
        ]
    )

    return memory.save(
        namespace=namespace,
        key=key,
        value=resolved_value,
        metadata=resolved_metadata,
        tags=resolved_tags,
    )


def test_constructor_creates_database_file(
    tmp_path: Path,
):
    """Constructor should create the SQLite database file."""

    database_path = (
        tmp_path
        / "nested"
        / "memory"
        / "finance_memory.db"
    )

    memory = LongTermMemory(
        database_path=database_path
    )

    try:
        assert database_path.exists()
        assert database_path.is_file()
        assert memory.database_path == str(database_path)
    finally:
        memory.close()


def test_constructor_creates_parent_directories(
    tmp_path: Path,
):
    """Missing database parent directories should be created."""

    parent_directory = (
        tmp_path
        / "finance"
        / "memory"
        / "database"
    )
    database_path = (
        parent_directory
        / "long_term_memory.db"
    )

    memory = LongTermMemory(
        database_path=database_path
    )

    try:
        assert parent_directory.exists()
        assert parent_directory.is_dir()
    finally:
        memory.close()


def test_constructor_accepts_string_database_path(
    tmp_path: Path,
):
    """A string database path should be accepted."""

    database_path = str(
        tmp_path / "finance_memory.db"
    )

    memory = LongTermMemory(
        database_path=database_path
    )

    try:
        assert memory.database_path == database_path
    finally:
        memory.close()


def test_constructor_accepts_memory_database():
    """The special SQLite in-memory path should work."""

    memory = LongTermMemory(database_path=":memory:")

    try:
        assert memory.database_path == ":memory:"
        assert memory.count == 0
    finally:
        memory.close()


@pytest.mark.parametrize(
    "database_path",
    [
        "",
        " ",
        "   ",
        None,
        100,
    ],
)
def test_constructor_rejects_invalid_database_path(
    database_path,
):
    """Database path must be a non-empty string or Path."""

    with pytest.raises(
        ValueError,
        match=(
            "database_path must be a non-empty "
            "string or Path"
        ),
    ):
        LongTermMemory(database_path=database_path)


@pytest.mark.parametrize(
    "timeout_seconds",
    [
        0,
        -1,
        -30,
    ],
)
def test_constructor_rejects_invalid_timeout(
    tmp_path: Path,
    timeout_seconds,
):
    """SQLite timeout must be positive."""

    with pytest.raises(
        ValueError,
        match="timeout_seconds must be positive",
    ):
        LongTermMemory(
            database_path=tmp_path / "memory.db",
            timeout_seconds=timeout_seconds,
        )


def test_database_contains_expected_table(
    tmp_path: Path,
):
    """Initialization should create the memory table."""

    database_path = tmp_path / "memory.db"
    memory = LongTermMemory(database_path)

    try:
        with sqlite3.connect(database_path) as connection:
            result = connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                  AND name = 'long_term_memory'
                """
            ).fetchone()

        assert result is not None
        assert result[0] == "long_term_memory"
    finally:
        memory.close()


def test_database_contains_expected_indexes(
    tmp_path: Path,
):
    """Initialization should create query indexes."""

    database_path = tmp_path / "memory.db"
    memory = LongTermMemory(database_path)

    try:
        with sqlite3.connect(database_path) as connection:
            rows = connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'index'
                  AND tbl_name = 'long_term_memory'
                """
            ).fetchall()

        index_names = {
            row[0]
            for row in rows
        }

        assert "idx_long_term_memory_namespace" in (
            index_names
        )
        assert "idx_long_term_memory_updated_at" in (
            index_names
        )
        assert (
            "idx_long_term_memory_namespace_updated"
            in index_names
        )
    finally:
        memory.close()


def test_count_is_zero_for_new_database(
    memory: LongTermMemory,
):
    """A new database should contain no memories."""

    assert memory.count == 0


def test_save_creates_memory(
    memory: LongTermMemory,
):
    """save() should persist a new memory."""

    memory_id = save_sample_memory(memory)

    assert isinstance(memory_id, str)
    assert memory_id
    assert memory.count == 1
    assert memory.exists(memory_id) is True


def test_save_accepts_custom_memory_id(
    memory: LongTermMemory,
):
    """A custom memory identifier should be retained."""

    memory_id = memory.save(
        namespace="reports",
        key="june-report",
        value={"status": "PASS"},
        memory_id="memory-001",
    )

    assert memory_id == "memory-001"
    assert memory.exists("memory-001") is True


def test_save_strips_string_fields(
    memory: LongTermMemory,
):
    """Namespace, key, memory ID, and tags should be normalized."""

    memory_id = memory.save(
        namespace="  reports  ",
        key="  june-report  ",
        value={"status": "PASS"},
        memory_id="  memory-001  ",
        tags=[
            "  monthly  ",
            "  report  ",
        ],
    )

    entry = memory.get(memory_id)

    assert memory_id == "memory-001"
    assert entry.namespace == "reports"
    assert entry.key == "june-report"
    assert entry.tags == [
        "monthly",
        "report",
    ]


def test_save_stores_complete_entry(
    memory: LongTermMemory,
):
    """Saved fields should be returned through get()."""

    memory_id = save_sample_memory(memory)
    entry = memory.get(memory_id)

    assert isinstance(entry, LongTermMemoryEntry)
    assert entry.memory_id == memory_id
    assert entry.namespace == "reports"
    assert entry.key == "chennai-2026-06"
    assert entry.value == {
        "status": "REVIEW",
        "revenue_variance": -200_000,
    }
    assert entry.metadata == {
        "branch": "Chennai",
        "period": "2026-06",
    }
    assert entry.tags == [
        "management-report",
        "monthly",
        "variance",
    ]
    assert entry.created_at.tzinfo is not None
    assert entry.updated_at.tzinfo is not None


def test_save_deep_copies_value_and_metadata(
    memory: LongTermMemory,
):
    """Changing source objects should not alter stored memory."""

    value = {
        "summary": {
            "status": "REVIEW",
        }
    }
    metadata = {
        "filters": {
            "branch": "Chennai",
        }
    }
    tags = [
        "monthly",
        "variance",
    ]

    memory_id = memory.save(
        namespace="reports",
        key="june",
        value=value,
        metadata=metadata,
        tags=tags,
    )

    value["summary"]["status"] = "PASS"
    metadata["filters"]["branch"] = "Changed"
    tags.append("changed")

    entry = memory.get(memory_id)

    assert entry.value == {
        "summary": {
            "status": "REVIEW",
        }
    }
    assert entry.metadata == {
        "filters": {
            "branch": "Chennai",
        }
    }
    assert entry.tags == [
        "monthly",
        "variance",
    ]


def test_get_returns_independent_values(
    memory: LongTermMemory,
):
    """Changing a retrieved value should not mutate the database."""

    memory_id = save_sample_memory(memory)

    first_entry = memory.get(memory_id)
    first_entry.value["status"] = "PASS"
    first_entry.metadata["branch"] = "Changed"
    first_entry.tags.append("changed")

    second_entry = memory.get(memory_id)

    assert second_entry.value["status"] == "REVIEW"
    assert second_entry.metadata["branch"] == "Chennai"
    assert "changed" not in second_entry.tags


def test_save_uses_empty_metadata_and_tags_by_default(
    memory: LongTermMemory,
):
    """Omitted metadata and tags should become empty collections."""

    memory_id = memory.save(
        namespace="questions",
        key="question-001",
        value="Analyze June.",
    )

    entry = memory.get(memory_id)

    assert entry.metadata == {}
    assert entry.tags == []


def test_save_deduplicates_and_sorts_tags(
    memory: LongTermMemory,
):
    """Tags should be normalized, deduplicated, and sorted."""

    memory_id = memory.save(
        namespace="reports",
        key="june",
        value="June report",
        tags=[
            "variance",
            "monthly",
            "variance",
            "report",
        ],
    )

    entry = memory.get(memory_id)

    assert entry.tags == [
        "monthly",
        "report",
        "variance",
    ]


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("namespace", ""),
        ("namespace", " "),
        ("namespace", None),
        ("namespace", 100),
        ("key", ""),
        ("key", " "),
        ("key", None),
        ("key", 100),
    ],
)
def test_save_rejects_invalid_required_fields(
    memory: LongTermMemory,
    field_name,
    field_value,
):
    """Namespace and key must be non-empty strings."""

    arguments = {
        "namespace": "reports",
        "key": "june",
        "value": {"status": "PASS"},
    }
    arguments[field_name] = field_value

    with pytest.raises(
        ValueError,
        match=(
            f"{field_name} must be a "
            "non-empty string"
        ),
    ):
        memory.save(**arguments)


@pytest.mark.parametrize(
    "memory_id",
    [
        "",
        " ",
        None,
        100,
    ],
)
def test_save_rejects_invalid_custom_memory_id(
    memory: LongTermMemory,
    memory_id,
):
    """Custom memory IDs must be non-empty strings."""

    if memory_id is None:
        generated_id = memory.save(
            namespace="reports",
            key="june",
            value="report",
            memory_id=memory_id,
        )

        assert isinstance(generated_id, str)
        assert generated_id
        return

    with pytest.raises(
        ValueError,
        match="memory_id must be a non-empty string",
    ):
        memory.save(
            namespace="reports",
            key="june",
            value="report",
            memory_id=memory_id,
        )


@pytest.mark.parametrize(
    "metadata",
    [
        "invalid",
        100,
        [],
        ("invalid",),
    ],
)
def test_save_rejects_non_dictionary_metadata(
    memory: LongTermMemory,
    metadata,
):
    """Metadata must be a dictionary."""

    with pytest.raises(
        TypeError,
        match="metadata must be a dictionary",
    ):
        memory.save(
            namespace="reports",
            key="june",
            value="report",
            metadata=metadata,
        )


def test_save_rejects_non_json_serializable_metadata(
    memory: LongTermMemory,
):
    """Metadata must be JSON serializable."""

    with pytest.raises(
        TypeError,
        match="metadata must be JSON serializable",
    ):
        memory.save(
            namespace="reports",
            key="june",
            value="report",
            metadata={
                "unsupported": object(),
            },
        )


@pytest.mark.parametrize(
    "tags",
    [
        "monthly",
        100,
        {"tag": "monthly"},
    ],
)
def test_save_rejects_invalid_tag_collection(
    memory: LongTermMemory,
    tags,
):
    """Tags must use a supported collection."""

    with pytest.raises(
        TypeError,
        match=(
            "tags must be a list, tuple, "
            "or set of strings"
        ),
    ):
        memory.save(
            namespace="reports",
            key="june",
            value="report",
            tags=tags,
        )


@pytest.mark.parametrize(
    "tags",
    [
        [""],
        [" "],
        [None],
        [100],
        ["monthly", ""],
    ],
)
def test_save_rejects_invalid_tag_values(
    memory: LongTermMemory,
    tags,
):
    """Every tag must be a non-empty string."""

    with pytest.raises(
        ValueError,
        match="tag must be a non-empty string",
    ):
        memory.save(
            namespace="reports",
            key="june",
            value="report",
            tags=tags,
        )


def test_save_rejects_duplicate_namespace_and_key(
    memory: LongTermMemory,
):
    """Duplicate business keys should require overwrite."""

    save_sample_memory(memory)

    with pytest.raises(
        KeyError,
        match="memory already exists",
    ):
        save_sample_memory(memory)


def test_save_overwrite_updates_existing_memory(
    memory: LongTermMemory,
):
    """overwrite=True should replace an existing memory."""

    first_memory_id = save_sample_memory(memory)

    second_memory_id = memory.save(
        namespace="reports",
        key="chennai-2026-06",
        value={
            "status": "PASS",
            "revenue_variance": -100_000,
        },
        metadata={
            "branch": "Chennai",
            "period": "2026-06",
            "reviewed": True,
        },
        tags=[
            "reviewed",
            "monthly",
        ],
        overwrite=True,
    )

    entry = memory.get(first_memory_id)

    assert second_memory_id == first_memory_id
    assert memory.count == 1
    assert entry.value == {
        "status": "PASS",
        "revenue_variance": -100_000,
    }
    assert entry.metadata["reviewed"] is True
    assert entry.tags == [
        "monthly",
        "reviewed",
    ]


def test_save_overwrite_preserves_created_at(
    memory: LongTermMemory,
):
    """Overwriting should retain the original creation time."""

    memory_id = save_sample_memory(memory)
    original_entry = memory.get(memory_id)

    memory.save(
        namespace="reports",
        key="chennai-2026-06",
        value={"status": "PASS"},
        overwrite=True,
    )

    updated_entry = memory.get(memory_id)

    assert (
        updated_entry.created_at
        == original_entry.created_at
    )
    assert (
        updated_entry.updated_at
        >= original_entry.updated_at
    )


def test_save_rejects_duplicate_memory_id(
    memory: LongTermMemory,
):
    """Memory IDs must be globally unique."""

    memory.save(
        namespace="reports",
        key="june",
        value="June report",
        memory_id="memory-001",
    )

    with pytest.raises(
        KeyError,
        match="memory_id already exists: memory-001",
    ):
        memory.save(
            namespace="reports",
            key="july",
            value="July report",
            memory_id="memory-001",
        )


def test_upsert_creates_new_memory(
    memory: LongTermMemory,
):
    """upsert() should create a missing memory."""

    memory_id = memory.upsert(
        namespace="reports",
        key="june",
        value={"status": "REVIEW"},
    )

    assert memory.exists(memory_id) is True
    assert memory.count == 1


def test_upsert_updates_existing_memory(
    memory: LongTermMemory,
):
    """upsert() should update an existing business key."""

    first_memory_id = memory.upsert(
        namespace="reports",
        key="june",
        value={"status": "REVIEW"},
    )

    second_memory_id = memory.upsert(
        namespace="reports",
        key="june",
        value={"status": "PASS"},
        metadata={"reviewed": True},
        tags=["reviewed"],
    )

    entry = memory.get(first_memory_id)

    assert second_memory_id == first_memory_id
    assert memory.count == 1
    assert entry.value == {"status": "PASS"}
    assert entry.metadata == {"reviewed": True}
    assert entry.tags == ["reviewed"]


def test_get_unknown_memory_raises_key_error(
    memory: LongTermMemory,
):
    """get() should reject an unknown memory ID."""

    with pytest.raises(
        KeyError,
        match="long-term memory not found: missing",
    ):
        memory.get("missing")


@pytest.mark.parametrize(
    "memory_id",
    [
        "",
        " ",
        None,
        100,
    ],
)
def test_get_rejects_invalid_memory_id(
    memory: LongTermMemory,
    memory_id,
):
    """get() should validate memory identifiers."""

    with pytest.raises(
        ValueError,
        match="memory_id must be a non-empty string",
    ):
        memory.get(memory_id)


def test_get_by_key_returns_matching_memory(
    memory: LongTermMemory,
):
    """Namespace and key should retrieve the memory."""

    memory_id = save_sample_memory(memory)

    entry = memory.get_by_key(
        namespace="reports",
        key="chennai-2026-06",
    )

    assert entry.memory_id == memory_id
    assert entry.value["status"] == "REVIEW"


def test_get_by_key_unknown_memory_raises_key_error(
    memory: LongTermMemory,
):
    """Unknown namespace and key should raise KeyError."""

    with pytest.raises(
        KeyError,
        match="long-term memory not found",
    ):
        memory.get_by_key(
            namespace="reports",
            key="missing",
        )


def test_exists_returns_false_for_missing_memory(
    memory: LongTermMemory,
):
    """exists() should return False for an unknown ID."""

    assert memory.exists("missing") is False


def test_key_exists_returns_true_for_existing_key(
    memory: LongTermMemory,
):
    """key_exists() should detect stored business keys."""

    save_sample_memory(memory)

    assert memory.key_exists(
        "reports",
        "chennai-2026-06",
    ) is True


def test_key_exists_returns_false_for_missing_key(
    memory: LongTermMemory,
):
    """key_exists() should return False for missing keys."""

    assert memory.key_exists(
        "reports",
        "missing",
    ) is False


def test_update_changes_existing_memory(
    memory: LongTermMemory,
):
    """update() should replace a stored value."""

    memory_id = save_sample_memory(memory)
    original_entry = memory.get(memory_id)

    updated_entry = memory.update(
        memory_id=memory_id,
        value={
            "status": "PASS",
            "revenue_variance": -50_000,
        },
        metadata={
            "branch": "Chennai",
            "period": "2026-06",
            "approved": True,
        },
        tags=[
            "approved",
            "monthly",
        ],
    )

    assert updated_entry.memory_id == memory_id
    assert updated_entry.value == {
        "status": "PASS",
        "revenue_variance": -50_000,
    }
    assert updated_entry.metadata["approved"] is True
    assert updated_entry.tags == [
        "approved",
        "monthly",
    ]
    assert (
        updated_entry.created_at
        == original_entry.created_at
    )
    assert (
        updated_entry.updated_at
        >= original_entry.updated_at
    )


def test_update_preserves_metadata_and_tags_by_default(
    memory: LongTermMemory,
):
    """Omitted metadata and tags should remain unchanged."""

    memory_id = save_sample_memory(memory)

    updated_entry = memory.update(
        memory_id=memory_id,
        value={"status": "PASS"},
    )

    assert updated_entry.metadata == {
        "branch": "Chennai",
        "period": "2026-06",
    }
    assert updated_entry.tags == [
        "management-report",
        "monthly",
        "variance",
    ]


def test_update_can_clear_metadata_and_tags(
    memory: LongTermMemory,
):
    """Preservation flags should allow clearing fields."""

    memory_id = save_sample_memory(memory)

    updated_entry = memory.update(
        memory_id=memory_id,
        value={"status": "PASS"},
        preserve_metadata=False,
        preserve_tags=False,
    )

    assert updated_entry.metadata == {}
    assert updated_entry.tags == []


def test_update_unknown_memory_raises_key_error(
    memory: LongTermMemory,
):
    """Updating an unknown memory should fail."""

    with pytest.raises(
        KeyError,
        match="long-term memory not found: missing",
    ):
        memory.update(
            memory_id="missing",
            value="updated",
        )


def test_update_metadata_merges_by_default(
    memory: LongTermMemory,
):
    """update_metadata() should merge metadata by default."""

    memory_id = save_sample_memory(memory)

    updated_metadata = memory.update_metadata(
        memory_id=memory_id,
        metadata={
            "reviewed": True,
            "period": "2026-06-final",
        },
    )

    assert updated_metadata == {
        "branch": "Chennai",
        "period": "2026-06-final",
        "reviewed": True,
    }


def test_update_metadata_can_replace_metadata(
    memory: LongTermMemory,
):
    """merge=False should replace all metadata."""

    memory_id = save_sample_memory(memory)

    updated_metadata = memory.update_metadata(
        memory_id=memory_id,
        metadata={
            "reviewed": True,
        },
        merge=False,
    )

    assert updated_metadata == {
        "reviewed": True,
    }


def test_rename_changes_namespace_and_key(
    memory: LongTermMemory,
):
    """rename() should update namespace and key."""

    memory_id = save_sample_memory(memory)

    renamed_entry = memory.rename(
        memory_id=memory_id,
        namespace="approved-reports",
        key="chennai-june-2026-final",
    )

    assert renamed_entry.memory_id == memory_id
    assert renamed_entry.namespace == (
        "approved-reports"
    )
    assert renamed_entry.key == (
        "chennai-june-2026-final"
    )
    assert memory.key_exists(
        "approved-reports",
        "chennai-june-2026-final",
    ) is True


def test_rename_can_change_only_namespace(
    memory: LongTermMemory,
):
    """rename() should support a namespace-only change."""

    memory_id = save_sample_memory(memory)

    renamed_entry = memory.rename(
        memory_id=memory_id,
        namespace="archived-reports",
    )

    assert renamed_entry.namespace == (
        "archived-reports"
    )
    assert renamed_entry.key == "chennai-2026-06"


def test_rename_can_change_only_key(
    memory: LongTermMemory,
):
    """rename() should support a key-only change."""

    memory_id = save_sample_memory(memory)

    renamed_entry = memory.rename(
        memory_id=memory_id,
        key="chennai-june-final",
    )

    assert renamed_entry.namespace == "reports"
    assert renamed_entry.key == "chennai-june-final"


def test_rename_requires_namespace_or_key(
    memory: LongTermMemory,
):
    """rename() should require at least one changed field."""

    memory_id = save_sample_memory(memory)

    with pytest.raises(
        ValueError,
        match="namespace or key must be provided",
    ):
        memory.rename(memory_id)


def test_rename_rejects_duplicate_business_key(
    memory: LongTermMemory,
):
    """rename() should not create duplicate namespace keys."""

    first_memory_id = memory.save(
        namespace="reports",
        key="june",
        value="June report",
    )
    memory.save(
        namespace="reports",
        key="july",
        value="July report",
    )

    with pytest.raises(
        KeyError,
        match="memory already exists",
    ):
        memory.rename(
            memory_id=first_memory_id,
            key="july",
        )


def test_delete_removes_existing_memory(
    memory: LongTermMemory,
):
    """delete() should remove an existing memory."""

    memory_id = save_sample_memory(memory)

    deleted = memory.delete(memory_id)

    assert deleted is True
    assert memory.exists(memory_id) is False
    assert memory.count == 0


def test_delete_returns_false_for_missing_memory(
    memory: LongTermMemory,
):
    """delete() should return False for unknown IDs."""

    assert memory.delete("missing") is False


def test_delete_by_key_removes_matching_memory(
    memory: LongTermMemory,
):
    """delete_by_key() should delete a business key."""

    save_sample_memory(memory)

    deleted = memory.delete_by_key(
        namespace="reports",
        key="chennai-2026-06",
    )

    assert deleted is True
    assert memory.count == 0


def test_delete_by_key_returns_false_for_missing_key(
    memory: LongTermMemory,
):
    """delete_by_key() should return False when absent."""

    deleted = memory.delete_by_key(
        namespace="reports",
        key="missing",
    )

    assert deleted is False


def test_list_memories_returns_summaries(
    memory: LongTermMemory,
):
    """list_memories() should return lightweight summaries."""

    save_sample_memory(
        memory,
        key="chennai-2026-06",
    )
    save_sample_memory(
        memory,
        key="chennai-2026-07",
        metadata={
            "branch": "Chennai",
            "period": "2026-07",
        },
    )

    results = memory.list_memories()

    assert len(results) == 2
    assert all(
        isinstance(item, LongTermMemorySummary)
        for item in results
    )
    assert {
        item.key
        for item in results
    } == {
        "chennai-2026-06",
        "chennai-2026-07",
    }


def test_list_memories_filters_by_namespace(
    memory: LongTermMemory,
):
    """Namespace filtering should isolate memory groups."""

    memory.save(
        namespace="reports",
        key="june-report",
        value="June report",
    )
    memory.save(
        namespace="questions",
        key="question-001",
        value="Analyze June.",
    )

    results = memory.list_memories(
        namespace="reports"
    )

    assert len(results) == 1
    assert results[0].namespace == "reports"
    assert results[0].key == "june-report"


def test_list_memories_filters_by_tag(
    memory: LongTermMemory,
):
    """Tag filtering should use exact tags."""

    memory.save(
        namespace="reports",
        key="june",
        value="June report",
        tags=[
            "monthly",
            "approved",
        ],
    )
    memory.save(
        namespace="reports",
        key="july",
        value="July report",
        tags=[
            "monthly",
            "review",
        ],
    )

    approved_results = memory.list_memories(
        tag="approved"
    )

    assert len(approved_results) == 1
    assert approved_results[0].key == "june"


def test_list_memories_filters_namespace_and_tag(
    memory: LongTermMemory,
):
    """Namespace and tag filters should work together."""

    memory.save(
        namespace="reports",
        key="june",
        value="June report",
        tags=["approved"],
    )
    memory.save(
        namespace="reports",
        key="july",
        value="July report",
        tags=["review"],
    )
    memory.save(
        namespace="forecasts",
        key="june",
        value="June forecast",
        tags=["approved"],
    )

    results = memory.list_memories(
        namespace="reports",
        tag="approved",
    )

    assert len(results) == 1
    assert results[0].namespace == "reports"
    assert results[0].key == "june"


def test_list_memories_supports_limit_and_offset(
    memory: LongTermMemory,
):
    """Pagination should apply limit and offset."""

    for index in range(5):
        memory.save(
            namespace="reports",
            key=f"report-{index}",
            value=index,
        )

    first_page = memory.list_memories(
        limit=2,
        offset=0,
        newest_first=False,
    )
    second_page = memory.list_memories(
        limit=2,
        offset=2,
        newest_first=False,
    )

    assert len(first_page) == 2
    assert len(second_page) == 2

    first_page_ids = {
        item.memory_id
        for item in first_page
    }
    second_page_ids = {
        item.memory_id
        for item in second_page
    }

    assert first_page_ids.isdisjoint(
        second_page_ids
    )


def test_list_memories_supports_offset_without_limit(
    memory: LongTermMemory,
):
    """Offset should work even when limit is omitted."""

    for index in range(4):
        memory.save(
            namespace="reports",
            key=f"report-{index}",
            value=index,
        )

    results = memory.list_memories(
        offset=2,
        newest_first=False,
    )

    assert len(results) == 2


@pytest.mark.parametrize(
    "limit",
    [
        0,
        -1,
        "10",
        1.5,
        True,
    ],
)
def test_list_memories_rejects_invalid_limit(
    memory: LongTermMemory,
    limit,
):
    """Result limit must be a positive integer."""

    with pytest.raises(
        ValueError,
        match=(
            "limit must be a positive "
            "integer or None"
        ),
    ):
        memory.list_memories(limit=limit)


@pytest.mark.parametrize(
    "offset",
    [
        -1,
        "1",
        1.5,
        True,
    ],
)
def test_list_memories_rejects_invalid_offset(
    memory: LongTermMemory,
    offset,
):
    """Result offset must be a non-negative integer."""

    with pytest.raises(
        ValueError,
        match=(
            "offset must be a non-negative integer"
        ),
    ):
        memory.list_memories(offset=offset)


def test_search_finds_key_text(
    memory: LongTermMemory,
):
    """Search should find matching memory keys."""

    memory.save(
        namespace="reports",
        key="chennai-june-variance",
        value="June report",
    )
    memory.save(
        namespace="reports",
        key="bangalore-july-forecast",
        value="July forecast",
    )

    results = memory.search("variance")

    assert len(results) == 1
    assert results[0].key == (
        "chennai-june-variance"
    )


def test_search_is_case_insensitive(
    memory: LongTermMemory,
):
    """Search should not depend on text casing."""

    memory.save(
        namespace="Management-Reports",
        key="Chennai-June",
        value="June report",
        metadata={
            "Status": "REVIEW",
        },
        tags=["Monthly"],
    )

    results = memory.search("chennai")

    assert len(results) == 1
    assert results[0].key == "Chennai-June"


def test_search_finds_metadata_text(
    memory: LongTermMemory,
):
    """Search should inspect metadata JSON."""

    memory.save(
        namespace="reports",
        key="june",
        value="June report",
        metadata={
            "branch": "Chennai",
            "status": "REVIEW",
        },
    )

    results = memory.search("Chennai")

    assert len(results) == 1
    assert results[0].key == "june"


def test_search_finds_tag_text(
    memory: LongTermMemory,
):
    """Search should inspect stored tags."""

    memory.save(
        namespace="reports",
        key="june",
        value="June report",
        tags=["management-report"],
    )

    results = memory.search("management-report")

    assert len(results) == 1
    assert results[0].key == "june"


def test_search_filters_by_namespace(
    memory: LongTermMemory,
):
    """Search should optionally restrict the namespace."""

    memory.save(
        namespace="reports",
        key="june-variance",
        value="June report",
    )
    memory.save(
        namespace="forecasts",
        key="june-variance",
        value="June forecast",
    )

    results = memory.search(
        query="variance",
        namespace="reports",
    )

    assert len(results) == 1
    assert results[0].namespace == "reports"


def test_search_returns_empty_list_when_no_match(
    memory: LongTermMemory,
):
    """Search should return an empty list when unmatched."""

    save_sample_memory(memory)

    assert memory.search("unrelated-text") == []


@pytest.mark.parametrize(
    "query",
    [
        "",
        " ",
        None,
        100,
    ],
)
def test_search_rejects_invalid_query(
    memory: LongTermMemory,
    query,
):
    """Search query must be a non-empty string."""

    with pytest.raises(
        ValueError,
        match="query must be a non-empty string",
    ):
        memory.search(query)


def test_get_latest_returns_latest_namespace_memory(
    memory: LongTermMemory,
):
    """get_latest() should return the most recent match."""

    first_memory_id = memory.save(
        namespace="reports",
        key="june",
        value="June report",
    )
    second_memory_id = memory.save(
        namespace="reports",
        key="july",
        value="July report",
    )

    with memory._lock, memory._connection() as connection:
        connection.execute(
            """
            UPDATE long_term_memory
            SET updated_at = ?
            WHERE memory_id = ?
            """,
            (
                "2026-06-01T00:00:00+00:00",
                first_memory_id,
            ),
        )
        connection.execute(
            """
            UPDATE long_term_memory
            SET updated_at = ?
            WHERE memory_id = ?
            """,
            (
                "2026-07-01T00:00:00+00:00",
                second_memory_id,
            ),
        )
        connection.commit()

    latest = memory.get_latest("reports")

    assert latest is not None
    assert latest.memory_id == second_memory_id
    assert latest.key == "july"


def test_get_latest_filters_by_tag(
    memory: LongTermMemory,
):
    """get_latest() should support tag filtering."""

    memory.save(
        namespace="reports",
        key="june-approved",
        value="June report",
        tags=["approved"],
    )
    memory.save(
        namespace="reports",
        key="july-review",
        value="July report",
        tags=["review"],
    )

    latest = memory.get_latest(
        namespace="reports",
        tag="approved",
    )

    assert latest is not None
    assert latest.key == "june-approved"


def test_get_latest_returns_none_when_no_match(
    memory: LongTermMemory,
):
    """get_latest() should return None when empty."""

    assert memory.get_latest("reports") is None


def test_count_by_namespace(
    memory: LongTermMemory,
):
    """count_by_namespace() should count matching records."""

    memory.save(
        namespace="reports",
        key="june",
        value="June report",
    )
    memory.save(
        namespace="reports",
        key="july",
        value="July report",
    )
    memory.save(
        namespace="forecasts",
        key="june",
        value="June forecast",
    )

    assert memory.count_by_namespace("reports") == 2
    assert memory.count_by_namespace("forecasts") == 1
    assert memory.count_by_namespace("questions") == 0


def test_clear_namespace_removes_only_matching_memories(
    memory: LongTermMemory,
):
    """clear_namespace() should preserve other namespaces."""

    memory.save(
        namespace="reports",
        key="june",
        value="June report",
    )
    memory.save(
        namespace="reports",
        key="july",
        value="July report",
    )
    memory.save(
        namespace="forecasts",
        key="june",
        value="June forecast",
    )

    removed_count = memory.clear_namespace("reports")

    assert removed_count == 2
    assert memory.count == 1
    assert memory.count_by_namespace("reports") == 0
    assert memory.count_by_namespace("forecasts") == 1


def test_clear_namespace_returns_zero_when_empty(
    memory: LongTermMemory,
):
    """Clearing an absent namespace should return zero."""

    assert memory.clear_namespace("missing") == 0


def test_clear_all_removes_every_memory(
    memory: LongTermMemory,
):
    """clear_all() should remove the complete database content."""

    memory.save(
        namespace="reports",
        key="june",
        value="June report",
    )
    memory.save(
        namespace="forecasts",
        key="june",
        value="June forecast",
    )
    memory.save(
        namespace="questions",
        key="question-001",
        value="Analyze June.",
    )

    removed_count = memory.clear_all()

    assert removed_count == 3
    assert memory.count == 0


def test_clear_all_returns_zero_for_empty_database(
    memory: LongTermMemory,
):
    """Clearing an empty database should return zero."""

    assert memory.clear_all() == 0


def test_memory_persists_across_instances(
    tmp_path: Path,
):
    """File-backed memories should survive instance restart."""

    database_path = tmp_path / "persistent_memory.db"

    first_instance = LongTermMemory(database_path)
    memory_id = first_instance.save(
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
        tags=["monthly", "variance"],
    )
    first_instance.close()

    second_instance = LongTermMemory(database_path)

    try:
        restored_entry = second_instance.get(memory_id)

        assert restored_entry.value == {
            "status": "REVIEW",
            "revenue_variance": -200_000,
        }
        assert restored_entry.metadata == {
            "branch": "Chennai",
            "period": "2026-06",
        }
        assert restored_entry.tags == [
            "monthly",
            "variance",
        ]
    finally:
        second_instance.close()


def test_in_memory_database_reuses_connection(
    in_memory_database: LongTermMemory,
):
    """The same in-memory connection should retain records."""

    memory_id = save_sample_memory(
        in_memory_database
    )

    assert in_memory_database.count == 1
    assert (
        in_memory_database.get(memory_id).key
        == "chennai-2026-06"
    )


def test_context_manager_support(
    tmp_path: Path,
):
    """LongTermMemory should work as a context manager."""

    database_path = tmp_path / "context_memory.db"

    with LongTermMemory(database_path) as memory:
        memory_id = memory.save(
            namespace="reports",
            key="june",
            value="June report",
        )

        assert memory.exists(memory_id) is True

    with LongTermMemory(database_path) as memory:
        assert memory.exists(memory_id) is True


def test_close_is_safe_for_file_backed_database(
    memory: LongTermMemory,
):
    """close() should be safe without a shared connection."""

    memory.close()
    memory.close()

    assert memory.count == 0


def test_long_term_memory_entry_to_dict():
    """LongTermMemoryEntry should serialize itself."""

    now = datetime.now(timezone.utc)

    entry = LongTermMemoryEntry(
        memory_id="memory-001",
        namespace="reports",
        key="june",
        value={"status": "REVIEW"},
        created_at=now,
        updated_at=now,
        metadata={"period": "2026-06"},
        tags=["monthly"],
    )

    result = entry.to_dict()

    assert result == {
        "memory_id": "memory-001",
        "namespace": "reports",
        "key": "june",
        "value": {"status": "REVIEW"},
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
        "metadata": {"period": "2026-06"},
        "tags": ["monthly"],
    }


def test_long_term_memory_entry_to_dict_deep_copies():
    """Entry dictionary output should not expose data."""

    entry = LongTermMemoryEntry(
        memory_id="memory-001",
        namespace="reports",
        key="june",
        value={
            "risks": [
                "Revenue below budget",
            ]
        },
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        metadata={
            "filters": {
                "branch": "Chennai",
            }
        },
        tags=["monthly"],
    )

    result = entry.to_dict()

    result["value"]["risks"].append("Changed")
    result["metadata"]["filters"]["branch"] = "Changed"
    result["tags"].append("changed")

    assert entry.value == {
        "risks": [
            "Revenue below budget",
        ]
    }
    assert entry.metadata == {
        "filters": {
            "branch": "Chennai",
        }
    }
    assert entry.tags == ["monthly"]


def test_long_term_memory_summary_to_dict():
    """LongTermMemorySummary should serialize itself."""

    now = datetime.now(timezone.utc)

    summary = LongTermMemorySummary(
        memory_id="memory-001",
        namespace="reports",
        key="june",
        created_at=now,
        updated_at=now,
        metadata={"period": "2026-06"},
        tags=["monthly"],
    )

    result = summary.to_dict()

    assert result == {
        "memory_id": "memory-001",
        "namespace": "reports",
        "key": "june",
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
        "metadata": {"period": "2026-06"},
        "tags": ["monthly"],
    }


def test_datetime_to_string_converts_to_utc():
    """Datetime serialization should normalize to UTC."""

    value = datetime.now(timezone.utc)

    result = _datetime_to_string(value)

    assert result == value.isoformat()


def test_datetime_to_string_rejects_naive_datetime():
    """Datetime serialization should reject naive values."""

    with pytest.raises(
        ValueError,
        match="datetime value must be timezone-aware",
    ):
        _datetime_to_string(datetime.now())


def test_datetime_from_string_returns_aware_datetime():
    """Datetime parsing should return a UTC-aware value."""

    parsed_value = _datetime_from_string(
        "2026-06-01T10:30:00+00:00"
    )

    assert parsed_value.tzinfo is not None
    assert parsed_value.utcoffset().total_seconds() == 0


def test_datetime_from_string_handles_naive_text():
    """Naive stored text should be interpreted as UTC."""

    parsed_value = _datetime_from_string(
        "2026-06-01T10:30:00"
    )

    assert parsed_value.tzinfo is not None
    assert parsed_value.utcoffset().total_seconds() == 0


def test_memory_stores_dataclass_agent_result(
    memory: LongTermMemory,
):
    """Custom dataclass agent results should persist."""

    result = SampleFinanceResult(
        status="REVIEW",
        actual_revenue=1_000_000,
        budget_revenue=1_200_000,
        revenue_variance=-200_000,
    )

    memory_id = memory.save(
        namespace="agent-results",
        key="variance-agent-2026-06",
        value=result,
        metadata={
            "agent": "variance_agent",
            "period": "2026-06",
        },
        tags=[
            "variance",
            "agent-result",
        ],
    )

    restored_result = memory.get(memory_id).value

    assert isinstance(
        restored_result,
        SampleFinanceResult,
    )
    assert restored_result.status == "REVIEW"
    assert (
        restored_result.revenue_variance
        == -200_000
    )


def test_memory_stores_simple_namespace_result(
    memory: LongTermMemory,
):
    """Simple agent result objects should persist."""

    result = SimpleNamespace(
        overall_status="REVIEW",
        risks=[
            "Revenue below budget",
        ],
    )

    memory_id = memory.save(
        namespace="agent-results",
        key="commentary-result",
        value=result,
    )

    restored_result = memory.get(memory_id).value

    assert restored_result.overall_status == "REVIEW"
    assert restored_result.risks == [
        "Revenue below budget"
    ]


def test_realistic_monthly_fp_and_a_workflow(
    tmp_path: Path,
):
    """Long-term memory should support monthly FP&A history."""

    database_path = (
        tmp_path
        / "finance_history.db"
    )

    with LongTermMemory(database_path) as memory:
        june_report_id = memory.save(
            namespace="management-reports",
            key="chennai-2026-06",
            value={
                "overall_status": "REVIEW",
                "executive_summary": (
                    "June revenue was below budget "
                    "by ₹2.00 lakh."
                ),
                "actual_revenue": 1_000_000,
                "budget_revenue": 1_200_000,
                "revenue_variance": -200_000,
                "recommendations": [
                    "Review category-level volume gaps.",
                    "Investigate price realization.",
                ],
            },
            metadata={
                "branch": "Chennai",
                "period": "2026-06",
                "report_type": "monthly",
                "currency": "INR",
            },
            tags=[
                "monthly",
                "management-report",
                "variance",
                "chennai",
            ],
        )

        memory.save(
            namespace="management-reports",
            key="chennai-2026-07",
            value={
                "overall_status": "PASS",
                "actual_revenue": 1_250_000,
                "budget_revenue": 1_200_000,
                "revenue_variance": 50_000,
            },
            metadata={
                "branch": "Chennai",
                "period": "2026-07",
                "report_type": "monthly",
                "currency": "INR",
            },
            tags=[
                "monthly",
                "management-report",
                "chennai",
            ],
        )

        memory.save(
            namespace="user-preferences",
            key="finance-user-001",
            value={
                "preferred_branch": "Chennai",
                "currency": "INR",
                "report_detail": "management",
            },
            metadata={
                "user_id": "finance-user-001",
            },
            tags=["preferences"],
        )

        assert memory.count == 3
        assert (
            memory.count_by_namespace(
                "management-reports"
            )
            == 2
        )

        june_report = memory.get(june_report_id)

        assert (
            june_report.value["revenue_variance"]
            == -200_000
        )
        assert (
            june_report.metadata["branch"]
            == "Chennai"
        )

        chennai_reports = memory.search(
            query="Chennai",
            namespace="management-reports",
        )

        assert len(chennai_reports) == 2

        monthly_reports = memory.list_memories(
            namespace="management-reports",
            tag="monthly",
        )

        assert len(monthly_reports) == 2

    with LongTermMemory(database_path) as restored_memory:
        assert restored_memory.count == 3

        restored_june_report = (
            restored_memory.get_by_key(
                namespace="management-reports",
                key="chennai-2026-06",
            )
        )

        assert (
            restored_june_report.value[
                "overall_status"
            ]
            == "REVIEW"
        )
        assert (
            restored_june_report.metadata[
                "currency"
            ]
            == "INR"
        )


def test_realistic_memory_update_after_approval(
    memory: LongTermMemory,
):
    """A report should be updateable after approval."""

    memory_id = save_sample_memory(memory)

    approved_report = memory.update(
        memory_id=memory_id,
        value={
            "status": "APPROVED",
            "revenue_variance": -200_000,
            "approval_comment": (
                "Reviewed and approved by finance."
            ),
        },
        metadata={
            "branch": "Chennai",
            "period": "2026-06",
            "approved": True,
            "approved_by": "finance-manager",
        },
        tags=[
            "monthly",
            "management-report",
            "approved",
        ],
    )

    assert (
        approved_report.value["status"]
        == "APPROVED"
    )
    assert (
        approved_report.metadata["approved"]
        is True
    )
    assert "approved" in approved_report.tags


def test_realistic_get_latest_report(
    memory: LongTermMemory,
):
    """The latest report should be retrievable."""

    june_id = memory.save(
        namespace="management-reports",
        key="chennai-2026-06",
        value={"period": "2026-06"},
        tags=["monthly"],
    )
    july_id = memory.save(
        namespace="management-reports",
        key="chennai-2026-07",
        value={"period": "2026-07"},
        tags=["monthly"],
    )

    with memory._lock, memory._connection() as connection:
        connection.execute(
            """
            UPDATE long_term_memory
            SET updated_at = ?
            WHERE memory_id = ?
            """,
            (
                "2026-06-30T23:59:59+00:00",
                june_id,
            ),
        )
        connection.execute(
            """
            UPDATE long_term_memory
            SET updated_at = ?
            WHERE memory_id = ?
            """,
            (
                "2026-07-31T23:59:59+00:00",
                july_id,
            ),
        )
        connection.commit()

    latest_report = memory.get_latest(
        namespace="management-reports",
        tag="monthly",
    )

    assert latest_report is not None
    assert latest_report.key == "chennai-2026-07"
    assert latest_report.value["period"] == "2026-07"
