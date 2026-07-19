
"""
Tests for the Finance Agentic AI session-memory module.

These tests verify:

- Constructor validation
- Session creation
- Custom session identifiers
- Session metadata
- Entry creation and retrieval
- Entry updates
- Entry deletion
- Session clearing
- Session deletion
- Session expiration
- Automatic TTL refresh
- Maximum-session limits
- Deep-copy protection
- Thread-safe public behaviour
- Validation and error handling
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.memory.session_memory import (
    SessionMemory,
    SessionMemoryEntry,
    SessionRecord,
)


def test_constructor_accepts_valid_configuration():
    """SessionMemory should accept valid configuration values."""

    memory = SessionMemory(
        session_ttl_minutes=30,
        max_sessions=100,
        auto_refresh_ttl=False,
    )

    assert memory.session_count == 0


@pytest.mark.parametrize(
    "session_ttl_minutes",
    [0, -1, -100],
)
def test_constructor_rejects_invalid_session_ttl(
    session_ttl_minutes,
):
    """Session TTL must be a positive number of minutes."""

    with pytest.raises(
        ValueError,
        match="session_ttl_minutes must be positive",
    ):
        SessionMemory(
            session_ttl_minutes=session_ttl_minutes
        )


@pytest.mark.parametrize(
    "max_sessions",
    [0, -1, -100],
)
def test_constructor_rejects_invalid_max_sessions(
    max_sessions,
):
    """Maximum session count must be positive."""

    with pytest.raises(
        ValueError,
        match="max_sessions must be positive",
    ):
        SessionMemory(max_sessions=max_sessions)


def test_create_session_generates_unique_session_id():
    """A session ID should be generated when one is not supplied."""

    memory = SessionMemory()

    first_session_id = memory.create_session()
    second_session_id = memory.create_session()

    assert isinstance(first_session_id, str)
    assert isinstance(second_session_id, str)
    assert first_session_id
    assert second_session_id
    assert first_session_id != second_session_id
    assert memory.session_count == 2


def test_create_session_accepts_custom_session_id():
    """A valid custom session ID should be retained."""

    memory = SessionMemory()

    session_id = memory.create_session(
        session_id="finance-session-001"
    )

    assert session_id == "finance-session-001"
    assert memory.session_exists("finance-session-001") is True


def test_create_session_strips_custom_session_id():
    """Leading and trailing spaces should be removed."""

    memory = SessionMemory()

    session_id = memory.create_session(
        session_id="  finance-session-001  "
    )

    assert session_id == "finance-session-001"
    assert memory.session_exists("finance-session-001") is True


@pytest.mark.parametrize(
    "session_id",
    ["", " ", "   "],
)
def test_create_session_rejects_empty_custom_session_id(
    session_id,
):
    """Empty custom session identifiers should be rejected."""

    memory = SessionMemory()

    with pytest.raises(
        ValueError,
        match="session_id must be a non-empty string",
    ):
        memory.create_session(session_id=session_id)


def test_create_session_rejects_duplicate_session_id():
    """Duplicate session identifiers should not be allowed."""

    memory = SessionMemory()
    memory.create_session(session_id="finance-session")

    with pytest.raises(
        ValueError,
        match="session already exists: finance-session",
    ):
        memory.create_session(session_id="finance-session")


def test_create_session_stores_metadata():
    """Session-level metadata should be stored."""

    memory = SessionMemory()

    session_id = memory.create_session(
        metadata={
            "user_id": "user-001",
            "company": "Sample Finance Company",
        }
    )

    session = memory.get_session(session_id)

    assert session.metadata == {
        "user_id": "user-001",
        "company": "Sample Finance Company",
    }


def test_create_session_deep_copies_metadata():
    """Changing source metadata should not mutate session memory."""

    metadata = {
        "user": {
            "name": "Karthik",
        }
    }

    memory = SessionMemory()
    session_id = memory.create_session(metadata=metadata)

    metadata["user"]["name"] = "Changed"

    session = memory.get_session(session_id)

    assert session.metadata["user"]["name"] == "Karthik"


def test_session_exists_returns_false_for_unknown_session():
    """Unknown sessions should not be reported as active."""

    memory = SessionMemory()

    assert memory.session_exists("missing-session") is False


def test_session_exists_returns_true_for_active_session():
    """An active session should be reported as existing."""

    memory = SessionMemory()
    session_id = memory.create_session()

    assert memory.session_exists(session_id) is True


def test_set_creates_memory_entry():
    """set() should create an entry in an existing session."""

    memory = SessionMemory()
    session_id = memory.create_session()

    entry = memory.set(
        session_id=session_id,
        key="last_question",
        value="Explain the June revenue variance.",
    )

    assert isinstance(entry, SessionMemoryEntry)
    assert entry.key == "last_question"
    assert entry.value == "Explain the June revenue variance."
    assert memory.get(
        session_id,
        "last_question",
    ) == "Explain the June revenue variance."


def test_set_strips_entry_key():
    """Entry keys should be normalized before storage."""

    memory = SessionMemory()
    session_id = memory.create_session()

    memory.set(
        session_id=session_id,
        key="  last_report  ",
        value="June management report",
    )

    assert memory.contains(session_id, "last_report") is True
    assert memory.get(
        session_id,
        "last_report",
    ) == "June management report"


@pytest.mark.parametrize(
    "key",
    ["", " ", "   ", None, 100],
)
def test_set_rejects_invalid_key(key):
    """Memory entry keys must be non-empty strings."""

    memory = SessionMemory()
    session_id = memory.create_session()

    with pytest.raises(
        ValueError,
        match="key must be a non-empty string",
    ):
        memory.set(
            session_id=session_id,
            key=key,
            value="value",
        )


def test_set_stores_entry_metadata():
    """Entry-level metadata should be stored."""

    memory = SessionMemory()
    session_id = memory.create_session()

    memory.set(
        session_id=session_id,
        key="last_report",
        value="June report",
        metadata={
            "report_type": "monthly",
            "period": "2026-06",
        },
    )

    entry = memory.get_entry(
        session_id,
        "last_report",
    )

    assert entry is not None
    assert entry.metadata == {
        "report_type": "monthly",
        "period": "2026-06",
    }


def test_set_replaces_existing_entry_value():
    """Calling set() for an existing key should update its value."""

    memory = SessionMemory()
    session_id = memory.create_session()

    memory.set(
        session_id=session_id,
        key="last_period",
        value="2026-05",
    )

    original_entry = memory.get_entry(
        session_id,
        "last_period",
    )

    memory.set(
        session_id=session_id,
        key="last_period",
        value="2026-06",
    )

    updated_entry = memory.get_entry(
        session_id,
        "last_period",
    )

    assert original_entry is not None
    assert updated_entry is not None
    assert updated_entry.value == "2026-06"
    assert updated_entry.created_at == original_entry.created_at
    assert updated_entry.updated_at >= original_entry.updated_at


def test_set_preserves_metadata_when_metadata_is_not_supplied():
    """Updating through set() should preserve existing metadata."""

    memory = SessionMemory()
    session_id = memory.create_session()

    memory.set(
        session_id=session_id,
        key="report",
        value="Old report",
        metadata={"period": "2026-06"},
    )

    memory.set(
        session_id=session_id,
        key="report",
        value="Updated report",
    )

    entry = memory.get_entry(session_id, "report")

    assert entry is not None
    assert entry.value == "Updated report"
    assert entry.metadata == {"period": "2026-06"}


def test_set_replaces_metadata_when_metadata_is_supplied():
    """Explicit metadata should replace existing entry metadata."""

    memory = SessionMemory()
    session_id = memory.create_session()

    memory.set(
        session_id=session_id,
        key="report",
        value="Old report",
        metadata={"period": "2026-05"},
    )

    memory.set(
        session_id=session_id,
        key="report",
        value="June report",
        metadata={
            "period": "2026-06",
            "status": "REVIEW",
        },
    )

    entry = memory.get_entry(session_id, "report")

    assert entry is not None
    assert entry.metadata == {
        "period": "2026-06",
        "status": "REVIEW",
    }


def test_set_deep_copies_value():
    """Mutating the original value should not affect stored memory."""

    source_value = {
        "variance": {
            "revenue": -200_000,
        }
    }

    memory = SessionMemory()
    session_id = memory.create_session()

    memory.set(
        session_id=session_id,
        key="variance_result",
        value=source_value,
    )

    source_value["variance"]["revenue"] = 0

    stored_value = memory.get(
        session_id,
        "variance_result",
    )

    assert stored_value["variance"]["revenue"] == -200_000


def test_get_returns_deep_copy():
    """Mutating a retrieved value should not affect memory."""

    memory = SessionMemory()
    session_id = memory.create_session()

    memory.set(
        session_id=session_id,
        key="report",
        value={
            "risks": [
                "Revenue below budget",
            ]
        },
    )

    retrieved_value = memory.get(session_id, "report")
    retrieved_value["risks"].append("Changed externally")

    stored_value = memory.get(session_id, "report")

    assert stored_value == {
        "risks": [
            "Revenue below budget",
        ]
    }


def test_get_returns_default_for_missing_key():
    """A missing key should return the supplied default value."""

    memory = SessionMemory()
    session_id = memory.create_session()

    result = memory.get(
        session_id=session_id,
        key="missing",
        default={"status": "not-found"},
    )

    assert result == {"status": "not-found"}


def test_get_deep_copies_default_value():
    """The returned default should be protected by deep copying."""

    default = {
        "items": [],
    }

    memory = SessionMemory()
    session_id = memory.create_session()

    result = memory.get(
        session_id=session_id,
        key="missing",
        default=default,
    )

    result["items"].append("external change")

    assert default == {"items": []}


def test_get_entry_returns_none_for_missing_key():
    """get_entry() should return None for an unknown key."""

    memory = SessionMemory()
    session_id = memory.create_session()

    assert memory.get_entry(
        session_id,
        "missing",
    ) is None


def test_get_entry_returns_deep_copy():
    """Changing a returned entry should not mutate stored memory."""

    memory = SessionMemory()
    session_id = memory.create_session()

    memory.set(
        session_id=session_id,
        key="report",
        value={"status": "REVIEW"},
        metadata={"period": "2026-06"},
    )

    entry = memory.get_entry(session_id, "report")

    assert entry is not None

    entry.value["status"] = "PASS"
    entry.metadata["period"] = "changed"

    stored_entry = memory.get_entry(session_id, "report")

    assert stored_entry is not None
    assert stored_entry.value == {"status": "REVIEW"}
    assert stored_entry.metadata == {"period": "2026-06"}


def test_update_changes_existing_entry():
    """update() should change an existing memory entry."""

    memory = SessionMemory()
    session_id = memory.create_session()

    memory.set(
        session_id=session_id,
        key="last_question",
        value="Analyze June.",
    )

    updated_entry = memory.update(
        session_id=session_id,
        key="last_question",
        value="Compare June with May.",
        metadata={"request_type": "comparison"},
    )

    assert updated_entry.value == "Compare June with May."
    assert updated_entry.metadata == {
        "request_type": "comparison"
    }
    assert memory.get(
        session_id,
        "last_question",
    ) == "Compare June with May."


def test_update_requires_existing_entry():
    """update() should not silently create a missing entry."""

    memory = SessionMemory()
    session_id = memory.create_session()

    with pytest.raises(
        KeyError,
        match="session entry not found: missing",
    ):
        memory.update(
            session_id=session_id,
            key="missing",
            value="value",
        )


def test_delete_removes_existing_entry():
    """delete() should remove an existing key."""

    memory = SessionMemory()
    session_id = memory.create_session()

    memory.set(
        session_id=session_id,
        key="temporary_value",
        value=100,
    )

    removed = memory.delete(
        session_id,
        "temporary_value",
    )

    assert removed is True
    assert memory.contains(
        session_id,
        "temporary_value",
    ) is False


def test_delete_returns_false_for_missing_entry():
    """delete() should return False when no entry is removed."""

    memory = SessionMemory()
    session_id = memory.create_session()

    assert memory.delete(
        session_id,
        "missing",
    ) is False


def test_clear_session_removes_all_entries():
    """clear_session() should keep the session but remove its entries."""

    memory = SessionMemory()
    session_id = memory.create_session()

    memory.set(session_id, "question", "Analyze June")
    memory.set(session_id, "report", "June report")
    memory.set(session_id, "status", "REVIEW")

    removed_count = memory.clear_session(session_id)

    assert removed_count == 3
    assert memory.session_exists(session_id) is True
    assert memory.list_keys(session_id) == []


def test_clear_empty_session_returns_zero():
    """Clearing an already empty session should return zero."""

    memory = SessionMemory()
    session_id = memory.create_session()

    assert memory.clear_session(session_id) == 0


def test_delete_session_removes_complete_session():
    """delete_session() should remove the complete session."""

    memory = SessionMemory()
    session_id = memory.create_session()

    memory.set(session_id, "question", "Analyze June")

    removed = memory.delete_session(session_id)

    assert removed is True
    assert memory.session_exists(session_id) is False
    assert memory.session_count == 0


def test_delete_session_returns_false_for_unknown_session():
    """Deleting an unknown session should return False."""

    memory = SessionMemory()

    assert memory.delete_session("missing-session") is False


def test_get_session_returns_session_record():
    """get_session() should return a complete SessionRecord."""

    memory = SessionMemory()
    session_id = memory.create_session(
        metadata={"user_id": "user-001"}
    )

    memory.set(
        session_id=session_id,
        key="last_question",
        value="Analyze June.",
    )

    session = memory.get_session(session_id)

    assert isinstance(session, SessionRecord)
    assert session.session_id == session_id
    assert session.metadata == {"user_id": "user-001"}
    assert "last_question" in session.entries


def test_get_session_returns_deep_copy():
    """Changing a returned SessionRecord should not mutate memory."""

    memory = SessionMemory()
    session_id = memory.create_session(
        metadata={"user": {"name": "Karthik"}}
    )

    memory.set(
        session_id=session_id,
        key="report",
        value={"status": "REVIEW"},
    )

    returned_session = memory.get_session(session_id)

    returned_session.metadata["user"]["name"] = "Changed"
    returned_session.entries["report"].value["status"] = "PASS"

    stored_session = memory.get_session(session_id)

    assert stored_session.metadata["user"]["name"] == "Karthik"
    assert (
        stored_session.entries["report"].value["status"]
        == "REVIEW"
    )


def test_list_sessions_returns_session_summaries():
    """list_sessions() should summarize every active session."""

    memory = SessionMemory()

    first_session_id = memory.create_session(
        session_id="session-001",
        metadata={"user_id": "user-001"},
    )
    second_session_id = memory.create_session(
        session_id="session-002",
        metadata={"user_id": "user-002"},
    )

    memory.set(
        first_session_id,
        "question",
        "Analyze June.",
    )
    memory.set(
        second_session_id,
        "question",
        "Analyze July.",
    )
    memory.set(
        second_session_id,
        "report",
        "July report",
    )

    sessions = memory.list_sessions()

    assert len(sessions) == 2

    session_by_id = {
        item["session_id"]: item
        for item in sessions
    }

    assert session_by_id["session-001"]["entry_count"] == 1
    assert session_by_id["session-002"]["entry_count"] == 2
    assert session_by_id["session-001"]["metadata"] == {
        "user_id": "user-001"
    }


def test_list_sessions_returns_deep_copied_metadata():
    """Changing list output should not mutate session metadata."""

    memory = SessionMemory()
    session_id = memory.create_session(
        metadata={
            "user": {
                "name": "Karthik",
            }
        }
    )

    sessions = memory.list_sessions()
    sessions[0]["metadata"]["user"]["name"] = "Changed"

    session = memory.get_session(session_id)

    assert session.metadata["user"]["name"] == "Karthik"


def test_list_keys_returns_sorted_keys():
    """Session entry keys should be listed alphabetically."""

    memory = SessionMemory()
    session_id = memory.create_session()

    memory.set(session_id, "variance", 1)
    memory.set(session_id, "budget", 2)
    memory.set(session_id, "actual", 3)

    assert memory.list_keys(session_id) == [
        "actual",
        "budget",
        "variance",
    ]


def test_contains_returns_true_for_existing_key():
    """contains() should detect an existing key."""

    memory = SessionMemory()
    session_id = memory.create_session()

    memory.set(
        session_id,
        "last_report",
        "June report",
    )

    assert memory.contains(
        session_id,
        "last_report",
    ) is True


def test_contains_returns_false_for_missing_key():
    """contains() should return False for an unknown key."""

    memory = SessionMemory()
    session_id = memory.create_session()

    assert memory.contains(
        session_id,
        "missing",
    ) is False


def test_update_session_metadata_merges_by_default():
    """Session metadata should be merged by default."""

    memory = SessionMemory()
    session_id = memory.create_session(
        metadata={
            "user_id": "user-001",
            "company": "Sample Company",
        }
    )

    metadata = memory.update_session_metadata(
        session_id=session_id,
        metadata={
            "period": "2026-06",
            "company": "Updated Company",
        },
    )

    assert metadata == {
        "user_id": "user-001",
        "company": "Updated Company",
        "period": "2026-06",
    }


def test_update_session_metadata_can_replace_metadata():
    """merge=False should replace all existing metadata."""

    memory = SessionMemory()
    session_id = memory.create_session(
        metadata={
            "user_id": "user-001",
            "company": "Sample Company",
        }
    )

    metadata = memory.update_session_metadata(
        session_id=session_id,
        metadata={
            "period": "2026-06",
        },
        merge=False,
    )

    assert metadata == {
        "period": "2026-06",
    }


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        "invalid",
        100,
        ["invalid"],
    ],
)
def test_update_session_metadata_requires_dictionary(
    metadata,
):
    """Session metadata updates must use a dictionary."""

    memory = SessionMemory()
    session_id = memory.create_session()

    with pytest.raises(
        TypeError,
        match="metadata must be a dictionary",
    ):
        memory.update_session_metadata(
            session_id=session_id,
            metadata=metadata,
        )


def test_update_session_metadata_deep_copies_input():
    """Changing metadata input should not mutate stored metadata."""

    metadata = {
        "filters": {
            "branch": "Chennai",
        }
    }

    memory = SessionMemory()
    session_id = memory.create_session()

    memory.update_session_metadata(
        session_id=session_id,
        metadata=metadata,
    )

    metadata["filters"]["branch"] = "Changed"

    session = memory.get_session(session_id)

    assert session.metadata["filters"]["branch"] == "Chennai"


def test_session_memory_entry_to_dict():
    """SessionMemoryEntry should convert itself to a dictionary."""

    now = datetime.now(timezone.utc)

    entry = SessionMemoryEntry(
        key="report",
        value={"status": "REVIEW"},
        created_at=now,
        updated_at=now,
        metadata={"period": "2026-06"},
    )

    result = entry.to_dict()

    assert result == {
        "key": "report",
        "value": {"status": "REVIEW"},
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
        "metadata": {"period": "2026-06"},
    }


def test_session_memory_entry_to_dict_deep_copies_data():
    """Entry dictionary output should not expose internal data."""

    entry = SessionMemoryEntry(
        key="report",
        value={"status": "REVIEW"},
        metadata={"periods": ["2026-06"]},
    )

    result = entry.to_dict()

    result["value"]["status"] = "PASS"
    result["metadata"]["periods"].append("2026-07")

    assert entry.value == {"status": "REVIEW"}
    assert entry.metadata == {
        "periods": ["2026-06"]
    }


def test_session_record_to_dict():
    """SessionRecord should serialize session and entry information."""

    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(minutes=60)

    entry = SessionMemoryEntry(
        key="question",
        value="Analyze June.",
        created_at=now,
        updated_at=now,
    )

    session = SessionRecord(
        session_id="session-001",
        created_at=now,
        updated_at=now,
        expires_at=expires_at,
        metadata={"user_id": "user-001"},
        entries={"question": entry},
    )

    result = session.to_dict()

    assert result["session_id"] == "session-001"
    assert result["created_at"] == now.isoformat()
    assert result["updated_at"] == now.isoformat()
    assert result["expires_at"] == expires_at.isoformat()
    assert result["metadata"] == {
        "user_id": "user-001"
    }
    assert result["entries"]["question"]["value"] == (
        "Analyze June."
    )


def test_session_record_is_expired():
    """SessionRecord should identify an expired session."""

    now = datetime.now(timezone.utc)

    session = SessionRecord(
        session_id="expired-session",
        created_at=now - timedelta(hours=2),
        updated_at=now - timedelta(hours=2),
        expires_at=now - timedelta(minutes=1),
    )

    assert session.is_expired(current_time=now) is True


def test_session_record_is_not_expired():
    """SessionRecord should identify an active session."""

    now = datetime.now(timezone.utc)

    session = SessionRecord(
        session_id="active-session",
        created_at=now,
        updated_at=now,
        expires_at=now + timedelta(minutes=60),
    )

    assert session.is_expired(current_time=now) is False


def test_cleanup_expired_sessions_removes_expired_sessions():
    """Expired sessions should be removed during cleanup."""

    memory = SessionMemory(
        session_ttl_minutes=60,
        auto_refresh_ttl=False,
    )

    expired_session_id = memory.create_session(
        session_id="expired-session"
    )
    active_session_id = memory.create_session(
        session_id="active-session"
    )

    memory._sessions[expired_session_id].expires_at = (
        datetime.now(timezone.utc) - timedelta(minutes=1)
    )
    memory._sessions[active_session_id].expires_at = (
        datetime.now(timezone.utc) + timedelta(minutes=30)
    )

    removed_count = memory.cleanup_expired_sessions()

    assert removed_count == 1
    assert memory.session_exists(expired_session_id) is False
    assert memory.session_exists(active_session_id) is True


def test_cleanup_expired_sessions_accepts_deterministic_time():
    """Cleanup should support a supplied timezone-aware comparison time."""

    memory = SessionMemory(
        session_ttl_minutes=60,
        auto_refresh_ttl=False,
    )
    session_id = memory.create_session()

    expires_at = memory.get_session(session_id).expires_at

    removed_count = memory.cleanup_expired_sessions(
        current_time=expires_at,
    )

    assert removed_count == 1
    assert memory.session_exists(session_id) is False


def test_cleanup_expired_sessions_rejects_naive_datetime():
    """Cleanup comparison time must include timezone information."""

    memory = SessionMemory()

    with pytest.raises(
        ValueError,
        match="current_time must be timezone-aware",
    ):
        memory.cleanup_expired_sessions(
            current_time=datetime.now()
        )


def test_expired_session_access_raises_key_error():
    """Accessing an expired session should raise a clear error."""

    memory = SessionMemory(
        session_ttl_minutes=60,
        auto_refresh_ttl=False,
    )
    session_id = memory.create_session()

    memory._sessions[session_id].expires_at = (
        datetime.now(timezone.utc) - timedelta(seconds=1)
    )

    with pytest.raises(
        KeyError,
        match=f"session expired: {session_id}",
    ):
        memory.get(session_id, "question")


def test_session_exists_removes_expired_session():
    """session_exists() should clean up an expired session."""

    memory = SessionMemory(
        session_ttl_minutes=60,
        auto_refresh_ttl=False,
    )
    session_id = memory.create_session()

    memory._sessions[session_id].expires_at = (
        datetime.now(timezone.utc) - timedelta(seconds=1)
    )

    assert memory.session_exists(session_id) is False
    assert session_id not in memory._sessions


def test_auto_refresh_ttl_extends_expiration():
    """Reading a session should extend TTL when auto-refresh is enabled."""

    memory = SessionMemory(
        session_ttl_minutes=60,
        auto_refresh_ttl=True,
    )
    session_id = memory.create_session()

    memory.set(
        session_id,
        "question",
        "Analyze June.",
    )

    earlier_expiration = (
        datetime.now(timezone.utc) + timedelta(minutes=5)
    )
    memory._sessions[session_id].expires_at = earlier_expiration

    memory.get(session_id, "question")

    refreshed_expiration = (
        memory._sessions[session_id].expires_at
    )

    assert refreshed_expiration > earlier_expiration


def test_disabled_auto_refresh_does_not_extend_expiration():
    """TTL should not move when auto-refresh is disabled."""

    memory = SessionMemory(
        session_ttl_minutes=60,
        auto_refresh_ttl=False,
    )
    session_id = memory.create_session()

    memory.set(
        session_id,
        "question",
        "Analyze June.",
    )

    fixed_expiration = (
        datetime.now(timezone.utc) + timedelta(minutes=30)
    )
    memory._sessions[session_id].expires_at = fixed_expiration

    memory.get(session_id, "question")

    assert (
        memory._sessions[session_id].expires_at
        == fixed_expiration
    )


def test_max_sessions_removes_least_recently_updated_session():
    """Oldest inactive session should be evicted at capacity."""

    memory = SessionMemory(
        max_sessions=2,
        auto_refresh_ttl=False,
    )

    first_session_id = memory.create_session(
        session_id="session-001"
    )
    second_session_id = memory.create_session(
        session_id="session-002"
    )

    memory._sessions[first_session_id].updated_at = (
        datetime.now(timezone.utc) - timedelta(hours=2)
    )
    memory._sessions[second_session_id].updated_at = (
        datetime.now(timezone.utc) - timedelta(hours=1)
    )

    third_session_id = memory.create_session(
        session_id="session-003"
    )

    assert memory.session_count == 2
    assert memory.session_exists(first_session_id) is False
    assert memory.session_exists(second_session_id) is True
    assert memory.session_exists(third_session_id) is True


def test_clear_all_removes_every_session():
    """clear_all() should remove all active sessions."""

    memory = SessionMemory()

    memory.create_session(session_id="session-001")
    memory.create_session(session_id="session-002")
    memory.create_session(session_id="session-003")

    removed_count = memory.clear_all()

    assert removed_count == 3
    assert memory.session_count == 0


def test_clear_all_on_empty_memory_returns_zero():
    """Clearing empty memory should return zero."""

    memory = SessionMemory()

    assert memory.clear_all() == 0


@pytest.mark.parametrize(
    "operation",
    [
        lambda memory, session_id: memory.get(
            session_id,
            "key",
        ),
        lambda memory, session_id: memory.get_entry(
            session_id,
            "key",
        ),
        lambda memory, session_id: memory.set(
            session_id,
            "key",
            "value",
        ),
        lambda memory, session_id: memory.update(
            session_id,
            "key",
            "value",
        ),
        lambda memory, session_id: memory.delete(
            session_id,
            "key",
        ),
        lambda memory, session_id: memory.clear_session(
            session_id
        ),
        lambda memory, session_id: memory.get_session(
            session_id
        ),
        lambda memory, session_id: memory.list_keys(
            session_id
        ),
        lambda memory, session_id: memory.contains(
            session_id,
            "key",
        ),
        lambda memory, session_id: (
            memory.update_session_metadata(
                session_id,
                {"period": "2026-06"},
            )
        ),
    ],
)
def test_session_operations_require_existing_session(
    operation,
):
    """Public session operations should reject unknown sessions."""

    memory = SessionMemory()

    with pytest.raises(
        KeyError,
        match="session not found: missing-session",
    ):
        operation(memory, "missing-session")


@pytest.mark.parametrize(
    "session_id",
    ["", " ", None, 100],
)
def test_session_operations_validate_session_id(
    session_id,
):
    """Session identifiers must be valid non-empty strings."""

    memory = SessionMemory()

    with pytest.raises(
        ValueError,
        match="session_id must be a non-empty string",
    ):
        memory.session_exists(session_id)


def test_memory_accepts_agent_result_objects():
    """Session memory should store arbitrary Python agent results."""

    result = SimpleNamespace(
        overall_status="REVIEW",
        revenue_variance=-200_000,
        risks=[
            "Revenue was below budget.",
        ],
    )

    memory = SessionMemory()
    session_id = memory.create_session()

    memory.set(
        session_id=session_id,
        key="variance_result",
        value=result,
        metadata={
            "agent": "variance_agent",
            "period": "2026-06",
        },
    )

    stored_result = memory.get(
        session_id,
        "variance_result",
    )

    assert stored_result.overall_status == "REVIEW"
    assert stored_result.revenue_variance == -200_000
    assert stored_result.risks == [
        "Revenue was below budget."
    ]


def test_realistic_finance_session_workflow():
    """Session memory should support a realistic finance workflow."""

    memory = SessionMemory(
        session_ttl_minutes=120,
        max_sessions=100,
    )

    session_id = memory.create_session(
        metadata={
            "user_id": "finance-user-001",
            "branch": "Chennai",
            "period": "2026-06",
        }
    )

    memory.set(
        session_id=session_id,
        key="last_question",
        value="Analyze June actual versus budget.",
    )

    memory.set(
        session_id=session_id,
        key="uploaded_files",
        value=[
            "june_actuals.csv",
            "june_budget.csv",
        ],
    )

    memory.set(
        session_id=session_id,
        key="variance_result",
        value={
            "actual_revenue": 1_000_000,
            "budget_revenue": 1_200_000,
            "revenue_variance": -200_000,
        },
    )

    memory.set(
        session_id=session_id,
        key="last_report",
        value={
            "overall_status": "REVIEW",
            "executive_summary": (
                "Revenue was below budget by ₹2.00 lakh."
            ),
        },
    )

    assert memory.get(
        session_id,
        "last_question",
    ) == "Analyze June actual versus budget."

    assert memory.get(
        session_id,
        "variance_result",
    )["revenue_variance"] == -200_000

    assert memory.get(
        session_id,
        "last_report",
    )["overall_status"] == "REVIEW"

    assert memory.list_keys(session_id) == [
        "last_question",
        "last_report",
        "uploaded_files",
        "variance_result",
    ]

    session = memory.get_session(session_id)

    assert session.metadata["branch"] == "Chennai"
    assert len(session.entries) == 4
