"""
In-memory session storage for the Finance Agentic AI System.

This module stores temporary conversation and workflow information for the
current application runtime. Session memory is intended for short-lived data,
such as:

- User questions
- Uploaded-file references
- Previous agent outputs
- Generated reports
- Workflow state
- Conversation context

Session memory is not permanent. All data is lost when the application process
stops. Persistent information should be stored using long-term memory.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Any
from uuid import uuid4


def _utc_now() -> datetime:
    """
    Return the current timezone-aware UTC datetime.

    Returns:
        Current UTC datetime.
    """

    return datetime.now(timezone.utc)


@dataclass
class SessionMemoryEntry:
    """
    Represents one key-value entry stored inside a session.

    Attributes:
        key: Unique key within the session.
        value: Stored Python object.
        created_at: Time when the entry was created.
        updated_at: Time when the entry was last updated.
        metadata: Optional descriptive information about the entry.
    """

    key: str
    value: Any
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def update(
        self,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Update the stored value and optional metadata.

        Args:
            value: New value to store.
            metadata: Optional metadata to replace the existing metadata.
        """

        self.value = deepcopy(value)

        if metadata is not None:
            self.metadata = deepcopy(metadata)

        self.updated_at = _utc_now()

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the entry to a serializable dictionary.

        Returns:
            Dictionary representation of the memory entry.
        """

        return {
            "key": self.key,
            "value": deepcopy(self.value),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": deepcopy(self.metadata),
        }


@dataclass
class SessionRecord:
    """
    Represents one complete session.

    Attributes:
        session_id: Unique session identifier.
        created_at: Time when the session was created.
        updated_at: Time when the session was last updated.
        expires_at: Time when the session becomes eligible for removal.
        metadata: Optional session-level metadata.
        entries: Key-value memory entries belonging to the session.
    """

    session_id: str
    created_at: datetime
    updated_at: datetime
    expires_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)
    entries: dict[str, SessionMemoryEntry] = field(default_factory=dict)

    def touch(self, ttl: timedelta) -> None:
        """
        Refresh the session update and expiration timestamps.

        Args:
            ttl: New time-to-live duration.
        """

        now = _utc_now()
        self.updated_at = now
        self.expires_at = now + ttl

    def is_expired(
        self,
        current_time: datetime | None = None,
    ) -> bool:
        """
        Check whether the session has expired.

        Args:
            current_time: Optional comparison time for deterministic testing.

        Returns:
            True when the session has expired, otherwise False.
        """

        comparison_time = current_time or _utc_now()
        return comparison_time >= self.expires_at

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the complete session to a dictionary.

        Returns:
            Dictionary representation of the session.
        """

        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "metadata": deepcopy(self.metadata),
            "entries": {
                key: entry.to_dict()
                for key, entry in self.entries.items()
            },
        }


class SessionMemory:
    """
    Thread-safe in-memory session manager.

    The manager keeps temporary information for active Finance Agentic AI
    conversations and workflows.
    """

    def __init__(
        self,
        session_ttl_minutes: int = 60,
        max_sessions: int = 1_000,
        auto_refresh_ttl: bool = True,
    ) -> None:
        """
        Initialize the session-memory manager.

        Args:
            session_ttl_minutes:
                Number of minutes before an inactive session expires.
            max_sessions:
                Maximum number of active sessions retained in memory.
            auto_refresh_ttl:
                Whether session access should refresh its expiration time.

        Raises:
            ValueError:
                If session_ttl_minutes or max_sessions is not positive.
        """

        if (
            not isinstance(session_ttl_minutes, int)
            or isinstance(session_ttl_minutes, bool)
            or session_ttl_minutes <= 0
        ):
            raise ValueError(
                "session_ttl_minutes must be positive"
            )

        if (
            not isinstance(max_sessions, int)
            or isinstance(max_sessions, bool)
            or max_sessions <= 0
        ):
            raise ValueError(
                "max_sessions must be positive"
            )

        if not isinstance(auto_refresh_ttl, bool):
            raise TypeError(
                "auto_refresh_ttl must be a boolean"
            )

        self._session_ttl = timedelta(
            minutes=session_ttl_minutes
        )
        self._max_sessions = max_sessions
        self._auto_refresh_ttl = auto_refresh_ttl
        self._sessions: dict[str, SessionRecord] = {}
        self._lock = RLock()

    @property
    def session_count(self) -> int:
        """
        Return the number of active sessions.

        Expired sessions are removed before the count is returned.
        """

        with self._lock:
            self.cleanup_expired_sessions()
            return len(self._sessions)

    def create_session(
        self,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Create and store a new session.

        Args:
            session_id:
                Optional custom session identifier.
            metadata:
                Optional session-level metadata.

        Returns:
            Created session identifier.
        """

        if metadata is not None and not isinstance(
            metadata,
            dict,
        ):
            raise TypeError(
                "metadata must be a dictionary"
            )

        if session_id is None:
            resolved_session_id = str(uuid4())
        elif isinstance(session_id, str):
            resolved_session_id = session_id.strip()
        else:
            raise ValueError(
                "session_id must be a non-empty string"
            )

        if not resolved_session_id:
            raise ValueError(
                "session_id must be a non-empty string"
            )

        with self._lock:
            self.cleanup_expired_sessions()

            if resolved_session_id in self._sessions:
                raise ValueError(
                    "session already exists: "
                    f"{resolved_session_id}"
                )

            if len(self._sessions) >= self._max_sessions:
                self._remove_oldest_session()

            now = _utc_now()

            self._sessions[
                resolved_session_id
            ] = SessionRecord(
                session_id=resolved_session_id,
                created_at=now,
                updated_at=now,
                expires_at=now + self._session_ttl,
                metadata=deepcopy(metadata or {}),
            )

            return resolved_session_id

    def session_exists(
        self,
        session_id: str,
    ) -> bool:
        """
        Check whether a non-expired session exists.
        """

        resolved_session_id = self._validate_session_id(
            session_id
        )

        with self._lock:
            session = self._sessions.get(
                resolved_session_id
            )

            if session is None:
                return False

            if session.is_expired():
                del self._sessions[
                    resolved_session_id
                ]
                return False

            return True

    def set(
        self,
        session_id: str,
        key: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> SessionMemoryEntry:
        """
        Create or replace a value in a session.
        """

        if metadata is not None and not isinstance(
            metadata,
            dict,
        ):
            raise TypeError(
                "metadata must be a dictionary"
            )

        resolved_key = self._validate_key(key)

        with self._lock:
            session = self._require_session(
                session_id
            )
            existing_entry = session.entries.get(
                resolved_key
            )

            if existing_entry is None:
                entry = SessionMemoryEntry(
                    key=resolved_key,
                    value=deepcopy(value),
                    metadata=deepcopy(
                        metadata or {}
                    ),
                )
                session.entries[
                    resolved_key
                ] = entry
            else:
                existing_entry.update(
                    value=value,
                    metadata=metadata,
                )
                entry = existing_entry

            self._touch_session(session)
            return deepcopy(entry)

    def get(
        self,
        session_id: str,
        key: str,
        default: Any = None,
    ) -> Any:
        """
        Retrieve a value from a session.
        """

        resolved_key = self._validate_key(key)

        with self._lock:
            session = self._require_session(
                session_id
            )
            entry = session.entries.get(
                resolved_key
            )
            self._touch_session(session)

            if entry is None:
                return deepcopy(default)

            return deepcopy(entry.value)

    def get_entry(
        self,
        session_id: str,
        key: str,
    ) -> SessionMemoryEntry | None:
        """
        Retrieve a complete session-memory entry.
        """

        resolved_key = self._validate_key(key)

        with self._lock:
            session = self._require_session(
                session_id
            )
            entry = session.entries.get(
                resolved_key
            )
            self._touch_session(session)
            return deepcopy(entry)

    def update(
        self,
        session_id: str,
        key: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> SessionMemoryEntry:
        """
        Update an existing entry.
        """

        if metadata is not None and not isinstance(
            metadata,
            dict,
        ):
            raise TypeError(
                "metadata must be a dictionary"
            )

        resolved_key = self._validate_key(key)

        with self._lock:
            session = self._require_session(
                session_id
            )
            entry = session.entries.get(
                resolved_key
            )

            if entry is None:
                raise KeyError(
                    "session entry not found: "
                    f"{resolved_key}"
                )

            entry.update(
                value=value,
                metadata=metadata,
            )
            self._touch_session(session)
            return deepcopy(entry)

    def delete(
        self,
        session_id: str,
        key: str,
    ) -> bool:
        """
        Delete one entry from a session.
        """

        resolved_key = self._validate_key(key)

        with self._lock:
            session = self._require_session(
                session_id
            )
            removed = session.entries.pop(
                resolved_key,
                None,
            )

            if removed is not None:
                self._touch_session(session)
                return True

            return False

    def clear_session(
        self,
        session_id: str,
    ) -> int:
        """
        Remove every entry without deleting the session.
        """

        with self._lock:
            session = self._require_session(
                session_id
            )
            removed_count = len(
                session.entries
            )
            session.entries.clear()
            self._touch_session(session)
            return removed_count

    def delete_session(
        self,
        session_id: str,
    ) -> bool:
        """
        Delete a complete session.
        """

        resolved_session_id = self._validate_session_id(
            session_id
        )

        with self._lock:
            return (
                self._sessions.pop(
                    resolved_session_id,
                    None,
                )
                is not None
            )

    def get_session(
        self,
        session_id: str,
    ) -> SessionRecord:
        """
        Retrieve a complete session record.
        """

        with self._lock:
            session = self._require_session(
                session_id
            )
            self._touch_session(session)
            return deepcopy(session)

    def list_sessions(
        self,
    ) -> list[dict[str, Any]]:
        """
        Return summaries for all active sessions.
        """

        with self._lock:
            self.cleanup_expired_sessions()

            sessions = sorted(
                self._sessions.values(),
                key=lambda item: item.created_at,
            )

            return [
                {
                    "session_id": session.session_id,
                    "created_at": (
                        session.created_at.isoformat()
                    ),
                    "updated_at": (
                        session.updated_at.isoformat()
                    ),
                    "expires_at": (
                        session.expires_at.isoformat()
                    ),
                    "entry_count": len(
                        session.entries
                    ),
                    "metadata": deepcopy(
                        session.metadata
                    ),
                }
                for session in sessions
            ]

    def list_keys(
        self,
        session_id: str,
    ) -> list[str]:
        """
        Return all keys stored in a session.
        """

        with self._lock:
            session = self._require_session(
                session_id
            )
            self._touch_session(session)
            return sorted(session.entries)

    def contains(
        self,
        session_id: str,
        key: str,
    ) -> bool:
        """
        Check whether a key exists in a session.
        """

        resolved_key = self._validate_key(key)

        with self._lock:
            session = self._require_session(
                session_id
            )
            self._touch_session(session)
            return resolved_key in session.entries

    def update_session_metadata(
        self,
        session_id: str,
        metadata: dict[str, Any],
        merge: bool = True,
    ) -> dict[str, Any]:
        """
        Update session-level metadata.
        """

        if not isinstance(metadata, dict):
            raise TypeError(
                "metadata must be a dictionary"
            )

        with self._lock:
            session = self._require_session(
                session_id
            )

            if merge:
                session.metadata.update(
                    deepcopy(metadata)
                )
            else:
                session.metadata = deepcopy(
                    metadata
                )

            self._touch_session(session)
            return deepcopy(session.metadata)

    def cleanup_expired_sessions(
        self,
        current_time: datetime | None = None,
    ) -> int:
        """
        Delete expired sessions.
        """

        comparison_time = (
            current_time or _utc_now()
        )

        if comparison_time.tzinfo is None:
            raise ValueError(
                "current_time must be timezone-aware"
            )

        with self._lock:
            expired_session_ids = [
                session_id
                for session_id, session
                in self._sessions.items()
                if session.is_expired(
                    comparison_time
                )
            ]

            for session_id in (
                expired_session_ids
            ):
                del self._sessions[session_id]

            return len(expired_session_ids)

    def clear_all(self) -> int:
        """
        Delete all sessions.
        """

        with self._lock:
            removed_count = len(
                self._sessions
            )
            self._sessions.clear()
            return removed_count

    def _require_session(
        self,
        session_id: str,
    ) -> SessionRecord:
        """
        Return an existing non-expired session.
        """

        resolved_session_id = self._validate_session_id(
            session_id
        )
        session = self._sessions.get(
            resolved_session_id
        )

        if session is None:
            raise KeyError(
                "session not found: "
                f"{resolved_session_id}"
            )

        if session.is_expired():
            del self._sessions[
                resolved_session_id
            ]
            raise KeyError(
                "session expired: "
                f"{resolved_session_id}"
            )

        return session

    def _touch_session(
        self,
        session: SessionRecord,
    ) -> None:
        """
        Refresh session timestamps when enabled.
        """

        if self._auto_refresh_ttl:
            session.touch(
                self._session_ttl
            )
        else:
            session.updated_at = _utc_now()

    def _remove_oldest_session(
        self,
    ) -> None:
        """
        Remove the least recently updated session.
        """

        if not self._sessions:
            return

        oldest_session_id = min(
            self._sessions,
            key=lambda session_id: (
                self._sessions[
                    session_id
                ].updated_at
            ),
        )

        del self._sessions[
            oldest_session_id
        ]

    @staticmethod
    def _validate_session_id(
        session_id: str,
    ) -> str:
        """
        Validate and normalize a session identifier.
        """

        if not isinstance(session_id, str):
            raise ValueError(
                "session_id must be a non-empty string"
            )

        normalized = session_id.strip()

        if not normalized:
            raise ValueError(
                "session_id must be a non-empty string"
            )

        return normalized

    @staticmethod
    def _validate_key(
        key: str,
    ) -> str:
        """
        Validate and normalize a memory key.
        """

        if not isinstance(key, str):
            raise ValueError(
                "key must be a non-empty string"
            )

        normalized = key.strip()

        if not normalized:
            raise ValueError(
                "key must be a non-empty string"
            )

        return normalized