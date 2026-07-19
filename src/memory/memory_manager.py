
"""
Unified memory manager for the Finance Agentic AI System.

This module coordinates:

- Session memory for temporary workflow context
- Long-term memory for persistent finance history
- User questions
- Uploaded file context
- Agent outputs
- Workflow results
- Management reports
- User preferences

The orchestrator should interact with this manager instead of directly
depending on the underlying session or SQLite memory implementations.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

from src.memory.long_term_memory import (
    LongTermMemory,
    LongTermMemoryEntry,
    LongTermMemorySummary,
)
from src.memory.session_memory import SessionMemory


@dataclass
class FinanceMemoryContext:
    """
    Combined memory context for a finance workflow.

    Attributes:
        session_id:
            Current temporary session identifier.

        user_id:
            Optional user or caller identifier.

        question:
            Current finance question.

        uploaded_files:
            File metadata attached to the workflow.

        workflow_context:
            Temporary workflow values stored in session memory.

        previous_reports:
            Relevant reports retrieved from long-term memory.

        user_preferences:
            Persistent user preferences.

        created_at:
            UTC timestamp when the context was created.
    """

    session_id: str
    user_id: str | None
    question: str | None
    uploaded_files: list[dict[str, Any]]
    workflow_context: dict[str, Any]
    previous_reports: list[LongTermMemoryEntry]
    user_preferences: dict[str, Any]
    created_at: datetime

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the memory context into a deep-copied dictionary.
        """

        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "question": self.question,
            "uploaded_files": deepcopy(
                self.uploaded_files
            ),
            "workflow_context": deepcopy(
                self.workflow_context
            ),
            "previous_reports": [
                report.to_dict()
                for report in self.previous_reports
            ],
            "user_preferences": deepcopy(
                self.user_preferences
            ),
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class WorkflowMemoryResult:
    """
    Result returned after a workflow is stored.

    Attributes:
        session_id:
            Session associated with the workflow.

        workflow_id:
            Unique identifier for the workflow execution.

        report_memory_id:
            Long-term memory ID of the saved report.

        agent_memory_ids:
            Long-term memory IDs of saved agent outputs.

        saved_at:
            UTC timestamp when persistence completed.
    """

    session_id: str
    workflow_id: str
    report_memory_id: str | None
    agent_memory_ids: dict[str, str]
    saved_at: datetime

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the result into a serializable dictionary.
        """

        return {
            "session_id": self.session_id,
            "workflow_id": self.workflow_id,
            "report_memory_id": self.report_memory_id,
            "agent_memory_ids": deepcopy(
                self.agent_memory_ids
            ),
            "saved_at": self.saved_at.isoformat(),
        }


class MemoryManager:
    """
    Coordinate temporary and persistent memory.

    Session memory stores the active workflow state.

    Long-term memory stores reusable finance history such as:

    - management reports
    - agent results
    - workflow summaries
    - user preferences
    - prior analysis context
    """

    SESSION_QUESTION_KEY = "current_question"
    SESSION_USER_ID_KEY = "user_id"
    SESSION_WORKFLOW_ID_KEY = "workflow_id"
    SESSION_FILES_KEY = "uploaded_files"
    SESSION_CONTEXT_KEY = "workflow_context"
    SESSION_AGENT_OUTPUTS_KEY = "agent_outputs"
    SESSION_FINAL_REPORT_KEY = "final_report"
    SESSION_STATUS_KEY = "workflow_status"
    SESSION_CREATED_AT_KEY = "workflow_created_at"
    SESSION_UPDATED_AT_KEY = "workflow_updated_at"

    REPORT_NAMESPACE = "management-reports"
    AGENT_RESULT_NAMESPACE = "agent-results"
    WORKFLOW_NAMESPACE = "workflow-history"
    USER_PREFERENCE_NAMESPACE = "user-preferences"

    def __init__(
        self,
        session_memory: SessionMemory | None = None,
        long_term_memory: LongTermMemory | None = None,
        *,
        database_path: str | Path = (
            "data/long_term_memory.db"
        ),
        session_ttl_seconds: int = 3_600,
        max_sessions: int = 1_000,
    ) -> None:
        """
        Initialize the memory manager.

        Args:
            session_memory:
                Optional existing SessionMemory instance.

            long_term_memory:
                Optional existing LongTermMemory instance.

            database_path:
                SQLite path used when long-term memory is not supplied.

            session_ttl_seconds:
                Default session expiration period.

            max_sessions:
                Maximum number of active sessions.

        Raises:
            ValueError:
                If configuration values are invalid.
        """

        if (
            not isinstance(session_ttl_seconds, int)
            or isinstance(session_ttl_seconds, bool)
            or session_ttl_seconds <= 0
        ):
            raise ValueError(
                "session_ttl_seconds must be a "
                "positive integer"
            )

        if (
            not isinstance(max_sessions, int)
            or isinstance(max_sessions, bool)
            or max_sessions <= 0
        ):
            raise ValueError(
                "max_sessions must be a positive integer"
            )

        session_ttl_minutes = max(
            1,
            ceil(session_ttl_seconds / 60),
        )

        self._session_memory = (
            session_memory
            if session_memory is not None
            else SessionMemory(
                session_ttl_minutes=session_ttl_minutes,
                max_sessions=max_sessions,
            )
        )

        self._long_term_memory = (
            long_term_memory
            if long_term_memory is not None
            else LongTermMemory(
                database_path=database_path
            )
        )

        self._owns_long_term_memory = (
            long_term_memory is None
        )
        self._lock = RLock()

    @property
    def session_memory(self) -> SessionMemory:
        """
        Return the configured session memory instance.
        """

        return self._session_memory

    @property
    def long_term_memory(self) -> LongTermMemory:
        """
        Return the configured long-term memory instance.
        """

        return self._long_term_memory

    def create_session(
        self,
        *,
        user_id: str | None = None,
        question: str | None = None,
        uploaded_files: (
            list[dict[str, Any]] | None
        ) = None,
        workflow_context: (
            dict[str, Any] | None
        ) = None,
        metadata: dict[str, Any] | None = None,
        ttl_seconds: int | None = None,
    ) -> str:
        """
        Create and initialize a finance workflow session.

        Args:
            user_id:
                Optional user identifier.

            question:
                Optional initial finance question.

            uploaded_files:
                Optional uploaded-file metadata.

            workflow_context:
                Optional initial workflow context.

            metadata:
                Optional session metadata.

            ttl_seconds:
                Optional custom session expiration.

        Returns:
            Newly created session ID.
        """

        normalized_user_id = self._optional_string(
            user_id,
            "user_id",
        )
        normalized_question = self._optional_string(
            question,
            "question",
        )
        normalized_files = self._validate_files(
            uploaded_files
        )
        normalized_context = self._validate_dictionary(
            workflow_context,
            "workflow_context",
            default={},
        )
        normalized_metadata = self._validate_dictionary(
            metadata,
            "metadata",
            default={},
        )

        if ttl_seconds is not None:
            if (
                not isinstance(ttl_seconds, int)
                or isinstance(ttl_seconds, bool)
                or ttl_seconds <= 0
            ):
                raise ValueError(
                    "ttl_seconds must be a positive "
                    "integer or None"
                )

        workflow_id = str(uuid4())
        now = self._utc_now()

        session_metadata = deepcopy(
            normalized_metadata
        )
        session_metadata.update(
            {
                "workflow_id": workflow_id,
                "user_id": normalized_user_id,
                "created_at": now.isoformat(),
            }
        )

        with self._lock:
            session_id = (
                self._session_memory.create_session(
                    metadata=session_metadata,
                )
            )

            self._session_memory.set(
                session_id,
                self.SESSION_WORKFLOW_ID_KEY,
                workflow_id,
            )
            self._session_memory.set(
                session_id,
                self.SESSION_USER_ID_KEY,
                normalized_user_id,
            )
            self._session_memory.set(
                session_id,
                self.SESSION_QUESTION_KEY,
                normalized_question,
            )
            self._session_memory.set(
                session_id,
                self.SESSION_FILES_KEY,
                normalized_files,
            )
            self._session_memory.set(
                session_id,
                self.SESSION_CONTEXT_KEY,
                normalized_context,
            )
            self._session_memory.set(
                session_id,
                self.SESSION_AGENT_OUTPUTS_KEY,
                {},
            )
            self._session_memory.set(
                session_id,
                self.SESSION_FINAL_REPORT_KEY,
                None,
            )
            self._session_memory.set(
                session_id,
                self.SESSION_STATUS_KEY,
                "CREATED",
            )
            self._session_memory.set(
                session_id,
                self.SESSION_CREATED_AT_KEY,
                now,
            )
            self._session_memory.set(
                session_id,
                self.SESSION_UPDATED_AT_KEY,
                now,
            )

        return session_id

    def session_exists(
        self,
        session_id: str,
    ) -> bool:
        """
        Return whether a workflow session exists.
        """

        normalized_session_id = (
            self._required_string(
                session_id,
                "session_id",
            )
        )

        return self._session_memory.session_exists(
            normalized_session_id
        )

    def get_workflow_id(
        self,
        session_id: str,
    ) -> str:
        """
        Return the workflow ID assigned to a session.
        """

        value = self._session_memory.get(
            self._required_string(
                session_id,
                "session_id",
            ),
            self.SESSION_WORKFLOW_ID_KEY,
        )

        if not isinstance(value, str) or not value:
            raise ValueError(
                "session does not contain a valid "
                "workflow_id"
            )

        return value

    def set_question(
        self,
        session_id: str,
        question: str,
    ) -> None:
        """
        Store or replace the current finance question.
        """

        self._set_session_value(
            session_id=session_id,
            key=self.SESSION_QUESTION_KEY,
            value=self._required_string(
                question,
                "question",
            ),
        )

    def get_question(
        self,
        session_id: str,
    ) -> str | None:
        """
        Return the current finance question.
        """

        value = self._session_memory.get(
            self._required_string(
                session_id,
                "session_id",
            ),
            self.SESSION_QUESTION_KEY,
            default=None,
        )

        if value is None:
            return None

        return str(value)

    def set_uploaded_files(
        self,
        session_id: str,
        uploaded_files: list[dict[str, Any]],
    ) -> None:
        """
        Replace uploaded-file context for a session.
        """

        self._set_session_value(
            session_id=session_id,
            key=self.SESSION_FILES_KEY,
            value=self._validate_files(
                uploaded_files
            ),
        )

    def add_uploaded_file(
        self,
        session_id: str,
        file_context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Add one uploaded-file record to a session.

        Returns:
            Updated uploaded-file list.
        """

        validated_file = self._validate_dictionary(
            file_context,
            "file_context",
        )

        normalized_session_id = (
            self._required_string(
                session_id,
                "session_id",
            )
        )

        with self._lock:
            uploaded_files = (
                self._session_memory.get(
                    normalized_session_id,
                    self.SESSION_FILES_KEY,
                    default=[],
                )
            )

            if not isinstance(uploaded_files, list):
                uploaded_files = []

            updated_files = deepcopy(uploaded_files)
            updated_files.append(validated_file)

            self._set_session_value(
                session_id=normalized_session_id,
                key=self.SESSION_FILES_KEY,
                value=updated_files,
            )

        return deepcopy(updated_files)

    def get_uploaded_files(
        self,
        session_id: str,
    ) -> list[dict[str, Any]]:
        """
        Return uploaded-file context for a session.
        """

        value = self._session_memory.get(
            self._required_string(
                session_id,
                "session_id",
            ),
            self.SESSION_FILES_KEY,
            default=[],
        )

        if not isinstance(value, list):
            return []

        return deepcopy(value)

    def set_workflow_context(
        self,
        session_id: str,
        context: dict[str, Any],
    ) -> None:
        """
        Replace the complete workflow context.
        """

        self._set_session_value(
            session_id=session_id,
            key=self.SESSION_CONTEXT_KEY,
            value=self._validate_dictionary(
                context,
                "context",
            ),
        )

    def update_workflow_context(
        self,
        session_id: str,
        updates: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Merge values into the workflow context.

        Returns:
            Updated workflow context.
        """

        normalized_updates = (
            self._validate_dictionary(
                updates,
                "updates",
            )
        )
        normalized_session_id = (
            self._required_string(
                session_id,
                "session_id",
            )
        )

        with self._lock:
            current_context = (
                self._session_memory.get(
                    normalized_session_id,
                    self.SESSION_CONTEXT_KEY,
                    default={},
                )
            )

            if not isinstance(
                current_context,
                dict,
            ):
                current_context = {}

            updated_context = deepcopy(
                current_context
            )
            updated_context.update(
                normalized_updates
            )

            self._set_session_value(
                session_id=normalized_session_id,
                key=self.SESSION_CONTEXT_KEY,
                value=updated_context,
            )

        return deepcopy(updated_context)

    def get_workflow_context(
        self,
        session_id: str,
    ) -> dict[str, Any]:
        """
        Return the active workflow context.
        """

        value = self._session_memory.get(
            self._required_string(
                session_id,
                "session_id",
            ),
            self.SESSION_CONTEXT_KEY,
            default={},
        )

        if not isinstance(value, dict):
            return {}

        return deepcopy(value)

    def store_agent_output(
        self,
        session_id: str,
        agent_name: str,
        output: Any,
    ) -> None:
        """
        Store one agent result temporarily in session memory.
        """

        normalized_agent_name = (
            self._required_string(
                agent_name,
                "agent_name",
            )
        )
        normalized_session_id = (
            self._required_string(
                session_id,
                "session_id",
            )
        )

        with self._lock:
            agent_outputs = (
                self._session_memory.get(
                    normalized_session_id,
                    self.SESSION_AGENT_OUTPUTS_KEY,
                    default={},
                )
            )

            if not isinstance(agent_outputs, dict):
                agent_outputs = {}

            updated_outputs = deepcopy(
                agent_outputs
            )
            updated_outputs[
                normalized_agent_name
            ] = deepcopy(output)

            self._set_session_value(
                session_id=normalized_session_id,
                key=self.SESSION_AGENT_OUTPUTS_KEY,
                value=updated_outputs,
            )

    def get_agent_output(
        self,
        session_id: str,
        agent_name: str,
        default: Any = None,
    ) -> Any:
        """
        Return one temporary agent result.
        """

        outputs = self.get_agent_outputs(
            session_id
        )

        return deepcopy(
            outputs.get(agent_name, default)
        )

    def get_agent_outputs(
        self,
        session_id: str,
    ) -> dict[str, Any]:
        """
        Return all temporary agent results.
        """

        value = self._session_memory.get(
            self._required_string(
                session_id,
                "session_id",
            ),
            self.SESSION_AGENT_OUTPUTS_KEY,
            default={},
        )

        if not isinstance(value, dict):
            return {}

        return deepcopy(value)

    def set_final_report(
        self,
        session_id: str,
        report: Any,
    ) -> None:
        """
        Store the completed report in session memory.
        """

        self._set_session_value(
            session_id=session_id,
            key=self.SESSION_FINAL_REPORT_KEY,
            value=report,
        )

    def get_final_report(
        self,
        session_id: str,
    ) -> Any:
        """
        Return the completed report from session memory.
        """

        return self._session_memory.get(
            self._required_string(
                session_id,
                "session_id",
            ),
            self.SESSION_FINAL_REPORT_KEY,
            default=None,
        )

    def set_workflow_status(
        self,
        session_id: str,
        status: str,
    ) -> None:
        """
        Set workflow status such as RUNNING or COMPLETED.
        """

        self._set_session_value(
            session_id=session_id,
            key=self.SESSION_STATUS_KEY,
            value=self._required_string(
                status,
                "status",
            ).upper(),
        )

    def get_workflow_status(
        self,
        session_id: str,
    ) -> str:
        """
        Return the current workflow status.
        """

        value = self._session_memory.get(
            self._required_string(
                session_id,
                "session_id",
            ),
            self.SESSION_STATUS_KEY,
            default="UNKNOWN",
        )

        return str(value)

    def build_context(
        self,
        session_id: str,
        *,
        report_namespace: str | None = None,
        report_tag: str | None = None,
        previous_report_limit: int = 5,
    ) -> FinanceMemoryContext:
        """
        Build combined temporary and persistent memory context.

        Args:
            session_id:
                Active session identifier.

            report_namespace:
                Optional report namespace override.

            report_tag:
                Optional report tag filter.

            previous_report_limit:
                Maximum number of prior reports loaded.

        Returns:
            Combined FinanceMemoryContext.
        """

        if (
            not isinstance(
                previous_report_limit,
                int,
            )
            or isinstance(
                previous_report_limit,
                bool,
            )
            or previous_report_limit < 0
        ):
            raise ValueError(
                "previous_report_limit must be a "
                "non-negative integer"
            )

        normalized_session_id = (
            self._required_string(
                session_id,
                "session_id",
            )
        )

        user_id = self._session_memory.get(
            normalized_session_id,
            self.SESSION_USER_ID_KEY,
            default=None,
        )

        reports: list[LongTermMemoryEntry] = []

        if previous_report_limit > 0:
            namespace = (
                report_namespace
                or self.REPORT_NAMESPACE
            )

            summaries = (
                self._long_term_memory.list_memories(
                    namespace=namespace,
                    tag=report_tag,
                    limit=previous_report_limit,
                    newest_first=True,
                )
            )

            reports = [
                self._long_term_memory.get(
                    summary.memory_id
                )
                for summary in summaries
            ]

        preferences = {}

        if isinstance(user_id, str) and user_id:
            preferences = self.get_user_preferences(
                user_id,
                default={},
            )

        return FinanceMemoryContext(
            session_id=normalized_session_id,
            user_id=(
                user_id
                if isinstance(user_id, str)
                else None
            ),
            question=self.get_question(
                normalized_session_id
            ),
            uploaded_files=self.get_uploaded_files(
                normalized_session_id
            ),
            workflow_context=(
                self.get_workflow_context(
                    normalized_session_id
                )
            ),
            previous_reports=reports,
            user_preferences=preferences,
            created_at=self._utc_now(),
        )

    def save_report(
        self,
        *,
        session_id: str,
        report: Any | None = None,
        namespace: str | None = None,
        key: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        overwrite: bool = True,
    ) -> str:
        """
        Save a management report into long-term memory.

        If report is omitted, the report stored in session memory is used.
        """

        normalized_session_id = (
            self._required_string(
                session_id,
                "session_id",
            )
        )

        resolved_report = (
            report
            if report is not None
            else self.get_final_report(
                normalized_session_id
            )
        )

        if resolved_report is None:
            raise ValueError(
                "report is required because the session "
                "does not contain a final report"
            )

        workflow_id = self.get_workflow_id(
            normalized_session_id
        )

        resolved_namespace = (
            namespace
            or self.REPORT_NAMESPACE
        )

        resolved_key = (
            key
            or self._default_report_key(
                normalized_session_id
            )
        )

        resolved_metadata = (
            self._base_persistent_metadata(
                normalized_session_id
            )
        )
        resolved_metadata.update(
            self._validate_dictionary(
                metadata,
                "metadata",
                default={},
            )
        )
        resolved_metadata.update(
            {
                "workflow_id": workflow_id,
                "memory_type": "management_report",
            }
        )

        resolved_tags = self._merge_tags(
            [
                "management-report",
                "finance",
            ],
            tags,
        )

        return self._long_term_memory.save(
            namespace=resolved_namespace,
            key=resolved_key,
            value=resolved_report,
            metadata=resolved_metadata,
            tags=resolved_tags,
            overwrite=overwrite,
        )

    def save_agent_output(
        self,
        *,
        session_id: str,
        agent_name: str,
        output: Any | None = None,
        namespace: str | None = None,
        key: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        overwrite: bool = True,
    ) -> str:
        """
        Persist one agent output into long-term memory.
        """

        normalized_session_id = (
            self._required_string(
                session_id,
                "session_id",
            )
        )
        normalized_agent_name = (
            self._required_string(
                agent_name,
                "agent_name",
            )
        )

        resolved_output = (
            output
            if output is not None
            else self.get_agent_output(
                normalized_session_id,
                normalized_agent_name,
                default=None,
            )
        )

        if resolved_output is None:
            raise ValueError(
                "agent output is required because the "
                "session does not contain the result"
            )

        workflow_id = self.get_workflow_id(
            normalized_session_id
        )

        resolved_key = (
            key
            or (
                f"{workflow_id}:"
                f"{normalized_agent_name}"
            )
        )

        resolved_metadata = (
            self._base_persistent_metadata(
                normalized_session_id
            )
        )
        resolved_metadata.update(
            self._validate_dictionary(
                metadata,
                "metadata",
                default={},
            )
        )
        resolved_metadata.update(
            {
                "workflow_id": workflow_id,
                "agent_name": (
                    normalized_agent_name
                ),
                "memory_type": "agent_result",
            }
        )

        resolved_tags = self._merge_tags(
            [
                "agent-result",
                normalized_agent_name,
                "finance",
            ],
            tags,
        )

        return self._long_term_memory.save(
            namespace=(
                namespace
                or self.AGENT_RESULT_NAMESPACE
            ),
            key=resolved_key,
            value=resolved_output,
            metadata=resolved_metadata,
            tags=resolved_tags,
            overwrite=overwrite,
        )

    def save_all_agent_outputs(
        self,
        *,
        session_id: str,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        overwrite: bool = True,
    ) -> dict[str, str]:
        """
        Persist every temporary agent output.

        Returns:
            Mapping of agent name to long-term memory ID.
        """

        agent_outputs = self.get_agent_outputs(
            session_id
        )

        memory_ids: dict[str, str] = {}

        for agent_name, output in (
            agent_outputs.items()
        ):
            memory_ids[agent_name] = (
                self.save_agent_output(
                    session_id=session_id,
                    agent_name=agent_name,
                    output=output,
                    metadata=metadata,
                    tags=tags,
                    overwrite=overwrite,
                )
            )

        return memory_ids

    def complete_workflow(
        self,
        *,
        session_id: str,
        report: Any | None = None,
        save_agents: bool = True,
        report_key: str | None = None,
        report_metadata: (
            dict[str, Any] | None
        ) = None,
        report_tags: list[str] | None = None,
    ) -> WorkflowMemoryResult:
        """
        Mark a workflow complete and persist its outputs.

        Returns:
            WorkflowMemoryResult with stored memory IDs.
        """

        normalized_session_id = (
            self._required_string(
                session_id,
                "session_id",
            )
        )

        if report is not None:
            self.set_final_report(
                normalized_session_id,
                report,
            )

        self.set_workflow_status(
            normalized_session_id,
            "COMPLETED",
        )

        workflow_id = self.get_workflow_id(
            normalized_session_id
        )

        final_report = self.get_final_report(
            normalized_session_id
        )

        report_memory_id = None

        if final_report is not None:
            report_memory_id = self.save_report(
                session_id=normalized_session_id,
                report=final_report,
                key=report_key,
                metadata=report_metadata,
                tags=report_tags,
                overwrite=True,
            )

        agent_memory_ids = {}

        if save_agents:
            agent_memory_ids = (
                self.save_all_agent_outputs(
                    session_id=(
                        normalized_session_id
                    ),
                    metadata={
                        "workflow_completed": True,
                    },
                    tags=["completed-workflow"],
                    overwrite=True,
                )
            )

        self._save_workflow_summary(
            normalized_session_id,
            report_memory_id=report_memory_id,
            agent_memory_ids=agent_memory_ids,
        )

        return WorkflowMemoryResult(
            session_id=normalized_session_id,
            workflow_id=workflow_id,
            report_memory_id=report_memory_id,
            agent_memory_ids=agent_memory_ids,
            saved_at=self._utc_now(),
        )

    def save_user_preferences(
        self,
        user_id: str,
        preferences: dict[str, Any],
        *,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> str:
        """
        Create or update persistent user preferences.
        """

        normalized_user_id = (
            self._required_string(
                user_id,
                "user_id",
            )
        )
        normalized_preferences = (
            self._validate_dictionary(
                preferences,
                "preferences",
            )
        )

        resolved_metadata = (
            self._validate_dictionary(
                metadata,
                "metadata",
                default={},
            )
        )
        resolved_metadata.update(
            {
                "user_id": normalized_user_id,
                "memory_type": "user_preferences",
            }
        )

        return self._long_term_memory.upsert(
            namespace=self.USER_PREFERENCE_NAMESPACE,
            key=normalized_user_id,
            value=normalized_preferences,
            metadata=resolved_metadata,
            tags=self._merge_tags(
                ["preferences", "finance"],
                tags,
            ),
        )

    def get_user_preferences(
        self,
        user_id: str,
        *,
        default: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Return persistent preferences for a user.
        """

        normalized_user_id = (
            self._required_string(
                user_id,
                "user_id",
            )
        )

        if not self._long_term_memory.key_exists(
            self.USER_PREFERENCE_NAMESPACE,
            normalized_user_id,
        ):
            return deepcopy(
                default
                if default is not None
                else {}
            )

        entry = self._long_term_memory.get_by_key(
            namespace=self.USER_PREFERENCE_NAMESPACE,
            key=normalized_user_id,
        )

        if not isinstance(entry.value, dict):
            return deepcopy(
                default
                if default is not None
                else {}
            )

        return deepcopy(entry.value)

    def restore_report_to_session(
        self,
        *,
        session_id: str,
        memory_id: str | None = None,
        namespace: str | None = None,
        key: str | None = None,
        context_key: str = "restored_report",
    ) -> LongTermMemoryEntry:
        """
        Restore a persistent report into session context.

        A memory ID or namespace/key pair must be supplied.
        """

        normalized_session_id = (
            self._required_string(
                session_id,
                "session_id",
            )
        )
        normalized_context_key = (
            self._required_string(
                context_key,
                "context_key",
            )
        )

        if memory_id is not None:
            entry = self._long_term_memory.get(
                self._required_string(
                    memory_id,
                    "memory_id",
                )
            )
        elif namespace is not None and key is not None:
            entry = (
                self._long_term_memory.get_by_key(
                    namespace=self._required_string(
                        namespace,
                        "namespace",
                    ),
                    key=self._required_string(
                        key,
                        "key",
                    ),
                )
            )
        else:
            raise ValueError(
                "memory_id or both namespace and "
                "key must be provided"
            )

        self.update_workflow_context(
            normalized_session_id,
            {
                normalized_context_key: (
                    deepcopy(entry.value)
                ),
                (
                    f"{normalized_context_key}"
                    "_metadata"
                ): {
                    "memory_id": entry.memory_id,
                    "namespace": entry.namespace,
                    "key": entry.key,
                    "metadata": deepcopy(
                        entry.metadata
                    ),
                    "tags": deepcopy(entry.tags),
                },
            },
        )

        return entry

    def get_previous_reports(
        self,
        *,
        limit: int = 5,
        namespace: str | None = None,
        tag: str | None = None,
    ) -> list[LongTermMemoryEntry]:
        """
        Return recent persistent reports.
        """

        summaries = (
            self._long_term_memory.list_memories(
                namespace=(
                    namespace
                    or self.REPORT_NAMESPACE
                ),
                tag=tag,
                limit=limit,
                newest_first=True,
            )
        )

        return [
            self._long_term_memory.get(
                summary.memory_id
            )
            for summary in summaries
        ]

    def search_memory(
        self,
        query: str,
        *,
        namespace: str | None = None,
        limit: int = 20,
    ) -> list[LongTermMemorySummary]:
        """
        Search persistent finance memory.
        """

        return self._long_term_memory.search(
            query=query,
            namespace=namespace,
            limit=limit,
        )

    def clear_session(
        self,
        session_id: str,
    ) -> bool:
        """
        Delete a temporary workflow session.
        """

        return self._session_memory.delete_session(
            self._required_string(
                session_id,
                "session_id",
            )
        )

    def cleanup_expired_sessions(self) -> int:
        """
        Remove expired sessions.

        Returns:
            Number of sessions removed.
        """

        return (
            self._session_memory
            .cleanup_expired_sessions()
        )

    def close(self) -> None:
        """
        Close owned persistent resources.
        """

        if self._owns_long_term_memory:
            self._long_term_memory.close()

    def __enter__(self) -> "MemoryManager":
        """
        Enter context-manager usage.
        """

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        """
        Close owned resources when leaving context.
        """

        self.close()

    def _set_session_value(
        self,
        *,
        session_id: str,
        key: str,
        value: Any,
    ) -> None:
        """
        Store one session value and refresh update time.
        """

        normalized_session_id = (
            self._required_string(
                session_id,
                "session_id",
            )
        )

        with self._lock:
            self._session_memory.set(
                normalized_session_id,
                key,
                deepcopy(value),
            )
            self._session_memory.set(
                normalized_session_id,
                self.SESSION_UPDATED_AT_KEY,
                self._utc_now(),
            )

    def _save_workflow_summary(
        self,
        session_id: str,
        *,
        report_memory_id: str | None,
        agent_memory_ids: dict[str, str],
    ) -> str:
        """
        Persist a lightweight workflow-history record.
        """

        workflow_id = self.get_workflow_id(
            session_id
        )

        value = {
            "workflow_id": workflow_id,
            "session_id": session_id,
            "question": self.get_question(
                session_id
            ),
            "status": self.get_workflow_status(
                session_id
            ),
            "report_memory_id": report_memory_id,
            "agent_memory_ids": deepcopy(
                agent_memory_ids
            ),
            "uploaded_files": (
                self.get_uploaded_files(
                    session_id
                )
            ),
            "completed_at": (
                self._utc_now().isoformat()
            ),
        }

        metadata = (
            self._base_persistent_metadata(
                session_id
            )
        )
        metadata.update(
            {
                "memory_type": "workflow_summary",
                "report_memory_id": (
                    report_memory_id
                ),
                "agent_count": len(
                    agent_memory_ids
                ),
            }
        )

        return self._long_term_memory.upsert(
            namespace=self.WORKFLOW_NAMESPACE,
            key=workflow_id,
            value=value,
            metadata=metadata,
            tags=[
                "workflow",
                "completed",
                "finance",
            ],
        )

    def _default_report_key(
        self,
        session_id: str,
    ) -> str:
        """
        Build a stable report key from session context.
        """

        context = self.get_workflow_context(
            session_id
        )

        branch = self._safe_key_component(
            context.get("branch")
            or context.get("location")
        )
        period = self._safe_key_component(
            context.get("period")
            or context.get("reporting_period")
        )

        workflow_id = self.get_workflow_id(
            session_id
        )

        parts = [
            part
            for part in [
                branch,
                period,
            ]
            if part
        ]

        if parts:
            return "-".join(parts)

        return workflow_id

    def _base_persistent_metadata(
        self,
        session_id: str,
    ) -> dict[str, Any]:
        """
        Build common persistent-memory metadata.
        """

        normalized_session_id = (
            self._required_string(
                session_id,
                "session_id",
            )
        )

        user_id = self._session_memory.get(
            normalized_session_id,
            self.SESSION_USER_ID_KEY,
            default=None,
        )
        context = self.get_workflow_context(
            normalized_session_id
        )

        metadata: dict[str, Any] = {
            "session_id": normalized_session_id,
            "user_id": user_id,
            "saved_at": self._utc_now().isoformat(),
        }

        for field_name in (
            "branch",
            "period",
            "reporting_period",
            "currency",
            "business_unit",
        ):
            if field_name in context:
                metadata[field_name] = deepcopy(
                    context[field_name]
                )

        return metadata

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        """
        Convert common objects into a dictionary when useful.

        Long-term memory can already pickle arbitrary Python objects.
        This helper mainly supports metadata or caller-facing use.
        """

        if is_dataclass(value):
            return asdict(value)

        if hasattr(value, "to_dict") and callable(
            value.to_dict
        ):
            return value.to_dict()

        return deepcopy(value)

    @staticmethod
    def _merge_tags(
        required_tags: list[str],
        optional_tags: list[str] | None,
    ) -> list[str]:
        """
        Combine, normalize, deduplicate, and sort tags.
        """

        combined = list(required_tags)

        if optional_tags is not None:
            if not isinstance(optional_tags, list):
                raise TypeError(
                    "tags must be a list of strings"
                )

            combined.extend(optional_tags)

        normalized_tags = []

        for tag in combined:
            if not isinstance(tag, str) or not tag.strip():
                raise ValueError(
                    "tag must be a non-empty string"
                )

            normalized_tags.append(
                tag.strip()
            )

        return sorted(set(normalized_tags))

    @staticmethod
    def _validate_files(
        files: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        """
        Validate uploaded-file context.
        """

        if files is None:
            return []

        if not isinstance(files, list):
            raise TypeError(
                "uploaded_files must be a list "
                "of dictionaries"
            )

        normalized_files = []

        for file_context in files:
            if not isinstance(file_context, dict):
                raise TypeError(
                    "each uploaded file must be "
                    "a dictionary"
                )

            normalized_files.append(
                deepcopy(file_context)
            )

        return normalized_files

    @staticmethod
    def _validate_dictionary(
        value: dict[str, Any] | None,
        field_name: str,
        *,
        default: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Validate and deep-copy a dictionary.
        """

        if value is None:
            return deepcopy(
                default
                if default is not None
                else {}
            )

        if not isinstance(value, dict):
            raise TypeError(
                f"{field_name} must be a dictionary"
            )

        return deepcopy(value)

    @staticmethod
    def _required_string(
        value: Any,
        field_name: str,
    ) -> str:
        """
        Validate a required non-empty string.
        """

        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{field_name} must be a non-empty string"
            )

        return value.strip()

    @staticmethod
    def _optional_string(
        value: Any,
        field_name: str,
    ) -> str | None:
        """
        Validate an optional non-empty string.
        """

        if value is None:
            return None

        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{field_name} must be a non-empty "
                "string or None"
            )

        return value.strip()

    @staticmethod
    def _safe_key_component(
        value: Any,
    ) -> str | None:
        """
        Convert a value into a safe memory-key component.
        """

        if value is None:
            return None

        text = str(value).strip().lower()

        if not text:
            return None

        safe_characters = []

        for character in text:
            if character.isalnum():
                safe_characters.append(character)
            elif character in {
                " ",
                "_",
                "-",
                "/",
            }:
                safe_characters.append("-")

        normalized = "".join(safe_characters)

        while "--" in normalized:
            normalized = normalized.replace(
                "--",
                "-",
            )

        return normalized.strip("-") or None

    @staticmethod
    def _utc_now() -> datetime:
        """
        Return the current timezone-aware UTC datetime.
        """

        return datetime.now(timezone.utc)