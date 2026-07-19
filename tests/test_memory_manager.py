
"""
Tests for the Finance Agentic AI unified memory manager.

These tests verify:

- MemoryManager initialization
- Session creation and workflow IDs
- Question management
- Uploaded-file context
- Workflow context
- Temporary agent outputs
- Final report handling
- Workflow status
- Combined memory context
- Persistent report storage
- Persistent agent-result storage
- Workflow completion
- User preferences
- Report restoration
- Long-term memory search
- Session deletion and cleanup
- Dataclass serialization
- Validation and defensive copying
- Context-manager behaviour
- Realistic FP&A workflows
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from collections.abc import Generator

import pytest

from src.memory.long_term_memory import LongTermMemory
from src.memory.memory_manager import (
    FinanceMemoryContext,
    MemoryManager,
    WorkflowMemoryResult,
)
from src.memory.session_memory import SessionMemory


@dataclass
class SampleAgentResult:
    """
    Serializable sample finance-agent output.
    """

    status: str
    actual_revenue: float
    budget_revenue: float
    variance: float


@pytest.fixture
def manager(tmp_path: Path) -> MemoryManager:
    """
    Create an isolated MemoryManager.
    """

    instance = MemoryManager(
        database_path=(
            tmp_path / "finance_memory.db"
        ),
        session_ttl_seconds=3_600,
        max_sessions=100,
    )

    yield instance

    instance.close()


@pytest.fixture
def supplied_memories(
    tmp_path: Path,
) -> Generator[
    tuple[
    MemoryManager,
    SessionMemory,
    LongTermMemory,
  ],
    None,
    None,
]:
    """
    Create a manager with externally supplied memory instances.
    """

    session_memory = SessionMemory(
        session_ttl_minutes=60,
        max_sessions=100,
    )
    long_term_memory = LongTermMemory(
        database_path=(
            tmp_path / "supplied_memory.db"
        )
    )

    instance = MemoryManager(
        session_memory=session_memory,
        long_term_memory=long_term_memory,
    )

    yield (
        instance,
        session_memory,
        long_term_memory,
    )

    instance.close()
    long_term_memory.close()


def create_finance_session(
    manager: MemoryManager,
    *,
    user_id: str = "finance-user-001",
    question: str = (
        "Analyze Chennai June revenue variance."
    ),
    workflow_context: (
        dict[str, Any] | None
    ) = None,
) -> str:
    """
    Create a reusable finance workflow session.
    """

    resolved_context = (
        workflow_context
        if workflow_context is not None
        else {
            "branch": "Chennai",
            "period": "2026-06",
            "currency": "INR",
        }
    )

    return manager.create_session(
        user_id=user_id,
        question=question,
        uploaded_files=[
            {
                "file_name": (
                    "sample_operational_data.csv"
                ),
                "file_type": "csv",
                "purpose": "actuals",
            },
            {
                "file_name": "sample_budget.csv",
                "file_type": "csv",
                "purpose": "budget",
            },
        ],
        workflow_context=resolved_context,
        metadata={
            "source": "pytest",
        },
    )


def test_constructor_creates_default_memories(
    tmp_path: Path,
):
    """Constructor should create both memory layers."""

    manager = MemoryManager(
        database_path=tmp_path / "memory.db"
    )

    try:
        assert isinstance(
            manager.session_memory,
            SessionMemory,
        )
        assert isinstance(
            manager.long_term_memory,
            LongTermMemory,
        )
    finally:
        manager.close()


def test_constructor_uses_supplied_memories(
    supplied_memories,
):
    """Externally supplied memory objects should be reused."""

    (
        manager,
        session_memory,
        long_term_memory,
    ) = supplied_memories

    assert (
        manager.session_memory
        is session_memory
    )
    assert (
        manager.long_term_memory
        is long_term_memory
    )


@pytest.mark.parametrize(
    "session_ttl_seconds",
    [
        0,
        -1,
        -100,
        1.5,
        "3600",
        True,
    ],
)
def test_constructor_rejects_invalid_session_ttl(
    tmp_path: Path,
    session_ttl_seconds,
):
    """Session TTL must be a positive integer."""

    with pytest.raises(
        ValueError,
        match=(
            "session_ttl_seconds must be a "
            "positive integer"
        ),
    ):
        MemoryManager(
            database_path=tmp_path / "memory.db",
            session_ttl_seconds=(
                session_ttl_seconds
            ),
        )


@pytest.mark.parametrize(
    "max_sessions",
    [
        0,
        -1,
        -100,
        1.5,
        "100",
        True,
    ],
)
def test_constructor_rejects_invalid_max_sessions(
    tmp_path: Path,
    max_sessions,
):
    """Maximum sessions must be a positive integer."""

    with pytest.raises(
        ValueError,
        match=(
            "max_sessions must be a positive integer"
        ),
    ):
        MemoryManager(
            database_path=tmp_path / "memory.db",
            max_sessions=max_sessions,
        )


def test_create_session_returns_session_id(
    manager: MemoryManager,
):
    """create_session() should return a valid session ID."""

    session_id = create_finance_session(manager)

    assert isinstance(session_id, str)
    assert session_id
    assert manager.session_exists(session_id)


def test_create_session_initializes_workflow_values(
    manager: MemoryManager,
):
    """A new session should initialize workflow memory."""

    session_id = create_finance_session(manager)

    assert manager.get_question(session_id) == (
        "Analyze Chennai June revenue variance."
    )
    assert manager.get_workflow_status(
        session_id
    ) == "CREATED"
    assert manager.get_agent_outputs(
        session_id
    ) == {}
    assert manager.get_final_report(
        session_id
    ) is None

    workflow_id = manager.get_workflow_id(
        session_id
    )

    assert isinstance(workflow_id, str)
    assert workflow_id


def test_create_session_generates_unique_workflow_ids(
    manager: MemoryManager,
):
    """Every session should receive a unique workflow ID."""

    first_session = create_finance_session(
        manager
    )
    second_session = create_finance_session(
        manager
    )

    assert (
        manager.get_workflow_id(first_session)
        != manager.get_workflow_id(second_session)
    )


def test_create_session_accepts_minimal_values(
    manager: MemoryManager,
):
    """A session should work without optional context."""

    session_id = manager.create_session()

    assert manager.session_exists(session_id)
    assert manager.get_question(session_id) is None
    assert manager.get_uploaded_files(
        session_id
    ) == []
    assert manager.get_workflow_context(
        session_id
    ) == {}


def test_create_session_normalizes_strings(
    manager: MemoryManager,
):
    """User ID and question should be stripped."""

    session_id = manager.create_session(
        user_id="  finance-user-001  ",
        question="  Analyze revenue.  ",
    )

    assert manager.get_question(session_id) == (
        "Analyze revenue."
    )

    context = manager.build_context(
        session_id,
        previous_report_limit=0,
    )

    assert context.user_id == "finance-user-001"


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("user_id", ""),
        ("user_id", " "),
        ("user_id", 100),
        ("question", ""),
        ("question", " "),
        ("question", 100),
    ],
)
def test_create_session_rejects_invalid_optional_strings(
    manager: MemoryManager,
    field_name,
    field_value,
):
    """Optional strings must be non-empty when supplied."""

    arguments = {
        "user_id": "finance-user-001",
        "question": "Analyze revenue.",
    }
    arguments[field_name] = field_value

    with pytest.raises(
        ValueError,
        match=(
            f"{field_name} must be a non-empty "
            "string or None"
        ),
    ):
        manager.create_session(**arguments)


def test_create_session_rejects_invalid_uploaded_files(
    manager: MemoryManager,
):
    """Uploaded files must be a list of dictionaries."""

    with pytest.raises(
        TypeError,
        match=(
            "uploaded_files must be a list "
            "of dictionaries"
        ),
    ):
        manager.create_session(
            uploaded_files="file.csv",
        )


def test_create_session_rejects_invalid_file_item(
    manager: MemoryManager,
):
    """Each uploaded-file item must be a dictionary."""

    with pytest.raises(
        TypeError,
        match=(
            "each uploaded file must be "
            "a dictionary"
        ),
    ):
        manager.create_session(
            uploaded_files=[
                {
                    "file_name": "actual.csv",
                },
                "budget.csv",
            ],
        )


def test_create_session_rejects_invalid_context(
    manager: MemoryManager,
):
    """Workflow context must be a dictionary."""

    with pytest.raises(
        TypeError,
        match=(
            "workflow_context must be a dictionary"
        ),
    ):
        manager.create_session(
            workflow_context=["invalid"],
        )


def test_session_exists_returns_false_for_missing_session(
    manager: MemoryManager,
):
    """Unknown sessions should not exist."""

    assert manager.session_exists(
        "missing-session"
    ) is False


@pytest.mark.parametrize(
    "session_id",
    [
        "",
        " ",
        None,
        100,
    ],
)
def test_session_exists_rejects_invalid_session_id(
    manager: MemoryManager,
    session_id,
):
    """Session identifiers must be valid strings."""

    with pytest.raises(
        ValueError,
        match=(
            "session_id must be a non-empty string"
        ),
    ):
        manager.session_exists(session_id)


def test_set_and_get_question(
    manager: MemoryManager,
):
    """The current question should be replaceable."""

    session_id = create_finance_session(manager)

    manager.set_question(
        session_id,
        "Explain the GP percentage variance.",
    )

    assert manager.get_question(session_id) == (
        "Explain the GP percentage variance."
    )


def test_set_question_rejects_blank_question(
    manager: MemoryManager,
):
    """A stored question cannot be blank."""

    session_id = create_finance_session(manager)

    with pytest.raises(
        ValueError,
        match="question must be a non-empty string",
    ):
        manager.set_question(session_id, " ")


def test_set_and_get_uploaded_files(
    manager: MemoryManager,
):
    """Uploaded-file context should be replaceable."""

    session_id = create_finance_session(manager)

    files = [
        {
            "file_name": "p_and_l.xlsx",
            "file_type": "xlsx",
        }
    ]

    manager.set_uploaded_files(
        session_id,
        files,
    )

    assert manager.get_uploaded_files(
        session_id
    ) == files


def test_uploaded_files_are_defensively_copied(
    manager: MemoryManager,
):
    """File context should not expose mutable session state."""

    session_id = manager.create_session()

    files = [
        {
            "file_name": "actual.csv",
            "metadata": {
                "branch": "Chennai",
            },
        }
    ]

    manager.set_uploaded_files(
        session_id,
        files,
    )

    files[0]["metadata"]["branch"] = "Changed"

    retrieved_files = manager.get_uploaded_files(
        session_id
    )
    retrieved_files[0]["metadata"][
        "branch"
    ] = "Changed again"

    final_files = manager.get_uploaded_files(
        session_id
    )

    assert final_files[0]["metadata"][
        "branch"
    ] == "Chennai"


def test_add_uploaded_file(
    manager: MemoryManager,
):
    """One file should be appendable to the session."""

    session_id = manager.create_session(
        uploaded_files=[
            {
                "file_name": "actual.csv",
            }
        ]
    )

    updated_files = manager.add_uploaded_file(
        session_id,
        {
            "file_name": "budget.csv",
        },
    )

    assert updated_files == [
        {
            "file_name": "actual.csv",
        },
        {
            "file_name": "budget.csv",
        },
    ]

    assert manager.get_uploaded_files(
        session_id
    ) == updated_files


def test_add_uploaded_file_rejects_non_dictionary(
    manager: MemoryManager,
):
    """Added file context must be a dictionary."""

    session_id = manager.create_session()

    with pytest.raises(
        TypeError,
        match="file_context must be a dictionary",
    ):
        manager.add_uploaded_file(
            session_id,
            "actual.csv",
        )


def test_set_and_get_workflow_context(
    manager: MemoryManager,
):
    """Workflow context should be replaceable."""

    session_id = create_finance_session(manager)

    context = {
        "branch": "Chennai",
        "period": "2026-07",
        "scenario": "forecast",
    }

    manager.set_workflow_context(
        session_id,
        context,
    )

    assert manager.get_workflow_context(
        session_id
    ) == context


def test_update_workflow_context_merges_values(
    manager: MemoryManager,
):
    """Context updates should preserve existing fields."""

    session_id = create_finance_session(manager)

    updated_context = (
        manager.update_workflow_context(
            session_id,
            {
                "scenario": "downside",
                "currency": "USD",
            },
        )
    )

    assert updated_context == {
        "branch": "Chennai",
        "period": "2026-06",
        "currency": "USD",
        "scenario": "downside",
    }


def test_workflow_context_is_defensively_copied(
    manager: MemoryManager,
):
    """Retrieved context should not mutate session state."""

    session_id = manager.create_session(
        workflow_context={
            "filters": {
                "branch": "Chennai",
            }
        }
    )

    context = manager.get_workflow_context(
        session_id
    )
    context["filters"]["branch"] = "Changed"

    restored_context = (
        manager.get_workflow_context(
            session_id
        )
    )

    assert restored_context["filters"][
        "branch"
    ] == "Chennai"


def test_store_and_get_agent_output(
    manager: MemoryManager,
):
    """Temporary agent output should be stored by name."""

    session_id = create_finance_session(manager)

    output = {
        "status": "REVIEW",
        "revenue_variance": -200_000,
    }

    manager.store_agent_output(
        session_id,
        "variance_agent",
        output,
    )

    assert manager.get_agent_output(
        session_id,
        "variance_agent",
    ) == output


def test_store_multiple_agent_outputs(
    manager: MemoryManager,
):
    """Multiple agents should retain separate results."""

    session_id = create_finance_session(manager)

    manager.store_agent_output(
        session_id,
        "variance_agent",
        {
            "variance": -200_000,
        },
    )
    manager.store_agent_output(
        session_id,
        "commentary_agent",
        {
            "commentary": (
                "Revenue was below budget."
            ),
        },
    )

    outputs = manager.get_agent_outputs(
        session_id
    )

    assert set(outputs) == {
        "variance_agent",
        "commentary_agent",
    }


def test_agent_output_is_defensively_copied(
    manager: MemoryManager,
):
    """Agent results should not expose mutable state."""

    session_id = create_finance_session(manager)

    output = {
        "risks": [
            "Revenue below budget",
        ]
    }

    manager.store_agent_output(
        session_id,
        "risk_agent",
        output,
    )

    output["risks"].append("Changed")

    retrieved_output = manager.get_agent_output(
        session_id,
        "risk_agent",
    )
    retrieved_output["risks"].append(
        "Changed again"
    )

    final_output = manager.get_agent_output(
        session_id,
        "risk_agent",
    )

    assert final_output == {
        "risks": [
            "Revenue below budget",
        ]
    }


def test_get_agent_output_returns_default(
    manager: MemoryManager,
):
    """Unknown agent output should return the supplied default."""

    session_id = create_finance_session(manager)

    assert manager.get_agent_output(
        session_id,
        "missing_agent",
        default={
            "status": "NOT_RUN",
        },
    ) == {
        "status": "NOT_RUN",
    }


def test_store_agent_output_rejects_blank_name(
    manager: MemoryManager,
):
    """Agent names must be non-empty."""

    session_id = create_finance_session(manager)

    with pytest.raises(
        ValueError,
        match=(
            "agent_name must be a non-empty string"
        ),
    ):
        manager.store_agent_output(
            session_id,
            " ",
            {},
        )


def test_set_and_get_final_report(
    manager: MemoryManager,
):
    """The final report should remain in session memory."""

    session_id = create_finance_session(manager)

    report = {
        "overall_status": "REVIEW",
        "executive_summary": (
            "Revenue was below budget."
        ),
    }

    manager.set_final_report(
        session_id,
        report,
    )

    assert manager.get_final_report(
        session_id
    ) == report


def test_set_and_get_workflow_status(
    manager: MemoryManager,
):
    """Workflow status should be normalized to uppercase."""

    session_id = create_finance_session(manager)

    manager.set_workflow_status(
        session_id,
        "running",
    )

    assert manager.get_workflow_status(
        session_id
    ) == "RUNNING"


def test_build_context_returns_combined_context(
    manager: MemoryManager,
):
    """build_context() should combine session values."""

    session_id = create_finance_session(manager)

    context = manager.build_context(
        session_id,
        previous_report_limit=0,
    )

    assert isinstance(
        context,
        FinanceMemoryContext,
    )
    assert context.session_id == session_id
    assert context.user_id == (
        "finance-user-001"
    )
    assert context.question == (
        "Analyze Chennai June revenue variance."
    )
    assert len(context.uploaded_files) == 2
    assert context.workflow_context[
        "branch"
    ] == "Chennai"
    assert context.previous_reports == []
    assert context.user_preferences == {}
    assert context.created_at.tzinfo is not None


def test_build_context_loads_user_preferences(
    manager: MemoryManager,
):
    """Persistent user preferences should enter context."""

    manager.save_user_preferences(
        "finance-user-001",
        {
            "preferred_branch": "Chennai",
            "currency": "INR",
            "report_detail": "management",
        },
    )

    session_id = create_finance_session(manager)

    context = manager.build_context(
        session_id,
        previous_report_limit=0,
    )

    assert context.user_preferences == {
        "preferred_branch": "Chennai",
        "currency": "INR",
        "report_detail": "management",
    }


def test_build_context_loads_previous_reports(
    manager: MemoryManager,
):
    """Previous reports should be loaded from long-term memory."""

    first_session = create_finance_session(
        manager,
        workflow_context={
            "branch": "Chennai",
            "period": "2026-05",
        },
    )
    manager.set_final_report(
        first_session,
        {
            "period": "2026-05",
            "status": "REVIEW",
        },
    )
    manager.save_report(
        session_id=first_session,
    )

    second_session = create_finance_session(
        manager,
        workflow_context={
            "branch": "Chennai",
            "period": "2026-06",
        },
    )

    context = manager.build_context(
        second_session,
        previous_report_limit=5,
    )

    assert len(context.previous_reports) == 1
    assert context.previous_reports[0].value[
        "period"
    ] == "2026-05"


@pytest.mark.parametrize(
    "limit",
    [
        -1,
        1.5,
        "5",
        True,
    ],
)
def test_build_context_rejects_invalid_report_limit(
    manager: MemoryManager,
    limit,
):
    """Previous-report limit must be non-negative."""

    session_id = create_finance_session(manager)

    with pytest.raises(
        ValueError,
        match=(
            "previous_report_limit must be a "
            "non-negative integer"
        ),
    ):
        manager.build_context(
            session_id,
            previous_report_limit=limit,
        )


def test_finance_memory_context_to_dict(
    manager: MemoryManager,
):
    """FinanceMemoryContext should serialize cleanly."""

    session_id = create_finance_session(manager)

    context = manager.build_context(
        session_id,
        previous_report_limit=0,
    )

    result = context.to_dict()

    assert result["session_id"] == session_id
    assert result["user_id"] == (
        "finance-user-001"
    )
    assert isinstance(
        result["created_at"],
        str,
    )
    assert result["previous_reports"] == []


def test_finance_memory_context_to_dict_deep_copies(
    manager: MemoryManager,
):
    """Serialized context should not expose internal values."""

    session_id = create_finance_session(manager)

    context = manager.build_context(
        session_id,
        previous_report_limit=0,
    )

    result = context.to_dict()
    result["uploaded_files"][0][
        "file_name"
    ] = "changed.csv"
    result["workflow_context"][
        "branch"
    ] = "Changed"

    assert context.uploaded_files[0][
        "file_name"
    ] == "sample_operational_data.csv"
    assert context.workflow_context[
        "branch"
    ] == "Chennai"


def test_save_report_uses_session_report(
    manager: MemoryManager,
):
    """save_report() should use the session report by default."""

    session_id = create_finance_session(manager)

    manager.set_final_report(
        session_id,
        {
            "status": "REVIEW",
            "revenue_variance": -200_000,
        },
    )

    memory_id = manager.save_report(
        session_id=session_id,
    )

    entry = manager.long_term_memory.get(
        memory_id
    )

    assert entry.namespace == (
        manager.REPORT_NAMESPACE
    )
    assert entry.key == "chennai-2026-06"
    assert entry.value["status"] == "REVIEW"
    assert entry.metadata["branch"] == "Chennai"
    assert entry.metadata["period"] == "2026-06"
    assert entry.metadata["memory_type"] == (
        "management_report"
    )
    assert "management-report" in entry.tags
    assert "finance" in entry.tags


def test_save_report_accepts_direct_report(
    manager: MemoryManager,
):
    """A report may be supplied directly."""

    session_id = create_finance_session(manager)

    memory_id = manager.save_report(
        session_id=session_id,
        report={
            "status": "PASS",
        },
        key="custom-report-key",
        metadata={
            "approved": True,
        },
        tags=[
            "approved",
        ],
    )

    entry = manager.long_term_memory.get(
        memory_id
    )

    assert entry.key == "custom-report-key"
    assert entry.value == {
        "status": "PASS",
    }
    assert entry.metadata["approved"] is True
    assert "approved" in entry.tags


def test_save_report_uses_workflow_id_without_context(
    manager: MemoryManager,
):
    """Default report key should fall back to workflow ID."""

    session_id = manager.create_session()
    workflow_id = manager.get_workflow_id(
        session_id
    )

    memory_id = manager.save_report(
        session_id=session_id,
        report={
            "status": "PASS",
        },
    )

    entry = manager.long_term_memory.get(
        memory_id
    )

    assert entry.key == workflow_id


def test_save_report_requires_report(
    manager: MemoryManager,
):
    """Report persistence should fail when no report exists."""

    session_id = create_finance_session(manager)

    with pytest.raises(
        ValueError,
        match=(
            "report is required because the session "
            "does not contain a final report"
        ),
    ):
        manager.save_report(
            session_id=session_id
        )


def test_save_report_overwrites_same_business_key(
    manager: MemoryManager,
):
    """Repeated report saves should update the same key."""

    session_id = create_finance_session(manager)

    first_memory_id = manager.save_report(
        session_id=session_id,
        report={
            "status": "REVIEW",
        },
    )
    second_memory_id = manager.save_report(
        session_id=session_id,
        report={
            "status": "APPROVED",
        },
    )

    assert second_memory_id == first_memory_id
    assert manager.long_term_memory.count == 1
    assert manager.long_term_memory.get(
        first_memory_id
    ).value == {
        "status": "APPROVED",
    }


def test_save_agent_output_from_session(
    manager: MemoryManager,
):
    """An agent output should persist from session memory."""

    session_id = create_finance_session(manager)

    manager.store_agent_output(
        session_id,
        "variance_agent",
        {
            "revenue_variance": -200_000,
        },
    )

    memory_id = manager.save_agent_output(
        session_id=session_id,
        agent_name="variance_agent",
    )

    entry = manager.long_term_memory.get(
        memory_id
    )

    assert entry.namespace == (
        manager.AGENT_RESULT_NAMESPACE
    )
    assert entry.value == {
        "revenue_variance": -200_000,
    }
    assert entry.metadata["agent_name"] == (
        "variance_agent"
    )
    assert entry.metadata["memory_type"] == (
        "agent_result"
    )
    assert "variance_agent" in entry.tags


def test_save_agent_output_accepts_direct_output(
    manager: MemoryManager,
):
    """An agent output may be supplied directly."""

    session_id = create_finance_session(manager)

    memory_id = manager.save_agent_output(
        session_id=session_id,
        agent_name="kpi_agent",
        output={
            "revenue_per_order": 250.0,
        },
        key="custom-kpi-result",
    )

    entry = manager.long_term_memory.get(
        memory_id
    )

    assert entry.key == "custom-kpi-result"
    assert entry.value[
        "revenue_per_order"
    ] == 250.0


def test_save_agent_output_requires_output(
    manager: MemoryManager,
):
    """Missing agent output should raise an error."""

    session_id = create_finance_session(manager)

    with pytest.raises(
        ValueError,
        match=(
            "agent output is required because the "
            "session does not contain the result"
        ),
    ):
        manager.save_agent_output(
            session_id=session_id,
            agent_name="missing_agent",
        )


def test_save_all_agent_outputs(
    manager: MemoryManager,
):
    """All temporary outputs should persist."""

    session_id = create_finance_session(manager)

    manager.store_agent_output(
        session_id,
        "variance_agent",
        {
            "variance": -200_000,
        },
    )
    manager.store_agent_output(
        session_id,
        "commentary_agent",
        {
            "commentary": (
                "Revenue was below budget."
            ),
        },
    )

    memory_ids = (
        manager.save_all_agent_outputs(
            session_id=session_id
        )
    )

    assert set(memory_ids) == {
        "variance_agent",
        "commentary_agent",
    }

    assert (
        manager.long_term_memory.count_by_namespace(
            manager.AGENT_RESULT_NAMESPACE
        )
        == 2
    )


def test_save_all_agent_outputs_returns_empty_mapping(
    manager: MemoryManager,
):
    """A session without outputs should save nothing."""

    session_id = create_finance_session(manager)

    assert manager.save_all_agent_outputs(
        session_id=session_id
    ) == {}


def test_complete_workflow_persists_report_and_agents(
    manager: MemoryManager,
):
    """Workflow completion should persist all outputs."""

    session_id = create_finance_session(manager)

    manager.store_agent_output(
        session_id,
        "variance_agent",
        {
            "variance": -200_000,
        },
    )
    manager.store_agent_output(
        session_id,
        "recommendation_agent",
        {
            "recommendations": [
                "Review volume gaps.",
            ]
        },
    )

    result = manager.complete_workflow(
        session_id=session_id,
        report={
            "overall_status": "REVIEW",
            "executive_summary": (
                "Revenue was below budget."
            ),
        },
    )

    assert isinstance(
        result,
        WorkflowMemoryResult,
    )
    assert result.session_id == session_id
    assert result.workflow_id == (
        manager.get_workflow_id(session_id)
    )
    assert result.report_memory_id is not None
    assert set(result.agent_memory_ids) == {
        "variance_agent",
        "recommendation_agent",
    }
    assert result.saved_at.tzinfo is not None
    assert manager.get_workflow_status(
        session_id
    ) == "COMPLETED"

    assert (
        manager.long_term_memory.count_by_namespace(
            manager.REPORT_NAMESPACE
        )
        == 1
    )
    assert (
        manager.long_term_memory.count_by_namespace(
            manager.AGENT_RESULT_NAMESPACE
        )
        == 2
    )
    assert (
        manager.long_term_memory.count_by_namespace(
            manager.WORKFLOW_NAMESPACE
        )
        == 1
    )


def test_complete_workflow_can_skip_agent_persistence(
    manager: MemoryManager,
):
    """Agent outputs should be optional at completion."""

    session_id = create_finance_session(manager)

    manager.store_agent_output(
        session_id,
        "variance_agent",
        {
            "variance": -200_000,
        },
    )

    result = manager.complete_workflow(
        session_id=session_id,
        report={
            "status": "PASS",
        },
        save_agents=False,
    )

    assert result.agent_memory_ids == {}
    assert (
        manager.long_term_memory.count_by_namespace(
            manager.AGENT_RESULT_NAMESPACE
        )
        == 0
    )


def test_complete_workflow_without_report(
    manager: MemoryManager,
):
    """A workflow summary can persist without a report."""

    session_id = create_finance_session(manager)

    result = manager.complete_workflow(
        session_id=session_id,
        save_agents=False,
    )

    assert result.report_memory_id is None
    assert result.agent_memory_ids == {}
    assert manager.get_workflow_status(
        session_id
    ) == "COMPLETED"
    assert (
        manager.long_term_memory.count_by_namespace(
            manager.WORKFLOW_NAMESPACE
        )
        == 1
    )


def test_workflow_memory_result_to_dict(
    manager: MemoryManager,
):
    """WorkflowMemoryResult should serialize itself."""

    session_id = create_finance_session(manager)

    result = manager.complete_workflow(
        session_id=session_id,
        report={
            "status": "PASS",
        },
        save_agents=False,
    )

    serialized = result.to_dict()

    assert serialized["session_id"] == session_id
    assert serialized["workflow_id"] == (
        result.workflow_id
    )
    assert serialized["report_memory_id"] == (
        result.report_memory_id
    )
    assert serialized["agent_memory_ids"] == {}
    assert isinstance(
        serialized["saved_at"],
        str,
    )


def test_save_and_get_user_preferences(
    manager: MemoryManager,
):
    """User preferences should persist by user ID."""

    memory_id = manager.save_user_preferences(
        "finance-user-001",
        {
            "preferred_branch": "Chennai",
            "currency": "INR",
        },
    )

    assert isinstance(memory_id, str)

    preferences = manager.get_user_preferences(
        "finance-user-001"
    )

    assert preferences == {
        "preferred_branch": "Chennai",
        "currency": "INR",
    }


def test_save_user_preferences_upserts_existing_value(
    manager: MemoryManager,
):
    """Saving preferences again should replace the value."""

    first_memory_id = (
        manager.save_user_preferences(
            "finance-user-001",
            {
                "currency": "INR",
            },
        )
    )

    second_memory_id = (
        manager.save_user_preferences(
            "finance-user-001",
            {
                "currency": "USD",
                "preferred_branch": "Chennai",
            },
        )
    )

    assert second_memory_id == first_memory_id
    assert manager.get_user_preferences(
        "finance-user-001"
    ) == {
        "currency": "USD",
        "preferred_branch": "Chennai",
    }


def test_get_user_preferences_returns_default(
    manager: MemoryManager,
):
    """Missing preferences should return a copy of default."""

    default = {
        "currency": "INR",
    }

    result = manager.get_user_preferences(
        "missing-user",
        default=default,
    )

    result["currency"] = "USD"

    assert default == {
        "currency": "INR",
    }


def test_save_user_preferences_rejects_invalid_value(
    manager: MemoryManager,
):
    """Preferences must be a dictionary."""

    with pytest.raises(
        TypeError,
        match="preferences must be a dictionary",
    ):
        manager.save_user_preferences(
            "finance-user-001",
            ["INR"],
        )


def test_restore_report_to_session_by_memory_id(
    manager: MemoryManager,
):
    """A report should restore by memory ID."""

    source_session = create_finance_session(
        manager
    )

    memory_id = manager.save_report(
        session_id=source_session,
        report={
            "period": "2026-06",
            "status": "REVIEW",
        },
    )

    target_session = manager.create_session(
        workflow_context={
            "branch": "Chennai",
            "period": "2026-07",
        }
    )

    entry = manager.restore_report_to_session(
        session_id=target_session,
        memory_id=memory_id,
    )

    context = manager.get_workflow_context(
        target_session
    )

    assert entry.memory_id == memory_id
    assert context["restored_report"] == {
        "period": "2026-06",
        "status": "REVIEW",
    }
    assert context[
        "restored_report_metadata"
    ]["memory_id"] == memory_id


def test_restore_report_to_session_by_key(
    manager: MemoryManager,
):
    """A report should restore using namespace and key."""

    source_session = create_finance_session(
        manager
    )

    manager.save_report(
        session_id=source_session,
        report={
            "status": "PASS",
        },
        key="approved-june-report",
    )

    target_session = manager.create_session()

    manager.restore_report_to_session(
        session_id=target_session,
        namespace=manager.REPORT_NAMESPACE,
        key="approved-june-report",
        context_key="comparison_report",
    )

    context = manager.get_workflow_context(
        target_session
    )

    assert context["comparison_report"] == {
        "status": "PASS",
    }
    assert context[
        "comparison_report_metadata"
    ]["key"] == "approved-june-report"


def test_restore_report_requires_locator(
    manager: MemoryManager,
):
    """A memory ID or namespace/key must be supplied."""

    session_id = manager.create_session()

    with pytest.raises(
        ValueError,
        match=(
            "memory_id or both namespace and "
            "key must be provided"
        ),
    ):
        manager.restore_report_to_session(
            session_id=session_id
        )


def test_get_previous_reports(
    manager: MemoryManager,
):
    """Recent report entries should be returned."""

    for period in (
        "2026-04",
        "2026-05",
        "2026-06",
    ):
        session_id = create_finance_session(
            manager,
            workflow_context={
                "branch": "Chennai",
                "period": period,
            },
        )

        manager.save_report(
            session_id=session_id,
            report={
                "period": period,
            },
            tags=["monthly"],
        )

    reports = manager.get_previous_reports(
        limit=2,
        tag="monthly",
    )

    assert len(reports) == 2
    assert all(
        report.namespace
        == manager.REPORT_NAMESPACE
        for report in reports
    )


def test_search_memory(
    manager: MemoryManager,
):
    """Persistent memory should support text search."""

    session_id = create_finance_session(manager)

    manager.save_report(
        session_id=session_id,
        report={
            "status": "REVIEW",
        },
        key="chennai-revenue-variance",
        metadata={
            "analysis_type": "variance",
        },
    )

    results = manager.search_memory(
        "variance",
        namespace=manager.REPORT_NAMESPACE,
    )

    assert len(results) == 1
    assert results[0].key == (
        "chennai-revenue-variance"
    )


def test_clear_session(
    manager: MemoryManager,
):
    """clear_session() should delete temporary memory."""

    session_id = create_finance_session(manager)

    deleted = manager.clear_session(
        session_id
    )

    assert deleted is True
    assert manager.session_exists(
        session_id
    ) is False


def test_clear_session_returns_false_when_missing(
    manager: MemoryManager,
):
    """Deleting a missing session should return False."""

    assert manager.clear_session(
        "missing-session"
    ) is False


def test_cleanup_expired_sessions_returns_integer(
    manager: MemoryManager,
):
    """Cleanup should return the number removed."""

    manager.create_session()

    removed_count = (
        manager.cleanup_expired_sessions()
    )

    assert isinstance(removed_count, int)
    assert removed_count >= 0


def test_context_manager_persists_long_term_memory(
    tmp_path: Path,
):
    """Context-manager use should preserve SQLite data."""

    database_path = (
        tmp_path / "persistent_manager.db"
    )

    with MemoryManager(
        database_path=database_path
    ) as manager:
        session_id = create_finance_session(
            manager
        )
        memory_id = manager.save_report(
            session_id=session_id,
            report={
                "status": "PASS",
            },
        )

    with MemoryManager(
        database_path=database_path
    ) as restored_manager:
        restored_entry = (
            restored_manager.long_term_memory.get(
                memory_id
            )
        )

        assert restored_entry.value == {
            "status": "PASS",
        }


def test_close_does_not_close_supplied_long_term_memory(
    supplied_memories,
):
    """Manager should not own an injected long-term store."""

    (
        manager,
        _,
        long_term_memory,
    ) = supplied_memories

    manager.close()

    memory_id = long_term_memory.save(
        namespace="reports",
        key="after-manager-close",
        value={
            "status": "PASS",
        },
    )

    assert long_term_memory.exists(
        memory_id
    ) is True


def test_merge_tags_normalizes_and_deduplicates():
    """Tag merging should normalize duplicate values."""

    result = MemoryManager._merge_tags(
        [
            "finance",
            "management-report",
        ],
        [
            "finance",
            " monthly ",
        ],
    )

    assert result == [
        "finance",
        "management-report",
        "monthly",
    ]


def test_merge_tags_rejects_non_list_optional_tags():
    """Optional tags must be supplied as a list."""

    with pytest.raises(
        TypeError,
        match="tags must be a list of strings",
    ):
        MemoryManager._merge_tags(
            ["finance"],
            ("monthly",),
        )


@pytest.mark.parametrize(
    "tags",
    [
        [""],
        [" "],
        [100],
        [None],
    ],
)
def test_merge_tags_rejects_invalid_tag_values(
    tags,
):
    """Every merged tag must be a non-empty string."""

    with pytest.raises(
        ValueError,
        match="tag must be a non-empty string",
    ):
        MemoryManager._merge_tags(
            ["finance"],
            tags,
        )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Chennai", "chennai"),
        ("June 2026", "june-2026"),
        ("GP_Percentage", "gp-percentage"),
        ("Actual/Budget", "actual-budget"),
        ("  Revenue  ", "revenue"),
        ("---", None),
        ("", None),
        (None, None),
    ],
)
def test_safe_key_component(
    value,
    expected,
):
    """Key values should be normalized safely."""

    assert (
        MemoryManager._safe_key_component(
            value
        )
        == expected
    )


def test_serialize_value_converts_dataclass():
    """Dataclass results should serialize to dictionaries."""

    result = SampleAgentResult(
        status="REVIEW",
        actual_revenue=1_000_000,
        budget_revenue=1_200_000,
        variance=-200_000,
    )

    serialized = (
        MemoryManager._serialize_value(
            result
        )
    )

    assert serialized == {
        "status": "REVIEW",
        "actual_revenue": 1_000_000,
        "budget_revenue": 1_200_000,
        "variance": -200_000,
    }


def test_utc_now_returns_timezone_aware_datetime():
    """UTC helper should return an aware datetime."""

    value = MemoryManager._utc_now()

    assert isinstance(value, datetime)
    assert value.tzinfo is not None
    assert value.utcoffset().total_seconds() == 0


def test_realistic_fp_and_a_memory_workflow(
    tmp_path: Path,
):
    """
    MemoryManager should support an end-to-end FP&A workflow.
    """

    database_path = (
        tmp_path / "fp_and_a_memory.db"
    )

    with MemoryManager(
        database_path=database_path
    ) as manager:
        manager.save_user_preferences(
            "finance-user-001",
            {
                "preferred_branch": "Chennai",
                "currency": "INR",
                "report_detail": "management",
            },
        )

        session_id = manager.create_session(
            user_id="finance-user-001",
            question=(
                "Analyze June actual versus budget "
                "for Chennai."
            ),
            uploaded_files=[
                {
                    "file_name": (
                        "sample_operational_data.csv"
                    ),
                    "purpose": "actuals",
                },
                {
                    "file_name": "sample_budget.csv",
                    "purpose": "budget",
                },
            ],
            workflow_context={
                "branch": "Chennai",
                "period": "2026-06",
                "currency": "INR",
                "analysis_type": (
                    "actual_vs_budget"
                ),
            },
        )

        manager.set_workflow_status(
            session_id,
            "RUNNING",
        )

        manager.store_agent_output(
            session_id,
            "validation_agent",
            {
                "status": "PASS",
                "issues": [],
            },
        )

        manager.store_agent_output(
            session_id,
            "variance_agent",
            SampleAgentResult(
                status="REVIEW",
                actual_revenue=1_000_000,
                budget_revenue=1_200_000,
                variance=-200_000,
            ),
        )

        manager.store_agent_output(
            session_id,
            "recommendation_agent",
            {
                "recommendations": [
                    (
                        "Review category-level "
                        "volume gaps."
                    ),
                    (
                        "Investigate weak price "
                        "realization."
                    ),
                ]
            },
        )

        report = {
            "overall_status": "REVIEW",
            "executive_summary": (
                "June revenue was ₹2.00 lakh "
                "below budget."
            ),
            "actual_revenue": 1_000_000,
            "budget_revenue": 1_200_000,
            "revenue_variance": -200_000,
        }

        result = manager.complete_workflow(
            session_id=session_id,
            report=report,
            report_tags=[
                "monthly",
                "actual-vs-budget",
                "chennai",
            ],
        )

        assert result.report_memory_id is not None
        assert len(
            result.agent_memory_ids
        ) == 3

        built_context = manager.build_context(
            session_id,
            previous_report_limit=5,
        )

        assert built_context.user_preferences[
            "preferred_branch"
        ] == "Chennai"
        assert len(
            built_context.previous_reports
        ) == 1

        persistent_report = (
            manager.long_term_memory.get(
                result.report_memory_id
            )
        )

        assert persistent_report.value[
            "revenue_variance"
        ] == -200_000
        assert persistent_report.metadata[
            "branch"
        ] == "Chennai"
        assert "monthly" in (
            persistent_report.tags
        )

    with MemoryManager(
        database_path=database_path
    ) as restored_manager:
        reports = (
            restored_manager.get_previous_reports(
                limit=5,
                tag="monthly",
            )
        )

        assert len(reports) == 1
        assert reports[0].value[
            "overall_status"
        ] == "REVIEW"

        preferences = (
            restored_manager.get_user_preferences(
                "finance-user-001"
            )
        )

        assert preferences[
            "currency"
        ] == "INR"

