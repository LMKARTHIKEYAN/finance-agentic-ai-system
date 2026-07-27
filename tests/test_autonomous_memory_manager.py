"""Tests for controlled autonomous memory coordination."""

from pathlib import Path

import pytest

from src.autonomous.schemas import (
    AutonomousExecutionResult,
    EvidenceReference,
    ManagementResponse,
    ReconciliationCheck,
    ReconciliationResult,
    ReviewResult,
)
from src.memory.long_term_memory import LongTermMemory
from src.memory.memory_manager import MemoryManager
from src.memory.session_memory import SessionMemory


def _result(
    *,
    status: str = "completed",
    decision: str = "approved",
) -> AutonomousExecutionResult:
    return AutonomousExecutionResult(
        status=status,
        management_response=(
            ManagementResponse(
                answer="Reviewed management answer.",
                evidence_ids=("pnl-001",),
                caveats=("Limited period.",),
            )
            if status == "completed"
            else None
        ),
        review_result=ReviewResult(
            decision=decision,
            approved_answer=(
                "Reviewed management answer."
                if decision in {"approved", "approved_with_caveats"}
                else None
            ),
            required_caveats=(
                ("Limited period.",)
                if decision == "approved_with_caveats"
                else ()
            ),
        ),
        reconciliation_result=ReconciliationResult(
            passed=True,
            checks=(
                ReconciliationCheck(
                    name="pnl",
                    passed=True,
                    details="P&L reconciled.",
                    evidence_ids=("pnl-001",),
                ),
            ),
        ),
        evidence=(
            EvidenceReference(
                evidence_id="pnl-001",
                source="pnl_tool",
                result_type="pnl",
                reconciled=True,
                period="2026-04",
            ),
        ),
        fallback_flow=(
            "deterministic_planner" if status == "fallback" else None
        ),
        fallback_reason=(
            "Reviewer rejected analysis." if status == "fallback" else None
        ),
    )


@pytest.fixture
def manager(tmp_path: Path) -> MemoryManager:
    session = SessionMemory()
    long_term = LongTermMemory(tmp_path / "memory.db")
    instance = MemoryManager(
        session_memory=session,
        long_term_memory=long_term,
    )
    yield instance
    long_term.close()


def test_approved_result_is_saved_and_restored(
    manager: MemoryManager,
) -> None:
    session_id = manager.create_session(question="Why?")

    memory_id = manager.save_autonomous_execution(
        session_id=session_id,
        result=_result(),
        reporting_scope={"period": "2026-04", "category": "A"},
        memory_key="april",
    )

    assert memory_id
    context = manager.get_autonomous_session_context(session_id)
    assert context["trusted"] is True
    assert context["reporting_scope"]["period"] == "2026-04"
    history = manager.get_autonomous_history("april")
    assert history["management_answer"] == "Reviewed management answer."
    assert history["evidence_ids"] == ["pnl-001"]


def test_approved_with_caveats_retains_caveats(
    manager: MemoryManager,
) -> None:
    session_id = manager.create_session()

    manager.save_autonomous_execution(
        session_id=session_id,
        result=_result(decision="approved_with_caveats"),
        memory_key="caveated",
    )

    history = manager.get_autonomous_history("caveated")
    assert history["required_caveats"] == ["Limited period."]


@pytest.mark.parametrize(
    ("status", "decision"),
    [
        ("fallback", "failed"),
        ("failed", "failed"),
        ("completed", "replan_required"),
    ],
)
def test_untrusted_results_remain_session_only(
    manager: MemoryManager,
    status: str,
    decision: str,
) -> None:
    session_id = manager.create_session()

    memory_id = manager.save_autonomous_execution(
        session_id=session_id,
        result=_result(status=status, decision=decision),
        memory_key=f"{status}-{decision}",
    )

    assert memory_id is None
    assert manager.get_autonomous_session_context(session_id)["trusted"] is False
    with pytest.raises(KeyError):
        manager.get_autonomous_history(f"{status}-{decision}")


def test_disabled_autonomous_memory_does_nothing(
    manager: MemoryManager,
) -> None:
    session_id = manager.create_session()

    assert manager.save_autonomous_execution(
        session_id=session_id,
        result=_result(),
        enabled=False,
    ) is None
    assert manager.get_autonomous_session_context(session_id) == {}
