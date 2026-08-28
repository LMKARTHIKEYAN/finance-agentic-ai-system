"""Phase 1 tests for the bounded observation-driven autonomous loop."""

from __future__ import annotations

from datetime import date

import pytest

from src.autonomous.agent_loop import AgentLoop
from src.autonomous.decision_engine import DecisionEngine
from src.autonomous.goal_builder import GoalBuilder
from src.autonomous.schemas import (
    AutonomousDecision,
    FinanceGoal,
    GoalCompletionCriterion,
    ReportingScope,
)
from src.autonomous.state import AutonomousState
from src.autonomous.stop_conditions import StopConditions
from src.autonomous.tool_executor import ToolExecutionError, ToolExecutor
from src.autonomous.tools.registry import ToolDefinition, ToolRegistry


def test_goal_builder_requests_clarification_for_annual_pnl() -> None:
    goal = GoalBuilder().build(
        "Show P&L for 2026",
        reference_date=date(2026, 8, 27),
        goal_id="goal-pnl",
    )

    assert goal.ambiguities == ("annual_period_basis",)
    assert "calendar year 2026" in goal.clarification_question
    assert "pnl_analysis" in {item.key for item in goal.criteria}


def test_goal_builder_resolves_explicit_month() -> None:
    goal = GoalBuilder().build(
        "Show P&L for August 2026",
        reference_date=date(2026, 8, 27),
    )

    assert goal.ambiguities == ()
    assert goal.reporting_scope.start_date == date(2026, 8, 1)
    assert goal.reporting_scope.end_date == date(2026, 8, 31)


def test_goal_builder_accepts_year_before_month() -> None:
    goal = GoalBuilder().build("P&L 2026 August")

    assert goal.ambiguities == ()
    assert goal.reporting_scope.start_date == date(2026, 8, 1)
    assert goal.reporting_scope.end_date == date(2026, 8, 31)


def test_tool_executor_rejects_arguments_not_in_allow_list() -> None:
    registry = ToolRegistry((
        ToolDefinition(
            name="safe_tool",
            description="Return one safe result.",
            function=lambda: {"value": 1},
            required_inputs=(),
            result_type="test",
            performs_finance_calculation=False,
        ),
    ))
    executor = ToolExecutor(registry=registry)
    decision = AutonomousDecision(
        decision_id="decision-1",
        action="call_tool",
        tool_name="safe_tool",
        arguments={"unsafe": True},
        rationale="Test the executor allow-list.",
    )

    with pytest.raises(ToolExecutionError, match="Unsupported tool arguments"):
        executor.execute(decision)


def test_agent_loop_selects_next_tool_from_previous_observation() -> None:
    def get_revenue() -> dict[str, object]:
        return {
            "summary": "Revenue declined by 12%.",
            "decline_percentage": 12.0,
        }

    def explain_decline(decline_percentage: float) -> dict[str, object]:
        assert decline_percentage == 12.0
        return {
            "summary": "Vehicle volume was the largest supported driver.",
            "driver": "vehicle_volume",
        }

    registry = ToolRegistry((
        ToolDefinition(
            "get_revenue",
            "Retrieve revenue performance.",
            get_revenue,
            (),
            "revenue",
            True,
        ),
        ToolDefinition(
            "explain_decline",
            "Explain a retrieved revenue decline.",
            explain_decline,
            ("decline_percentage",),
            "root_cause",
            True,
        ),
    ))

    def policy(state: AutonomousState, _: ToolRegistry) -> AutonomousDecision:
        if "revenue" not in state.completed_criteria:
            return AutonomousDecision(
                decision_id=f"decision-{state.iteration}",
                action="call_tool",
                tool_name="get_revenue",
                rationale="Revenue evidence is missing.",
                target_criteria=("revenue",),
            )
        if "root_cause" not in state.completed_criteria:
            return AutonomousDecision(
                decision_id=f"decision-{state.iteration}",
                action="call_tool",
                tool_name="explain_decline",
                arguments={
                    "decline_percentage": state.context["decline_percentage"]
                },
                rationale="The revenue observation requires driver analysis.",
                target_criteria=("root_cause",),
            )
        return AutonomousDecision(
            decision_id=f"decision-{state.iteration}",
            action="validate",
            rationale="Evidence is ready for final validation.",
            target_criteria=("final_validation",),
        )

    goal = FinanceGoal(
        goal_id="goal-dynamic",
        original_request="Why did revenue decline?",
        objective="Explain the revenue decline.",
        reporting_scope=ReportingScope(
            start_date=date(2026, 4, 1),
            end_date=date(2026, 6, 30),
        ),
        criteria=(
            GoalCompletionCriterion(key="revenue", description="Revenue retrieved."),
            GoalCompletionCriterion(key="root_cause", description="Driver identified."),
            GoalCompletionCriterion(key="final_validation", description="Evidence validated."),
        ),
    )
    state = AutonomousState(goal)
    engine = DecisionEngine(registry=registry, policy=policy)
    result = AgentLoop(
        decision_engine=engine,
        tool_executor=ToolExecutor(registry=registry),
    ).run(state)

    assert result.status == "completed"
    assert result.stop.outcome == "complete"
    assert result.tool_calls == 2
    assert [item.source for item in result.observations] == [
        "get_revenue",
        "explain_decline",
        "final_validation",
    ]
    assert "12%" in result.answer
    assert "largest supported driver" in result.answer


def test_agent_loop_pauses_for_goal_clarification() -> None:
    registry = ToolRegistry((
        ToolDefinition(
            "noop",
            "A no-op tool.",
            lambda: {},
            (),
            "test",
            False,
        ),
    ))
    goal = GoalBuilder().build("Show P&L for 2026", goal_id="goal-clarify")
    result = AgentLoop(
        decision_engine=DecisionEngine(registry=registry),
        tool_executor=ToolExecutor(registry=registry),
    ).run(AutonomousState(goal))

    assert result.status == "waiting_for_user"
    assert result.stop.outcome == "ask_user"
    assert "calendar year" in result.question


def test_stop_conditions_bound_repeated_tool_failure() -> None:
    def fail() -> None:
        raise RuntimeError("database unavailable")

    registry = ToolRegistry((
        ToolDefinition(
            "failing_tool",
            "Always fail for bounded recovery testing.",
            fail,
            (),
            "test",
            False,
        ),
    ))

    def policy(state: AutonomousState, _: ToolRegistry) -> AutonomousDecision:
        return AutonomousDecision(
            decision_id=f"decision-{state.iteration}",
            action="call_tool",
            tool_name="failing_tool",
            rationale="Retry missing evidence within the bounded loop.",
            target_criteria=("result",),
        )

    goal = FinanceGoal(
        goal_id="goal-failure",
        original_request="Run a bounded failing goal.",
        objective="Verify failure handling.",
        criteria=(GoalCompletionCriterion(key="result", description="Result exists."),),
    )
    state = AutonomousState(goal)
    result = AgentLoop(
        decision_engine=DecisionEngine(registry=registry, policy=policy),
        tool_executor=ToolExecutor(registry=registry),
        stop_conditions=StopConditions(max_consecutive_failures=2),
    ).run(state)

    assert result.status == "failed"
    assert result.stop.outcome == "failed"
    assert result.tool_calls == 2
    assert len(result.observations) == 2
