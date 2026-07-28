"""Tests for the service-facing autonomous runtime callable."""

from types import SimpleNamespace
from contextvars import ContextVar

import pandas as pd

from src.autonomous.runtime import AutonomousRuntime
from src.autonomous.schemas import (
    AutonomousExecutionResult,
    ComplexityDecision,
    ManagementResponse,
)
from src.api.service import AskServiceResult, FinanceAskService
from src.autonomous.service_executor import AutonomousServiceExecutor


class FakeGraph:
    def __init__(self, result: AutonomousExecutionResult) -> None:
        self.result = result
        self.states: list[dict] = []

    def invoke(self, state: dict) -> dict:
        self.states.append(state)
        return {"result": self.result}


def _runtime(graph: FakeGraph) -> AutonomousRuntime:
    return AutonomousRuntime(
        llm_client=SimpleNamespace(),
        supervisor=SimpleNamespace(),
        reviewer=SimpleNamespace(),
        validator=SimpleNamespace(),
        coordinator=SimpleNamespace(),
        graph=graph,
        limits=SimpleNamespace(),
    )


def test_service_executor_invokes_graph_with_internal_context() -> None:
    completed = AutonomousExecutionResult(
        status="completed",
        management_response=ManagementResponse(
            answer="Autonomous answer.",
            evidence_ids=("pnl-001",),
        ),
    )
    graph = FakeGraph(completed)
    executor = AutonomousServiceExecutor(_runtime(graph))
    operations = pd.DataFrame({"commission_amount": [100.0]})

    result = executor(
        "Why did profit improve?",
        SimpleNamespace(answer="Deterministic draft."),
        {
            "graph_state": {
                "operations_data": operations,
                "operations_result": {"total_revenue": 100.0},
                "anomaly_result": {"anomalies": []},
            }
        },
    )

    assert result == completed
    arguments = graph.states[0]["execution_arguments"]
    assert arguments["context"].operations_data is operations
    assert graph.states[0]["datasets"][0].row_count == 1


def test_service_executor_falls_back_without_internal_context() -> None:
    graph = FakeGraph(AutonomousExecutionResult(status="completed"))
    executor = AutonomousServiceExecutor(_runtime(graph))

    result = executor(
        "Why did profit improve?",
        SimpleNamespace(answer="Draft."),
        None,
    )

    assert result.status == "fallback"
    assert result.fallback_flow == "deterministic_planner"
    assert graph.states == []


def test_service_executor_hides_runtime_exception() -> None:
    class BrokenGraph(FakeGraph):
        def invoke(self, state: dict) -> dict:
            raise RuntimeError("secret API key and dataframe details")

    executor = AutonomousServiceExecutor(
        _runtime(BrokenGraph(AutonomousExecutionResult(status="failed")))
    )
    result = executor(
        "Why?",
        SimpleNamespace(answer="Draft."),
        {"graph_state": {}},
    )

    assert result.status == "fallback"
    assert "secret" not in result.fallback_reason


def test_finance_service_passes_request_local_internal_context() -> None:
    service = object.__new__(FinanceAskService)
    service._autonomous_context = ContextVar("test_context", default=None)
    service._autonomous_enabled = True
    service._autonomous_shadow_mode = False

    class Classifier:
        def classify(self, request: str) -> ComplexityDecision:
            return ComplexityDecision(
                execution_mode="autonomous",
                request_type="diagnostic",
                confidence=1,
                reasons=("test",),
                fallback_flow="pnl",
            )

    service._complexity_classifier = Classifier()
    deterministic = AskServiceResult(
        answer="Draft.",
        sources=[],
        selected_flow="pnl",
        execution_status="completed",
        used_fallback=False,
    )

    def deterministic_run(*args, **kwargs):
        service._autonomous_context.set(
            {"graph_state": {"operations_result": {"revenue": 100}}}
        )
        return deterministic

    received: list[dict] = []

    def autonomous_run(question, result, internal_context):
        received.append(internal_context)
        return AutonomousExecutionResult(
            status="completed",
            management_response=ManagementResponse(
                answer="Autonomous.",
                evidence_ids=("pnl-001",),
            ),
        )

    service._ask_deterministic = deterministic_run
    service._autonomous_executor = autonomous_run

    result = service.ask("Why did profit improve?")

    assert result.answer == "Autonomous."
    assert received[0]["graph_state"]["operations_result"]["revenue"] == 100
