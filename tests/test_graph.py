"""Tests for the Finance Agentic AI LangGraph workflow."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
import pytest

import src.orchestrator.graph as graph_module
from src.orchestrator.state import FinanceGraphState


NodeFunction = Callable[[FinanceGraphState], FinanceGraphState]


# ----------------------------------------------------------------------
# Stub-node helpers
# ----------------------------------------------------------------------


def make_success_node(
    node_name: str,
    output_field: str | None = None,
    output_value: Any = None,
    *,
    complete_execution: bool = False,
) -> NodeFunction:
    """
    Create a successful graph-node stub.

    The stub records its execution and optionally writes an output field.
    """

    def node(state: FinanceGraphState) -> FinanceGraphState:
        executed_nodes = list(state.get("executed_nodes", []))
        executed_nodes.append(node_name)

        updates: dict[str, Any] = {}

        if output_field is not None:
            updates[output_field] = (
                output_value
                if output_value is not None
                else {"node": node_name}
            )

        return {
            **state,
            **updates,
            "execution_status": (
                "completed"
                if complete_execution
                else "running"
            ),
            "error_message": "",
            "failed_node": "",
            "errors": list(state.get("errors", [])),
            "executed_nodes": executed_nodes,
        }

    return node


def make_failure_node(
    node_name: str,
    error_message: str,
) -> NodeFunction:
    """Create a node stub that returns a failed graph state."""

    def node(state: FinanceGraphState) -> FinanceGraphState:
        errors = list(state.get("errors", []))
        errors.append(f"{node_name}: {error_message}")

        return {
            **state,
            "execution_status": "failed",
            "failed_node": node_name,
            "error_message": error_message,
            "errors": errors,
        }

    return node


def stub_all_graph_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Replace agent wrappers with deterministic successful test nodes.

    The graph is rebuilt after these replacements, so the compiled graph uses
    the test node functions rather than the real finance agents.
    """

    node_replacements: dict[str, NodeFunction] = {
        "validate_operations_node": make_success_node(
            "validate_operations",
            "operations_validation",
        ),
        "validate_budget_node": make_success_node(
            "validate_budget",
            "budget_validation",
        ),
        "clean_operations_node": make_success_node(
            "clean_operations",
            "cleaned_operations_data",
            pd.DataFrame({"cleaned": [True]}),
        ),
        "clean_budget_node": make_success_node(
            "clean_budget",
            "cleaned_budget_data",
            pd.DataFrame({"cleaned": [True]}),
        ),
        "profile_operations_node": make_success_node(
            "profile_operations",
            "operations_profile",
        ),
        "profile_budget_node": make_success_node(
            "profile_budget",
            "budget_profile",
        ),
        "operations_analysis_node": make_success_node(
            "operations_analysis",
            "operations_result",
        ),
        "budget_node": make_success_node(
            "budget",
            "budget_result",
        ),
        "forecast_node": make_success_node(
            "forecast",
            "forecast_result",
        ),
        "scenario_node": make_success_node(
            "scenario",
            "scenario_result",
        ),
        "variance_node": make_success_node(
            "variance",
            "variance_result",
        ),
        "finance_rules_node": make_success_node(
            "finance_rules",
            "finance_rules_result",
        ),
        "anomaly_node": make_success_node(
            "anomaly",
            "anomaly_result",
        ),
        "root_cause_node": make_success_node(
            "root_cause",
            "root_cause_result",
        ),
        "recommendation_node": make_success_node(
            "recommendation",
            "recommendation_result",
        ),
        "kpi_node": make_success_node(
            "kpi",
            "kpi_result",
        ),
        "commentary_node": make_success_node(
            "commentary",
            "commentary_result",
        ),
        "pnl_node": make_success_node(
            "pnl",
            "pnl_result",
        ),
        "report_node": make_success_node(
            "report",
            "report_result",
            complete_execution=True,
        ),
        "complete_node": make_success_node(
            "complete",
            complete_execution=True,
        ),
    }

    for attribute_name, replacement in node_replacements.items():
        monkeypatch.setattr(
            graph_module,
            attribute_name,
            replacement,
        )


def build_stubbed_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    """Build a graph whose finance nodes use deterministic stubs."""

    stub_all_graph_nodes(monkeypatch)
    return graph_module.build_finance_graph()


@pytest.fixture
def base_state() -> FinanceGraphState:
    """Return common initial graph-state values."""

    return {
        "execution_status": "pending",
        "errors": [],
        "executed_nodes": [],
        "filters": {},
    }


# ----------------------------------------------------------------------
# Graph-construction tests
# ----------------------------------------------------------------------


def test_graph_contains_required_control_nodes() -> None:
    """The compiled graph should contain start, router, error and end nodes."""

    graph = graph_module.build_finance_graph()
    node_names = set(graph.get_graph().nodes.keys())

    assert "__start__" in node_names
    assert "router" in node_names
    assert "error" in node_names
    assert "__end__" in node_names


@pytest.mark.parametrize(
    "expected_node",
    [
        "kpi__validate_operations",
        "budget__validate_budget",
        "forecast__forecast",
        "variance__variance",
        "scenario__scenario",
        "full__report",
        "pnl__pnl",
    ],
)
def test_graph_contains_route_specific_nodes(
    expected_node: str,
) -> None:
    """Every supported workflow should have registered graph nodes."""

    graph = graph_module.build_finance_graph()
    node_names = set(graph.get_graph().nodes.keys())

    assert expected_node in node_names


# ----------------------------------------------------------------------
# Successful route tests
# ----------------------------------------------------------------------


def test_kpi_route(
    monkeypatch: pytest.MonkeyPatch,
    base_state: FinanceGraphState,
) -> None:
    """A KPI request should execute only the KPI workflow."""

    graph = build_stubbed_graph(monkeypatch)

    initial_state: FinanceGraphState = {
        **base_state,
        "user_request": "Show KPI performance",
        "operations_data": pd.DataFrame({"value": [1]}),
    }

    result = graph.invoke(initial_state)

    assert result["selected_flow"] == "kpi"
    assert result["execution_status"] == "completed"
    assert result["executed_nodes"] == [
        "validate_operations",
        "clean_operations",
        "operations_analysis",
        "kpi",
        "commentary",
        "complete",
    ]

    assert "budget_result" not in result
    assert "forecast_result" not in result
    assert "report_result" not in result


def test_budget_route(
    monkeypatch: pytest.MonkeyPatch,
    base_state: FinanceGraphState,
) -> None:
    """A budget request should execute the budget workflow."""

    graph = build_stubbed_graph(monkeypatch)

    initial_state: FinanceGraphState = {
        **base_state,
        "user_request": "Prepare budget analysis",
        "budget_data": pd.DataFrame({"value": [1]}),
    }

    result = graph.invoke(initial_state)

    assert result["selected_flow"] == "budget"
    assert result["execution_status"] == "completed"
    assert result["executed_nodes"] == [
        "validate_budget",
        "clean_budget",
        "profile_budget",
        "budget",
        "kpi",
        "commentary",
        "complete",
    ]

    assert "budget_result" in result
    assert "operations_result" not in result


def test_forecast_route(
    monkeypatch: pytest.MonkeyPatch,
    base_state: FinanceGraphState,
) -> None:
    """A forecast request should execute the forecast workflow."""

    graph = build_stubbed_graph(monkeypatch)

    initial_state: FinanceGraphState = {
        **base_state,
        "user_request": "Forecast next 3 months",
        "operations_data": pd.DataFrame({"value": [1]}),
    }

    result = graph.invoke(initial_state)

    assert result["selected_flow"] == "forecast"
    assert result["execution_status"] == "completed"
    assert result["executed_nodes"] == [
        "validate_operations",
        "clean_operations",
        "operations_analysis",
        "forecast",
        "kpi",
        "commentary",
        "complete",
    ]

    assert "forecast_result" in result
    assert "scenario_result" not in result


def test_variance_route(
    monkeypatch: pytest.MonkeyPatch,
    base_state: FinanceGraphState,
) -> None:
    """A variance request should include root-cause analysis."""

    graph = build_stubbed_graph(monkeypatch)

    initial_state: FinanceGraphState = {
        **base_state,
        "user_request": "Explain revenue variance",
        "operations_data": pd.DataFrame({"value": [1]}),
        "budget_data": pd.DataFrame({"value": [1]}),
    }

    result = graph.invoke(initial_state)

    assert result["selected_flow"] == "variance"
    assert result["execution_status"] == "completed"
    assert result["executed_nodes"] == [
        "validate_operations",
        "validate_budget",
        "clean_operations",
        "clean_budget",
        "operations_analysis",
        "budget",
        "variance",
        "anomaly",
        "root_cause",
        "recommendation",
        "kpi",
        "commentary",
        "complete",
    ]

    assert "variance_result" in result
    assert "root_cause_result" in result
    assert "recommendation_result" in result


def test_scenario_route(
    monkeypatch: pytest.MonkeyPatch,
    base_state: FinanceGraphState,
) -> None:
    """A scenario request should run forecast before scenario analysis."""

    graph = build_stubbed_graph(monkeypatch)

    initial_state: FinanceGraphState = {
        **base_state,
        "user_request": "Run scenario analysis",
        "operations_data": pd.DataFrame({"value": [1]}),
        "business_assumptions": [
            {
                "metric": "revenue",
                "change_percentage": 5.0,
            }
        ],
    }

    result = graph.invoke(initial_state)

    assert result["selected_flow"] == "scenario"
    assert result["execution_status"] == "completed"
    assert result["executed_nodes"] == [
        "validate_operations",
        "clean_operations",
        "operations_analysis",
        "forecast",
        "scenario",
        "kpi",
        "commentary",
        "complete",
    ]

    assert "forecast_result" in result
    assert "scenario_result" in result


def test_full_route(
    monkeypatch: pytest.MonkeyPatch,
    base_state: FinanceGraphState,
) -> None:
    """A complete-management request should execute the full workflow."""

    graph = build_stubbed_graph(monkeypatch)

    initial_state: FinanceGraphState = {
        **base_state,
        "user_request": "Run complete management report",
        "operations_data": pd.DataFrame({"value": [1]}),
        "budget_data": pd.DataFrame({"value": [1]}),
        "business_assumptions": [
            {
                "metric": "revenue",
                "change_percentage": 5.0,
            }
        ],
    }

    result = graph.invoke(initial_state)

    assert result["selected_flow"] == "full"
    assert result["execution_status"] == "completed"
    assert result["executed_nodes"] == [
        "validate_operations",
        "validate_budget",
        "clean_operations",
        "clean_budget",
        "profile_operations",
        "profile_budget",
        "operations_analysis",
        "budget",
        "forecast",
        "scenario",
        "variance",
        "finance_rules",
        "anomaly",
        "root_cause",
        "recommendation",
        "kpi",
        "commentary",
        "report",
    ]

    assert "operations_result" in result
    assert "budget_result" in result
    assert "forecast_result" in result
    assert "scenario_result" in result
    assert "variance_result" in result
    assert "finance_rules_result" in result
    assert "anomaly_result" in result
    assert "root_cause_result" in result
    assert "recommendation_result" in result
    assert "kpi_result" in result
    assert "commentary_result" in result
    assert "report_result" in result



def test_pnl_route(
    monkeypatch: pytest.MonkeyPatch,
    base_state: FinanceGraphState,
) -> None:
    """A P&L request should execute only the dedicated P&L workflow."""

    graph = build_stubbed_graph(monkeypatch)

    initial_state: FinanceGraphState = {
        **base_state,
        "user_request": "Generate profit and loss statement",
        "operations_data": pd.DataFrame({"value":[1]}),
        "budget_data": pd.DataFrame({"value":[1]}),
        "corporate_expenses_data": pd.DataFrame({"value":[1]}),
        "budget_corporate_expenses_data": pd.DataFrame({"value":[1]}),
    }

    result = graph.invoke(initial_state)

    assert result["selected_flow"]=="pnl"
    assert result["execution_status"]=="completed"
    assert result["executed_nodes"]==[
        "validate_operations",
        "validate_budget",
        "clean_operations",
        "clean_budget",
        "pnl",
        "complete",
    ]
    assert "pnl_result" in result
    assert "report_result" not in result
    assert "forecast_result" not in result


def test_pnl_route_failure(
    monkeypatch: pytest.MonkeyPatch,
    base_state: FinanceGraphState,
)->None:
    """P&L workflow should stop if pnl node fails."""

    stub_all_graph_nodes(monkeypatch)
    monkeypatch.setattr(
        graph_module,
        "pnl_node",
        make_failure_node("pnl","P&L calculation failed."),
    )
    graph=graph_module.build_finance_graph()
    initial_state={
        **base_state,
        "user_request":"Generate profit and loss statement",
        "operations_data":pd.DataFrame({"value":[1]}),
        "budget_data":pd.DataFrame({"value":[1]}),
        "corporate_expenses_data":pd.DataFrame({"value":[1]}),
        "budget_corporate_expenses_data":pd.DataFrame({"value":[1]}),
    }
    result=graph.invoke(initial_state)
    assert result["selected_flow"]=="pnl"
    assert result["execution_status"]=="failed"
    assert result["failed_node"]=="pnl"
    assert result["executed_nodes"]==[
        "validate_operations",
        "validate_budget",
        "clean_operations",
        "clean_budget",
    ]

# ----------------------------------------------------------------------
# Error-route tests
# ----------------------------------------------------------------------


def test_unknown_request_routes_to_error(
    monkeypatch: pytest.MonkeyPatch,
    base_state: FinanceGraphState,
) -> None:
    """An unsupported request should end safely through the error node."""

    graph = build_stubbed_graph(monkeypatch)

    initial_state: FinanceGraphState = {
        **base_state,
        "user_request": "Tell me a joke",
    }

    result = graph.invoke(initial_state)

    assert result["selected_flow"] == "unknown"
    assert result["execution_status"] == "failed"
    assert result["error_message"] == (
        "Unable to identify a supported workflow."
    )
    assert result["errors"] == [
        "Unable to identify a supported workflow."
    ]
    assert result["executed_nodes"] == []


def test_missing_operations_data_stops_graph(
    monkeypatch: pytest.MonkeyPatch,
    base_state: FinanceGraphState,
) -> None:
    """A validation failure should stop later KPI nodes from executing."""

    stub_all_graph_nodes(monkeypatch)

    monkeypatch.setattr(
        graph_module,
        "validate_operations_node",
        make_failure_node(
            "validate_operations",
            "Required state field 'operations_data' is missing.",
        ),
    )

    graph = graph_module.build_finance_graph()

    initial_state: FinanceGraphState = {
        **base_state,
        "user_request": "Show KPI performance",
    }

    result = graph.invoke(initial_state)

    assert result["selected_flow"] == "kpi"
    assert result["execution_status"] == "failed"
    assert result["failed_node"] == "validate_operations"
    assert "operations_data" in result["error_message"]
    assert result["executed_nodes"] == []

    assert "operations_result" not in result
    assert "kpi_result" not in result
    assert "commentary_result" not in result


def test_missing_budget_data_stops_variance_route(
    monkeypatch: pytest.MonkeyPatch,
    base_state: FinanceGraphState,
) -> None:
    """The variance workflow should stop when budget validation fails."""

    stub_all_graph_nodes(monkeypatch)

    monkeypatch.setattr(
        graph_module,
        "validate_budget_node",
        make_failure_node(
            "validate_budget",
            "Required state field 'budget_data' is missing.",
        ),
    )

    graph = graph_module.build_finance_graph()

    initial_state: FinanceGraphState = {
        **base_state,
        "user_request": "Explain revenue variance",
        "operations_data": pd.DataFrame({"value": [1]}),
    }

    result = graph.invoke(initial_state)

    assert result["selected_flow"] == "variance"
    assert result["execution_status"] == "failed"
    assert result["failed_node"] == "validate_budget"
    assert "budget_data" in result["error_message"]

    assert result["executed_nodes"] == [
        "validate_operations",
    ]

    assert "variance_result" not in result
    assert "root_cause_result" not in result


def test_agent_failure_routes_to_error_node(
    monkeypatch: pytest.MonkeyPatch,
    base_state: FinanceGraphState,
) -> None:
    """An agent failure should prevent every later workflow node."""

    stub_all_graph_nodes(monkeypatch)

    monkeypatch.setattr(
        graph_module,
        "forecast_node",
        make_failure_node(
            "forecast",
            "Forecast Agent produced an empty forecast output.",
        ),
    )

    graph = graph_module.build_finance_graph()

    initial_state: FinanceGraphState = {
        **base_state,
        "user_request": "Forecast next 3 months",
        "operations_data": pd.DataFrame({"value": [1]}),
    }

    result = graph.invoke(initial_state)

    assert result["selected_flow"] == "forecast"
    assert result["execution_status"] == "failed"
    assert result["failed_node"] == "forecast"
    assert result["error_message"] == (
        "Forecast Agent produced an empty forecast output."
    )

    assert result["executed_nodes"] == [
        "validate_operations",
        "clean_operations",
        "operations_analysis",
    ]

    assert "forecast_result" not in result
    assert "kpi_result" not in result
    assert "commentary_result" not in result


# ----------------------------------------------------------------------
# Public graph-runner test
# ----------------------------------------------------------------------


def test_run_finance_graph_uses_compiled_graph(
    monkeypatch: pytest.MonkeyPatch,
    base_state: FinanceGraphState,
) -> None:
    """The public runner should return the final FinanceGraphState."""

    graph = build_stubbed_graph(monkeypatch)

    monkeypatch.setattr(
        graph_module,
        "finance_graph",
        graph,
    )

    initial_state: FinanceGraphState = {
        **base_state,
        "user_request": "Show KPI performance",
        "operations_data": pd.DataFrame({"value": [1]}),
    }

    result = graph_module.run_finance_graph(initial_state)

    assert result["selected_flow"] == "kpi"
    assert result["execution_status"] == "completed"
    assert result["executed_nodes"][-1] == "complete"