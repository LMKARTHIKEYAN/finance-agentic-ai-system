import pandas as pd

from src.autonomous.langgraph_runtime.observation_node import observation_node
from src.autonomous.langgraph_runtime.tool_executor_node import ToolExecutorNode
from src.autonomous.tools.data_tools import FinanceDataContext


def _state(arguments=None):
    frame = pd.DataFrame({
        "order_date": pd.to_datetime(["2026-06-30", "2026-07-01", "2026-07-02"]),
        "vehicle_category": ["2W", "Tata Ace", "2W"],
        "order_id": ["a", "b", "c"],
        "order_status": ["Completed", "Completed", "Completed"],
        "commission_amount": [10.0, 20.0, 30.0],
        "incentive": [0.0, 1.0, 0.0],
        "goodwill": [0.0, 0.0, 0.0],
        "dry_run": [0.0, 0.0, 0.0],
        "surge": [0.0, 0.0, 0.0],
        "pickup_cluster": ["A", "B", "C"],
    })
    return {
        "request": "Investigate Tata Ace by pickup cluster",
        "decision": {"tool_name": "analyze_drilldown", "arguments": arguments or {
            "start_date": "2026-07-01", "end_date": "2026-07-31",
            "category": "Tata Ace", "dimension": "pickup_cluster",
        }},
        "context": {"finance_context": FinanceDataContext(operations_data=frame)},
        "executed_calls": [],
    }


def test_executes_with_resolved_filtered_context():
    update = ToolExecutorNode()(_state())
    result = update["context"]["last_tool_result"]
    assert result["status"] == "completed"
    assert result["payload"]["rows"][0]["pickup_cluster"] == "B"


def test_unknown_tool_is_a_safe_failure():
    state = _state()
    state["decision"] = {"tool_name": "delete_database", "arguments": {}}
    result = ToolExecutorNode()(state)["context"]["last_tool_result"]
    assert result["status"] == "failed"
    assert "unapproved" not in repr(state["context"]).lower()


def test_trusted_context_override_is_blocked():
    result = ToolExecutorNode()(_state({
        "finance_context": "fake", "dimension": "pickup_cluster",
    }))["context"]["last_tool_result"]
    assert result["status"] == "failed"
    assert "cannot supply trusted" in result["error"]


def test_duplicate_identical_call_is_blocked():
    node = ToolExecutorNode()
    state = _state()
    first = node(state)
    state.update(first)
    second = node(state)
    assert second["context"]["last_tool_result"]["error"] == "Duplicate tool call blocked."


def test_observation_does_not_contain_dataframe():
    update = ToolExecutorNode()(_state())
    observed = observation_node({"context": update["context"], "observations": [], "evidence": [], "step_count": 0})
    assert "DataFrame" not in repr(observed)
