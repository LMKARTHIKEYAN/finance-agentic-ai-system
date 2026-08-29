from src.autonomous.langgraph_runtime.graph_routes import route_after_supervisor


def test_supervisor_tool_action_routes_to_executor():
    assert route_after_supervisor({"next_action": "call_tool", "step_count": 0, "max_steps": 8}) == "execute_tool"


def test_execution_limit_routes_to_recovery():
    assert route_after_supervisor({"next_action": "call_tool", "step_count": 8, "max_steps": 8}) == "recover"
