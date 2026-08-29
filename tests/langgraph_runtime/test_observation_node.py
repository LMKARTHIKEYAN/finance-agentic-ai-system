from src.autonomous.langgraph_runtime.observation_node import observation_node


def test_completed_result_becomes_evidence():
    update = observation_node({"context": {"last_tool_result": {"tool_name": "x", "status": "completed", "payload": {"value": 1}, "evidence_id": "e-1"}}, "observations": [], "evidence": [], "step_count": 0})
    assert update["next_action"] == "supervise"
    assert update["evidence"][0]["evidence_id"] == "e-1"
