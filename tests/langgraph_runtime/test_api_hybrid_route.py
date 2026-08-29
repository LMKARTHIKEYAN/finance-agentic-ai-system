from types import SimpleNamespace

from fastapi.testclient import TestClient

from src.api.app import app
from src.api.autonomous_routes import get_autonomous_service
from src.autonomous.hybrid_runtime_router import HybridRuntimeResult


class Router:
    def run(self, request):
        response = SimpleNamespace(
            status="completed", answer="existing answer", question=None,
            goal_id="g-1", evidence=(), metrics={}, management_commentary=None,
        )
        return HybridRuntimeResult(
            response=response, execution_mode="complex", runtime="python",
            shadow_executed=True, shadow_result={
                "completed": True, "step_count": 1,
                "evidence": [{"evidence_id": "e-1", "payload": {"secret": "hidden"}}],
                "validation": {"passed": True},
                "review": {"decision": "approved"},
                "final_answer": "Validated [e-1].",
                "execution_trace": [
                    {"step": 0, "node": "supervisor", "action": "call_tool",
                     "tool_name": "compare_periods", "rationale": "hidden reasoning"},
                ],
                "context": {"password": "must-not-leak"},
            },
        )


def test_api_preserves_response_and_adds_shadow_metadata():
    app.dependency_overrides[get_autonomous_service] = lambda: Router()
    try:
        response = TestClient(app).post(
            "/api/v1/autonomous/ask", json={"question": "Why did revenue decline?"}
        )
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "existing answer"
    assert payload["runtime"] == "python"
    assert payload["langgraph_shadow"]["succeeded"] is True
    summary = payload["langgraph_shadow"]["summary"]
    assert summary["selected_tools"] == ["compare_periods"]
    assert summary["validation_passed"] is True
    assert "hidden reasoning" not in response.text
    assert "must-not-leak" not in response.text
