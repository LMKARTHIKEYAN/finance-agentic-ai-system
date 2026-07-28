"""End-to-end hybrid metadata flow through API, client, and presentation."""

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.dependencies import get_finance_service
from src.api.routes import router
from src.api.service import AskServiceResult
from src.ui.api_client import _parse_ask_response
from src.ui.streamlit import _build_hybrid_status_messages


class HybridService:
    def ask(self, question: str, **kwargs) -> AskServiceResult:
        return AskServiceResult(
            answer="Reviewed autonomous answer.",
            sources=[],
            selected_flow="pnl",
            execution_status="completed",
            used_fallback=False,
            hybrid_metadata={
                "execution_mode": "autonomous",
                "autonomous_status": "completed",
                "fallback_used": False,
                "review_decision": "approved",
                "evidence_ids": ["pnl-001"],
                "usage": {"total_tokens": 400},
            },
        )


def test_hybrid_metadata_reaches_streamlit_without_internal_context() -> None:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_finance_service] = HybridService
    response = TestClient(app).post(
        "/ask",
        json={"question": "Why did April profit improve?"},
    )

    assert response.status_code == 200
    body = response.json()
    parsed = _parse_ask_response(json.dumps(body))
    metadata = parsed["hybrid_metadata"]
    assert metadata["execution_mode"] == "autonomous"
    assert metadata["review_decision"] == "approved"
    assert metadata["evidence_ids"] == ["pnl-001"]
    assert "graph_state" not in json.dumps(body)
    messages = _build_hybrid_status_messages(parsed)
    visible = " ".join(message for _, message in messages)
    assert "Reviewer status: Approved" in visible
    assert "Evidence: pnl-001" in visible


def test_legacy_api_response_still_parses_without_hybrid_metadata() -> None:
    parsed = _parse_ask_response(
        json.dumps(
            {
                "answer": "Legacy.",
                "sources": [],
                "execution_status": "completed",
                "used_fallback": False,
            }
        )
    )

    assert parsed["hybrid_metadata"] is None
