"""API tests for asynchronous finance request submission and lookup."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.dependencies import get_finance_request_lifecycle_service
from src.api.routes import router
from tests.test_finance_request_lifecycle import (
    InMemoryRequestRepository,
    SuccessfulFinanceService,
)
from src.services.finance_request_lifecycle_service import (
    FinanceRequestLifecycleService,
)


def create_client():
    repository = InMemoryRequestRepository()
    service = FinanceRequestLifecycleService(
        repository, SuccessfulFinanceService()
    )
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[
        get_finance_request_lifecycle_service
    ] = lambda: service
    return TestClient(app)


def test_submit_then_get_completed_request() -> None:
    client = create_client()

    submitted = client.post(
        "/api/v1/finance/ask",
        json={"question": "Explain April variance", "user_id": "u1"},
    )

    assert submitted.status_code == 202
    assert submitted.json()["status"] == "pending"
    request_id = submitted.json()["request_id"]

    fetched = client.get(f"/api/v1/finance/requests/{request_id}")
    assert fetched.status_code == 200
    assert fetched.json()["status"] == "completed"
    assert fetched.json()["answer"] == "Variance answer"


def test_get_unknown_request_returns_404() -> None:
    response = create_client().get(
        "/api/v1/finance/requests/00000000-0000-0000-0000-000000000000"
    )

    assert response.status_code == 404
