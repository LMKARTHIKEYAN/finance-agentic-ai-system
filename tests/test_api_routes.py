"""
Tests for Finance Agentic AI API routes.

These tests use a fake FinanceAskService and do not require:

- PostgreSQL
- pgvector
- Docker
- OpenAI
- Local finance CSV files
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.dependencies import get_finance_service
from src.api.routes import router
from src.api.service import (
    AskServiceResult,
    FinanceAskServiceError,
)


class FakeFinanceService:
    """
    Fake service used for successful route tests.
    """

    def ask(
        self,
        question: str,
        *,
        top_k: int | None = None,
        score_threshold: float | None = None,
        metadata_filter: dict | None = None,
    ) -> AskServiceResult:
        assert question == "What is the revenue variance?"
        assert top_k == 5
        assert metadata_filter == {}

        return AskServiceResult(
            answer=(
                "Revenue variance was mainly driven "
                "by price and volume."
            ),
            sources=[
                {
                    "id": "chunk_001",
                    "score": 0.95,
                    "rank": 1,
                    "metadata": {
                        "filename": "finance_policy.pdf",
                    },
                    "excerpt": (
                        "Revenue variance was caused "
                        "by price and volume."
                    ),
                }
            ],
            selected_flow="variance",
            execution_status="completed",
            used_fallback=False,
        )


def create_test_app(
    service: object,
) -> FastAPI:
    """
    Create a small FastAPI application with a dependency override.
    """

    app = FastAPI()

    app.include_router(router)

    app.dependency_overrides[
        get_finance_service
    ] = lambda: service

    return app


def test_health_endpoint() -> None:
    """
    GET /health should confirm that the API is running.
    """

    app = create_test_app(
        FakeFinanceService()
    )

    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json() == {
        "status": "healthy",
        "service": "finance-agentic-ai-api",
    }


def test_ask_endpoint_returns_answer() -> None:
    """
    POST /ask should return the service answer and source details.
    """

    app = create_test_app(
        FakeFinanceService()
    )

    client = TestClient(app)

    response = client.post(
        "/ask",
        json={
            "question": (
                "What is the revenue variance?"
            ),
            "top_k": 5,
            "metadata_filter": {},
        },
    )

    assert response.status_code == 200

    response_body = response.json()

    assert response_body["answer"] == (
        "Revenue variance was mainly driven "
        "by price and volume."
    )

    assert (
        response_body["selected_flow"]
        == "variance"
    )

    assert (
        response_body["execution_status"]
        == "completed"
    )

    assert response_body["used_fallback"] is False

    assert len(response_body["sources"]) == 1

    assert (
        response_body["sources"][0]["id"]
        == "chunk_001"
    )

    assert (
        response_body["sources"][0]["score"]
        == 0.95
    )


def test_ask_endpoint_rejects_empty_question() -> None:
    """
    Pydantic should reject an empty question before calling the service.
    """

    app = create_test_app(
        FakeFinanceService()
    )

    client = TestClient(app)

    response = client.post(
        "/ask",
        json={
            "question": "",
            "top_k": 5,
            "metadata_filter": {},
        },
    )

    assert response.status_code == 422


def test_ask_endpoint_rejects_invalid_top_k() -> None:
    """
    Pydantic should reject top_k values below one.
    """

    app = create_test_app(
        FakeFinanceService()
    )

    client = TestClient(app)

    response = client.post(
        "/ask",
        json={
            "question": "Revenue variance",
            "top_k": 0,
            "metadata_filter": {},
        },
    )

    assert response.status_code == 422


def test_ask_endpoint_translates_value_error_to_400() -> None:
    """
    Service validation errors should become HTTP 400 responses.
    """

    class InvalidQuestionService:
        def ask(
            self,
            question: str,
            **kwargs,
        ):
            raise ValueError(
                "Question is invalid."
            )

    app = create_test_app(
        InvalidQuestionService()
    )

    client = TestClient(app)

    response = client.post(
        "/ask",
        json={
            "question": "Revenue variance",
        },
    )

    assert response.status_code == 400

    assert response.json() == {
        "detail": "Question is invalid."
    }


def test_ask_endpoint_translates_service_error_to_500() -> None:
    """
    Known service failures should become HTTP 500 responses.
    """

    class FailingFinanceService:
        def ask(
            self,
            question: str,
            **kwargs,
        ):
            raise FinanceAskServiceError(
                "Finance workflow failed."
            )

    app = create_test_app(
        FailingFinanceService()
    )

    client = TestClient(app)

    response = client.post(
        "/ask",
        json={
            "question": "Revenue variance",
        },
    )

    assert response.status_code == 500

    assert response.json() == {
        "detail": "Finance workflow failed."
    }


def test_ask_endpoint_hides_unexpected_error_details() -> None:
    """
    Unexpected internal errors should not expose sensitive implementation
    details to API clients.
    """

    class UnexpectedFailureService:
        def ask(
            self,
            question: str,
            **kwargs,
        ):
            raise RuntimeError(
                "Database password was exposed."
            )

    app = create_test_app(
        UnexpectedFailureService()
    )

    client = TestClient(app)

    response = client.post(
        "/ask",
        json={
            "question": "Revenue variance",
        },
    )

    assert response.status_code == 500

    assert response.json() == {
        "detail": (
            "An unexpected error occurred while "
            "processing the finance request."
        )
    }

    assert (
        "Database password"
        not in response.text
    )