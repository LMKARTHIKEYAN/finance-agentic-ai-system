"""
Tests for the Finance Agentic AI FastAPI application.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.app import app, create_app


def test_create_app_returns_fastapi_instance() -> None:
    """
    create_app should return a configured FastAPI application.
    """

    application = create_app()

    assert isinstance(application, FastAPI)
    assert application.title == "Finance Agentic AI API"
    assert application.version == "1.0.0"


def test_module_exposes_application() -> None:
    """
    The module-level app should be available for Uvicorn.
    """

    assert isinstance(app, FastAPI)
    assert app.title == "Finance Agentic AI API"


def test_openapi_document_is_available() -> None:
    """
    The application should expose its OpenAPI document.
    """

    client = TestClient(app)

    response = client.get("/openapi.json")

    assert response.status_code == 200

    document = response.json()

    assert document["info"]["title"] == (
        "Finance Agentic AI API"
    )

    assert "/health" in document["paths"]
    assert "/ask" in document["paths"]


def test_swagger_documentation_is_available() -> None:
    """
    Swagger UI should be available at /docs.
    """

    client = TestClient(app)

    response = client.get("/docs")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_health_route_is_registered_in_application() -> None:
    """
    The application should include the API router.
    """

    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json() == {
        "status": "healthy",
        "service": "finance-agentic-ai-api",
    }