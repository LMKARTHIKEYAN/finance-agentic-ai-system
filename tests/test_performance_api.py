"""Tests for GET /api/v1/data/performance."""

from datetime import date
from decimal import Decimal

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.dependencies import get_performance_service
from src.api.routes import router
from src.integrations.snowflake_connection import SnowflakeConnectionError


class FakePerformanceService:
    def get_performance(self, *, month, vehicle_category=None):
        assert month == date(2026, 4, 1)
        assert vehicle_category == "10 FT"
        return [
            {
                "month": month,
                "vehicle_category": vehicle_category,
                "actual_orders": 110,
                "budget_orders": 100,
                "orders_variance": 10,
                "actual_revenue": Decimal("1200"),
                "budget_revenue": Decimal("1000"),
                "revenue_variance": Decimal("200"),
                "revenue_variance_pct": Decimal("20.00"),
                "actual_cogs": Decimal("700"),
                "budget_cogs": Decimal("650"),
                "cogs_variance": Decimal("50"),
            }
        ]


def create_client(service) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_performance_service] = lambda: service
    return TestClient(app)


def test_performance_endpoint_returns_validated_metrics() -> None:
    response = create_client(FakePerformanceService()).get(
        "/api/v1/data/performance",
        params={"month": "2026-04-15", "vehicle_category": "10 FT"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["month"] == "2026-04-01"
    assert body["row_count"] == 1
    assert body["rows"][0]["orders_variance"] == 10
    assert body["rows"][0]["revenue_variance_pct"] == "20.00"


def test_performance_endpoint_requires_valid_month() -> None:
    response = create_client(FakePerformanceService()).get(
        "/api/v1/data/performance",
        params={"month": "April 2026"},
    )

    assert response.status_code == 422


def test_performance_endpoint_hides_snowflake_error() -> None:
    class FailingService:
        def get_performance(self, **kwargs):
            raise SnowflakeConnectionError("sensitive detail")

    response = create_client(FailingService()).get(
        "/api/v1/data/performance",
        params={"month": "2026-04-01"},
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Performance data is temporarily unavailable."
    }
    assert "sensitive detail" not in response.text
