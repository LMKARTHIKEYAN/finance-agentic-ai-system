"""Tests for the read-only Snowflake performance repository."""

from contextlib import contextmanager
from datetime import date
from types import SimpleNamespace

from src.repositories.snowflake_performance_repository import (
    SnowflakePerformanceRepository,
)


class FakeCursor:
    description = [("MONTH",), ("VEHICLE_CATEGORY",), ("ACTUAL_ORDERS",)]

    def __init__(self) -> None:
        self.execution = None

    def execute(self, query, params, **kwargs):
        self.execution = (query, params, kwargs)

    def fetchall(self):
        return [(date(2026, 4, 1), "10 FT", 110)]


class FakeFactory:
    def __init__(self) -> None:
        self.config = SimpleNamespace(query_timeout_seconds=30)
        self.fake_cursor = FakeCursor()

    @contextmanager
    def cursor(self):
        yield self.fake_cursor


def test_repository_uses_parameterized_read_query() -> None:
    factory = FakeFactory()
    repository = object.__new__(SnowflakePerformanceRepository)
    repository._factory = factory

    rows = repository.get_monthly_performance(
        month=date(2026, 4, 1),
        vehicle_category=" 10 FT ",
    )

    query, params, kwargs = factory.fake_cursor.execution
    assert "FINANCE_AI.RAW.ORDERS" in query
    assert "FINANCE_AI.RAW.BUDGET" in query
    assert "INSERT" not in query.upper()
    assert "UPDATE" not in query.upper()
    assert "DELETE" not in query.upper()
    assert params == (
        date(2026, 4, 1),
        "10 FT",
        "10 FT",
        date(2026, 4, 1),
        "10 FT",
        "10 FT",
    )
    assert kwargs["timeout"] == 30
    assert rows[0]["vehicle_category"] == "10 FT"
