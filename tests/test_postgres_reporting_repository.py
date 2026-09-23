"""Unit tests for the PostgreSQL reporting repository."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date
from types import SimpleNamespace

import pytest

from src.integrations.postgres_connection import (
    PostgresConfigurationError,
    PostgresConnectionConfig,
)
from src.repositories.postgres_reporting_repository import (
    PostgresReportingRepository,
)


class FakeCursor:
    def __init__(self, rows, columns) -> None:
        self._rows = rows
        self.description = [SimpleNamespace(name=name) for name in columns]
        self.executions = []

    def execute(self, query, params) -> None:
        self.executions.append((query, params))

    def fetchall(self):
        return self._rows

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        return None


class FakeConnection:
    def __init__(self, cursor) -> None:
        self._cursor = cursor

    def cursor(self):
        return self._cursor


class FakeFactory:
    def __init__(self, cursor) -> None:
        self._connection = FakeConnection(cursor)

    @contextmanager
    def connection(self):
        yield self._connection


def repository_with(cursor: FakeCursor) -> PostgresReportingRepository:
    repository = object.__new__(PostgresReportingRepository)
    repository._factory = FakeFactory(cursor)
    return repository


def test_config_uses_postgres_url_without_exposing_it() -> None:
    settings = SimpleNamespace(
        FINANCE_DATABASE_URL="postgresql://postgres:secret@localhost/finance_agentic_ai",
        POSTGRES_CONNECT_TIMEOUT_SECONDS=5,
    )

    config = PostgresConnectionConfig.from_settings(settings)

    assert config.connect_timeout_seconds == 5
    assert "secret" not in repr(config)


def test_config_rejects_an_empty_database_url() -> None:
    settings = SimpleNamespace(
        FINANCE_DATABASE_URL="",
        POSTGRES_CONNECT_TIMEOUT_SECONDS=5,
    )

    with pytest.raises(PostgresConfigurationError, match="FINANCE_DATABASE_URL"):
        PostgresConnectionConfig.from_settings(settings)


def test_orders_query_is_parameterized_and_returns_dataframe() -> None:
    cursor = FakeCursor(
        [("ORD1", date(2026, 4, 1), "Completed")],
        ["order_id", "order_date", "order_status"],
    )
    repository = repository_with(cursor)

    frame = repository.get_orders(date(2026, 4, 1), date(2026, 4, 30))

    assert frame.to_dict("records") == [{
        "order_id": "ORD1",
        "order_date": date(2026, 4, 1),
        "order_status": "Completed",
    }]
    query, params = cursor.executions[0]
    assert "public.orders" in query
    assert params == (date(2026, 4, 1), date(2026, 4, 30))


def test_budget_months_are_normalized_for_finance_agents() -> None:
    cursor = FakeCursor(
        [(date(2026, 4, 1), "2W", 100, 2500, None)],
        ["month", "vehicle_category", "budget_orders", "budget_revenue", "budget_cogs"],
    )
    repository = repository_with(cursor)

    frame = repository.get_budget("2026-04", "2026-04")

    assert frame.loc[0, "month"] == "2026-04"
    assert cursor.executions[0][1] == (date(2026, 4, 1), date(2026, 4, 1))
