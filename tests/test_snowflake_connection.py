"""Unit tests for secure Snowflake connection lifecycle handling."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from src.integrations import snowflake_connection
from src.integrations.snowflake_connection import (
    SnowflakeConfigurationError,
    SnowflakeConnectionConfig,
    SnowflakeConnectionError,
    SnowflakeConnectionFactory,
    SnowflakeQueryError,
    run_snowflake_diagnostic,
)


def build_config(**overrides: Any) -> SnowflakeConnectionConfig:
    """Return valid test configuration with optional replacements."""

    values = {
        "account": "example-account",
        "user": "finance-app-user",
        "password": "private-test-password",
        "warehouse": "FINANCE_AI_WH",
        "database": "FINANCE_AI",
        "schema": "ANALYTICS",
        "role": "FINANCE_AI_APP_ROLE",
        "connect_timeout_seconds": 10,
        "query_timeout_seconds": 30,
    }
    values.update(overrides)
    return SnowflakeConnectionConfig(**values)


class FakeCursor:
    """Small cursor double that records execution and close behaviour."""

    def __init__(
        self,
        *,
        row: tuple[Any, ...] | None = (1,),
        description: list[tuple[Any, ...]] | None = None,
        execute_error: Exception | None = None,
    ) -> None:
        self.row = row
        self.description = description or [("VALUE",)]
        self.execute_error = execute_error
        self.executions: list[tuple[str, Any, Any]] = []
        self.closed = False

    def execute(
        self,
        command: str,
        params: tuple[Any, ...] | None = None,
        **kwargs: Any,
    ) -> "FakeCursor":
        self.executions.append((command, params, kwargs.get("timeout")))
        if self.execute_error is not None:
            raise self.execute_error
        return self

    def fetchone(self) -> tuple[Any, ...] | None:
        return self.row

    def close(self) -> None:
        self.closed = True


class FakeConnection:
    """Small connection double that returns a configured fake cursor."""

    def __init__(self, cursor: FakeCursor) -> None:
        self.fake_cursor = cursor
        self.closed = False

    def cursor(self) -> FakeCursor:
        return self.fake_cursor

    def close(self) -> None:
        self.closed = True


def test_config_from_settings_returns_clean_values() -> None:
    app_settings = SimpleNamespace(
        SNOWFLAKE_ACCOUNT="example-account",
        SNOWFLAKE_USER="finance-user",
        SNOWFLAKE_PASSWORD="secret",
        SNOWFLAKE_WAREHOUSE="FINANCE_AI_WH",
        SNOWFLAKE_DATABASE="FINANCE_AI",
        SNOWFLAKE_SCHEMA="ANALYTICS",
        SNOWFLAKE_ROLE="FINANCE_AI_APP_ROLE",
        SNOWFLAKE_CONNECT_TIMEOUT_SECONDS=10,
        SNOWFLAKE_QUERY_TIMEOUT_SECONDS=30,
    )

    config = SnowflakeConnectionConfig.from_settings(app_settings)

    assert config.account == "example-account"
    assert config.user == "finance-user"
    assert config.role == "FINANCE_AI_APP_ROLE"
    assert "secret" not in repr(config)


def test_config_from_settings_rejects_missing_values() -> None:
    app_settings = SimpleNamespace(
        SNOWFLAKE_ACCOUNT="",
        SNOWFLAKE_USER="finance-user",
        SNOWFLAKE_PASSWORD="",
        SNOWFLAKE_WAREHOUSE="FINANCE_AI_WH",
        SNOWFLAKE_DATABASE="FINANCE_AI",
        SNOWFLAKE_SCHEMA="ANALYTICS",
        SNOWFLAKE_ROLE="FINANCE_AI_APP_ROLE",
        SNOWFLAKE_CONNECT_TIMEOUT_SECONDS=10,
        SNOWFLAKE_QUERY_TIMEOUT_SECONDS=30,
    )

    with pytest.raises(
        SnowflakeConfigurationError,
        match="SNOWFLAKE_ACCOUNT, SNOWFLAKE_PASSWORD",
    ):
        SnowflakeConnectionConfig.from_settings(app_settings)


def test_connection_uses_restricted_configuration_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cursor = FakeCursor()
    connection = FakeConnection(cursor)
    captured: dict[str, Any] = {}

    def fake_connect(**kwargs: Any) -> FakeConnection:
        captured.update(kwargs)
        return connection

    monkeypatch.setattr(
        snowflake_connection.snowflake.connector,
        "connect",
        fake_connect,
    )

    factory = SnowflakeConnectionFactory(build_config())
    with factory.connection() as yielded:
        assert yielded is connection

    assert captured["role"] == "FINANCE_AI_APP_ROLE"
    assert captured["warehouse"] == "FINANCE_AI_WH"
    assert captured["database"] == "FINANCE_AI"
    assert captured["schema"] == "ANALYTICS"
    assert captured["login_timeout"] == 10
    assert captured["network_timeout"] == 30
    assert connection.closed is True


def test_connection_failure_does_not_expose_password(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_connect(**kwargs: Any) -> None:
        raise RuntimeError(
            f"authentication failed for {kwargs['password']}"
        )

    monkeypatch.setattr(
        snowflake_connection.snowflake.connector,
        "connect",
        fake_connect,
    )
    factory = SnowflakeConnectionFactory(build_config())

    with pytest.raises(SnowflakeConnectionError) as error:
        with factory.connection():
            pass

    assert "private-test-password" not in str(error.value)


def test_execute_one_returns_lowercase_mapping_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cursor = FakeCursor(
        row=("FINANCE_AI_APP_ROLE", 500000),
        description=[("CURRENT_ROLE",), ("ORDER_COUNT",)],
    )
    connection = FakeConnection(cursor)
    monkeypatch.setattr(
        snowflake_connection.snowflake.connector,
        "connect",
        lambda **kwargs: connection,
    )
    factory = SnowflakeConnectionFactory(build_config())

    result = factory.execute_one("SELECT 1")

    assert result == {
        "current_role": "FINANCE_AI_APP_ROLE",
        "order_count": 500000,
    }
    assert cursor.executions == [("SELECT 1", None, 30)]
    assert cursor.closed is True
    assert connection.closed is True


def test_execute_one_translates_query_failure_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cursor = FakeCursor(execute_error=RuntimeError("query failed"))
    connection = FakeConnection(cursor)
    monkeypatch.setattr(
        snowflake_connection.snowflake.connector,
        "connect",
        lambda **kwargs: connection,
    )
    factory = SnowflakeConnectionFactory(build_config())

    with pytest.raises(SnowflakeQueryError):
        factory.execute_one("SELECT 1")

    assert cursor.closed is True
    assert connection.closed is True


def test_run_diagnostic_combines_identity_and_order_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = SnowflakeConnectionFactory(build_config())
    results = iter(
        [
            {
                "current_user": "FINANCE_USER",
                "current_role": "FINANCE_AI_APP_ROLE",
                "current_warehouse": "FINANCE_AI_WH",
                "current_database": "FINANCE_AI",
                "current_schema": "ANALYTICS",
            },
            {"order_count": 500000},
        ]
    )
    monkeypatch.setattr(
        factory,
        "execute_one",
        lambda query: next(results),
    )

    result = run_snowflake_diagnostic(factory)

    assert result["current_role"] == "FINANCE_AI_APP_ROLE"
    assert result["order_count"] == 500000
