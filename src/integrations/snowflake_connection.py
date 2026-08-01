"""Secure Snowflake connection lifecycle and read-only diagnostics."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Protocol

import snowflake.connector

from src.config.settings import Settings, settings


class SnowflakeConfigurationError(RuntimeError):
    """Raised when required Snowflake configuration is unavailable."""


class SnowflakeConnectionError(RuntimeError):
    """Raised when a Snowflake connection cannot be established."""


class SnowflakeQueryError(RuntimeError):
    """Raised when an approved Snowflake diagnostic query fails."""


class SnowflakeCursorProtocol(Protocol):
    """Minimal cursor contract used by the connection factory."""

    description: Any

    def execute(
        self,
        command: str,
        params: tuple[Any, ...] | None = None,
        **kwargs: Any,
    ) -> Any: ...

    def fetchone(self) -> Any: ...

    def fetchall(self) -> Any: ...

    def fetch_pandas_all(self) -> Any: ...

    def close(self) -> None: ...


class SnowflakeConnectionProtocol(Protocol):
    """Minimal Snowflake connection contract used by this module."""

    def cursor(self) -> SnowflakeCursorProtocol: ...

    def commit(self) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class SnowflakeConnectionConfig:
    """Validated settings required to connect to Snowflake."""

    account: str
    user: str
    password: str = field(repr=False)
    warehouse: str = "FINANCE_AI_WH"
    database: str = "FINANCE_AI"
    schema: str = "ANALYTICS"
    role: str = "FINANCE_AI_APP_ROLE"
    connect_timeout_seconds: int = 10
    query_timeout_seconds: int = 30

    @classmethod
    def from_settings(
        cls,
        app_settings: Settings = settings,
    ) -> "SnowflakeConnectionConfig":
        """Create and validate connection configuration from app settings."""

        values = {
            "account": app_settings.SNOWFLAKE_ACCOUNT,
            "user": app_settings.SNOWFLAKE_USER,
            "password": app_settings.SNOWFLAKE_PASSWORD,
            "warehouse": app_settings.SNOWFLAKE_WAREHOUSE,
            "database": app_settings.SNOWFLAKE_DATABASE,
            "schema": app_settings.SNOWFLAKE_SCHEMA,
            "role": app_settings.SNOWFLAKE_ROLE,
        }
        missing = [
            name.upper()
            for name, value in values.items()
            if not isinstance(value, str) or not value.strip()
        ]
        if missing:
            variable_names = ", ".join(
                f"SNOWFLAKE_{name}" for name in missing
            )
            raise SnowflakeConfigurationError(
                "Required Snowflake configuration is missing: "
                f"{variable_names}"
            )

        connect_timeout = app_settings.SNOWFLAKE_CONNECT_TIMEOUT_SECONDS
        query_timeout = app_settings.SNOWFLAKE_QUERY_TIMEOUT_SECONDS
        if connect_timeout <= 0:
            raise SnowflakeConfigurationError(
                "SNOWFLAKE_CONNECT_TIMEOUT_SECONDS must be positive."
            )
        if query_timeout <= 0:
            raise SnowflakeConfigurationError(
                "SNOWFLAKE_QUERY_TIMEOUT_SECONDS must be positive."
            )

        return cls(
            account=values["account"],
            user=values["user"],
            password=values["password"],
            warehouse=values["warehouse"],
            database=values["database"],
            schema=values["schema"],
            role=values["role"],
            connect_timeout_seconds=connect_timeout,
            query_timeout_seconds=query_timeout,
        )


class SnowflakeConnectionFactory:
    """Create Snowflake connections and close them deterministically."""

    def __init__(
        self,
        config: SnowflakeConnectionConfig,
    ) -> None:
        if not isinstance(config, SnowflakeConnectionConfig):
            raise TypeError(
                "config must be a SnowflakeConnectionConfig."
            )
        self._config = config

    @property
    def config(self) -> SnowflakeConnectionConfig:
        """Return the immutable connection configuration."""

        return self._config

    @contextmanager
    def connection(self) -> Iterator[SnowflakeConnectionProtocol]:
        """Yield one configured Snowflake connection and always close it."""

        try:
            connection = snowflake.connector.connect(
                account=self._config.account,
                user=self._config.user,
                password=self._config.password,
                warehouse=self._config.warehouse,
                database=self._config.database,
                schema=self._config.schema,
                role=self._config.role,
                login_timeout=self._config.connect_timeout_seconds,
                network_timeout=self._config.query_timeout_seconds,
                client_session_keep_alive=False,
            )
        except Exception as exc:
            raise SnowflakeConnectionError(
                "Unable to connect to Snowflake. Check the account, user, "
                "authentication method, network access, and role grants."
            ) from exc

        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def cursor(self) -> Iterator[SnowflakeCursorProtocol]:
        """Yield one cursor and close both cursor and connection."""

        with self.connection() as connection:
            cursor = connection.cursor()
            try:
                yield cursor
            finally:
                cursor.close()

    def execute_one(
        self,
        query: str,
        params: tuple[Any, ...] | None = None,
    ) -> Mapping[str, Any]:
        """Execute one approved read query and return its first row."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        try:
            with self.cursor() as cursor:
                cursor.execute(
                    query,
                    params,
                    timeout=self._config.query_timeout_seconds,
                )
                row = cursor.fetchone()
                description = cursor.description or []
        except SnowflakeConnectionError:
            raise
        except Exception as exc:
            raise SnowflakeQueryError(
                "Snowflake read query failed. Review object permissions, "
                "query parameters, and warehouse availability."
            ) from exc

        if row is None:
            return {}

        column_names = [
            str(column[0]).lower()
            for column in description
        ]
        return dict(zip(column_names, row, strict=False))


def run_snowflake_diagnostic(
    factory: SnowflakeConnectionFactory,
) -> dict[str, Any]:
    """Run read-only identity and order-count checks for Phase 1."""

    identity = factory.execute_one(
        """
        SELECT
            CURRENT_USER() AS current_user,
            CURRENT_ROLE() AS current_role,
            CURRENT_WAREHOUSE() AS current_warehouse,
            CURRENT_DATABASE() AS current_database,
            CURRENT_SCHEMA() AS current_schema
        """
    )
    order_count = factory.execute_one(
        """
        SELECT COUNT(*) AS order_count
        FROM FINANCE_AI.RAW.ORDERS
        """
    )
    return {
        **identity,
        **order_count,
    }
