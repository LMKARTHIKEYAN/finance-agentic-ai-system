"""PostgreSQL connection lifecycle for the local finance database."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import psycopg

from src.config.settings import Settings, settings


class PostgresConfigurationError(RuntimeError):
    """Raised when PostgreSQL configuration is incomplete or invalid."""


class PostgresConnectionError(RuntimeError):
    """Raised when PostgreSQL cannot be reached."""


@dataclass(frozen=True)
class PostgresConnectionConfig:
    """Validated, secret-safe PostgreSQL connection settings."""

    dsn: str = field(repr=False)
    connect_timeout_seconds: int = 10

    @classmethod
    def from_settings(
        cls,
        app_settings: Settings = settings,
    ) -> "PostgresConnectionConfig":
        dsn = app_settings.FINANCE_DATABASE_URL.strip()
        if not dsn:
            raise PostgresConfigurationError(
                "FINANCE_DATABASE_URL must be configured."
            )
        if not dsn.startswith(("postgresql://", "postgres://")):
            raise PostgresConfigurationError(
                "FINANCE_DATABASE_URL must be a PostgreSQL connection URL."
            )

        timeout = app_settings.POSTGRES_CONNECT_TIMEOUT_SECONDS
        if timeout <= 0:
            raise PostgresConfigurationError(
                "POSTGRES_CONNECT_TIMEOUT_SECONDS must be positive."
            )
        return cls(dsn=dsn, connect_timeout_seconds=timeout)


class PostgresConnectionFactory:
    """Open and deterministically close PostgreSQL connections."""

    def __init__(self, config: PostgresConnectionConfig) -> None:
        if not isinstance(config, PostgresConnectionConfig):
            raise TypeError("config must be a PostgresConnectionConfig.")
        self._config = config

    @property
    def config(self) -> PostgresConnectionConfig:
        return self._config

    @contextmanager
    def connection(self) -> Iterator[Any]:
        try:
            connection = psycopg.connect(
                self._config.dsn,
                connect_timeout=self._config.connect_timeout_seconds,
            )
        except Exception as exc:
            raise PostgresConnectionError(
                "Unable to connect to PostgreSQL. Check FINANCE_DATABASE_URL "
                "and that the local PostgreSQL service is running."
            ) from exc

        try:
            yield connection
        finally:
            connection.close()
