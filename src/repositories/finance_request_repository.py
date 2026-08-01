"""Request lifecycle persistence for asynchronous finance questions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from src.integrations.snowflake_connection import SnowflakeConnectionFactory


class FinanceRequestRepositoryError(RuntimeError):
    """Raised when request lifecycle persistence fails."""


@dataclass(frozen=True)
class FinanceRequestRecord:
    request_id: str
    user_id: str | None
    question: str
    status: str
    selected_flow: str | None = None
    answer: str | None = None
    error_message: str | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


class FinanceRequestRepository(Protocol):
    def create(self, record: FinanceRequestRecord) -> None: ...
    def mark_processing(self, request_id: str) -> None: ...
    def mark_completed(
        self, request_id: str, *, selected_flow: str, answer: str
    ) -> None: ...
    def mark_failed(self, request_id: str, *, error_message: str) -> None: ...
    def get(self, request_id: str) -> FinanceRequestRecord | None: ...


class SnowflakeFinanceRequestRepository:
    """Persist request state in FINANCE_AI.AUDIT.FINANCE_REQUESTS."""

    _TABLE = "FINANCE_AI.AUDIT.FINANCE_REQUESTS"

    def __init__(self, factory: SnowflakeConnectionFactory) -> None:
        self._factory = factory

    def create(self, record: FinanceRequestRecord) -> None:
        self._write(
            f"INSERT INTO {self._TABLE} "
            "(request_id, user_id, question, status, created_at) "
            "VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP())",
            (record.request_id, record.user_id, record.question, "pending"),
        )

    def mark_processing(self, request_id: str) -> None:
        self._write(
            f"UPDATE {self._TABLE} SET status = 'processing', "
            "started_at = CURRENT_TIMESTAMP() WHERE request_id = %s",
            (request_id,),
        )

    def mark_completed(
        self, request_id: str, *, selected_flow: str, answer: str
    ) -> None:
        self._write(
            f"UPDATE {self._TABLE} SET status = 'completed', "
            "selected_flow = %s, answer = %s, error_message = NULL, "
            "completed_at = CURRENT_TIMESTAMP() WHERE request_id = %s",
            (selected_flow, answer, request_id),
        )

    def mark_failed(self, request_id: str, *, error_message: str) -> None:
        self._write(
            f"UPDATE {self._TABLE} SET status = 'failed', "
            "error_message = %s, completed_at = CURRENT_TIMESTAMP() "
            "WHERE request_id = %s",
            (error_message, request_id),
        )

    def get(self, request_id: str) -> FinanceRequestRecord | None:
        query = f"""
            SELECT request_id, user_id, question, status, selected_flow,
                   answer, error_message, created_at, started_at, completed_at
            FROM {self._TABLE}
            WHERE request_id = %s
        """
        try:
            with self._factory.cursor() as cursor:
                cursor.execute(
                    query, (request_id,),
                    timeout=self._factory.config.query_timeout_seconds,
                )
                row = cursor.fetchone()
        except Exception as exc:
            raise FinanceRequestRepositoryError(
                "Unable to retrieve finance request."
            ) from exc
        return FinanceRequestRecord(*row) if row else None

    def _write(self, query: str, params: tuple[object, ...]) -> None:
        try:
            with self._factory.connection() as connection:
                cursor = connection.cursor()
                try:
                    cursor.execute(
                        query, params,
                        timeout=self._factory.config.query_timeout_seconds,
                    )
                    connection.commit()
                finally:
                    cursor.close()
        except Exception as exc:
            raise FinanceRequestRepositoryError(
                "Unable to persist finance request state."
            ) from exc
