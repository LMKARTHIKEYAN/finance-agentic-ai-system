"""SQLite execution history and duplicate-delivery protection."""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class ReportExecution:
    execution_id: str
    report_type: str
    period_key: str
    status: str


class ReportExecutionService:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def start(self, report_type: str, period_key: str) -> ReportExecution:
        if self.was_delivered(report_type, period_key):
            raise ValueError(f"{report_type} report {period_key} was already delivered.")
        execution = ReportExecution(uuid.uuid4().hex, report_type, period_key, "RUNNING")
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO report_executions VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL)",
                (execution.execution_id, report_type, period_key, "RUNNING", self._now()),
            )
        return execution

    def complete(self, execution_id: str, email_status: str = "NOT_REQUESTED") -> None:
        self._finish(execution_id, "SUCCESS", email_status, None)

    def fail(self, execution_id: str, error_message: str) -> None:
        self._finish(execution_id, "FAILED", "NOT_SENT", error_message[:2000])

    def was_delivered(self, report_type: str, period_key: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM report_executions WHERE report_type=? AND period_key=? AND status='SUCCESS' AND email_status='SENT' LIMIT 1",
                (report_type, period_key),
            ).fetchone()
        return row is not None

    def _finish(self, execution_id: str, status: str, email_status: str, error: str | None) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE report_executions SET status=?, completed_at=?, email_status=?, error_message=? WHERE execution_id=?",
                (status, self._now(), email_status, error, execution_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("Unknown execution_id.")

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS report_executions (
                execution_id TEXT PRIMARY KEY, report_type TEXT NOT NULL, period_key TEXT NOT NULL,
                status TEXT NOT NULL, started_at TEXT NOT NULL, completed_at TEXT,
                email_status TEXT, error_message TEXT)""")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database_path)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()
