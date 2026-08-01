"""Tests for the Phase 4 finance request lifecycle."""

from dataclasses import replace
from types import SimpleNamespace

from src.repositories.finance_request_repository import FinanceRequestRecord
from src.services.finance_request_lifecycle_service import (
    FinanceRequestLifecycleService,
    FinanceRequestNotFoundError,
)


class InMemoryRequestRepository:
    def __init__(self) -> None:
        self.records = {}

    def create(self, record):
        self.records[record.request_id] = record

    def mark_processing(self, request_id):
        self.records[request_id] = replace(
            self.records[request_id], status="processing"
        )

    def mark_completed(self, request_id, *, selected_flow, answer):
        self.records[request_id] = replace(
            self.records[request_id], status="completed",
            selected_flow=selected_flow, answer=answer,
        )

    def mark_failed(self, request_id, *, error_message):
        self.records[request_id] = replace(
            self.records[request_id], status="failed",
            error_message=error_message,
        )

    def get(self, request_id):
        return self.records.get(request_id)


class SuccessfulFinanceService:
    def ask(self, question, **kwargs):
        return SimpleNamespace(
            selected_flow="variance", answer="Variance answer"
        )


def test_lifecycle_transitions_to_completed() -> None:
    repository = InMemoryRequestRepository()
    service = FinanceRequestLifecycleService(
        repository, SuccessfulFinanceService()
    )

    request_id = service.submit(question="Explain variance", user_id="u1")
    assert service.get(request_id).status == "pending"

    service.process(request_id)

    record = service.get(request_id)
    assert record.status == "completed"
    assert record.selected_flow == "variance"
    assert record.answer == "Variance answer"


def test_lifecycle_records_safe_failure_message() -> None:
    class FailingFinanceService:
        def ask(self, question, **kwargs):
            raise RuntimeError("secret internal detail")

    repository = InMemoryRequestRepository()
    service = FinanceRequestLifecycleService(
        repository, FailingFinanceService()
    )
    request_id = service.submit(question="Explain variance", user_id=None)

    service.process(request_id)

    record = service.get(request_id)
    assert record.status == "failed"
    assert record.error_message == "Finance request processing failed."
    assert "secret" not in record.error_message


def test_lifecycle_rejects_unknown_request() -> None:
    service = FinanceRequestLifecycleService(
        InMemoryRequestRepository(), SuccessfulFinanceService()
    )
    try:
        service.get("00000000-0000-0000-0000-000000000000")
    except FinanceRequestNotFoundError:
        pass
    else:
        raise AssertionError("Expected FinanceRequestNotFoundError")
