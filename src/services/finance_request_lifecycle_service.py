"""Application service for finance request submission and processing."""

from __future__ import annotations

from uuid import UUID, uuid4

from src.api.service import FinanceAskService
from src.repositories.finance_request_repository import (
    FinanceRequestRecord,
    FinanceRequestRepository,
)


class FinanceRequestNotFoundError(LookupError):
    """Raised when a request identifier does not exist."""


class FinanceRequestLifecycleService:
    """Coordinate request state transitions around FinanceAskService."""

    def __init__(
        self,
        repository: FinanceRequestRepository,
        finance_service: FinanceAskService,
    ) -> None:
        self._repository = repository
        self._finance_service = finance_service

    def submit(self, *, question: str, user_id: str | None) -> str:
        request_id = str(uuid4())
        self._repository.create(
            FinanceRequestRecord(
                request_id=request_id,
                user_id=user_id,
                question=question,
                status="pending",
            )
        )
        return request_id

    def process(self, request_id: str) -> None:
        record = self.get(request_id)
        self._repository.mark_processing(request_id)
        try:
            result = self._finance_service.ask(
                record.question,
                user_id=record.user_id,
            )
            self._repository.mark_completed(
                request_id,
                selected_flow=result.selected_flow,
                answer=result.answer,
            )
        except Exception:
            self._repository.mark_failed(
                request_id,
                error_message="Finance request processing failed.",
            )

    def get(self, request_id: str) -> FinanceRequestRecord:
        UUID(request_id)
        record = self._repository.get(request_id)
        if record is None:
            raise FinanceRequestNotFoundError(request_id)
        return record
