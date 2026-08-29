"""Shared state contract for complex autonomous finance execution."""

from __future__ import annotations

from typing import Any, TypedDict


class FinanceGraphState(TypedDict, total=False):
    request: str
    context: dict[str, Any]
    decision: dict[str, Any]
    observations: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    validation: dict[str, Any]
    review: dict[str, Any]
    final_answer: str
    pending_question: str
    approval_request: str
    error: str
    step_count: int
    max_steps: int
    next_action: str
    completed: bool
    executed_calls: list[str]
    execution_trace: list[dict[str, Any]]
