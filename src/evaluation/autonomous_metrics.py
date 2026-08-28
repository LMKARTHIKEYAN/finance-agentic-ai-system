"""Metrics for autonomy, safety, evidence, and completion."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AutonomousMetrics:
    goals: int = 0
    completed: int = 0
    clarification_requests: int = 0
    tool_calls: int = 0
    validated_answers: int = 0
    unsafe_actions_blocked: int = 0
    reconciliations_passed: int = 0
    reconciliations_failed: int = 0
    cited_answers: int = 0
    tool_failures: int = 0

    def completion_rate(self) -> float:
        return self.completed / self.goals if self.goals else 0.0

    def evidence_validation_rate(self) -> float:
        return self.validated_answers / self.completed if self.completed else 0.0

    def reconciliation_rate(self) -> float:
        total = self.reconciliations_passed + self.reconciliations_failed
        return self.reconciliations_passed / total if total else 0.0

    def citation_coverage(self) -> float:
        return self.cited_answers / self.completed if self.completed else 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            **self.__dict__,
            "completion_rate": self.completion_rate(),
            "evidence_validation_rate": self.evidence_validation_rate(),
            "reconciliation_rate": self.reconciliation_rate(),
            "citation_coverage": self.citation_coverage(),
        }
