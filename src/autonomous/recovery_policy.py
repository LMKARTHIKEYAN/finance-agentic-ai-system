"""Bounded retry and safe-recovery decisions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RecoveryDecision:
    action: str
    reason: str


class RecoveryPolicy:
    def __init__(self, max_retries: int = 1) -> None:
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative.")
        self.max_retries = max_retries

    def decide(self, *, attempt: int, recoverable: bool) -> RecoveryDecision:
        if recoverable and attempt <= self.max_retries:
            return RecoveryDecision("retry", "Recoverable failure within retry limit.")
        return RecoveryDecision("stop", "Failure is unsafe or retry limit was reached.")
