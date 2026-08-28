"""Explicit permissions for autonomous reads, calculations, and releases."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PermissionDecision:
    allowed: bool
    requires_approval: bool
    reason: str


class PermissionPolicy:
    """Keep autonomy inside an application-owned allow list."""

    def __init__(
        self,
        *,
        allowed_tools: set[str],
        external_write_tools: set[str] | None = None,
    ) -> None:
        self.allowed_tools = frozenset(allowed_tools)
        self.external_write_tools = frozenset(external_write_tools or set())

    def evaluate(self, tool_name: str) -> PermissionDecision:
        if tool_name not in self.allowed_tools:
            return PermissionDecision(False, False, "Tool is not allow-listed.")
        if tool_name in self.external_write_tools:
            return PermissionDecision(True, True, "External write requires approval.")
        return PermissionDecision(True, False, "Read/calculation tool is approved.")
