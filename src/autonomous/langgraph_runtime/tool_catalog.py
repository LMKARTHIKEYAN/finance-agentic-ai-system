"""LLM-safe catalog over the existing allow-listed Python tools."""

from __future__ import annotations

from typing import Any

from src.autonomous.tools.registry import DEFAULT_TOOL_REGISTRY, ToolRegistry


class LangGraphToolCatalog:
    def __init__(self, registry: ToolRegistry = DEFAULT_TOOL_REGISTRY) -> None:
        self.registry = registry

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "name": item.name,
                "description": item.description,
                "required_inputs": list(item.required_inputs),
                "result_type": item.result_type,
                "external_write": item.external_write,
            }
            for item in self.registry.as_mapping().values()
        ]

    def contains(self, name: str) -> bool:
        return name in self.registry.names
