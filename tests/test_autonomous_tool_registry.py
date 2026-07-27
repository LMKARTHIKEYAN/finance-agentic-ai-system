"""Tests for the autonomous deterministic-tool allow list."""

import pytest

from src.autonomous.tools.registry import (
    DEFAULT_TOOL_REGISTRY,
    ToolDefinition,
    ToolRegistry,
)


def _noop() -> None:
    return None


def test_default_registry_contains_approved_tools() -> None:
    assert set(DEFAULT_TOOL_REGISTRY.names) == {
        "calculate_validated_kpis",
        "generate_validated_pnl_analysis",
        "calculate_validated_revenue_variance",
        "calculate_validated_gp_decomposition",
        "identify_supported_root_causes",
        "generate_supported_recommendations",
        "retrieve_company_context",
    }


def test_registry_rejects_unknown_tool() -> None:
    with pytest.raises(KeyError, match="Unknown"):
        DEFAULT_TOOL_REGISTRY.get("arbitrary_python")


def test_registry_rejects_duplicate_names() -> None:
    definition = ToolDefinition(
        "duplicate",
        "Test tool.",
        _noop,
        (),
        "test",
        False,
    )
    with pytest.raises(ValueError, match="unique"):
        ToolRegistry((definition, definition))


def test_registry_rejects_external_write_tool() -> None:
    definition = ToolDefinition(
        "send_email",
        "External write.",
        _noop,
        (),
        "external",
        False,
        external_write=True,
    )
    with pytest.raises(ValueError, match="External-write"):
        ToolRegistry((definition,))


def test_registry_mapping_is_immutable() -> None:
    with pytest.raises(TypeError):
        DEFAULT_TOOL_REGISTRY.as_mapping()["new"] = (  # type: ignore[index]
            DEFAULT_TOOL_REGISTRY.get("calculate_validated_kpis")
        )
