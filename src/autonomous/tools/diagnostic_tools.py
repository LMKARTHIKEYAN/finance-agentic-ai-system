"""Deterministic root-cause and recommendation tool wrappers."""

from __future__ import annotations

from typing import Any

from src.agents.analytics.recommendation_agent import RecommendationAgent
from src.agents.analytics.root_cause_agent import RootCauseAgent
from src.autonomous.schemas import ToolResult
from src.autonomous.tools.data_tools import serialize_tool_payload


def identify_supported_root_causes(
    *,
    anomaly_result: Any,
    operations_result: Any,
    revenue_variance_result: Any | None = None,
    agent: Any | None = None,
) -> ToolResult:
    """Run the existing deterministic root-cause analysis."""

    root_cause_agent = agent if agent is not None else RootCauseAgent()
    result = root_cause_agent.analyze(
        anomaly_result=anomaly_result,
        operations_result=operations_result,
        variance_result=revenue_variance_result,
    )
    return ToolResult(
        call_id="identify_supported_root_causes",
        tool_name="identify_supported_root_causes",
        status="completed",
        payload=serialize_tool_payload(result),
    )


def generate_supported_recommendations(
    *,
    root_cause_result: Any,
    agent: Any | None = None,
) -> ToolResult:
    """Run the existing deterministic recommendation analysis."""

    recommendation_agent = (
        agent if agent is not None else RecommendationAgent()
    )
    result = recommendation_agent.analyze(root_cause_result)
    return ToolResult(
        call_id="generate_supported_recommendations",
        tool_name="generate_supported_recommendations",
        status="completed",
        payload=serialize_tool_payload(result),
    )
