"""Construction of the reusable supervised autonomous runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.autonomous.agents.reviewer_agent import ReviewerAgent
from src.autonomous.agents.supervisor_agent import FinanceSupervisorAgent
from src.autonomous.execution_coordinator import (
    AutonomousExecutionCoordinator,
)
from src.autonomous.execution_limits import AutonomousExecutionLimits
from src.autonomous.graph import build_autonomous_graph
from src.autonomous.plan_validator import AutonomousPlanValidator
from src.config.settings import Settings, settings
from src.llm.client import StructuredLLMClient
from src.llm.factory import create_llm_client


@dataclass(frozen=True)
class AutonomousRuntime:
    """Reusable runtime dependencies for autonomous service execution."""

    llm_client: StructuredLLMClient
    supervisor: FinanceSupervisorAgent
    reviewer: ReviewerAgent
    validator: AutonomousPlanValidator
    coordinator: AutonomousExecutionCoordinator
    graph: Any
    limits: AutonomousExecutionLimits


def build_autonomous_runtime(
    *,
    app_settings: Settings = settings,
    llm_client: StructuredLLMClient | None = None,
    api_key: str | None = None,
) -> AutonomousRuntime:
    """Build one runtime; API keys remain inside the LLM client."""

    limits = AutonomousExecutionLimits(
        max_agents=app_settings.AUTONOMOUS_MAX_AGENTS,
        max_replans=app_settings.AUTONOMOUS_MAX_REPLANS,
        max_retries_per_agent=(
            app_settings.AUTONOMOUS_MAX_RETRIES_PER_AGENT
        ),
        max_tool_calls=app_settings.AUTONOMOUS_MAX_TOOL_CALLS,
        max_input_tokens=app_settings.AUTONOMOUS_MAX_INPUT_TOKENS,
        max_output_tokens=app_settings.AUTONOMOUS_MAX_OUTPUT_TOKENS,
        max_total_tokens=app_settings.AUTONOMOUS_MAX_TOTAL_TOKENS,
        max_cost_usd=app_settings.AUTONOMOUS_MAX_COST_USD,
        max_execution_seconds=(
            app_settings.AUTONOMOUS_MAX_EXECUTION_SECONDS
        ),
    )
    client = llm_client or create_llm_client(
        app_settings=app_settings,
        api_key=api_key,
    )
    supervisor = FinanceSupervisorAgent(
        client,
        limits=limits,
    )
    supervisor.instructions = (
        supervisor.instructions
        + """

Runtime plan contract:
- KPI step: capability "kpi_analysis", agent_name "kpi_agent",
  tool_name "calculate_validated_kpis", and requested_kpis as a non-empty list.
- P&L step: capability "pnl_analysis", agent_name "pnl_agent",
  tool_name "generate_validated_pnl_analysis", and required reconciliation
  "pnl_structure".
- Revenue variance step: capability "revenue_variance", agent_name
  "revenue_variance_agent", tool_name
  "calculate_validated_revenue_variance", and required reconciliation
  "revenue_variance".
- GP% step: capability "gp_decomposition", agent_name
  "gp_decomposition_agent", tool_name
  "calculate_validated_gp_decomposition", and required reconciliation
  "gp_decomposition".
- Diagnostic step: capability "root_cause_recommendation", agent_name
  "root_cause_recommendation_agent", tool_name
  "identify_supported_root_causes", and recommendation_tool_name
  "generate_supported_recommendations". It must depend on every selected
  finance step.
- Final step: capability "review", agent_name "reviewer_agent". It must
  depend on the diagnostic step.
- Do not add any other capability, tool, agent, or argument name.
- For a P&L diagnostic request, select only pnl_analysis,
  root_cause_recommendation, and review unless the request explicitly asks
  for KPI, revenue variance, or GP% decomposition.
- Do not select retrieval tools for finance calculations or diagnostics.
""".strip()
    )
    reviewer = ReviewerAgent(client, limits=limits)
    validator = AutonomousPlanValidator(limits=limits)
    coordinator = AutonomousExecutionCoordinator(
        reviewer=reviewer,
        limits=limits,
    )
    graph = build_autonomous_graph(
        supervisor=supervisor,
        validator=validator,
        coordinator=coordinator,
        limits=limits,
    )
    return AutonomousRuntime(
        llm_client=client,
        supervisor=supervisor,
        reviewer=reviewer,
        validator=validator,
        coordinator=coordinator,
        graph=graph,
        limits=limits,
    )
