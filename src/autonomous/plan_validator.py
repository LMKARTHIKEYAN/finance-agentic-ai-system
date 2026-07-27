"""Deterministic validation for LLM-proposed autonomous plans."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

from src.autonomous.execution_limits import AutonomousExecutionLimits
from src.autonomous.schemas import (
    PlanValidationIssue,
    PlanValidationResult,
    SupervisorPlan,
)
from src.autonomous.tools.registry import (
    DEFAULT_TOOL_REGISTRY,
    ToolRegistry,
)


_RECONCILIATION_BY_TOOL = {
    "generate_validated_pnl_analysis": "pnl_structure",
    "calculate_validated_revenue_variance": "revenue_variance",
    "calculate_validated_gp_decomposition": "gp_decomposition",
}

_FINANCE_CAPABILITY_TERMS = (
    "kpi",
    "pnl",
    "p&l",
    "revenue_variance",
    "gp_decomposition",
    "gross_margin",
)


class AutonomousPlanValidator:
    """Approve only bounded plans using allow-listed tools."""

    def __init__(
        self,
        *,
        registry: ToolRegistry = DEFAULT_TOOL_REGISTRY,
        limits: AutonomousExecutionLimits | None = None,
    ) -> None:
        self._registry = registry
        self._limits = limits or AutonomousExecutionLimits()

    def validate(
        self,
        plan: SupervisorPlan,
        *,
        available_inputs: Collection[str] = (),
    ) -> PlanValidationResult:
        """Validate a plan without executing it or mutating its contents."""

        if not isinstance(plan, SupervisorPlan):
            raise TypeError("plan must be SupervisorPlan.")

        issues: list[PlanValidationIssue] = []
        selected_agents: list[str] = []
        selected_tools: list[str] = []
        available = set(available_inputs)

        self._validate_cycles(plan, issues)

        for step in plan.steps:
            arguments = step.arguments
            agent_name = _optional_text(arguments.get("agent_name"))
            tool_name = _optional_text(arguments.get("tool_name"))

            if not agent_name and not tool_name:
                issues.append(
                    _error(
                        "missing_execution_target",
                        "Plan step must select an approved agent or tool.",
                        step.step_id,
                    )
                )

            if agent_name and agent_name not in selected_agents:
                selected_agents.append(agent_name)

            if tool_name:
                try:
                    definition = self._registry.get(tool_name)
                except KeyError:
                    issues.append(
                        _error(
                            "unknown_tool",
                            f"Tool {tool_name!r} is not allow-listed.",
                            step.step_id,
                        )
                    )
                    continue

                if tool_name not in selected_tools:
                    selected_tools.append(tool_name)

                missing = [
                    name
                    for name in definition.required_inputs
                    if name not in available
                    and name not in arguments
                ]
                if missing:
                    issues.append(
                        _error(
                            "missing_tool_inputs",
                            f"Tool {tool_name!r} is missing inputs: "
                            f"{', '.join(missing)}.",
                            step.step_id,
                        )
                    )

                required_reconciliation = (
                    _RECONCILIATION_BY_TOOL.get(tool_name)
                )
                if (
                    required_reconciliation
                    and required_reconciliation
                    not in plan.required_reconciliations
                ):
                    issues.append(
                        _error(
                            "missing_reconciliation",
                            "Plan is missing required reconciliation "
                            f"{required_reconciliation!r}.",
                            step.step_id,
                        )
                    )

            if (
                agent_name
                and _is_finance_capability(step.capability)
                and not tool_name
            ):
                issues.append(
                    _error(
                        "llm_calculation_forbidden",
                        "Finance analysis agents must use an approved "
                        "deterministic calculation tool.",
                        step.step_id,
                    )
                )

            if _scope_override_is_inconsistent(
                arguments,
                plan,
            ):
                issues.append(
                    _error(
                        "inconsistent_reporting_scope",
                        "Step reporting scope conflicts with the approved "
                        "plan scope.",
                        step.step_id,
                    )
                )

        if not any(
            "review" in step.capability.lower()
            or _optional_text(step.arguments.get("agent_name"))
            in {"reviewer", "reviewer_agent"}
            for step in plan.steps
        ):
            issues.append(
                _error(
                    "reviewer_required",
                    "Every autonomous plan must include a reviewer step.",
                )
            )

        if len(selected_agents) > self._limits.max_agents:
            issues.append(
                _error(
                    "agent_limit_exceeded",
                    "Plan selects more than the maximum allowed agents.",
                )
            )

        has_errors = any(
            issue.severity == "error"
            for issue in issues
        )
        return PlanValidationResult(
            valid=not has_errors,
            approved_plan=None if has_errors else plan,
            issues=tuple(issues),
            selected_agents=tuple(selected_agents),
            selected_tools=tuple(selected_tools),
        )

    @staticmethod
    def _validate_cycles(
        plan: SupervisorPlan,
        issues: list[PlanValidationIssue],
    ) -> None:
        graph = {
            step.step_id: tuple(step.depends_on)
            for step in plan.steps
        }
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(step_id: str) -> bool:
            if step_id in visiting:
                return True
            if step_id in visited:
                return False
            visiting.add(step_id)
            if any(visit(item) for item in graph[step_id]):
                return True
            visiting.remove(step_id)
            visited.add(step_id)
            return False

        if any(visit(step_id) for step_id in graph):
            issues.append(
                _error(
                    "circular_dependency",
                    "Plan contains a circular dependency.",
                )
            )


def _scope_override_is_inconsistent(
    arguments: dict[str, Any],
    plan: SupervisorPlan,
) -> bool:
    scope = plan.reporting_scope
    comparisons = (
        ("start_date", scope.start_date),
        ("end_date", scope.end_date),
        ("category", scope.category),
    )
    for key, expected in comparisons:
        supplied = arguments.get(key)
        if supplied is None or expected is None:
            continue
        if str(supplied) != str(expected):
            return True
    return False


def _is_finance_capability(capability: str) -> bool:
    normalized = capability.lower()
    return any(term in normalized for term in _FINANCE_CAPABILITY_TERMS)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _error(
    code: str,
    message: str,
    step_id: str | None = None,
) -> PlanValidationIssue:
    return PlanValidationIssue(
        code=code,
        message=message,
        severity="error",
        step_id=step_id,
    )
