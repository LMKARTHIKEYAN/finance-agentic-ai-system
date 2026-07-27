"""Structured, provider-neutral supervisor for autonomous finance planning."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from src.autonomous.execution_limits import AutonomousExecutionLimits
from src.autonomous.schemas import (
    DatasetAvailability,
    PlanValidationIssue,
    ReportingScope,
    SupervisorPlan,
)
from src.autonomous.tools.registry import (
    DEFAULT_TOOL_REGISTRY,
    ToolRegistry,
)
from src.llm.client import LLMMessage, StructuredLLMClient


class SupervisorAgentError(RuntimeError):
    """Raised when the supervisor cannot produce a structured plan."""


class FinanceSupervisorAgent:
    """Propose finance-analysis plans without executing calculations."""

    name = "Finance Supervisor Agent"
    description = (
        "Selects approved finance specialists and deterministic tools for "
        "complex diagnostic and decision-support requests."
    )
    instructions = """
You are the supervisor for a controlled FP&A analysis workflow.
Return only a structured SupervisorPlan matching the supplied schema.

Planning rules:
- Select only tools listed in approved_tools.
- Never calculate, estimate, infer, or invent financial values.
- Financial calculations must be performed by deterministic Python tools.
- Use calculate_validated_kpis for KPI analysis.
- Use generate_validated_pnl_analysis for Actual, Budget, and P&L variance.
- Use calculate_validated_revenue_variance for revenue variance.
- Use calculate_validated_gp_decomposition for both Product-Level and
  Portfolio-Level GP% decomposition.
- Add root-cause or recommendation analysis only when the request needs it.
- Every plan must finish with a reviewer step that depends on all conclusions.
- Request the reconciliation checks needed for every financial result.
- Keep dependencies explicit and acyclic.
- Do not exceed the supplied execution limits.
- Do not request external writes or communications.
- Treat previous_validation_issues as deterministic correction requirements.
- Use only supplied request context and metadata; no raw datasets are available.
""".strip()

    def __init__(
        self,
        llm_client: StructuredLLMClient,
        *,
        registry: ToolRegistry = DEFAULT_TOOL_REGISTRY,
        limits: AutonomousExecutionLimits | None = None,
        max_output_tokens: int | None = None,
    ) -> None:
        if not isinstance(llm_client, StructuredLLMClient):
            raise TypeError(
                "llm_client must implement StructuredLLMClient."
            )
        if not isinstance(registry, ToolRegistry):
            raise TypeError("registry must be a ToolRegistry.")
        if not registry.names:
            raise ValueError("registry must contain at least one approved tool.")

        resolved_limits = limits or AutonomousExecutionLimits()
        resolved_max_tokens = (
            resolved_limits.max_output_tokens
            if max_output_tokens is None
            else max_output_tokens
        )
        if (
            isinstance(resolved_max_tokens, bool)
            or not isinstance(resolved_max_tokens, int)
        ):
            raise TypeError("max_output_tokens must be an integer.")
        if resolved_max_tokens <= 0:
            raise ValueError("max_output_tokens must be positive.")
        if resolved_max_tokens > resolved_limits.max_output_tokens:
            raise ValueError(
                "max_output_tokens cannot exceed the autonomous limit."
            )

        self._llm_client = llm_client
        self._registry = registry
        self._limits = resolved_limits
        self._max_output_tokens = resolved_max_tokens

    def propose_plan(
        self,
        request: str,
        *,
        reporting_scope: ReportingScope | None = None,
        datasets: Sequence[DatasetAvailability] = (),
        validation_issues: Sequence[PlanValidationIssue] = (),
    ) -> SupervisorPlan:
        """Ask the LLM for one structured plan using safe metadata only."""

        cleaned_request = _required_text(request, "request")
        scope = reporting_scope or ReportingScope()
        if not isinstance(scope, ReportingScope):
            raise TypeError("reporting_scope must be a ReportingScope.")

        dataset_metadata = _validated_sequence(
            datasets,
            DatasetAvailability,
            "datasets",
        )
        issues = _validated_sequence(
            validation_issues,
            PlanValidationIssue,
            "validation_issues",
        )
        messages = self._build_messages(
            request=cleaned_request,
            reporting_scope=scope,
            datasets=dataset_metadata,
            validation_issues=issues,
        )

        try:
            response = self._llm_client.generate_structured(
                messages=messages,
                response_model=SupervisorPlan,
                max_output_tokens=self._max_output_tokens,
                timeout_seconds=self._limits.max_execution_seconds,
            )
        except Exception as exc:
            raise SupervisorAgentError(
                "Supervisor plan generation failed."
            ) from exc

        if not isinstance(response.output, SupervisorPlan):
            raise SupervisorAgentError(
                "Supervisor returned an invalid structured plan."
            )
        return response.output

    def _build_messages(
        self,
        *,
        request: str,
        reporting_scope: ReportingScope,
        datasets: tuple[DatasetAvailability, ...],
        validation_issues: tuple[PlanValidationIssue, ...],
    ) -> tuple[LLMMessage, ...]:
        approved_tools = []
        for definition in self._registry.as_mapping().values():
            approved_tools.append(
                {
                    "name": definition.name,
                    "description": definition.description,
                    "required_inputs": definition.required_inputs,
                    "result_type": definition.result_type,
                    "performs_finance_calculation": (
                        definition.performs_finance_calculation
                    ),
                    "external_write": definition.external_write,
                }
            )

        context: dict[str, Any] = {
            "request": request,
            "reporting_scope": reporting_scope.model_dump(mode="json"),
            "dataset_availability": [
                item.model_dump(mode="json") for item in datasets
            ],
            "approved_tools": approved_tools,
            "execution_limits": self._limits.model_dump(mode="json"),
            "previous_validation_issues": [
                item.model_dump(mode="json") for item in validation_issues
            ],
        }
        return (
            {"role": "system", "content": self.instructions},
            {
                "role": "user",
                "content": json.dumps(
                    context,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            },
        )


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty.")
    return cleaned


def _validated_sequence(
    values: object,
    expected_type: type[Any],
    field_name: str,
) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{field_name} must be a sequence.")
    result = tuple(values)
    if any(not isinstance(item, expected_type) for item in result):
        raise TypeError(
            f"every {field_name} item must be a "
            f"{expected_type.__name__}."
        )
    return result
