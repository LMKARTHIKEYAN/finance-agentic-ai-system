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
from src.llm.schemas import (
    LLMBudgetExceededError,
    LLMError,
    LLMOutputLimitError,
    LLMStructuredOutputError,
    LLMTimeoutError,
)


class SupervisorAgentError(RuntimeError):
    """Raised when the supervisor cannot produce a structured plan."""

    def __init__(self, message: str, *, failure_code: str) -> None:
        super().__init__(message)
        self.failure_code = failure_code


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

        attempts = self._limits.max_retries_per_agent + 1
        current_messages = messages
        last_error: Exception | None = None

        for attempt in range(attempts):
            try:
                response = self._llm_client.generate_structured(
                    messages=current_messages,
                    response_model=SupervisorPlan,
                    max_output_tokens=self._max_output_tokens,
                    timeout_seconds=self._limits.max_execution_seconds,
                )
            except Exception as exc:
                failure_code = _safe_failure_code(exc)
                if (
                    failure_code == "structured_output_invalid"
                    and attempt + 1 < attempts
                ):
                    last_error = exc
                    current_messages = _retry_messages(messages)
                    continue
                raise SupervisorAgentError(
                    "Supervisor plan generation failed.",
                    failure_code=failure_code,
                ) from exc

            if isinstance(response.output, SupervisorPlan):
                return response.output

            if attempt + 1 < attempts:
                current_messages = _retry_messages(messages)
                continue
            raise SupervisorAgentError(
                "Supervisor returned an invalid structured plan.",
                failure_code="structured_output_invalid",
            )

        raise SupervisorAgentError(
            "Supervisor plan generation failed.",
            failure_code="structured_output_invalid",
        ) from last_error

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


def _retry_messages(
    original_messages: tuple[LLMMessage, ...],
) -> tuple[LLMMessage, ...]:
    """Add a bounded, data-free correction request for one retry."""

    return (
        *original_messages,
        {
            "role": "user",
            "content": (
                "The previous plan failed structured contract validation. "
                "Return a corrected SupervisorPlan with unique step IDs, "
                "valid acyclic dependencies, a valid reporting date range, "
                "only approved argument fields, and the required final "
                "reviewer step. Do not calculate financial values."
            ),
        },
    )


def _safe_failure_code(error: Exception) -> str:
    """Classify provider failures without exposing exception details."""

    current: BaseException | None = error
    generic_llm_error = False
    while current is not None:
        if isinstance(current, LLMTimeoutError):
            return "timeout"
        if isinstance(current, LLMOutputLimitError):
            return "output_token_limit"
        if isinstance(current, LLMBudgetExceededError):
            return "token_budget_exceeded"
        if isinstance(current, LLMStructuredOutputError):
            return "structured_output_invalid"
        if isinstance(current, LLMError):
            generic_llm_error = True

        provider_code = {
            "AuthenticationError": "authentication_error",
            "PermissionDeniedError": "permission_or_model_access_denied",
            "RateLimitError": "rate_limit_or_quota_exceeded",
            "BadRequestError": "invalid_provider_request",
            "UnprocessableEntityError": "unprocessable_request",
            "NotFoundError": "model_or_endpoint_not_found",
            "APIConnectionError": "provider_connection_error",
            "InternalServerError": "provider_server_error",
            "APIResponseValidationError": "response_validation_error",
            "LengthFinishReasonError": "output_token_limit",
            "ContentFilterFinishReasonError": "content_filter",
            "ValidationError": "structured_output_invalid",
            "OpenAIError": "provider_sdk_error",
            "APIError": "provider_sdk_error",
        }.get(type(current).__name__)
        if provider_code is not None:
            return provider_code

        status_code = getattr(current, "status_code", None)
        if status_code == 401:
            return "authentication_error"
        if status_code == 403:
            return "permission_or_model_access_denied"
        if status_code == 404:
            return "model_or_endpoint_not_found"
        if status_code == 429:
            return "rate_limit_or_quota_exceeded"
        if status_code == 400:
            return "invalid_provider_request"
        if isinstance(status_code, int) and status_code >= 500:
            return "provider_server_error"

        current = current.__cause__
    return "provider_api_error" if generic_llm_error else "unexpected_error"
