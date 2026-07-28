"""Tests for the structured autonomous finance supervisor."""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import pytest

from src.autonomous.agents.supervisor_agent import (
    FinanceSupervisorAgent,
    SupervisorAgentError,
)
from src.autonomous.execution_limits import AutonomousExecutionLimits
from src.autonomous.schemas import (
    DatasetAvailability,
    PlanStep,
    PlanValidationIssue,
    ReportingScope,
    SupervisorPlan,
)
from src.autonomous.tools.registry import ToolRegistry
from src.llm.client import StructuredLLMClient
from src.llm.schemas import (
    LLMRequestMetadata,
    LLMUsage,
    StructuredLLMResponse,
)


def _plan() -> SupervisorPlan:
    return SupervisorPlan(
        objective="Analyze April profitability",
        reporting_scope=ReportingScope(
            start_date=date(2025, 4, 1),
            end_date=date(2025, 4, 30),
        ),
        steps=(
            PlanStep(
                step_id="pnl",
                capability="pnl_analysis",
                arguments={
                    "agent_name": "pnl_agent",
                    "tool_name": "generate_validated_pnl_analysis",
                },
            ),
            PlanStep(
                step_id="review",
                capability="review",
                depends_on=("pnl",),
                arguments={"agent_name": "reviewer_agent"},
            ),
        ),
        required_reconciliations=("pnl_structure",),
        expected_outputs=("management_answer",),
    )


class FakeStructuredLLMClient(StructuredLLMClient):
    def __init__(
        self,
        output: SupervisorPlan | object,
        *,
        error: Exception | None = None,
    ) -> None:
        self.output = output
        self.error = error
        self.calls: list[dict[str, Any]] = []

    @property
    def provider(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return "fake-supervisor"

    def generate_structured(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        if isinstance(self.output, SupervisorPlan):
            return StructuredLLMResponse[SupervisorPlan](
                output=self.output,
                usage=LLMUsage(),
                metadata=LLMRequestMetadata(
                    provider=self.provider,
                    model=self.model,
                    elapsed_seconds=0.01,
                ),
            )
        return type("InvalidResponse", (), {"output": self.output})()


def _user_context(client: FakeStructuredLLMClient) -> dict[str, Any]:
    messages = client.calls[0]["messages"]
    return json.loads(messages[1]["content"])


def test_supervisor_returns_structured_plan() -> None:
    expected = _plan()
    client = FakeStructuredLLMClient(expected)

    result = FinanceSupervisorAgent(client).propose_plan(
        "Why did April profit improve?"
    )

    assert result == expected
    assert client.calls[0]["response_model"] is SupervisorPlan
    assert client.calls[0]["max_output_tokens"] == 1_200
    assert client.calls[0]["timeout_seconds"] == 120.0


def test_supervisor_plan_schema_is_strict_openai_compatible() -> None:
    schema = SupervisorPlan.model_json_schema()
    arguments_schema = schema["$defs"]["PlanStepArguments"]

    assert arguments_schema["additionalProperties"] is False
    assert "agent_name" in arguments_schema["properties"]
    assert "tool_name" in arguments_schema["properties"]
    assert "requested_kpis" in arguments_schema["properties"]


def test_supervisor_sends_only_safe_dataset_and_tool_metadata() -> None:
    client = FakeStructuredLLMClient(_plan())
    dataset = DatasetAvailability(
        dataset_name="operations",
        available=True,
        row_count=42,
        period_start=date(2025, 4, 1),
        period_end=date(2025, 4, 30),
    )

    FinanceSupervisorAgent(client).propose_plan(
        "Analyze April performance",
        datasets=(dataset,),
    )
    context = _user_context(client)

    assert context["dataset_availability"][0]["row_count"] == 42
    assert "calculate_validated_kpis" in {
        item["name"] for item in context["approved_tools"]
    }
    assert all("function" not in item for item in context["approved_tools"])
    assert "commission_amount" not in client.calls[0]["messages"][1]["content"]


def test_supervisor_includes_scope_limits_and_validator_feedback() -> None:
    client = FakeStructuredLLMClient(_plan())
    limits = AutonomousExecutionLimits(
        max_agents=4,
        max_replans=1,
        max_output_tokens=600,
    )
    scope = ReportingScope(category="Widgets")
    issue = PlanValidationIssue(
        code="missing_reviewer",
        message="Add a reviewer step.",
        severity="error",
    )

    FinanceSupervisorAgent(client, limits=limits).propose_plan(
        "Diagnose margin movement",
        reporting_scope=scope,
        validation_issues=(issue,),
    )
    context = _user_context(client)

    assert context["reporting_scope"]["category"] == "Widgets"
    assert context["execution_limits"]["max_agents"] == 4
    assert context["execution_limits"]["max_replans"] == 1
    assert context["previous_validation_issues"][0]["code"] == (
        "missing_reviewer"
    )
    assert client.calls[0]["max_output_tokens"] == 600


def test_supervisor_instructions_enforce_finance_safety() -> None:
    instructions = FinanceSupervisorAgent.instructions.lower()

    assert "never calculate" in instructions
    assert "deterministic python tools" in instructions
    assert "reviewer step" in instructions
    assert "reconciliation" in instructions
    assert "external writes" in instructions
    assert "product-level" in instructions
    assert "portfolio-level" in instructions


@pytest.mark.parametrize("request_text", ["", "   "])
def test_supervisor_rejects_empty_request_without_llm_call(
    request_text: str,
) -> None:
    client = FakeStructuredLLMClient(_plan())

    with pytest.raises(ValueError, match="request"):
        FinanceSupervisorAgent(client).propose_plan(request_text)

    assert client.calls == []


def test_supervisor_rejects_raw_or_unvalidated_dataset_objects() -> None:
    client = FakeStructuredLLMClient(_plan())

    with pytest.raises(TypeError, match="DatasetAvailability"):
        FinanceSupervisorAgent(client).propose_plan(
            "Analyze performance",
            datasets=({"raw_rows": [{"revenue": 100}]},),  # type: ignore[arg-type]
        )

    assert client.calls == []


def test_supervisor_wraps_provider_error_without_exposing_details() -> None:
    client = FakeStructuredLLMClient(
        _plan(),
        error=RuntimeError("secret provider response"),
    )

    with pytest.raises(
        SupervisorAgentError,
        match="Supervisor plan generation failed",
    ) as error:
        FinanceSupervisorAgent(client).propose_plan("Analyze performance")

    assert "secret provider response" not in str(error.value)


@pytest.mark.parametrize(
    ("provider_error_name", "expected_code"),
    [
        ("AuthenticationError", "authentication_error"),
        (
            "PermissionDeniedError",
            "permission_or_model_access_denied",
        ),
        ("RateLimitError", "rate_limit_or_quota_exceeded"),
        ("BadRequestError", "invalid_provider_request"),
        ("UnprocessableEntityError", "unprocessable_request"),
        ("NotFoundError", "model_or_endpoint_not_found"),
        ("APIConnectionError", "provider_connection_error"),
        ("InternalServerError", "provider_server_error"),
        ("APIResponseValidationError", "response_validation_error"),
        ("LengthFinishReasonError", "output_token_limit"),
        ("ContentFilterFinishReasonError", "content_filter"),
        ("ValidationError", "structured_output_invalid"),
        ("OpenAIError", "provider_sdk_error"),
        ("APIError", "provider_sdk_error"),
    ],
)
def test_supervisor_classifies_safe_provider_error_types(
    provider_error_name: str,
    expected_code: str,
) -> None:
    provider_error_type = type(provider_error_name, (RuntimeError,), {})
    client = FakeStructuredLLMClient(
        _plan(),
        error=provider_error_type("secret provider details"),
    )

    with pytest.raises(SupervisorAgentError) as captured:
        FinanceSupervisorAgent(client).propose_plan("Analyze performance")

    assert captured.value.failure_code == expected_code
    assert "secret provider details" not in str(captured.value)


def test_supervisor_rejects_non_structured_provider_output() -> None:
    client = FakeStructuredLLMClient({"objective": "unvalidated"})

    with pytest.raises(SupervisorAgentError, match="invalid structured plan"):
        FinanceSupervisorAgent(client).propose_plan("Analyze performance")

    assert len(client.calls) == 2


def test_supervisor_retries_invalid_structure_then_succeeds() -> None:
    class RetryClient(FakeStructuredLLMClient):
        def __init__(self) -> None:
            super().__init__(_plan())
            self.attempt = 0

        def generate_structured(self, **kwargs: Any) -> Any:
            self.calls.append(kwargs)
            self.attempt += 1
            if self.attempt == 1:
                invalid_type = type("ValidationError", (RuntimeError,), {})
                raise invalid_type("secret invalid response")
            return StructuredLLMResponse[SupervisorPlan](
                output=_plan(),
                usage=LLMUsage(),
                metadata=LLMRequestMetadata(
                    provider=self.provider,
                    model=self.model,
                    elapsed_seconds=0.01,
                ),
            )

    client = RetryClient()

    result = FinanceSupervisorAgent(client).propose_plan(
        "Analyze performance"
    )

    assert result == _plan()
    assert len(client.calls) == 2
    retry_messages = client.calls[1]["messages"]
    assert "failed structured contract validation" in (
        retry_messages[-1]["content"]
    )
    assert "secret invalid response" not in str(retry_messages)


def test_supervisor_stops_after_configured_structure_retry() -> None:
    invalid_type = type("ValidationError", (RuntimeError,), {})
    client = FakeStructuredLLMClient(
        _plan(),
        error=invalid_type("secret invalid response"),
    )
    limits = AutonomousExecutionLimits(max_retries_per_agent=1)

    with pytest.raises(SupervisorAgentError) as captured:
        FinanceSupervisorAgent(
            client,
            limits=limits,
        ).propose_plan("Analyze performance")

    assert captured.value.failure_code == "structured_output_invalid"
    assert len(client.calls) == 2
    assert "secret invalid response" not in str(captured.value)


def test_supervisor_does_not_retry_non_structural_provider_failure() -> None:
    error_type = type("AuthenticationError", (RuntimeError,), {})
    client = FakeStructuredLLMClient(
        _plan(),
        error=error_type("secret credentials"),
    )

    with pytest.raises(SupervisorAgentError) as captured:
        FinanceSupervisorAgent(client).propose_plan(
            "Analyze performance"
        )

    assert captured.value.failure_code == "authentication_error"
    assert len(client.calls) == 1


def test_supervisor_rejects_empty_tool_registry() -> None:
    client = FakeStructuredLLMClient(_plan())

    with pytest.raises(ValueError, match="approved tool"):
        FinanceSupervisorAgent(client, registry=ToolRegistry(()))


def test_supervisor_rejects_output_limit_above_execution_limit() -> None:
    client = FakeStructuredLLMClient(_plan())

    with pytest.raises(ValueError, match="cannot exceed"):
        FinanceSupervisorAgent(
            client,
            limits=AutonomousExecutionLimits(max_output_tokens=100),
            max_output_tokens=101,
        )


def test_supervisor_does_not_mutate_inputs() -> None:
    client = FakeStructuredLLMClient(_plan())
    datasets = (
        DatasetAvailability(dataset_name="budget", available=True),
    )
    issues = (
        PlanValidationIssue(
            code="unknown_tool",
            message="Use an approved tool.",
            severity="error",
        ),
    )
    before = (datasets, issues)

    FinanceSupervisorAgent(client).propose_plan(
        "Analyze performance",
        datasets=datasets,
        validation_issues=issues,
    )

    assert (datasets, issues) == before
