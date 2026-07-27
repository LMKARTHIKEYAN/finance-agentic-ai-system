"""Mocked tests for the OpenAI structured LLM client."""

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from src.llm.openai_client import OpenAIStructuredLLMClient
from src.llm.schemas import (
    LLMBudgetExceededError,
    LLMError,
    LLMStructuredOutputError,
    LLMTimeoutError,
)


class DecisionOutput(BaseModel):
    decision: str


class FakeResponsesAPI:
    def __init__(
        self,
        *,
        output: object,
        usage: object | None = None,
        error: Exception | None = None,
    ) -> None:
        self.output = output
        self.usage = usage
        self.error = error
        self.calls: list[dict[str, object]] = []

    def parse(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(dict(kwargs))

        if self.error is not None:
            raise self.error

        return SimpleNamespace(
            id="response-123",
            output_parsed=self.output,
            usage=self.usage,
        )


class FakeOpenAIClient:
    def __init__(self, responses: FakeResponsesAPI) -> None:
        self.responses = responses


def build_client(
    responses: FakeResponsesAPI,
    *,
    max_total_tokens: int | None = 1_000,
) -> OpenAIStructuredLLMClient:
    return OpenAIStructuredLLMClient(
        model="test-model",
        client=FakeOpenAIClient(responses),
        timeout_seconds=10,
        max_output_tokens=200,
        max_total_tokens=max_total_tokens,
    )


def test_generate_structured_returns_output_and_usage() -> None:
    usage = SimpleNamespace(
        input_tokens=100,
        output_tokens=25,
        total_tokens=125,
        input_tokens_details=SimpleNamespace(cached_tokens=10),
    )
    responses = FakeResponsesAPI(
        output=DecisionOutput(decision="autonomous"),
        usage=usage,
    )

    result = build_client(responses).generate_structured(
        messages=[{"role": "user", "content": "Classify request"}],
        response_model=DecisionOutput,
    )

    assert result.output.decision == "autonomous"
    assert result.usage.input_tokens == 100
    assert result.usage.cached_input_tokens == 10
    assert result.usage.output_tokens == 25
    assert result.metadata.request_id == "response-123"
    assert responses.calls[0]["model"] == "test-model"
    assert responses.calls[0]["max_output_tokens"] == 200
    assert responses.calls[0]["timeout"] == 10.0


def test_generate_structured_rejects_invalid_output() -> None:
    responses = FakeResponsesAPI(output={"decision": "autonomous"})

    with pytest.raises(LLMStructuredOutputError):
        build_client(responses).generate_structured(
            messages=[{"role": "user", "content": "Classify"}],
            response_model=DecisionOutput,
        )


def test_generate_structured_converts_timeout() -> None:
    responses = FakeResponsesAPI(
        output=None,
        error=TimeoutError("slow"),
    )

    with pytest.raises(LLMTimeoutError):
        build_client(responses).generate_structured(
            messages=[{"role": "user", "content": "Classify"}],
            response_model=DecisionOutput,
        )


def test_generate_structured_converts_provider_error() -> None:
    responses = FakeResponsesAPI(
        output=None,
        error=RuntimeError("provider failure"),
    )

    with pytest.raises(LLMError, match="provider failure"):
        build_client(responses).generate_structured(
            messages=[{"role": "user", "content": "Classify"}],
            response_model=DecisionOutput,
        )


def test_generate_structured_enforces_total_token_limit() -> None:
    usage = SimpleNamespace(
        input_tokens=90,
        output_tokens=20,
        total_tokens=110,
        input_tokens_details=None,
    )
    responses = FakeResponsesAPI(
        output=DecisionOutput(decision="autonomous"),
        usage=usage,
    )

    with pytest.raises(LLMBudgetExceededError):
        build_client(
            responses,
            max_total_tokens=100,
        ).generate_structured(
            messages=[{"role": "user", "content": "Classify"}],
            response_model=DecisionOutput,
        )


def test_client_rejects_missing_api_key_without_injected_client() -> None:
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        OpenAIStructuredLLMClient(
            model="test-model",
            api_key=None,
        )


def test_client_error_does_not_include_api_key() -> None:
    secret = "secret-key-value"

    with pytest.raises(ValueError) as captured:
        OpenAIStructuredLLMClient(
            model="test-model",
            api_key=secret,
            timeout_seconds=0,
        )

    assert secret not in str(captured.value)


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "", "content": "text"}],
        [{"role": "user", "content": ""}],
    ],
)
def test_generate_structured_validates_messages(
    messages: list[dict[str, str]],
) -> None:
    responses = FakeResponsesAPI(
        output=DecisionOutput(decision="simple")
    )

    with pytest.raises((TypeError, ValueError)):
        build_client(responses).generate_structured(
            messages=messages,
            response_model=DecisionOutput,
        )
