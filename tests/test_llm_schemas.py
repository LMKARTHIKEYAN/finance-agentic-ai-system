"""Tests for provider-neutral LLM schemas."""

import pytest
from pydantic import BaseModel, ValidationError

from src.llm.schemas import (
    LLMRequestMetadata,
    LLMUsage,
    StructuredLLMResponse,
)


class ExampleOutput(BaseModel):
    decision: str


def test_llm_usage_accepts_valid_counts() -> None:
    usage = LLMUsage(
        input_tokens=100,
        cached_input_tokens=20,
        output_tokens=30,
        total_tokens=130,
        estimated_cost_usd=0.01,
    )

    assert usage.total_tokens == 130
    assert usage.estimated_cost_usd == pytest.approx(0.01)


@pytest.mark.parametrize(
    "field_name",
    [
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "total_tokens",
    ],
)
def test_llm_usage_rejects_negative_tokens(field_name: str) -> None:
    with pytest.raises(ValidationError):
        LLMUsage(**{field_name: -1})


def test_llm_usage_rejects_negative_cost() -> None:
    with pytest.raises(ValidationError):
        LLMUsage(estimated_cost_usd=-0.01)


def test_structured_response_preserves_validated_output() -> None:
    response = StructuredLLMResponse[ExampleOutput](
        output=ExampleOutput(decision="autonomous"),
        usage=LLMUsage(total_tokens=5),
        metadata=LLMRequestMetadata(
            provider="openai",
            model="test-model",
            request_id="response-1",
            elapsed_seconds=0.1,
        ),
    )

    assert response.output.decision == "autonomous"
    assert response.metadata.provider == "openai"


def test_request_metadata_rejects_blank_provider() -> None:
    with pytest.raises(ValidationError):
        LLMRequestMetadata(
            provider="",
            model="test-model",
            elapsed_seconds=0.1,
        )
