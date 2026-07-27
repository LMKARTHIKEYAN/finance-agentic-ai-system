"""Structured contracts shared by LLM providers."""

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field


StructuredOutputT = TypeVar("StructuredOutputT", bound=BaseModel)


class LLMError(RuntimeError):
    """Base error raised by the provider-neutral LLM layer."""


class LLMTimeoutError(LLMError):
    """Raised when an LLM request exceeds its timeout."""


class LLMStructuredOutputError(LLMError):
    """Raised when an LLM response does not match the requested schema."""


class LLMBudgetExceededError(LLMError):
    """Raised when an LLM request or response exceeds an approved limit."""


class LLMUsage(BaseModel):
    """Token and cost information for one structured LLM request."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int = Field(default=0, ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0.0)


class LLMRequestMetadata(BaseModel):
    """Provider metadata retained for observability and auditability."""

    model_config = ConfigDict(frozen=True)

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    request_id: str | None = None
    elapsed_seconds: float = Field(ge=0.0)


class StructuredLLMResponse(
    BaseModel,
    Generic[StructuredOutputT],
):
    """Validated structured output plus provider usage metadata."""

    model_config = ConfigDict(frozen=True)

    output: StructuredOutputT
    usage: LLMUsage
    metadata: LLMRequestMetadata
