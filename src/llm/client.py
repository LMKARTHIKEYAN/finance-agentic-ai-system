"""Provider-neutral interface for structured LLM generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, TypeVar

from pydantic import BaseModel

from src.llm.schemas import StructuredLLMResponse


StructuredOutputT = TypeVar("StructuredOutputT", bound=BaseModel)
LLMMessage = dict[str, Any]


class StructuredLLMClient(ABC):
    """Interface implemented by structured-output LLM providers."""

    @property
    @abstractmethod
    def provider(self) -> str:
        """Return the configured provider name."""

    @property
    @abstractmethod
    def model(self) -> str:
        """Return the configured model name."""

    @abstractmethod
    def generate_structured(
        self,
        *,
        messages: Sequence[LLMMessage],
        response_model: type[StructuredOutputT],
        max_output_tokens: int | None = None,
        timeout_seconds: float | None = None,
    ) -> StructuredLLMResponse[StructuredOutputT]:
        """Generate and validate one structured response."""
