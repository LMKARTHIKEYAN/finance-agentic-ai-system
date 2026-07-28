"""OpenAI implementation of the structured LLM client."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any, TypeVar

from pydantic import BaseModel

from src.llm.client import LLMMessage, StructuredLLMClient
from src.llm.schemas import (
    LLMBudgetExceededError,
    LLMError,
    LLMOutputLimitError,
    LLMRequestMetadata,
    LLMStructuredOutputError,
    LLMTimeoutError,
    LLMUsage,
    StructuredLLMResponse,
)


StructuredOutputT = TypeVar("StructuredOutputT", bound=BaseModel)


class OpenAIStructuredLLMClient(StructuredLLMClient):
    """Generate schema-validated responses through the OpenAI SDK."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        max_output_tokens: int = 1_200,
        max_total_tokens: int | None = None,
        client: Any | None = None,
    ) -> None:
        self._model = _required_text(model, "model")
        self._timeout_seconds = _positive_number(
            timeout_seconds,
            "timeout_seconds",
        )
        self._max_output_tokens = _positive_integer(
            max_output_tokens,
            "max_output_tokens",
        )
        self._max_total_tokens = (
            _positive_integer(max_total_tokens, "max_total_tokens")
            if max_total_tokens is not None
            else None
        )
        self._client = (
            client
            if client is not None
            else self._create_client(
                api_key=api_key,
                timeout_seconds=self._timeout_seconds,
            )
        )

    @property
    def provider(self) -> str:
        return "openai"

    @property
    def model(self) -> str:
        return self._model

    @property
    def timeout_seconds(self) -> float:
        return self._timeout_seconds

    @property
    def max_output_tokens(self) -> int:
        return self._max_output_tokens

    @property
    def max_total_tokens(self) -> int | None:
        return self._max_total_tokens

    def generate_structured(
        self,
        *,
        messages: Sequence[LLMMessage],
        response_model: type[StructuredOutputT],
        max_output_tokens: int | None = None,
        timeout_seconds: float | None = None,
    ) -> StructuredLLMResponse[StructuredOutputT]:
        validated_messages = _validate_messages(messages)
        validated_response_model = _validate_response_model(response_model)
        output_limit = (
            self._max_output_tokens
            if max_output_tokens is None
            else _positive_integer(max_output_tokens, "max_output_tokens")
        )
        request_timeout = (
            self._timeout_seconds
            if timeout_seconds is None
            else _positive_number(timeout_seconds, "timeout_seconds")
        )

        started_at = time.perf_counter()

        try:
            response = self._client.responses.parse(
                model=self._model,
                input=validated_messages,
                text_format=validated_response_model,
                max_output_tokens=output_limit,
                timeout=request_timeout,
            )
        except Exception as exc:
            if _is_timeout_error(exc):
                raise LLMTimeoutError(
                    "OpenAI structured response timed out."
                ) from exc
            raise LLMError(
                f"OpenAI structured response failed: {exc}"
            ) from exc

        elapsed_seconds = time.perf_counter() - started_at
        parsed_output = getattr(response, "output_parsed", None)

        if not isinstance(parsed_output, validated_response_model):
            incomplete_details = getattr(
                response,
                "incomplete_details",
                None,
            )
            incomplete_reason = getattr(
                incomplete_details,
                "reason",
                None,
            )
            if (
                getattr(response, "status", None) == "incomplete"
                and incomplete_reason == "max_output_tokens"
            ):
                raise LLMOutputLimitError(
                    "OpenAI response reached the configured output-token "
                    "limit."
                )
            raise LLMStructuredOutputError(
                "OpenAI did not return output matching the requested schema."
            )

        usage = _build_usage(getattr(response, "usage", None))

        if (
            self._max_total_tokens is not None
            and usage.total_tokens > self._max_total_tokens
        ):
            raise LLMBudgetExceededError(
                "OpenAI response exceeded the configured total-token limit."
            )

        request_id = getattr(response, "id", None)

        return StructuredLLMResponse[StructuredOutputT](
            output=parsed_output,
            usage=usage,
            metadata=LLMRequestMetadata(
                provider=self.provider,
                model=self.model,
                request_id=(
                    request_id
                    if isinstance(request_id, str) and request_id
                    else None
                ),
                elapsed_seconds=elapsed_seconds,
            ),
        )

    @staticmethod
    def _create_client(
        *,
        api_key: str | None,
        timeout_seconds: float,
    ) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMError(
                "The OpenAI package is not installed."
            ) from exc

        if not isinstance(api_key, str) or not api_key.strip():
            raise LLMError(
                "OPENAI_API_KEY is required for the OpenAI LLM provider."
            )

        try:
            return OpenAI(
                api_key=api_key.strip(),
                timeout=timeout_seconds,
            )
        except Exception as exc:
            raise LLMError(
                "Unable to initialize the OpenAI client."
            ) from exc


def _build_usage(raw_usage: Any) -> LLMUsage:
    if raw_usage is None:
        return LLMUsage()

    input_tokens = _non_negative_int(
        getattr(raw_usage, "input_tokens", 0)
    )
    output_tokens = _non_negative_int(
        getattr(raw_usage, "output_tokens", 0)
    )
    total_tokens = _non_negative_int(
        getattr(
            raw_usage,
            "total_tokens",
            input_tokens + output_tokens,
        )
    )

    input_details = getattr(
        raw_usage,
        "input_tokens_details",
        None,
    )
    cached_input_tokens = _non_negative_int(
        getattr(input_details, "cached_tokens", 0)
    )

    return LLMUsage(
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def _validate_messages(
    messages: Sequence[LLMMessage],
) -> list[LLMMessage]:
    if isinstance(messages, (str, bytes)) or not isinstance(
        messages,
        Sequence,
    ):
        raise TypeError("messages must be a sequence of dictionaries.")

    if not messages:
        raise ValueError("messages cannot be empty.")

    validated: list[LLMMessage] = []

    for message in messages:
        if not isinstance(message, dict):
            raise TypeError("each message must be a dictionary.")

        role = message.get("role")
        content = message.get("content")

        if not isinstance(role, str) or not role.strip():
            raise ValueError("each message requires a non-empty role.")

        if not isinstance(content, str) or not content.strip():
            raise ValueError("each message requires non-empty text content.")

        validated.append(dict(message))

    return validated


def _validate_response_model(
    response_model: type[StructuredOutputT],
) -> type[StructuredOutputT]:
    if not isinstance(response_model, type) or not issubclass(
        response_model,
        BaseModel,
    ):
        raise TypeError("response_model must be a Pydantic BaseModel type.")

    return response_model


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")

    cleaned = value.strip()

    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty.")

    return cleaned


def _positive_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer.")

    if value <= 0:
        raise ValueError(f"{field_name} must be greater than zero.")

    return value


def _positive_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric.")

    numeric_value = float(value)

    if numeric_value <= 0:
        raise ValueError(f"{field_name} must be greater than zero.")

    return numeric_value


def _non_negative_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(value, 0)


def _is_timeout_error(error: Exception) -> bool:
    try:
        from openai import APITimeoutError
    except ImportError:
        APITimeoutError = ()  # type: ignore[assignment,misc]

    return isinstance(error, (TimeoutError, APITimeoutError))
