"""
Reusable HTTP client for the Finance Agentic AI FastAPI backend.

This module is responsible only for:

- Building requests for the existing POST /ask endpoint
- Sending JSON payloads to FastAPI
- Parsing successful API responses
- Translating HTTP, connection, and response errors into client exceptions

It must not contain Streamlit UI code, finance calculations, LangGraph logic,
RAG retrieval logic, or database access logic.
"""

from __future__ import annotations

import json
import os
from json import JSONDecodeError
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT_SECONDS = 60.0


class FinanceApiClientError(RuntimeError):
    """Base exception raised by the Finance API client."""


class FinanceApiConnectionError(FinanceApiClientError):
    """Raised when the FastAPI server cannot be reached."""


class FinanceApiResponseError(FinanceApiClientError):
    """Raised when FastAPI returns an error or an invalid response."""


def ask_finance_question(
    question: str,
    top_k: int = 5,
    metadata_filter: dict[str, Any] | None = None,
    *,
    base_url: str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """
    Submit a finance question to the existing FastAPI ``POST /ask`` route.

    Args:
        question:
            Finance-related question entered by the user.

        top_k:
            Maximum number of RAG sources requested from the API.
            The current API accepts values from 1 to 20.

        metadata_filter:
            Optional metadata values used by the RAG retriever.

        base_url:
            FastAPI server base URL. When omitted, the function uses the
            ``FINANCE_API_BASE_URL`` environment variable and falls back to
            ``http://127.0.0.1:8000``.

        timeout_seconds:
            Maximum number of seconds to wait for the API response.

    Returns:
        Parsed API response containing ``answer``, ``sources``,
        ``selected_flow``, ``execution_status``, and ``used_fallback``.

    Raises:
        ValueError:
            If local input values are invalid.

        FinanceApiConnectionError:
            If the FastAPI application cannot be reached.

        FinanceApiResponseError:
            If the API returns an HTTP error, invalid JSON, or an unexpected
            response structure.
    """

    validated_question = _validate_question(question)
    validated_top_k = _validate_top_k(top_k)
    validated_timeout = _validate_timeout(timeout_seconds)

    payload = {
        "question": validated_question,
        "top_k": validated_top_k,
        "metadata_filter": dict(metadata_filter or {}),
    }

    endpoint = f"{_resolve_base_url(base_url)}/ask"

    request = Request(
        url=endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=validated_timeout) as response:
            response_body = response.read().decode("utf-8")

    except HTTPError as exc:
        error_message = _extract_http_error_message(exc)

        raise FinanceApiResponseError(
            f"Finance API returned HTTP {exc.code}: {error_message}"
        ) from exc

    except URLError as exc:
        reason = getattr(exc, "reason", exc)

        raise FinanceApiConnectionError(
            "Could not connect to the Finance API at "
            f"{endpoint}. Confirm that FastAPI is running. Reason: {reason}"
        ) from exc

    except TimeoutError as exc:
        raise FinanceApiConnectionError(
            "The Finance API request timed out after "
            f"{validated_timeout:g} seconds."
        ) from exc

    return _parse_ask_response(response_body)


def _resolve_base_url(base_url: str | None) -> str:
    """Resolve and normalize the FastAPI base URL."""

    resolved_url = (
        base_url
        or os.getenv("FINANCE_API_BASE_URL")
        or DEFAULT_API_BASE_URL
    ).strip()

    if not resolved_url:
        raise ValueError("Finance API base URL must not be empty.")

    return resolved_url.rstrip("/")


def _validate_question(question: str) -> str:
    """Validate and normalize a finance question."""

    if not isinstance(question, str):
        raise ValueError("Question must be a string.")

    normalized_question = question.strip()

    if not normalized_question:
        raise ValueError("Question must not be empty.")

    return normalized_question


def _validate_top_k(top_k: int) -> int:
    """Validate the retrieval result limit used by the API."""

    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise ValueError("top_k must be an integer.")

    if not 1 <= top_k <= 20:
        raise ValueError("top_k must be between 1 and 20.")

    return top_k


def _validate_timeout(timeout_seconds: float) -> float:
    """Validate the HTTP request timeout."""

    if isinstance(timeout_seconds, bool) or not isinstance(
        timeout_seconds,
        (int, float),
    ):
        raise ValueError("timeout_seconds must be a number.")

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than zero.")

    return float(timeout_seconds)


def _parse_ask_response(response_body: str) -> dict[str, Any]:
    """Parse and validate the successful ``POST /ask`` response."""

    try:
        parsed_response = json.loads(response_body)

    except JSONDecodeError as exc:
        raise FinanceApiResponseError(
            "Finance API returned a response that is not valid JSON."
        ) from exc

    if not isinstance(parsed_response, dict):
        raise FinanceApiResponseError(
            "Finance API returned an unexpected response structure."
        )

    required_fields = {
        "answer",
        "sources",
        "execution_status",
        "used_fallback",
    }

    missing_fields = required_fields.difference(parsed_response)

    if missing_fields:
        missing_fields_text = ", ".join(sorted(missing_fields))

        raise FinanceApiResponseError(
            "Finance API response is missing required field(s): "
            f"{missing_fields_text}."
        )

    if not isinstance(parsed_response["answer"], str):
        raise FinanceApiResponseError(
            "Finance API response field 'answer' must be a string."
        )

    if not isinstance(parsed_response["sources"], list):
        raise FinanceApiResponseError(
            "Finance API response field 'sources' must be a list."
        )

    return parsed_response


def _extract_http_error_message(error: HTTPError) -> str:
    """Extract a readable FastAPI error message from an HTTP error."""

    try:
        response_body = error.read().decode("utf-8")
        parsed_error = json.loads(response_body)

    except (JSONDecodeError, UnicodeDecodeError):
        return error.reason or "Unknown API error"

    if isinstance(parsed_error, dict):
        detail = parsed_error.get("detail")

        if isinstance(detail, str):
            return detail

        if detail is not None:
            return json.dumps(detail, ensure_ascii=False)

    return error.reason or "Unknown API error"