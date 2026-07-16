"""
Tests for the Finance Agentic AI API client.

These tests verify:

- Request payload creation
- Successful API response parsing
- Dashboard validation
- Backward compatibility
- Input validation
- HTTP error handling
- Connection error handling
- Timeout handling
"""

from __future__ import annotations

import io
import json
from urllib.error import HTTPError, URLError

import pytest

from src.ui import api_client
from src.ui.api_client import (
    FinanceApiConnectionError,
    FinanceApiResponseError,
    ask_finance_question,
)


class FakeHttpResponse:
    """Minimal context-manager HTTP response used by tests."""

    def __init__(self, payload: dict | list | str) -> None:
        """Initialize the fake response body."""

        if isinstance(payload, str):
            self._body = payload.encode("utf-8")
        else:
            self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "FakeHttpResponse":
        """Enter the response context."""

        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        """Exit the response context."""

    def read(self) -> bytes:
        """Return the encoded response body."""

        return self._body


@pytest.fixture
def valid_dashboard() -> dict:
    """Return a valid structured dashboard response."""

    return {
        "report_metadata": {
            "title": "June 2026 Management Report",
            "period": "June 2026",
            "analysis_type": "Management Report",
            "comparison": "Actual vs Budget",
            "category": "All Categories",
            "generated_at": "2026-07-16T12:00:00+00:00",
            "overall_status": "FAVOURABLE",
        },
        "kpi_cards": [
            {
                "key": "revenue",
                "label": "Revenue",
                "value": 720000,
                "formatted_value": "₹720,000.00",
                "unit": "currency",
            }
        ],
        "kpi_table": {
            "title": "KPI Summary",
            "columns": ["KPI", "Actual"],
            "rows": [
                {
                    "KPI": "Revenue",
                    "Actual": 720000,
                }
            ],
        },
        "variance_table": None,
        "category_table": None,
        "trend_data": [],
        "waterfall_data": [],
        "recommendations": [],
        "risks": [],
        "executive_summary": "Revenue exceeded budget.",
        "commentary": {
            "executive_summary": "Revenue exceeded budget.",
            "management_attention": [],
        },
        "data_limitations": [],
        "available_sections": [
            "kpi_cards",
            "kpi_table",
            "executive_summary",
        ],
        "unavailable_sections": [
            "variance_table",
            "category_table",
            "trend_data",
            "waterfall_data",
            "recommendations",
            "risks",
        ],
    }


@pytest.fixture
def valid_api_response(
    valid_dashboard: dict,
) -> dict:
    """Return a complete valid POST /ask response."""

    return {
        "answer": "Revenue exceeded budget by ₹30,000.",
        "sources": [
            {
                "id": "source-1",
                "score": 0.91,
                "rank": 1,
                "metadata": {
                    "document": "budget_assumptions",
                },
                "excerpt": "Revenue budget assumptions.",
            }
        ],
        "selected_flow": "full",
        "execution_status": "completed",
        "used_fallback": False,
        "dashboard": valid_dashboard,
    }


def test_ask_finance_question_returns_parsed_response(
    monkeypatch: pytest.MonkeyPatch,
    valid_api_response: dict,
) -> None:
    """Successful requests should return the parsed API response."""

    def fake_urlopen(
        request: object,
        timeout: float,
    ) -> FakeHttpResponse:
        assert timeout == 30.0

        return FakeHttpResponse(
            valid_api_response
        )

    monkeypatch.setattr(
        api_client,
        "urlopen",
        fake_urlopen,
    )

    result = ask_finance_question(
        question="Show June revenue performance",
        top_k=5,
        base_url="http://localhost:8000/",
        timeout_seconds=30,
    )

    assert result["answer"] == (
        "Revenue exceeded budget by ₹30,000."
    )
    assert result["selected_flow"] == "full"
    assert result["execution_status"] == "completed"
    assert result["used_fallback"] is False
    assert result["dashboard"] is not None
    assert len(result["sources"]) == 1


def test_ask_finance_question_builds_correct_request(
    monkeypatch: pytest.MonkeyPatch,
    valid_api_response: dict,
) -> None:
    """Client should send the expected JSON request payload."""

    captured: dict = {}

    def fake_urlopen(
        request: object,
        timeout: float,
    ) -> FakeHttpResponse:
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["headers"] = dict(
            request.header_items()
        )
        captured["payload"] = json.loads(
            request.data.decode("utf-8")
        )
        captured["timeout"] = timeout

        return FakeHttpResponse(
            valid_api_response
        )

    monkeypatch.setattr(
        api_client,
        "urlopen",
        fake_urlopen,
    )

    ask_finance_question(
        question="  Show KPI performance  ",
        top_k=7,
        metadata_filter={
            "report_type": "monthly",
        },
        base_url="http://localhost:8000/",
        timeout_seconds=45,
    )

    assert captured["url"] == (
        "http://localhost:8000/ask"
    )
    assert captured["method"] == "POST"
    assert captured["timeout"] == 45.0
    assert captured["payload"] == {
        "question": "Show KPI performance",
        "top_k": 7,
        "metadata_filter": {
            "report_type": "monthly",
        },
    }


def test_response_without_dashboard_remains_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Older API responses should receive dashboard=None."""

    response = {
        "answer": "Finance answer",
        "sources": [],
        "selected_flow": "kpi",
        "execution_status": "completed",
        "used_fallback": False,
    }

    monkeypatch.setattr(
        api_client,
        "urlopen",
        lambda request, timeout: FakeHttpResponse(
            response
        ),
    )

    result = ask_finance_question(
        "Show KPI performance"
    )

    assert result["dashboard"] is None


def test_parse_response_rejects_invalid_json() -> None:
    """Invalid JSON should raise a response error."""

    with pytest.raises(
        FinanceApiResponseError,
        match="not valid JSON",
    ):
        api_client._parse_ask_response(
            "not-json"
        )


def test_parse_response_rejects_non_dictionary() -> None:
    """A JSON list is not a valid API response."""

    with pytest.raises(
        FinanceApiResponseError,
        match="unexpected response structure",
    ):
        api_client._parse_ask_response(
            json.dumps(
                ["invalid"]
            )
        )


@pytest.mark.parametrize(
    "missing_field",
    [
        "answer",
        "sources",
        "execution_status",
        "used_fallback",
    ],
)
def test_parse_response_rejects_missing_fields(
    missing_field: str,
) -> None:
    """Required response fields must be present."""

    response = {
        "answer": "Answer",
        "sources": [],
        "execution_status": "completed",
        "used_fallback": False,
    }

    response.pop(
        missing_field
    )

    with pytest.raises(
        FinanceApiResponseError,
        match="missing required field",
    ):
        api_client._parse_ask_response(
            json.dumps(response)
        )


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "expected_message"),
    [
        (
            "answer",
            123,
            "'answer' must be a string",
        ),
        (
            "sources",
            {},
            "'sources' must be a list",
        ),
        (
            "execution_status",
            123,
            "'execution_status' must be a string",
        ),
        (
            "used_fallback",
            "false",
            "'used_fallback' must be a boolean",
        ),
        (
            "selected_flow",
            100,
            "'selected_flow' must be a string or null",
        ),
    ],
)
def test_parse_response_validates_field_types(
    field_name: str,
    invalid_value: object,
    expected_message: str,
) -> None:
    """Response fields must use their declared types."""

    response = {
        "answer": "Answer",
        "sources": [],
        "selected_flow": "full",
        "execution_status": "completed",
        "used_fallback": False,
    }

    response[field_name] = invalid_value

    with pytest.raises(
        FinanceApiResponseError,
        match=expected_message,
    ):
        api_client._parse_ask_response(
            json.dumps(response)
        )


def test_dashboard_must_be_dictionary() -> None:
    """Dashboard must be an object or null."""

    response = {
        "answer": "Answer",
        "sources": [],
        "selected_flow": "full",
        "execution_status": "completed",
        "used_fallback": False,
        "dashboard": [],
    }

    with pytest.raises(
        FinanceApiResponseError,
        match="'dashboard' must be an object or null",
    ):
        api_client._parse_ask_response(
            json.dumps(response)
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "kpi_cards",
        "trend_data",
        "waterfall_data",
        "recommendations",
        "risks",
        "data_limitations",
        "available_sections",
        "unavailable_sections",
    ],
)
def test_dashboard_list_fields_must_be_lists(
    field_name: str,
) -> None:
    """Dashboard collection fields must be lists."""

    dashboard = {
        field_name: {
            "invalid": True,
        }
    }

    with pytest.raises(
        FinanceApiResponseError,
        match=f"'{field_name}' must be a list",
    ):
        api_client._validate_dashboard_response(
            dashboard
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "kpi_table",
        "variance_table",
        "category_table",
    ],
)
def test_dashboard_table_fields_must_be_objects(
    field_name: str,
) -> None:
    """Dashboard tables must be objects or null."""

    dashboard = {
        field_name: []
    }

    with pytest.raises(
        FinanceApiResponseError,
        match=(
            f"'{field_name}' must be an object or null"
        ),
    ):
        api_client._validate_dashboard_response(
            dashboard
        )


def test_dashboard_report_metadata_must_be_object() -> None:
    """Report metadata must be an object."""

    with pytest.raises(
        FinanceApiResponseError,
        match="'report_metadata' must be an object",
    ):
        api_client._validate_dashboard_response(
            {
                "report_metadata": [],
            }
        )


def test_dashboard_commentary_must_be_object() -> None:
    """Commentary must be an object."""

    with pytest.raises(
        FinanceApiResponseError,
        match="'commentary' must be an object",
    ):
        api_client._validate_dashboard_response(
            {
                "commentary": [],
            }
        )


def test_dashboard_summary_must_be_string_or_null() -> None:
    """Executive summary must be text or null."""

    with pytest.raises(
        FinanceApiResponseError,
        match=(
            "'executive_summary' must be "
            "a string or null"
        ),
    ):
        api_client._validate_dashboard_response(
            {
                "executive_summary": 123,
            }
        )


@pytest.mark.parametrize(
    "question",
    [
        "",
        "   ",
    ],
)
def test_question_cannot_be_empty(
    question: str,
) -> None:
    """Blank questions should fail before HTTP execution."""

    with pytest.raises(
        ValueError,
        match="Question must not be empty",
    ):
        ask_finance_question(
            question
        )


def test_question_must_be_string() -> None:
    """Question must be text."""

    with pytest.raises(
        ValueError,
        match="Question must be a string",
    ):
        ask_finance_question(
            123,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "top_k",
    [
        True,
        1.5,
        "5",
    ],
)
def test_top_k_must_be_integer(
    top_k: object,
) -> None:
    """Boolean, float and string top_k values should fail."""

    with pytest.raises(
        ValueError,
        match="top_k must be an integer",
    ):
        ask_finance_question(
            "Show KPI performance",
            top_k=top_k,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "top_k",
    [
        0,
        21,
    ],
)
def test_top_k_must_be_within_range(
    top_k: int,
) -> None:
    """top_k must stay between 1 and 20."""

    with pytest.raises(
        ValueError,
        match="between 1 and 20",
    ):
        ask_finance_question(
            "Show KPI performance",
            top_k=top_k,
        )


@pytest.mark.parametrize(
    "timeout",
    [
        True,
        "60",
    ],
)
def test_timeout_must_be_numeric(
    timeout: object,
) -> None:
    """Timeout must be an integer or float."""

    with pytest.raises(
        ValueError,
        match="timeout_seconds must be a number",
    ):
        ask_finance_question(
            "Show KPI performance",
            timeout_seconds=timeout,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "timeout",
    [
        0,
        -1,
    ],
)
def test_timeout_must_be_positive(
    timeout: float,
) -> None:
    """Timeout must be greater than zero."""

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        ask_finance_question(
            "Show KPI performance",
            timeout_seconds=timeout,
        )


def test_environment_base_url_is_used(
    monkeypatch: pytest.MonkeyPatch,
    valid_api_response: dict,
) -> None:
    """Environment base URL should be used when no URL is supplied."""

    captured: dict = {}

    monkeypatch.setenv(
        "FINANCE_API_BASE_URL",
        "http://finance-api:9000/",
    )

    def fake_urlopen(
        request: object,
        timeout: float,
    ) -> FakeHttpResponse:
        captured["url"] = request.full_url

        return FakeHttpResponse(
            valid_api_response
        )

    monkeypatch.setattr(
        api_client,
        "urlopen",
        fake_urlopen,
    )

    ask_finance_question(
        "Show revenue"
    )

    assert captured["url"] == (
        "http://finance-api:9000/ask"
    )


def test_http_error_is_converted_to_response_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FastAPI HTTP errors should become client response errors."""

    error_body = io.BytesIO(
        json.dumps(
            {
                "detail": "Finance workflow failed.",
            }
        ).encode("utf-8")
    )

    error = HTTPError(
        url="http://localhost:8000/ask",
        code=500,
        msg="Internal Server Error",
        hdrs=None,
        fp=error_body,
    )

    def fake_urlopen(
        request: object,
        timeout: float,
    ) -> FakeHttpResponse:
        raise error

    monkeypatch.setattr(
        api_client,
        "urlopen",
        fake_urlopen,
    )

    with pytest.raises(
        FinanceApiResponseError,
        match=(
            "HTTP 500: Finance workflow failed"
        ),
    ):
        ask_finance_question(
            "Show finance performance"
        )


def test_connection_error_is_converted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """URL connection failures should become connection errors."""

    def fake_urlopen(
        request: object,
        timeout: float,
    ) -> FakeHttpResponse:
        raise URLError(
            "Connection refused"
        )

    monkeypatch.setattr(
        api_client,
        "urlopen",
        fake_urlopen,
    )

    with pytest.raises(
        FinanceApiConnectionError,
        match="Could not connect",
    ):
        ask_finance_question(
            "Show finance performance"
        )


def test_timeout_error_is_converted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Timeouts should become connection errors."""

    def fake_urlopen(
        request: object,
        timeout: float,
    ) -> FakeHttpResponse:
        raise TimeoutError

    monkeypatch.setattr(
        api_client,
        "urlopen",
        fake_urlopen,
    )

    with pytest.raises(
        FinanceApiConnectionError,
        match="timed out after 10 seconds",
    ):
        ask_finance_question(
            "Show finance performance",
            timeout_seconds=10,
        )