"""Tests for the structured LLM client factory."""

from types import SimpleNamespace

import pytest

from src.config.settings import Settings
from src.llm.factory import create_llm_client
from src.llm.openai_client import OpenAIStructuredLLMClient


class TestSettings(Settings):
    AUTONOMOUS_LLM_PROVIDER = "openai"
    AUTONOMOUS_LLM_MODEL = "factory-model"
    AUTONOMOUS_LLM_TIMEOUT_SECONDS = 12.0
    AUTONOMOUS_MAX_OUTPUT_TOKENS = 300
    AUTONOMOUS_MAX_TOTAL_TOKENS = 900


def test_factory_creates_openai_client() -> None:
    client = create_llm_client(
        api_key="test-key",
        app_settings=TestSettings(),
        client=SimpleNamespace(),
    )

    assert isinstance(client, OpenAIStructuredLLMClient)
    assert client.provider == "openai"
    assert client.model == "factory-model"
    assert client.timeout_seconds == 12.0
    assert client.max_output_tokens == 300
    assert client.max_total_tokens == 900


def test_factory_allows_model_override() -> None:
    client = create_llm_client(
        model="override-model",
        api_key="test-key",
        app_settings=TestSettings(),
        client=SimpleNamespace(),
    )

    assert client.model == "override-model"


def test_factory_rejects_unsupported_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        create_llm_client(
            provider="unsupported",
            app_settings=TestSettings(),
            client=SimpleNamespace(),
        )


def test_factory_requires_api_key_without_injected_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(Exception, match="OPENAI_API_KEY"):
        create_llm_client(
            app_settings=TestSettings(),
        )
