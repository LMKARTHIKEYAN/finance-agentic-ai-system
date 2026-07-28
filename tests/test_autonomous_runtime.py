"""Tests for provider-neutral autonomous runtime construction."""

from typing import Any

from src.autonomous.runtime import build_autonomous_runtime
from src.api.dependencies import build_autonomous_service_executor
from src.config.settings import Settings
from src.config.settings import settings
from src.llm.client import StructuredLLMClient


class FakeClient(StructuredLLMClient):
    @property
    def provider(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return "fake"

    def generate_structured(self, **kwargs: Any) -> Any:
        raise AssertionError("Runtime construction must not call the LLM.")


class TestSettings(Settings):
    AUTONOMOUS_MAX_AGENTS = 6
    AUTONOMOUS_MAX_REPLANS = 2
    AUTONOMOUS_MAX_RETRIES_PER_AGENT = 1
    AUTONOMOUS_MAX_TOOL_CALLS = 10
    AUTONOMOUS_MAX_INPUT_TOKENS = 1_000
    AUTONOMOUS_MAX_OUTPUT_TOKENS = 500
    AUTONOMOUS_MAX_TOTAL_TOKENS = 2_000
    AUTONOMOUS_MAX_COST_USD = 0.02
    AUTONOMOUS_MAX_EXECUTION_SECONDS = 60


def test_runtime_builds_all_dependencies_without_llm_call() -> None:
    client = FakeClient()

    runtime = build_autonomous_runtime(
        app_settings=TestSettings(),
        llm_client=client,
    )

    assert runtime.llm_client is client
    assert runtime.supervisor._llm_client is client
    assert runtime.reviewer._llm_client is client
    assert runtime.limits.max_output_tokens == 500
    assert callable(runtime.graph.invoke)


def test_runtime_shares_limits_across_control_components() -> None:
    runtime = build_autonomous_runtime(
        app_settings=TestSettings(),
        llm_client=FakeClient(),
    )

    assert runtime.coordinator._limits == runtime.limits
    assert runtime.validator._limits == runtime.limits


def test_runtime_activation_is_disabled_by_default(
    monkeypatch,
) -> None:
    monkeypatch.setattr(settings, "AUTONOMOUS_ENABLED", False)
    monkeypatch.delenv("AUTONOMOUS_SHADOW_MODE", raising=False)

    assert build_autonomous_service_executor() is None


def test_runtime_activation_uses_injected_runtime(
    monkeypatch,
) -> None:
    runtime = build_autonomous_runtime(
        app_settings=TestSettings(),
        llm_client=FakeClient(),
    )
    monkeypatch.setattr(settings, "AUTONOMOUS_ENABLED", True)

    executor = build_autonomous_service_executor(runtime=runtime)

    assert executor._runtime is runtime
