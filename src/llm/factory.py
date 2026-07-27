"""Factory for provider-neutral structured LLM clients."""

from __future__ import annotations

import os
from typing import Any

from src.config.settings import Settings, settings
from src.llm.client import StructuredLLMClient
from src.llm.openai_client import OpenAIStructuredLLMClient


def create_llm_client(
    *,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    app_settings: Settings = settings,
    client: Any | None = None,
) -> StructuredLLMClient:
    """Create the configured structured LLM provider client."""

    resolved_provider = (
        provider or app_settings.AUTONOMOUS_LLM_PROVIDER
    ).strip().lower()

    if resolved_provider != "openai":
        raise ValueError(
            f"Unsupported autonomous LLM provider: {resolved_provider!r}."
        )

    resolved_model = model or app_settings.AUTONOMOUS_LLM_MODEL
    resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")

    return OpenAIStructuredLLMClient(
        model=resolved_model,
        api_key=resolved_api_key,
        timeout_seconds=app_settings.AUTONOMOUS_LLM_TIMEOUT_SECONDS,
        max_output_tokens=app_settings.AUTONOMOUS_MAX_OUTPUT_TOKENS,
        max_total_tokens=app_settings.AUTONOMOUS_MAX_TOTAL_TOKENS,
        client=client,
    )
