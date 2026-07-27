"""
Application settings for the Finance Agentic AI System.

This file centralizes basic application configuration and project paths.
It will be expanded later when new tools such as OpenAI, Snowflake,
Email, Power BI, and Vector Databases are added.
"""

import os
from pathlib import Path


class Settings:
    """
    Stores application-level settings and important project paths.

    Other modules will import this class instance instead of hardcoding
    project paths or application configuration.
    """

    APP_NAME: str = "Finance Agentic AI System"
    APP_ENV: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

    DATA_PATH: Path = PROJECT_ROOT / "data"
    REPORT_PATH: Path = PROJECT_ROOT / "reports"
    LOG_PATH: Path = PROJECT_ROOT / "logs"

    # Autonomous LLM execution remains disabled until the later hybrid
    # orchestration phases are implemented and explicitly enabled.
    AUTONOMOUS_ENABLED: bool = (
        os.getenv("AUTONOMOUS_ENABLED", "false").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    AUTONOMOUS_SHADOW_MODE: bool = (
        os.getenv("AUTONOMOUS_SHADOW_MODE", "true").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    AUTONOMOUS_CLASSIFIER_LLM_ENABLED: bool = (
        os.getenv(
            "AUTONOMOUS_CLASSIFIER_LLM_ENABLED",
            "false",
        ).strip().lower()
        in {"1", "true", "yes", "on"}
    )
    AUTONOMOUS_CLASSIFIER_CONFIDENCE_THRESHOLD: float = float(
        os.getenv(
            "AUTONOMOUS_CLASSIFIER_CONFIDENCE_THRESHOLD",
            "0.8",
        )
    )
    AUTONOMOUS_LLM_PROVIDER: str = os.getenv(
        "AUTONOMOUS_LLM_PROVIDER",
        "openai",
    ).strip()
    AUTONOMOUS_LLM_MODEL: str = os.getenv(
        "AUTONOMOUS_LLM_MODEL",
        "gpt-5-mini",
    ).strip()
    AUTONOMOUS_LLM_TIMEOUT_SECONDS: float = float(
        os.getenv("AUTONOMOUS_LLM_TIMEOUT_SECONDS", "30")
    )
    AUTONOMOUS_MAX_INPUT_TOKENS: int = int(
        os.getenv("AUTONOMOUS_MAX_INPUT_TOKENS", "12000")
    )
    AUTONOMOUS_MAX_OUTPUT_TOKENS: int = int(
        os.getenv("AUTONOMOUS_MAX_OUTPUT_TOKENS", "1200")
    )
    AUTONOMOUS_MAX_TOTAL_TOKENS: int = int(
        os.getenv("AUTONOMOUS_MAX_TOTAL_TOKENS", "20000")
    )
    AUTONOMOUS_MAX_COST_USD: float = float(
        os.getenv("AUTONOMOUS_MAX_COST_USD", "0.03")
    )
    AUTONOMOUS_MAX_EXECUTION_SECONDS: float = float(
        os.getenv("AUTONOMOUS_MAX_EXECUTION_SECONDS", "120")
    )
    AUTONOMOUS_MAX_AGENTS: int = int(
        os.getenv("AUTONOMOUS_MAX_AGENTS", "6")
    )
    AUTONOMOUS_MAX_REPLANS: int = int(
        os.getenv("AUTONOMOUS_MAX_REPLANS", "2")
    )
    AUTONOMOUS_MAX_RETRIES_PER_AGENT: int = int(
        os.getenv("AUTONOMOUS_MAX_RETRIES_PER_AGENT", "1")
    )
    AUTONOMOUS_MAX_TOOL_CALLS: int = int(
        os.getenv("AUTONOMOUS_MAX_TOOL_CALLS", "10")
    )


settings = Settings()
