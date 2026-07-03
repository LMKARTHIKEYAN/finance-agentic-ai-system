"""
Application settings for the Finance Agentic AI System.

This file centralizes basic application configuration and project paths.
It will be expanded later when new tools such as OpenAI, Snowflake,
Email, Power BI, and Vector Databases are added.
"""

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


settings = Settings()