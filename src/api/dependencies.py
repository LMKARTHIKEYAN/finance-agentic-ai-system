"""
Dependency construction for the Finance Agentic AI API.

This module creates the real infrastructure dependencies used by FastAPI.

It contains wiring only and does not contain:

- Finance calculations
- LangGraph business logic
- RAG prompt logic
- HTTP route logic
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.api.service import (
    FinanceAskService,
    FinanceDataPaths,
)
from src.autonomous.runtime import (
    AutonomousRuntime,
    build_autonomous_runtime,
)
from src.autonomous.service_executor import AutonomousServiceExecutor
from src.config.settings import settings
from src.integrations.snowflake_connection import (
    SnowflakeConnectionConfig,
    SnowflakeConnectionFactory,
)
from src.repositories.snowflake_performance_repository import (
    SnowflakePerformanceRepository,
)
from src.repositories.finance_data_repository import (
    FinanceDataRepository,
    LocalCsvFinanceRepository,
    SnowflakeFinanceRepository,
)
from src.repositories.finance_request_repository import (
    SnowflakeFinanceRequestRepository,
)
from src.services.performance_service import PerformanceService
from src.services.finance_request_lifecycle_service import (
    FinanceRequestLifecycleService,
)
from src.rag.embeddings import (
    DeterministicEmbeddingService,
)
from src.rag.pgvector_store import (
    PGVectorConfig,
    PGVectorStore,
)
from src.rag.rag_agent import (
    ExecutionMode,
    FinanceRAGAgent,
    OpenAIResponseGenerator,
    RAGAgentConfig,
)
from src.rag.retriever import FinanceRetriever

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


class DependencyConfigurationError(RuntimeError):
    """
    Raised when an API dependency cannot be configured.
    """


def _get_required_environment_variable(
    name: str,
) -> str:
    """
    Return a required environment-variable value.

    Args:
        name:
            Environment variable name.

    Returns:
        Normalized environment-variable value.

    Raises:
        DependencyConfigurationError:
            If the environment variable is missing or empty.
    """

    value = os.getenv(name)

    if value is None or not value.strip():
        raise DependencyConfigurationError(
            f"Required environment variable is missing: {name}"
        )

    return value.strip()


def _get_optional_environment_variable(
    name: str,
    default: str,
) -> str:
    """
    Return an optional environment variable or its default value.

    Args:
        name:
            Environment variable name.

        default:
            Value returned when the environment variable is missing
            or empty.

    Returns:
        Normalized environment-variable value.
    """

    value = os.getenv(name)

    if value is None or not value.strip():
        return default

    return value.strip()


def _get_environment_path(
    name: str,
    default: str,
) -> Path:
    """
    Return a path from an environment variable or default value.

    Args:
        name:
            Environment variable name.

        default:
            Default path when the environment variable is unavailable.

    Returns:
        Expanded filesystem path.
    """

    value = os.getenv(
        name,
        default,
    )

    return Path(value).expanduser()


def build_data_paths() -> FinanceDataPaths:
    """
    Build paths for local finance datasets.

    Optional environment variables:

    - FINANCE_OPERATIONS_DATA_PATH
    - FINANCE_BUDGET_DATA_PATH
    - FINANCE_ASSUMPTIONS_DATA_PATH
    - FINANCE_CORPORATE_EXPENSES_DATA_PATH
    - FINANCE_BUDGET_CORPORATE_EXPENSES_DATA_PATH

    Returns:
        Configured FinanceDataPaths containing all local datasets required
        by the finance workflows, including P&L analysis.
    """

    return FinanceDataPaths(
        operations=_get_environment_path(
            "FINANCE_OPERATIONS_DATA_PATH",
            "data/operations/sample_orders.csv",
        ),
        budget=_get_environment_path(
            "FINANCE_BUDGET_DATA_PATH",
            "data/planning/sample_budget.csv",
        ),
        assumptions=_get_environment_path(
            "FINANCE_ASSUMPTIONS_DATA_PATH",
            "data/assumptions/business_assumptions.csv",
        ),
        corporate_expenses=_get_environment_path(
            "FINANCE_CORPORATE_EXPENSES_DATA_PATH",
            "data/operations/sample_corporate_expenses.csv",
        ),
        budget_corporate_expenses=_get_environment_path(
            "FINANCE_BUDGET_CORPORATE_EXPENSES_DATA_PATH",
            (
                "data/planning/"
                "sample_budget_corporate_expenses.csv"
            ),
        ),
    )


def build_embedding_service() -> DeterministicEmbeddingService:
    """
    Build the existing deterministic embedding service.

    Returns:
        Configured deterministic embedding service.
    """

    return DeterministicEmbeddingService()


def build_pgvector_store() -> PGVectorStore:
    """
    Build the existing PostgreSQL and pgvector store.

    Required environment variable:

    FINANCE_DATABASE_URL

    Example:

    postgresql://postgres:postgres@localhost:5432/finance_agentic_ai

    Returns:
        Configured PostgreSQL pgvector store.
    """

    database_url = _get_required_environment_variable(
        "FINANCE_DATABASE_URL"
    )

    embedding_service = build_embedding_service()

    config = PGVectorConfig(
        dsn=database_url,
    )

    return PGVectorStore(
        config=config,
        embedding_service=embedding_service,
        initialize=True,
    )


def build_retriever(
    vector_store: PGVectorStore,
) -> FinanceRetriever:
    """
    Build the existing finance retriever.

    Args:
        vector_store:
            PostgreSQL pgvector store used for document retrieval.

    Returns:
        Configured finance retriever.

    Raises:
        TypeError:
            If vector_store is not a PGVectorStore instance.
    """

    if not isinstance(
        vector_store,
        PGVectorStore,
    ):
        raise TypeError(
            "vector_store must be a PGVectorStore."
        )

    return FinanceRetriever(
        vector_store=vector_store,
    )


def build_rag_agent(
    retriever: FinanceRetriever,
) -> FinanceRAGAgent:
    """
    Build the Finance RAG agent in OpenAI LLM mode.

    The OpenAI response generator produces the final business-friendly
    answer. The existing deterministic generator remains available as a
    fallback if the OpenAI generation request fails.

    Args:
        retriever:
            Retriever used to fetch relevant finance documents.

    Returns:
        Configured FinanceRAGAgent using OpenAI response generation.

    Raises:
        TypeError:
            If retriever is not a FinanceRetriever instance.

        DependencyConfigurationError:
            If OPENAI_API_KEY is missing or empty.
    """

    if not isinstance(
        retriever,
        FinanceRetriever,
    ):
        raise TypeError(
            "retriever must be a FinanceRetriever."
        )

    api_key = _get_required_environment_variable(
        "OPENAI_API_KEY"
    )

    model = _get_optional_environment_variable(
        "OPENAI_MODEL",
        DEFAULT_OPENAI_MODEL,
    )

    response_generator = OpenAIResponseGenerator(
        model=model,
        api_key=api_key,
        temperature=0.0,
    )

    config = RAGAgentConfig(
        execution_mode=ExecutionMode.LLM,
        deterministic_fallback=True,
        include_prompt_messages=False,
    )

    return FinanceRAGAgent(
        retriever=retriever,
        config=config,
        response_generator=response_generator,
    )


def build_finance_data_repository() -> FinanceDataRepository:
    """Select local CSV or Snowflake finance data access."""

    source = os.getenv("FINANCE_DATA_SOURCE", "local").strip().lower()
    if source == "local":
        return LocalCsvFinanceRepository(build_data_paths())
    if source == "snowflake":
        config = SnowflakeConnectionConfig.from_settings(settings)
        return SnowflakeFinanceRepository(
            SnowflakeConnectionFactory(config)
        )
    raise DependencyConfigurationError(
        "FINANCE_DATA_SOURCE must be 'local' or 'snowflake'."
    )


def build_autonomous_service_executor(
    *,
    runtime: AutonomousRuntime | None = None,
) -> AutonomousServiceExecutor | None:
    """Build autonomous execution only after explicit environment activation."""

    shadow_explicitly_enabled = (
        os.getenv("AUTONOMOUS_SHADOW_MODE", "").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    if not settings.AUTONOMOUS_ENABLED and not shadow_explicitly_enabled:
        return None
    resolved_runtime = runtime or build_autonomous_runtime(
        app_settings=settings,
        api_key=_get_required_environment_variable("OPENAI_API_KEY"),
    )
    return AutonomousServiceExecutor(resolved_runtime)


@lru_cache(maxsize=1)
def get_finance_service() -> FinanceAskService:
    """
    Build and cache the complete FinanceAskService.

    The same dependency objects are reused across API requests.

    Returns:
        Configured and cached FinanceAskService.
    """

    vector_store = build_pgvector_store()

    retriever = build_retriever(
        vector_store
    )

    rag_agent = build_rag_agent(
        retriever
    )

    data_repository = build_finance_data_repository()

    service_kwargs = {
        "rag_agent": rag_agent,
        "data_repository": data_repository,
    }
    autonomous_executor = build_autonomous_service_executor()
    if autonomous_executor is not None:
        service_kwargs.update(
            {
                "autonomous_executor": autonomous_executor,
                "autonomous_enabled": settings.AUTONOMOUS_ENABLED,
                "autonomous_shadow_mode": (
                    settings.AUTONOMOUS_SHADOW_MODE
                ),
            }
        )
    return FinanceAskService(
        **service_kwargs,
    )


@lru_cache(maxsize=1)
def get_performance_service() -> PerformanceService:
    """Build the read-only Snowflake-backed performance service."""

    config = SnowflakeConnectionConfig.from_settings(settings)
    factory = SnowflakeConnectionFactory(config)
    repository = SnowflakePerformanceRepository(factory)
    return PerformanceService(repository)


@lru_cache(maxsize=1)
def get_finance_request_lifecycle_service() -> FinanceRequestLifecycleService:
    """Build the Snowflake-backed asynchronous request service."""

    config = SnowflakeConnectionConfig.from_settings(settings)
    repository = SnowflakeFinanceRequestRepository(
        SnowflakeConnectionFactory(config)
    )
    return FinanceRequestLifecycleService(
        repository=repository,
        finance_service=get_finance_service(),
    )


def clear_dependency_cache() -> None:
    """
    Clear the cached FinanceAskService.

    Primarily used by tests.
    """

    get_finance_service.cache_clear()
    get_performance_service.cache_clear()
    get_finance_request_lifecycle_service.cache_clear()
