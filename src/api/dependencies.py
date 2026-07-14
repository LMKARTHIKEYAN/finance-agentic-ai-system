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

from src.api.service import (
    FinanceAskService,
    FinanceDataPaths,
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


load_dotenv()

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

    data_paths = build_data_paths()

    return FinanceAskService(
        rag_agent=rag_agent,
        data_paths=data_paths,
    )


def clear_dependency_cache() -> None:
    """
    Clear the cached FinanceAskService.

    Primarily used by tests.
    """

    get_finance_service.cache_clear()