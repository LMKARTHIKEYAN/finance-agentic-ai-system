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
from src.rag.rag_agent import FinanceRAGAgent
from src.rag.retriever import FinanceRetriever


class DependencyConfigurationError(RuntimeError):
    """
    Raised when an API dependency cannot be configured.
    """


def _get_required_environment_variable(
    name: str,
) -> str:
    """
    Return a required environment-variable value.
    """

    value = os.getenv(name)

    if value is None or not value.strip():
        raise DependencyConfigurationError(
            f"Required environment variable is missing: {name}"
        )

    return value.strip()


def _get_environment_path(
    name: str,
    default: str,
) -> Path:
    """
    Return a path from an environment variable or default value.
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
    """

    return DeterministicEmbeddingService()


def build_pgvector_store() -> PGVectorStore:
    """
    Build the existing PostgreSQL + pgvector store.

    Required environment variable:

    FINANCE_DATABASE_URL

    Example:

    postgresql://postgres:postgres@localhost:5432/finance_ai
    """

    database_url = (
        _get_required_environment_variable(
            "FINANCE_DATABASE_URL"
        )
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
    Build the existing Finance RAG agent.
    """

    if not isinstance(
        retriever,
        FinanceRetriever,
    ):
        raise TypeError(
            "retriever must be a FinanceRetriever."
        )

    return FinanceRAGAgent(
        retriever=retriever,
    )


@lru_cache(maxsize=1)
def get_finance_service() -> FinanceAskService:
    """
    Build and cache the complete FinanceAskService.

    The same dependency objects are reused across API requests.
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