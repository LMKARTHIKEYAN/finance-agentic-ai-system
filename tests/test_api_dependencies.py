"""
Tests for FastAPI dependency construction.

These tests validate dependency wiring without connecting to:

- PostgreSQL
- pgvector
- Docker
- OpenAI
"""

from pathlib import Path

import pytest

from src.api import dependencies
from src.api.dependencies import (
    DependencyConfigurationError,
    build_data_paths,
    build_finance_data_repository,
    clear_dependency_cache,
    get_finance_service,
)
from src.api.service import FinanceDataPaths
from src.repositories.finance_data_repository import (
    LocalCsvFinanceRepository,
    SnowflakeFinanceRepository,
)
from src.integrations.snowflake_connection import SnowflakeConnectionConfig


class FakeFinanceService:
    """Simple fake service used for dependency tests."""


def test_build_data_paths_uses_default_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Default local data paths should be used when environment variables
    are not configured.
    """

    monkeypatch.delenv(
        "FINANCE_OPERATIONS_DATA_PATH",
        raising=False,
    )
    monkeypatch.delenv(
        "FINANCE_BUDGET_DATA_PATH",
        raising=False,
    )
    monkeypatch.delenv(
        "FINANCE_ASSUMPTIONS_DATA_PATH",
        raising=False,
    )

    data_paths = build_data_paths()

    assert isinstance(
        data_paths,
        FinanceDataPaths,
    )

    assert data_paths.operations == Path(
        "data/operations/sample_orders.csv"
    )

    assert data_paths.budget == Path(
        "data/planning/sample_budget.csv"
    )

    assert data_paths.assumptions == Path(
        "data/assumptions/business_assumptions.csv"
    )


def test_build_data_paths_uses_environment_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Environment variables should override the default local paths.
    """

    monkeypatch.setenv(
        "FINANCE_OPERATIONS_DATA_PATH",
        "custom/operations_data.csv",
    )

    monkeypatch.setenv(
        "FINANCE_BUDGET_DATA_PATH",
        "custom/budget_data.csv",
    )

    monkeypatch.setenv(
        "FINANCE_ASSUMPTIONS_DATA_PATH",
        "custom/assumptions_data.csv",
    )

    data_paths = build_data_paths()

    assert data_paths.operations == Path(
        "custom/operations_data.csv"
    )

    assert data_paths.budget == Path(
        "custom/budget_data.csv"
    )

    assert data_paths.assumptions == Path(
        "custom/assumptions_data.csv"
    )


def test_repository_defaults_to_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FINANCE_DATA_SOURCE", raising=False)

    repository = build_finance_data_repository()

    assert isinstance(repository, LocalCsvFinanceRepository)


def test_repository_selects_snowflake(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FINANCE_DATA_SOURCE", "snowflake")
    config = SnowflakeConnectionConfig(
        account="test-account",
        user="test-user",
        password="test-password",
    )
    monkeypatch.setattr(
        dependencies.SnowflakeConnectionConfig,
        "from_settings",
        lambda settings: config,
    )

    repository = build_finance_data_repository()

    assert isinstance(repository, SnowflakeFinanceRepository)


def test_repository_rejects_invalid_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FINANCE_DATA_SOURCE", "unknown")

    with pytest.raises(
        DependencyConfigurationError,
        match="FINANCE_DATA_SOURCE",
    ):
        build_finance_data_repository()


def test_required_environment_variable_raises_error_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Missing required environment variables should raise a clear
    configuration error.
    """

    monkeypatch.delenv(
        "FINANCE_DATABASE_URL",
        raising=False,
    )

    with pytest.raises(
        DependencyConfigurationError,
        match=(
            "Required environment variable is missing: "
            "FINANCE_DATABASE_URL"
        ),
    ):
        dependencies._get_required_environment_variable(
            "FINANCE_DATABASE_URL"
        )


def test_required_environment_variable_returns_cleaned_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Required environment-variable values should be stripped of
    surrounding whitespace.
    """

    monkeypatch.setenv(
        "FINANCE_DATABASE_URL",
        "  postgresql://localhost/finance_ai  ",
    )

    value = (
        dependencies._get_required_environment_variable(
            "FINANCE_DATABASE_URL"
        )
    )

    assert value == (
        "postgresql://localhost/finance_ai"
    )


def test_get_finance_service_is_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    get_finance_service should build the service once and reuse the
    cached instance.
    """

    clear_dependency_cache()

    fake_vector_store = object()
    fake_retriever = object()
    fake_rag_agent = object()
    fake_repository = object()
    fake_service = FakeFinanceService()

    call_counts = {
        "vector_store": 0,
        "retriever": 0,
        "rag_agent": 0,
        "repository": 0,
        "service": 0,
    }

    def fake_build_pgvector_store():
        call_counts["vector_store"] += 1
        return fake_vector_store

    def fake_build_retriever(
        vector_store,
    ):
        call_counts["retriever"] += 1
        assert vector_store is fake_vector_store
        return fake_retriever

    def fake_build_rag_agent(
        retriever,
    ):
        call_counts["rag_agent"] += 1
        assert retriever is fake_retriever
        return fake_rag_agent

    def fake_build_repository():
        call_counts["repository"] += 1
        return fake_repository

    def fake_finance_ask_service(
        *,
        rag_agent,
        data_repository,
    ):
        call_counts["service"] += 1
        assert rag_agent is fake_rag_agent
        assert data_repository is fake_repository
        return fake_service

    monkeypatch.setattr(
        dependencies,
        "build_pgvector_store",
        fake_build_pgvector_store,
    )

    monkeypatch.setattr(
        dependencies,
        "build_retriever",
        fake_build_retriever,
    )

    monkeypatch.setattr(
        dependencies,
        "build_rag_agent",
        fake_build_rag_agent,
    )

    monkeypatch.setattr(
        dependencies,
        "build_finance_data_repository",
        fake_build_repository,
    )

    monkeypatch.setattr(
        dependencies,
        "FinanceAskService",
        fake_finance_ask_service,
    )

    first_service = get_finance_service()
    second_service = get_finance_service()

    assert first_service is fake_service
    assert second_service is fake_service
    assert first_service is second_service

    assert call_counts == {
        "vector_store": 1,
        "retriever": 1,
        "rag_agent": 1,
        "repository": 1,
        "service": 1,
    }

    clear_dependency_cache()


def test_clear_dependency_cache_forces_rebuild(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Clearing the cache should force get_finance_service to build a new
    service instance.
    """

    clear_dependency_cache()

    created_services: list[object] = []

    monkeypatch.setattr(
        dependencies,
        "build_pgvector_store",
        lambda: object(),
    )

    monkeypatch.setattr(
        dependencies,
        "build_retriever",
        lambda vector_store: object(),
    )

    monkeypatch.setattr(
        dependencies,
        "build_rag_agent",
        lambda retriever: object(),
    )

    monkeypatch.setattr(
        dependencies,
        "build_finance_data_repository",
        lambda: object(),
    )

    def fake_finance_ask_service(
        *,
        rag_agent,
        data_repository,
    ):
        service = object()
        created_services.append(service)
        return service

    monkeypatch.setattr(
        dependencies,
        "FinanceAskService",
        fake_finance_ask_service,
    )

    first_service = get_finance_service()

    clear_dependency_cache()

    second_service = get_finance_service()

    assert first_service is not second_service
    assert len(created_services) == 2

    clear_dependency_cache()
