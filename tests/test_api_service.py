"""
Tests for the Finance API service layer.

These tests use fake dependencies.

They do not require:

- PostgreSQL
- pgvector
- Docker
- OpenAI
- Real CSV files
"""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.api.service import (
    AskServiceResult,
    FinanceAskService,
    FinanceAskServiceError,
    FinanceDataPaths,
)


class FakeDocument:
    """Simple fake retrieved document."""

    def __init__(
        self,
        document_id: str,
        text: str,
        metadata: dict,
    ) -> None:
        self.id = document_id
        self.text = text
        self.metadata = metadata


class FakeRetrievedItem:
    """Simple fake similarity-search result."""

    def __init__(
        self,
        document: FakeDocument,
        score: float,
        rank: int,
    ) -> None:
        self.document = document
        self.score = score
        self.rank = rank


class FakeRAGAgent:
    """Fake RAG agent used only for service tests."""

    def run(
        self,
        *,
        user_request: str,
        finance_analysis: dict,
        retrieval_query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        metadata_filter: dict | None = None,
    ):
        document = FakeDocument(
            document_id="chunk_001",
            text=(
                "Revenue variance was caused by "
                "price and volume movements."
            ),
            metadata={
                "filename": "finance_policy.pdf",
            },
        )

        retrieved_item = FakeRetrievedItem(
            document=document,
            score=0.95,
            rank=1,
        )

        retrieval_result = SimpleNamespace(
            documents=[retrieved_item],
        )

        return SimpleNamespace(
            response=(
                "Revenue variance was mainly driven "
                "by price and volume."
            ),
            retrieval_result=retrieval_result,
            used_fallback=False,
        )


def fake_graph_executor(state: dict) -> dict:
    """Return a completed fake LangGraph state."""

    return {
        **state,
        "selected_flow": "variance",
        "execution_status": "completed",
        "variance_result": {
            "revenue_variance": 125000,
        },
    }


@pytest.fixture
def data_paths(
    tmp_path: Path,
) -> FinanceDataPaths:
    """
    Create temporary CSV files required by the service.
    """

    operations_path = tmp_path / "operations.csv"
    budget_path = tmp_path / "budget.csv"
    assumptions_path = tmp_path / "assumptions.csv"

    pd.DataFrame(
        {
            "month": ["2026-01"],
            "revenue": [100000],
        }
    ).to_csv(
        operations_path,
        index=False,
    )

    pd.DataFrame(
        {
            "month": ["2026-01"],
            "budget_revenue": [90000],
        }
    ).to_csv(
        budget_path,
        index=False,
    )

    pd.DataFrame(
        {
            "assumption": ["growth_rate"],
            "value": [0.10],
        }
    ).to_csv(
        assumptions_path,
        index=False,
    )

    return FinanceDataPaths(
        operations=operations_path,
        budget=budget_path,
        assumptions=assumptions_path,
    )


def test_service_returns_structured_result(
    data_paths: FinanceDataPaths,
) -> None:
    """
    Service should return the final answer,
    source information and execution details.
    """

    service = FinanceAskService(
        rag_agent=FakeRAGAgent(),
        data_paths=data_paths,
        graph_executor=fake_graph_executor,
    )

    result = service.ask(
        "What is the revenue variance?",
        top_k=5,
    )

    assert isinstance(
        result,
        AskServiceResult,
    )

    assert result.answer == (
        "Revenue variance was mainly driven "
        "by price and volume."
    )

    assert result.selected_flow == "variance"
    assert result.execution_status == "completed"
    assert result.used_fallback is False

    assert len(result.sources) == 1
    assert result.sources[0]["id"] == "chunk_001"
    assert result.sources[0]["rank"] == 1
    assert result.sources[0]["score"] == 0.95


def test_service_rejects_empty_question(
    data_paths: FinanceDataPaths,
) -> None:
    """
    Empty questions should be rejected before
    graph and RAG execution.
    """

    service = FinanceAskService(
        rag_agent=FakeRAGAgent(),
        data_paths=data_paths,
        graph_executor=fake_graph_executor,
    )

    with pytest.raises(
        ValueError,
        match="question cannot be empty",
    ):
        service.ask("   ")


def test_service_raises_when_graph_fails(
    data_paths: FinanceDataPaths,
) -> None:
    """
    Graph execution exceptions should become
    FinanceAskServiceError.
    """

    def failing_graph_executor(
        state: dict,
    ) -> dict:
        raise RuntimeError(
            "Graph unavailable"
        )

    service = FinanceAskService(
        rag_agent=FakeRAGAgent(),
        data_paths=data_paths,
        graph_executor=failing_graph_executor,
    )

    with pytest.raises(
        FinanceAskServiceError,
        match="Finance graph execution failed",
    ):
        service.ask(
            "What is the revenue variance?"
        )


def test_service_raises_for_missing_csv(
    tmp_path: Path,
) -> None:
    """
    Missing required local data should produce
    a clear service error.
    """

    missing_paths = FinanceDataPaths(
        operations=tmp_path / "missing_operations.csv",
        budget=tmp_path / "missing_budget.csv",
        assumptions=tmp_path / "missing_assumptions.csv",
    )

    service = FinanceAskService(
        rag_agent=FakeRAGAgent(),
        data_paths=missing_paths,
        graph_executor=fake_graph_executor,
    )

    with pytest.raises(
        FinanceAskServiceError,
        match="Required data file not found",
    ):
        service.ask(
            "What is the revenue variance?"
        )


def test_excerpt_shortens_long_source_text() -> None:
    """
    Long retrieved source text should be shortened
    for the API response.
    """

    long_text = "A" * 500

    excerpt = FinanceAskService._excerpt(
        long_text,
        limit=100,
    )

    assert len(excerpt) == 100
    assert excerpt.endswith("...")