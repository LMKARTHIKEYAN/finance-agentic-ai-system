"""Tests for the deterministic revenue-variance wrapper."""

from dataclasses import dataclass
from typing import Any

import pytest

from src.autonomous.tools.data_tools import FinanceDataContext
from src.autonomous.tools.revenue_variance_tools import (
    calculate_validated_revenue_variance,
)


@dataclass
class FakeVarianceResult:
    actual_revenue: float = 110
    budget_revenue: float = 100
    revenue_variance: float = 10
    price_effect: float = 6
    volume_effect: float = 4
    variance_check: float = 0


class FakeVarianceAgent:
    def __init__(self) -> None:
        self.actual: Any = None
        self.budget: Any = None

    def analyze(
        self,
        actual_result: Any,
        budget_result: Any,
    ) -> FakeVarianceResult:
        self.actual = actual_result
        self.budget = budget_result
        return FakeVarianceResult()


def test_revenue_variance_wrapper_delegates_and_preserves_result() -> None:
    actual = object()
    budget = object()
    context = FinanceDataContext(
        operations_result=actual,
        budget_result=budget,
    )
    agent = FakeVarianceAgent()

    result = calculate_validated_revenue_variance(
        context,
        agent=agent,
    )

    assert result.payload["revenue_variance"] == 10
    assert result.payload["variance_check"] == 0
    assert "variance_pnl" not in result.payload
    assert agent.actual is actual
    assert agent.budget is budget


def test_revenue_variance_requires_existing_results() -> None:
    with pytest.raises(ValueError, match="operations_result"):
        calculate_validated_revenue_variance(
            FinanceDataContext(),
            agent=FakeVarianceAgent(),
        )
