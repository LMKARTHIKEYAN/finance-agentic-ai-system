"""Tests for Product- and Portfolio-Level GP% wrapper output."""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.autonomous.tools.data_tools import FinanceDataContext
from src.autonomous.tools.gp_decomposition_tools import (
    calculate_validated_gp_decomposition,
)


@dataclass
class FakeGpResult:
    budget_gp_percentage: float = 30
    actual_gp_percentage: float = 32
    mix_effect_percentage_points: float = 0.5
    price_effect_percentage_points: float = 1.0
    cost_effect_percentage_points: float = 0.5
    reconciliation_difference: float = 0
    reconciliation_status: str = "passed"
    category_analysis: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {"vehicle_category": "2W", "actual_gp_percentage": 32}
        ]
    )
    available_months: list[str] = field(
        default_factory=lambda: ["2026-04"]
    )
    excluded_actual_categories: list[dict[str, str]] = field(
        default_factory=list
    )
    excluded_budget_categories: list[dict[str, str]] = field(
        default_factory=list
    )


class FakeGpAgent:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def analyze(self, **kwargs: Any) -> FakeGpResult:
        self.kwargs = kwargs
        return FakeGpResult()


def test_gp_wrapper_returns_both_analysis_levels() -> None:
    frame = pd.DataFrame({"value": [1]})
    agent = FakeGpAgent()

    result = calculate_validated_gp_decomposition(
        FinanceDataContext(
            operations_data=frame,
            budget_data=frame,
        ),
        start_month="2026-04",
        end_month="2026-04",
        agent=agent,
    )

    assert result.payload["product_level"][0]["vehicle_category"] == "2W"
    assert result.payload["portfolio_level"]["actual_gp_percentage"] == 32
    assert result.payload["portfolio_level"]["reconciliation_status"] == (
        "passed"
    )
    assert agent.kwargs["start_month"] == "2026-04"
