"""Tests for the deterministic P&L wrapper."""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import pytest

from src.autonomous.tools.data_tools import FinanceDataContext
from src.autonomous.tools.pnl_tools import (
    generate_validated_pnl_analysis,
)


@dataclass
class FakePnlResult:
    actual_pnl: list[dict[str, Any]] = field(
        default_factory=lambda: [{"net_profit": 75}]
    )
    budget_pnl: list[dict[str, Any]] = field(
        default_factory=lambda: [{"net_profit": 70}]
    )
    variance_pnl: list[dict[str, Any]] = field(
        default_factory=lambda: [{"net_profit_variance": 5}]
    )
    summary: dict[str, Any] = field(
        default_factory=lambda: {"actual_net_profit": 75}
    )
    available_months: list[str] = field(
        default_factory=lambda: ["2026-04"]
    )
    excluded_actual_months: list[str] = field(default_factory=list)
    excluded_budget_months: list[str] = field(default_factory=list)


class FakePnlAgent:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def analyze(self, **kwargs: Any) -> FakePnlResult:
        self.kwargs = kwargs
        return FakePnlResult()


def _context() -> FinanceDataContext:
    frame = pd.DataFrame({"value": [1]})
    return FinanceDataContext(
        operations_data=frame,
        budget_data=frame,
        corporate_expenses_data=frame,
        budget_corporate_expenses_data=frame,
    )


def test_pnl_wrapper_preserves_actual_budget_and_variance() -> None:
    agent = FakePnlAgent()

    result = generate_validated_pnl_analysis(
        _context(),
        start_month="2026-04",
        end_month="2026-04",
        agent=agent,
    )

    assert result.payload["actual_pnl"][0]["net_profit"] == 75
    assert result.payload["budget_pnl"][0]["net_profit"] == 70
    assert result.payload["variance_pnl"][0]["net_profit_variance"] == 5
    assert result.payload["pnl_summary"]["actual_net_profit"] == 75
    assert agent.kwargs["start_month"] == "2026-04"


def test_pnl_wrapper_requires_all_dataframes() -> None:
    with pytest.raises(ValueError, match="corporate_expenses_data"):
        generate_validated_pnl_analysis(
            FinanceDataContext(
                operations_data=pd.DataFrame({"value": [1]}),
                budget_data=pd.DataFrame({"value": [1]}),
            ),
            agent=FakePnlAgent(),
        )


def test_pnl_wrapper_does_not_return_dataframe() -> None:
    result = generate_validated_pnl_analysis(
        _context(),
        agent=FakePnlAgent(),
    )

    assert not any(
        isinstance(value, pd.DataFrame)
        for value in result.payload.values()
    )
