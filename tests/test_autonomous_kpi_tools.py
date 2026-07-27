"""Tests for the deterministic KPI wrapper."""

from dataclasses import dataclass
from typing import Any

import pandas as pd
import pytest

from src.autonomous.tools.data_tools import (
    FinanceDataContext,
    serialize_tool_payload,
)
from src.autonomous.tools.kpi_tools import calculate_validated_kpis


@dataclass
class FakeKpiResult:
    requested_kpis: list[str]
    selected_kpis: list[dict[str, Any]]
    unavailable_kpis: list[str]
    unknown_kpis: list[str]


class FakeKpiAgent:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def analyze(self, **kwargs: Any) -> FakeKpiResult:
        self.kwargs = kwargs
        return FakeKpiResult(
            requested_kpis=kwargs["requested_kpis"],
            selected_kpis=[{"kpi": "actual_revenue", "value": 100}],
            unavailable_kpis=[],
            unknown_kpis=[],
        )


def test_kpi_wrapper_delegates_and_serializes() -> None:
    agent = FakeKpiAgent()
    context = FinanceDataContext(operations_result=object())

    result = calculate_validated_kpis(
        context,
        ["actual_revenue"],
        dimension="vehicle_category",
        dimension_value="2W",
        agent=agent,
    )

    assert result.payload["selected_kpis"][0]["value"] == 100
    assert agent.kwargs["operations_result"] is context.operations_result
    assert agent.kwargs["dimension_value"] == "2W"


def test_kpi_wrapper_rejects_empty_request() -> None:
    with pytest.raises(ValueError):
        calculate_validated_kpis(
            FinanceDataContext(),
            [],
            agent=FakeKpiAgent(),
        )


def test_tool_payload_rejects_dataframe() -> None:
    with pytest.raises(TypeError, match="DataFrames"):
        serialize_tool_payload(pd.DataFrame({"value": [1]}))


def test_dataset_summary_contains_metadata_not_rows() -> None:
    context = FinanceDataContext(
        operations_data=pd.DataFrame({"secret": [100, 200]})
    )
    summary = context.summarize_dataset("operations_data")

    assert summary.row_count == 2
    assert summary.column_names == ("secret",)
    assert not hasattr(summary, "rows")
