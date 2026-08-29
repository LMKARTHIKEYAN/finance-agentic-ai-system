import pandas as pd
import pytest

from src.autonomous.langgraph_runtime.tool_argument_resolver import (
    ToolArgumentResolutionError,
    ToolArgumentResolver,
)
from src.autonomous.tools.data_tools import FinanceDataContext


@pytest.fixture
def context():
    return FinanceDataContext(operations_data=pd.DataFrame({
        "order_date": ["2026-06-30", "2026-07-01", "2026-07-02"],
        "vehicle_category": ["2W", "Tata Ace", "2W"],
        "commission_amount": [1.0, 2.0, 3.0],
    }))


def test_injects_context_and_filters_period_and_category(context):
    result = ToolArgumentResolver().resolve(
        "analyze_drilldown",
        {"start_date": "01-07-2026", "end_date": "2026-07-31", "category": "tata ace", "dimension": "pickup_cluster"},
        {"finance_context": context},
    )
    assert result["dimension"] == "pickup_cluster"
    assert len(result["finance_context"].operations_data) == 1
    assert len(context.operations_data) == 3


def test_rejects_unknown_tool(context):
    with pytest.raises(ToolArgumentResolutionError, match="unapproved"):
        ToolArgumentResolver().resolve("delete_database", {}, {"finance_context": context})


def test_rejects_unknown_argument(context):
    with pytest.raises(ToolArgumentResolutionError, match="Unsupported"):
        ToolArgumentResolver().resolve("analyze_drilldown", {"shell_command": "x"}, {"finance_context": context})


def test_rejects_llm_override_of_trusted_context(context):
    with pytest.raises(ToolArgumentResolutionError, match="cannot supply trusted"):
        ToolArgumentResolver().resolve("analyze_drilldown", {"finance_context": "fake", "dimension": "weekday"}, {"finance_context": context})


def test_missing_context_is_rejected():
    with pytest.raises(ToolArgumentResolutionError, match="finance_context"):
        ToolArgumentResolver().resolve("analyze_drilldown", {"dimension": "weekday"}, {})


def test_date_arguments_are_normalized(context):
    result = ToolArgumentResolver().resolve(
        "compare_periods",
        {"current_start": "01-07-2026", "current_end": "31-07-2026", "comparison_start": "01-06-2026", "comparison_end": "30-06-2026"},
        {"finance_context": context},
    )
    assert result["current_start"] == "2026-07-01"
    assert result["comparison_end"] == "2026-06-30"


def test_retriever_is_injected_and_query_comes_from_request():
    retriever = object()
    result = ToolArgumentResolver().resolve(
        "retrieve_company_context", {"top_k": 2}, {"retriever": retriever}, request="Why did revenue decline?"
    )
    assert result["retriever"] is retriever
    assert result["query"] == "Why did revenue decline?"
