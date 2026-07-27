"""Safe internal data context and serialization helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class DatasetSummary:
    """LLM-safe metadata for one internally held dataset."""

    dataset_name: str
    available: bool
    row_count: int
    column_names: tuple[str, ...]


@dataclass
class FinanceDataContext:
    """Internal inputs available to approved deterministic tools."""

    operations_data: pd.DataFrame | None = None
    budget_data: pd.DataFrame | None = None
    corporate_expenses_data: pd.DataFrame | None = None
    budget_corporate_expenses_data: pd.DataFrame | None = None
    business_assumptions: Any = None
    operations_result: Any = None
    budget_result: Any = None
    revenue_variance_result: Any = None
    gp_portfolio_result: Any = None
    forecast_result: Any = None
    scenario_result: Any = None
    finance_rules_result: Any = None

    def require_dataframe(
        self,
        field_name: str,
    ) -> pd.DataFrame:
        """Return a non-empty DataFrame without copying it into a payload."""

        if field_name not in {
            item.name for item in fields(self)
        }:
            raise ValueError(f"Unknown data-context field: {field_name!r}.")

        value = getattr(self, field_name)

        if not isinstance(value, pd.DataFrame):
            raise ValueError(
                f"Required DataFrame {field_name!r} is unavailable."
            )
        if value.empty:
            raise ValueError(
                f"Required DataFrame {field_name!r} is empty."
            )
        return value

    def require_result(self, field_name: str) -> Any:
        """Return a required previously calculated result."""

        if field_name not in {
            item.name for item in fields(self)
        }:
            raise ValueError(f"Unknown data-context field: {field_name!r}.")

        value = getattr(self, field_name)
        if value is None:
            raise ValueError(
                f"Required result {field_name!r} is unavailable."
            )
        return value

    def summarize_dataset(self, field_name: str) -> DatasetSummary:
        """Return metadata only; never include DataFrame rows."""

        value = getattr(self, field_name, None)
        if not isinstance(value, pd.DataFrame):
            return DatasetSummary(field_name, False, 0, ())

        return DatasetSummary(
            dataset_name=field_name,
            available=not value.empty,
            row_count=len(value),
            column_names=tuple(str(item) for item in value.columns),
        )


def serialize_tool_payload(value: Any) -> Any:
    """Convert deterministic results to JSON-safe values.

    DataFrames are deliberately rejected so an approved tool cannot
    accidentally expose row-level data to an LLM-facing payload.
    """

    if isinstance(value, pd.DataFrame):
        raise TypeError("DataFrames cannot be included in tool payloads.")

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if is_dataclass(value):
        return serialize_tool_payload(asdict(value))

    if isinstance(value, dict):
        return {
            str(key): serialize_tool_payload(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [serialize_tool_payload(item) for item in value]

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return serialize_tool_payload(to_dict())

    if hasattr(value, "__dict__"):
        return {
            key: serialize_tool_payload(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }

    return str(value)
