"""Resolve LLM business arguments into trusted Python-tool arguments."""

from __future__ import annotations

from dataclasses import fields, replace
from datetime import date, datetime
import inspect
import re
from typing import Any, Mapping

import pandas as pd

from src.autonomous.tools.data_tools import FinanceDataContext
from src.autonomous.tools.registry import DEFAULT_TOOL_REGISTRY, ToolRegistry


class ToolArgumentResolutionError(ValueError):
    """Raised when a proposed tool call cannot be resolved safely."""


TRUSTED_ARGUMENTS = {
    "finance_context", "context", "operations_data", "budget_data",
    "corporate_expenses_data", "budget_corporate_expenses_data",
    "operations_result", "budget_result", "root_cause_result", "retriever",
    "repository", "connection", "cursor", "sql", "query_text",
}
BUSINESS_FILTERS = {"start_date", "end_date", "category"}
DATE_ARGUMENTS = {
    "start_date", "end_date", "current_start", "current_end",
    "comparison_start", "comparison_end", "target_date",
}


class ToolArgumentResolver:
    """Inject trusted dependencies while keeping them outside LLM control."""

    def __init__(self, registry: ToolRegistry = DEFAULT_TOOL_REGISTRY) -> None:
        self.registry = registry

    def resolve(
        self,
        tool_name: str,
        llm_arguments: Mapping[str, Any] | None,
        trusted_context: Mapping[str, Any],
        *,
        request: str = "",
    ) -> dict[str, Any]:
        try:
            definition = self.registry.get(tool_name)
        except KeyError as exc:
            raise ToolArgumentResolutionError(
                f"Unknown or unapproved tool: {tool_name!r}."
            ) from exc

        proposed = dict(llm_arguments or {})
        forbidden = set(proposed) & TRUSTED_ARGUMENTS
        if forbidden:
            raise ToolArgumentResolutionError(
                "The LLM cannot supply trusted arguments: "
                + ", ".join(sorted(forbidden))
                + "."
            )

        signature = inspect.signature(definition.function)
        accepted = {
            name for name, parameter in signature.parameters.items()
            if name not in TRUSTED_ARGUMENTS
            and name not in {"agent"}
            and parameter.kind not in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }
        }
        unknown = set(proposed) - accepted - BUSINESS_FILTERS
        if unknown:
            raise ToolArgumentResolutionError(
                "Unsupported LLM arguments for " + tool_name + ": "
                + ", ".join(sorted(unknown)) + "."
            )

        finance_context = trusted_context.get("finance_context")
        if isinstance(finance_context, FinanceDataContext):
            finance_context = _filtered_context(
                finance_context,
                start_date=proposed.pop("start_date", None),
                end_date=proposed.pop("end_date", None),
                category=proposed.pop("category", None),
            )

        resolved = {
            key: _normalize_date(key, value)
            for key, value in proposed.items()
            if key in accepted
        }
        parameter_names = set(signature.parameters)
        if "finance_context" in parameter_names:
            resolved["finance_context"] = _require_context(finance_context)
        if "context" in parameter_names:
            resolved["context"] = _require_context(finance_context)

        for name in (
            "operations_data", "budget_data", "corporate_expenses_data",
            "budget_corporate_expenses_data", "operations_result",
            "budget_result", "root_cause_result", "actuals", "forecasts",
        ):
            if name not in parameter_names or name in resolved:
                continue
            value = trusted_context.get(name)
            if value is None and isinstance(finance_context, FinanceDataContext):
                context_names = {item.name for item in fields(finance_context)}
                if name in context_names:
                    value = getattr(finance_context, name)
            if value is not None:
                resolved[name] = value

        if "retriever" in parameter_names:
            retriever = trusted_context.get("retriever")
            if retriever is None:
                raise ToolArgumentResolutionError("Trusted retriever is unavailable.")
            resolved["retriever"] = retriever
        if "query" in parameter_names and "query" not in resolved:
            resolved["query"] = request
        if "requested_kpis" in parameter_names and "requested_kpis" not in resolved:
            resolved["requested_kpis"] = list(
                trusted_context.get("default_requested_kpis", [
                    "total orders", "completed orders", "cancelled orders",
                    "actual revenue", "actual aov", "fulfillment", "cancellation",
                ])
            )

        missing = [
            name for name, parameter in signature.parameters.items()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind not in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }
            and name not in resolved
            and name != "agent"
        ]
        if missing:
            raise ToolArgumentResolutionError(
                "Trusted inputs are unavailable for " + tool_name + ": "
                + ", ".join(sorted(missing)) + "."
            )
        return resolved


def _require_context(value: Any) -> FinanceDataContext:
    if not isinstance(value, FinanceDataContext):
        raise ToolArgumentResolutionError("Trusted finance_context is unavailable.")
    return value


def _normalize_date(name: str, value: Any) -> Any:
    if name not in DATE_ARGUMENTS or value is None:
        return value
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    try:
        text = str(value).strip()
        if re.fullmatch(r"\d{1,2}-\d{1,2}-\d{4}", text):
            return datetime.strptime(text, "%d-%m-%Y").date().isoformat()
        return pd.Timestamp(text).date().isoformat()
    except (TypeError, ValueError) as exc:
        raise ToolArgumentResolutionError(
            f"Invalid date supplied for {name}: {value!r}."
        ) from exc


def _filtered_context(
    context: FinanceDataContext,
    *,
    start_date: Any = None,
    end_date: Any = None,
    category: Any = None,
) -> FinanceDataContext:
    if start_date is None and end_date is None and category is None:
        return context
    start = pd.Timestamp(_normalize_date("start_date", start_date)) if start_date is not None else None
    end = pd.Timestamp(_normalize_date("end_date", end_date)) if end_date is not None else None
    updates: dict[str, Any] = {}
    for name in (
        "operations_data", "budget_data", "corporate_expenses_data",
        "budget_corporate_expenses_data",
    ):
        frame = getattr(context, name)
        if not isinstance(frame, pd.DataFrame):
            continue
        filtered = frame.copy()
        date_column = next((item for item in ("order_date", "date", "month") if item in filtered.columns), None)
        if date_column and (start is not None or end is not None):
            values = pd.to_datetime(filtered[date_column], errors="coerce")
            if start is not None:
                filtered = filtered.loc[values >= start]
                values = values.loc[filtered.index]
            if end is not None:
                filtered = filtered.loc[values <= end + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)]
        if category is not None and "vehicle_category" in filtered.columns:
            filtered = filtered.loc[
                filtered["vehicle_category"].astype(str).str.casefold()
                == str(category).strip().casefold()
            ]
        updates[name] = filtered.reset_index(drop=True)
    return replace(context, **updates)
