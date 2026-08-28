"""Shared deterministic preparation for autonomous analysis tools."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from src.autonomous.tools.data_tools import FinanceDataContext


DIRECT_COST_COLUMNS = ("incentive", "goodwill", "dry_run", "surge")


def prepared_orders(context: FinanceDataContext) -> pd.DataFrame:
    frame = context.require_dataframe("operations_data").copy()
    frame["order_date"] = pd.to_datetime(frame["order_date"], errors="coerce")
    frame = frame.dropna(subset=["order_date"])
    status = frame.get("order_status", pd.Series("", index=frame.index)).astype(str).str.lower()
    frame["is_completed"] = status.eq("completed").astype(int)
    frame["is_cancelled"] = status.eq("cancelled").astype(int)
    commission = numeric(frame, "commission_amount")
    frame["revenue"] = commission.where(status.eq("completed"), 0.0)
    direct_cost = pd.Series(0.0, index=frame.index)
    for column in DIRECT_COST_COLUMNS:
        direct_cost = direct_cost.add(numeric(frame, column), fill_value=0.0)
    frame["direct_cost"] = direct_cost.where(status.eq("completed"), 0.0)
    frame["gross_profit"] = frame["revenue"] - frame["direct_cost"]
    return frame


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def aggregate(frame: pd.DataFrame, dimensions: Iterable[str]) -> pd.DataFrame:
    dims = list(dimensions)
    grouped = frame.groupby(dims, dropna=False) if dims else [((), frame)]
    rows = []
    for key, group in grouped:
        keys = key if isinstance(key, tuple) else (key,)
        row = {name: str(value) for name, value in zip(dims, keys)}
        orders = len(group)
        completed = int(group["is_completed"].sum())
        cancelled = int(group["is_cancelled"].sum())
        revenue = float(group["revenue"].sum())
        cost = float(group["direct_cost"].sum())
        gp = revenue - cost
        row.update({
            "total_orders": orders,
            "completed_orders": completed,
            "cancelled_orders": cancelled,
            "revenue": round(revenue, 2),
            "aov": round(revenue / completed if completed else 0.0, 2),
            "fulfillment_percentage": round(completed / orders * 100 if orders else 0.0, 2),
            "cancellation_percentage": round(cancelled / orders * 100 if orders else 0.0, 2),
            "direct_cost": round(cost, 2),
            "gross_profit": round(gp, 2),
            "gp_percentage": round(gp / revenue * 100 if revenue else 0.0, 2),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def safe_records(frame: pd.DataFrame) -> list[dict]:
    output = frame.copy()
    for column in output.columns:
        if pd.api.types.is_datetime64_any_dtype(output[column]):
            output[column] = output[column].dt.strftime("%Y-%m-%d")
    return output.where(pd.notna(output), None).to_dict(orient="records")
