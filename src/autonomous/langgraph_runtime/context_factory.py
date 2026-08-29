"""Build trusted, non-LLM finance dependencies for one complex goal."""

from __future__ import annotations

from calendar import monthrange
from datetime import date, timedelta
import re
from typing import Any

from src.autonomous.goal_builder import GoalBuilder
from src.autonomous.tools.snowflake_tools import load_finance_data


MONTHS = {
    name: number for number, name in enumerate(
        ("january", "february", "march", "april", "may", "june",
         "july", "august", "september", "october", "november", "december"),
        1,
    )
}


class ComplexFinanceContextFactory:
    """Load the widest explicitly requested period into trusted context."""

    def __init__(self, *, repository: Any, retriever: Any = None, goal_builder: GoalBuilder | None = None) -> None:
        self.repository = repository
        self.retriever = retriever
        self.goal_builder = goal_builder or GoalBuilder()

    def __call__(self, request: str) -> dict[str, Any]:
        goal = self.goal_builder.build(request)
        if goal.ambiguities:
            raise ValueError(goal.clarification_question or "The reporting scope is unclear.")
        scope = goal.reporting_scope
        if scope.start_date is None or scope.end_date is None:
            raise ValueError("Please provide the reporting period.")

        start_date, end_date = _expanded_period(
            request, scope.start_date, scope.end_date
        )
        loaded = load_finance_data(
            repository=self.repository,
            start_date=start_date,
            end_date=end_date,
            category=scope.category,
        )
        return {
            "finance_context": loaded["finance_context"],
            "retriever": self.retriever,
            "goal_id": goal.goal_id,
            "reporting_scope": goal.reporting_scope.model_dump(mode="json"),
            "loaded_period": loaded["period"],
            "revenue_definition": "commission_amount",
        }


def _expanded_period(request: str, default_start: date, default_end: date) -> tuple[date, date]:
    matches = re.findall(
        r"\b(" + "|".join(MONTHS) + r")\s+(20\d{2})\b",
        request.casefold(),
    )
    periods = [
        (date(int(year), MONTHS[name], 1),
         date(int(year), MONTHS[name], monthrange(int(year), MONTHS[name])[1]))
        for name, year in matches
    ]
    if len(periods) >= 2:
        return min(item[0] for item in periods), max(item[1] for item in periods)
    normalized = request.casefold()
    if any(word in normalized for word in ("compare", "versus", " vs ")):
        previous_end = default_start - timedelta(days=1)
        return previous_end.replace(day=1), default_end
    return default_start, default_end
