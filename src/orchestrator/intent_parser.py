"""
Deterministic intent parser for Finance Agentic AI requests.

This module converts natural-language finance requests into structured
execution information for the existing router, planner and LangGraph flow.

Examples:

    "Show Actual vs Budget for January 2026"

becomes:

    selected_flow = "variance"
    comparison = "actual_vs_budget"
    start_date = "2026-01-01"
    end_date = "2026-01-31"

The parser also supports temporary slot filling. If required information is
missing, it returns a clarification question instead of guessing.

Reporting-period behaviour:

- A reporting period is optional.
- When no period is supplied, all available data may be analyzed.
- When the user supplies an invalid date or reversed range, clarification
  is required.

This module must not:

- Execute finance agents
- Run LangGraph
- Load or filter DataFrames
- Call OpenAI
- Store long-term conversation memory
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

from src.orchestrator.router import identify_flow
from src.orchestrator.state import FlowType


ComparisonType = Literal[
    "actual_vs_budget",
    "actual_vs_forecast",
    "actual_vs_last_year",
    "budget_vs_forecast",
    "none",
]

PeriodGranularity = Literal[
    "day",
    "month",
    "year",
    "range",
    "unknown",
]


MONTH_NAMES: dict[str, int] = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}


CATEGORY_ALIASES: dict[str, str] = {
    "2w": "2W",
    "two wheeler": "2W",
    "two-wheeler": "2W",
    "3w": "3W",
    "three wheeler": "3W",
    "three-wheeler": "3W",
    "tata ace open": "Tata Ace Open",
    "tata ace closed": "Tata Ace Closed",
    "tata ace close": "Tata Ace Closed",
    "tata ace": "Tata Ace",
    "packer and movers": "Packer & Movers",
    "packers and movers": "Packer & Movers",
    "packer & movers": "Packer & Movers",
    "compact auto": "Compact Auto",
}


KPI_ALIASES: dict[str, str] = {
    "average order value": "average_order_value",
    "fulfillment rate": "fulfillment_rate",
    "fulfilment rate": "fulfillment_rate",
    "gross margin percentage": "gross_margin_percentage",
    "margin percentage": "margin_percentage",
    "gross profit": "gross_profit",
    "gp percentage": "gross_margin_percentage",
    "gp%": "gross_margin_percentage",
    "revenue": "revenue",
    "orders": "orders",
    "order": "orders",
    "aov": "average_order_value",
    "fulfillment": "fulfillment_rate",
    "fulfilment": "fulfillment_rate",
    "cost": "cost",
    "profit": "profit",
    "margin": "margin_percentage",
}


@dataclass(frozen=True)
class ParsedPeriod:
    """
    Structured reporting period extracted from a request.

    Attributes:
        start_date:
            Inclusive reporting-period start date in ISO format.

        end_date:
            Inclusive reporting-period end date in ISO format.

        display_value:
            Human-readable period shown to the user.

        granularity:
            Day, month, year, range or unknown.
    """

    start_date: str | None = None
    end_date: str | None = None
    display_value: str | None = None
    granularity: PeriodGranularity = "unknown"


@dataclass(frozen=True)
class FinanceIntent:
    """
    Structured intent extracted from a finance request.

    Attributes:
        original_request:
            Original user request after whitespace normalization.

        selected_flow:
            Existing workflow selected by the deterministic router.

        comparison:
            Requested financial comparison.

        period:
            Parsed reporting period.

        category:
            Requested business or vehicle category.

        scenario_name:
            Requested scenario when scenario analysis is selected.

        requested_kpis:
            KPI names explicitly found in the request.

        missing_fields:
            Information required before execution can begin.

        clarification_question:
            One concise question for missing information.

        is_complete:
            Whether the request contains enough information to execute.
    """

    original_request: str
    selected_flow: FlowType
    comparison: ComparisonType = "none"
    period: ParsedPeriod = field(
        default_factory=ParsedPeriod
    )
    category: str | None = None
    scenario_name: str | None = None
    requested_kpis: tuple[str, ...] = ()
    missing_fields: tuple[str, ...] = ()
    clarification_question: str | None = None
    is_complete: bool = False

    def to_filters(self) -> dict[str, object]:
        """
        Convert parsed intent into graph-compatible filters.

        Returns:
            Dictionary suitable for FinanceGraphState.
        """

        filters: dict[str, object] = {}

        if self.period.start_date:
            filters["start_date"] = self.period.start_date

        if self.period.end_date:
            filters["end_date"] = self.period.end_date

        if self.period.display_value:
            filters["period"] = self.period.display_value

        if self.period.granularity != "unknown":
            filters["period_granularity"] = (
                self.period.granularity
            )

        if self.category:
            filters["category"] = self.category

        if self.comparison != "none":
            filters["comparison"] = self.comparison

        if self.scenario_name:
            filters["scenario_name"] = self.scenario_name

        if self.requested_kpis:
            filters["requested_kpis"] = list(
                self.requested_kpis
            )

        return filters


def parse_finance_intent(
    user_request: str,
    *,
    reference_date: date | None = None,
) -> FinanceIntent:
    """
    Parse one natural-language finance request.

    Args:
        user_request:
            Natural-language request from the user.

        reference_date:
            Date used to resolve phrases such as ``today`` or
            ``this month``. Defaults to the system date.

    Returns:
        Structured FinanceIntent.

    Raises:
        TypeError:
            If user_request is not a string.

        ValueError:
            If user_request is empty.
    """

    cleaned_request = _validate_request(
        user_request
    )

    normalized_request = _normalize_text(
        cleaned_request
    )

    effective_reference_date = (
        reference_date or date.today()
    )

    selected_flow = identify_flow(
        cleaned_request
    )

    comparison = _extract_comparison(
        normalized_request
    )

    period_was_requested = _contains_period_expression(
        normalized_request
    )

    period = _extract_period(
        normalized_request,
        reference_date=effective_reference_date,
    )

    category = _extract_category(
        normalized_request
    )

    scenario_name = _extract_scenario(
        normalized_request
    )

    requested_kpis = tuple(
        _extract_requested_kpis(
            normalized_request
        )
    )

    if (
        selected_flow == "unknown"
        and requested_kpis
    ):
        selected_flow = "kpi"

    missing_fields = tuple(
        _identify_missing_fields(
            selected_flow=selected_flow,
            comparison=comparison,
            period=period,
            period_was_requested=period_was_requested,
        )
    )

    clarification_question = (
        _build_clarification_question(
            missing_fields=missing_fields,
        )
    )

    is_complete = (
        selected_flow != "unknown"
        and not missing_fields
    )

    return FinanceIntent(
        original_request=cleaned_request,
        selected_flow=selected_flow,
        comparison=comparison,
        period=period,
        category=category,
        scenario_name=scenario_name,
        requested_kpis=requested_kpis,
        missing_fields=missing_fields,
        clarification_question=clarification_question,
        is_complete=is_complete,
    )


def merge_finance_intent(
    pending_intent: FinanceIntent,
    user_reply: str,
    *,
    reference_date: date | None = None,
) -> FinanceIntent:
    """
    Merge a clarification reply into a pending finance intent.

    Example:

        Pending request:
            "Show Actual vs Budget"

        Clarification reply:
            "January 2026"

        Result:
            Complete variance intent for January 2026.
    """

    if not isinstance(
        pending_intent,
        FinanceIntent,
    ):
        raise TypeError(
            "pending_intent must be a FinanceIntent."
        )

    cleaned_reply = _validate_request(
        user_reply
    )

    combined_request = (
        f"{pending_intent.original_request} "
        f"{cleaned_reply}"
    )

    reparsed = parse_finance_intent(
        combined_request,
        reference_date=reference_date,
    )

    selected_flow = (
        reparsed.selected_flow
        if reparsed.selected_flow != "unknown"
        else pending_intent.selected_flow
    )

    comparison = (
        reparsed.comparison
        if reparsed.comparison != "none"
        else pending_intent.comparison
    )

    period = (
        reparsed.period
        if reparsed.period.start_date
        else pending_intent.period
    )

    category = (
        reparsed.category
        or pending_intent.category
    )

    scenario_name = (
        reparsed.scenario_name
        or pending_intent.scenario_name
    )

    requested_kpis = tuple(
        dict.fromkeys(
            [
                *pending_intent.requested_kpis,
                *reparsed.requested_kpis,
            ]
        )
    )

    combined_normalized_request = _normalize_text(
        combined_request
    )

    period_was_requested = _contains_period_expression(
        combined_normalized_request
    )

    missing_fields = tuple(
        _identify_missing_fields(
            selected_flow=selected_flow,
            comparison=comparison,
            period=period,
            period_was_requested=period_was_requested,
        )
    )

    return FinanceIntent(
        original_request=combined_request,
        selected_flow=selected_flow,
        comparison=comparison,
        period=period,
        category=category,
        scenario_name=scenario_name,
        requested_kpis=requested_kpis,
        missing_fields=missing_fields,
        clarification_question=(
            _build_clarification_question(
                missing_fields=missing_fields,
            )
        ),
        is_complete=(
            selected_flow != "unknown"
            and not missing_fields
        ),
    )


def _extract_comparison(
    normalized_request: str,
) -> ComparisonType:
    """Extract the requested finance comparison."""

    comparison_patterns: tuple[
        tuple[ComparisonType, tuple[str, ...]],
        ...,
    ] = (
        (
            "actual_vs_budget",
            (
                "actual vs budget",
                "actual versus budget",
                "budget vs actual",
                "budget versus actual",
                "actual against budget",
            ),
        ),
        (
            "actual_vs_forecast",
            (
                "actual vs forecast",
                "actual versus forecast",
                "forecast vs actual",
                "forecast versus actual",
                "actual against forecast",
            ),
        ),
        (
            "actual_vs_last_year",
            (
                "actual vs last year",
                "actual versus last year",
                "actual vs prior year",
                "actual versus prior year",
                "year on year",
                "year-on-year",
                "yoy",
            ),
        ),
        (
            "budget_vs_forecast",
            (
                "budget vs forecast",
                "budget versus forecast",
                "forecast vs budget",
                "forecast versus budget",
            ),
        ),
    )

    for comparison, patterns in comparison_patterns:
        if any(
            pattern in normalized_request
            for pattern in patterns
        ):
            return comparison

    if (
        "variance" in normalized_request
        or "below budget" in normalized_request
        or "above budget" in normalized_request
    ):
        return "actual_vs_budget"

    return "none"


def _contains_period_expression(
    normalized_request: str,
) -> bool:
    """
    Check whether the user attempted to provide a reporting period.

    This distinguishes between:

    - No period supplied:
      "Show KPI"

    - Invalid period supplied:
      "Show KPI for 31 February 2026"

    A missing period is allowed. An invalid supplied period requires
    clarification.
    """

    relative_patterns = (
        r"\btoday\b",
        r"\byesterday\b",
        r"\bthis month\b",
        r"\blast month\b",
        r"\bthis year\b",
    )

    if any(
        re.search(pattern, normalized_request)
        for pattern in relative_patterns
    ):
        return True

    if re.search(
        r"\bfrom\b.+\bto\b",
        normalized_request,
    ):
        return True

    if re.search(
        r"\bbetween\b.+\band\b",
        normalized_request,
    ):
        return True

    if re.search(
        r"\b20\d{2}-\d{1,2}-\d{1,2}\b",
        normalized_request,
    ):
        return True

    if re.search(
        r"\b\d{1,2}[./-]\d{1,2}[./-]20\d{2}\b",
        normalized_request,
    ):
        return True

    month_pattern = "|".join(
        sorted(
            MONTH_NAMES,
            key=len,
            reverse=True,
        )
    )

    if re.search(
        rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+"
        rf"(?:{month_pattern})\s+20\d{{2}}\b",
        normalized_request,
    ):
        return True

    if re.search(
        rf"\b(?:{month_pattern})\s+20\d{{2}}\b",
        normalized_request,
    ):
        return True

    if re.search(
        r"\b20\d{2}\b",
        normalized_request,
    ):
        return True

    return False


def _extract_period(
    normalized_request: str,
    *,
    reference_date: date,
) -> ParsedPeriod:
    """
    Extract a date range, single day, month or year.

    Explicit invalid dates remain unresolved and must not fall back to
    month or year parsing.
    """

    has_range_expression = bool(
        re.search(
            r"\bfrom\b.+\bto\b|\bbetween\b.+\band\b",
            normalized_request,
        )
    )

    if has_range_expression:
        return _extract_date_range(
            normalized_request
        )

    relative_period = _extract_relative_period(
        normalized_request,
        reference_date=reference_date,
    )

    if relative_period.start_date:
        return relative_period

    iso_date_match = re.search(
        r"\b20\d{2}-\d{1,2}-\d{1,2}\b",
        normalized_request,
    )

    if iso_date_match:
        iso_date = _extract_iso_date(
            normalized_request
        )

        if iso_date:
            return _day_period(
                iso_date
            )

        return ParsedPeriod()

    numeric_date_match = re.search(
        r"\b\d{1,2}[./-]\d{1,2}[./-]20\d{2}\b",
        normalized_request,
    )

    if numeric_date_match:
        numeric_date = _extract_numeric_date(
            normalized_request
        )

        if numeric_date:
            return _day_period(
                numeric_date
            )

        return ParsedPeriod()

    month_pattern = "|".join(
        sorted(
            MONTH_NAMES,
            key=len,
            reverse=True,
        )
    )

    named_day_match = re.search(
        rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+"
        rf"(?:{month_pattern})\s+20\d{{2}}\b",
        normalized_request,
    )

    if named_day_match:
        named_day = _extract_named_day(
            normalized_request
        )

        if named_day:
            return _day_period(
                named_day
            )

        return ParsedPeriod()

    named_month = _extract_named_month(
        normalized_request
    )

    if named_month:
        return _month_period(
            year=named_month[0],
            month=named_month[1],
        )

    year_match = re.search(
        r"\b(20\d{2})\b",
        normalized_request,
    )

    if year_match:
        year = int(
            year_match.group(1)
        )

        return ParsedPeriod(
            start_date=f"{year:04d}-01-01",
            end_date=f"{year:04d}-12-31",
            display_value=str(year),
            granularity="year",
        )

    return ParsedPeriod()


def _extract_date_range(
    normalized_request: str,
) -> ParsedPeriod:
    """
    Extract a start date and end date.

    Supported examples:

    - from 1 January 2026 to 31 March 2026
    - between 01/01/2026 and 31/01/2026
    - from 2026-01-01 to 2026-03-31
    - from 1 Jan to 31 Mar 2026
    """

    explicit_range_patterns = (
        r"\bfrom\s+(.+?)\s+to\s+(.+?)(?:$|[?.!,])",
        r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:$|[?.!,])",
    )

    for pattern in explicit_range_patterns:
        match = re.search(
            pattern,
            normalized_request,
        )

        if not match:
            continue

        return _parse_range_fragments(
            match.group(1),
            match.group(2),
        )

    return ParsedPeriod()


def _parse_range_fragments(
    start_text: str,
    end_text: str,
) -> ParsedPeriod:
    """Parse two range endpoints."""

    cleaned_start = _clean_date_fragment(
        start_text
    )
    cleaned_end = _clean_date_fragment(
        end_text
    )

    start_date = _parse_date_fragment(
        cleaned_start
    )

    end_date = _parse_date_fragment(
        cleaned_end
    )

    if start_date is None and end_date is not None:
        start_date = _parse_date_fragment(
            cleaned_start,
            default_year=end_date.year,
        )

    if end_date is None and start_date is not None:
        end_date = _parse_date_fragment(
            cleaned_end,
            default_year=start_date.year,
        )

    if (
        start_date is None
        or end_date is None
    ):
        return ParsedPeriod()

    return _range_period(
        start_date=start_date,
        end_date=end_date,
    )


def _clean_date_fragment(
    value: str,
) -> str:
    """Remove common trailing request wording."""

    cleaned_value = " ".join(
        value.lower().split()
    )

    trailing_patterns = (
        r"\s+for\s+(?:2w|3w|tata ace.*|compact auto).*$",
        r"\s+using\s+actual.*$",
        r"\s+include\s+.*$",
        r"\s+showing\s+.*$",
    )

    for pattern in trailing_patterns:
        cleaned_value = re.sub(
            pattern,
            "",
            cleaned_value,
        )

    return cleaned_value.strip()


def _parse_date_fragment(
    value: str,
    *,
    default_year: int | None = None,
) -> date | None:
    """Parse one date fragment inside a date range."""

    cleaned_value = " ".join(
        value.lower().split()
    )

    iso_match = re.search(
        r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b",
        cleaned_value,
    )

    if iso_match:
        return _safe_date(
            year=int(iso_match.group(1)),
            month=int(iso_match.group(2)),
            day=int(iso_match.group(3)),
        )

    numeric_match = re.search(
        r"\b(\d{1,2})[./-](\d{1,2})[./-](20\d{2})\b",
        cleaned_value,
    )

    if numeric_match:
        return _safe_date(
            year=int(numeric_match.group(3)),
            month=int(numeric_match.group(2)),
            day=int(numeric_match.group(1)),
        )

    month_pattern = "|".join(
        sorted(
            MONTH_NAMES,
            key=len,
            reverse=True,
        )
    )

    named_with_year = re.search(
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+"
        rf"({month_pattern})\s+(20\d{{2}})\b",
        cleaned_value,
    )

    if named_with_year:
        return _safe_date(
            year=int(
                named_with_year.group(3)
            ),
            month=MONTH_NAMES[
                named_with_year.group(2)
            ],
            day=int(
                named_with_year.group(1)
            ),
        )

    named_without_year = re.search(
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+"
        rf"({month_pattern})\b",
        cleaned_value,
    )

    if (
        named_without_year
        and default_year is not None
    ):
        return _safe_date(
            year=default_year,
            month=MONTH_NAMES[
                named_without_year.group(2)
            ],
            day=int(
                named_without_year.group(1)
            ),
        )

    return None


def _extract_relative_period(
    normalized_request: str,
    *,
    reference_date: date,
) -> ParsedPeriod:
    """Extract supported relative date expressions."""

    if re.search(
        r"\btoday\b",
        normalized_request,
    ):
        return _day_period(
            reference_date
        )

    if re.search(
        r"\byesterday\b",
        normalized_request,
    ):
        return _day_period(
            date.fromordinal(
                reference_date.toordinal() - 1
            )
        )

    if re.search(
        r"\bthis month\b",
        normalized_request,
    ):
        return _month_period(
            year=reference_date.year,
            month=reference_date.month,
        )

    if re.search(
        r"\blast month\b",
        normalized_request,
    ):
        if reference_date.month == 1:
            year = reference_date.year - 1
            month = 12
        else:
            year = reference_date.year
            month = reference_date.month - 1

        return _month_period(
            year=year,
            month=month,
        )

    if re.search(
        r"\bthis year\b",
        normalized_request,
    ):
        year = reference_date.year

        return ParsedPeriod(
            start_date=f"{year:04d}-01-01",
            end_date=f"{year:04d}-12-31",
            display_value=str(year),
            granularity="year",
        )

    return ParsedPeriod()


def _extract_iso_date(
    normalized_request: str,
) -> date | None:
    """Extract an ISO date."""

    match = re.search(
        r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b",
        normalized_request,
    )

    if not match:
        return None

    return _safe_date(
        year=int(match.group(1)),
        month=int(match.group(2)),
        day=int(match.group(3)),
    )


def _extract_numeric_date(
    normalized_request: str,
) -> date | None:
    """Extract DD.MM.YYYY or DD/MM/YYYY."""

    match = re.search(
        r"\b(\d{1,2})[./-](\d{1,2})[./-](20\d{2})\b",
        normalized_request,
    )

    if not match:
        return None

    return _safe_date(
        year=int(match.group(3)),
        month=int(match.group(2)),
        day=int(match.group(1)),
    )


def _extract_named_day(
    normalized_request: str,
) -> date | None:
    """Extract dates such as 20 October 2026."""

    month_pattern = "|".join(
        sorted(
            MONTH_NAMES,
            key=len,
            reverse=True,
        )
    )

    match = re.search(
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+"
        rf"({month_pattern})\s+(20\d{{2}})\b",
        normalized_request,
    )

    if not match:
        return None

    return _safe_date(
        year=int(match.group(3)),
        month=MONTH_NAMES[
            match.group(2)
        ],
        day=int(match.group(1)),
    )


def _extract_named_month(
    normalized_request: str,
) -> tuple[int, int] | None:
    """Extract a month and year."""

    month_pattern = "|".join(
        sorted(
            MONTH_NAMES,
            key=len,
            reverse=True,
        )
    )

    match = re.search(
        rf"\b({month_pattern})\s+(20\d{{2}})\b",
        normalized_request,
    )

    if not match:
        return None

    return (
        int(match.group(2)),
        MONTH_NAMES[
            match.group(1)
        ],
    )


def _extract_category(
    normalized_request: str,
) -> str | None:
    """Extract a supported category."""

    aliases = sorted(
        CATEGORY_ALIASES,
        key=len,
        reverse=True,
    )

    for alias in aliases:
        pattern = (
            rf"(?<![a-z0-9])"
            rf"{re.escape(alias)}"
            rf"(?![a-z0-9])"
        )

        if re.search(
            pattern,
            normalized_request,
        ):
            return CATEGORY_ALIASES[
                alias
            ]

    return None


def _extract_scenario(
    normalized_request: str,
) -> str | None:
    """Extract a supported scenario."""

    scenario_patterns = {
        "Management Case": (
            "management case",
        ),
        "Base Case": (
            "base case",
        ),
        "Upside Case": (
            "upside case",
            "best case",
        ),
        "Downside Case": (
            "downside case",
            "worst case",
        ),
    }

    for scenario_name, patterns in scenario_patterns.items():
        if any(
            pattern in normalized_request
            for pattern in patterns
        ):
            return scenario_name

    return None


def _extract_requested_kpis(
    normalized_request: str,
) -> list[str]:
    """Extract explicitly requested KPI names."""

    requested_kpis: list[str] = []

    aliases = sorted(
        KPI_ALIASES,
        key=len,
        reverse=True,
    )

    for alias in aliases:
        pattern = (
            rf"(?<![a-z0-9])"
            rf"{re.escape(alias)}"
            rf"(?![a-z0-9])"
        )

        if not re.search(
            pattern,
            normalized_request,
        ):
            continue

        canonical_name = KPI_ALIASES[
            alias
        ]

        if canonical_name not in requested_kpis:
            requested_kpis.append(
                canonical_name
            )

    return requested_kpis


def _identify_missing_fields(
    *,
    selected_flow: FlowType,
    comparison: ComparisonType,
    period: ParsedPeriod,
    period_was_requested: bool,
) -> list[str]:
    """
    Identify information required before execution.

    Rules:

    - Unknown workflow requires the analysis type.
    - Variance analysis requires a comparison.
    - Reporting period is optional.
    - An invalid explicitly supplied period requires clarification.
    """

    missing_fields: list[str] = []

    if selected_flow == "unknown":
        missing_fields.append(
            "analysis_type"
        )
        return missing_fields

    if (
        selected_flow == "variance"
        and comparison == "none"
    ):
        missing_fields.append(
            "comparison"
        )

    if (
        period_was_requested
        and (
            period.start_date is None
            or period.end_date is None
        )
    ):
        missing_fields.append(
            "period"
        )

    return missing_fields


def _build_clarification_question(
    *,
    missing_fields: tuple[str, ...],
) -> str | None:
    """Create one concise clarification question."""

    if not missing_fields:
        return None

    if missing_fields == (
        "analysis_type",
    ):
        return (
            "What analysis would you like to run: KPI, budget, "
            "forecast, variance, scenario, P&L or management report?"
        )

    if missing_fields == (
        "period",
    ):
        return (
            "The reporting period appears invalid. Please provide a "
            "valid period, such as January 2026, 20 October 2026, "
            "or 1 January 2026 to 31 March 2026."
        )

    if missing_fields == (
        "comparison",
    ):
        return (
            "Which comparison would you like: Actual vs Budget, "
            "Actual vs Forecast or Actual vs Last Year?"
        )

    if (
        "period" in missing_fields
        and "comparison" in missing_fields
    ):
        return (
            "Please provide a valid reporting period and comparison. "
            "For example: January 2026, Actual vs Budget."
        )

    return (
        "Please provide the missing information: "
        + ", ".join(missing_fields)
        + "."
    )


def _month_period(
    *,
    year: int,
    month: int,
) -> ParsedPeriod:
    """Create a complete month period."""

    final_day = calendar.monthrange(
        year,
        month,
    )[1]

    return ParsedPeriod(
        start_date=(
            f"{year:04d}-{month:02d}-01"
        ),
        end_date=(
            f"{year:04d}-{month:02d}-"
            f"{final_day:02d}"
        ),
        display_value=(
            f"{calendar.month_name[month]} {year}"
        ),
        granularity="month",
    )


def _day_period(
    value: date,
) -> ParsedPeriod:
    """Create a one-day period."""

    iso_value = value.isoformat()

    return ParsedPeriod(
        start_date=iso_value,
        end_date=iso_value,
        display_value=value.strftime(
            "%d %B %Y"
        ),
        granularity="day",
    )


def _range_period(
    *,
    start_date: date,
    end_date: date,
) -> ParsedPeriod:
    """Create a validated date range."""

    if end_date < start_date:
        return ParsedPeriod()

    return ParsedPeriod(
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        display_value=(
            f"{start_date.strftime('%d %B %Y')} "
            f"to {end_date.strftime('%d %B %Y')}"
        ),
        granularity="range",
    )


def _safe_date(
    *,
    year: int,
    month: int,
    day: int,
) -> date | None:
    """Return a valid date or None."""

    try:
        return date(
            year,
            month,
            day,
        )
    except ValueError:
        return None


def _normalize_text(
    value: str,
) -> str:
    """Normalize whitespace and casing."""

    return " ".join(
        value.lower().split()
    )


def _validate_request(
    user_request: str,
) -> str:
    """Validate and normalize the request."""

    if not isinstance(
        user_request,
        str,
    ):
        raise TypeError(
            "user_request must be a string."
        )

    cleaned_request = " ".join(
        user_request.split()
    )

    if not cleaned_request:
        raise ValueError(
            "user_request cannot be empty."
        )

    return cleaned_request