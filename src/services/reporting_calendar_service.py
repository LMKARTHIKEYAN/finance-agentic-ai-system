"""Determine complete daily, weekly, and monthly reporting periods."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class ReportingPeriod:
    report_type: str
    start_date: date
    end_date: date
    data_cutoff: date

    @property
    def period_key(self) -> str:
        if self.report_type == "daily":
            return self.end_date.isoformat()
        if self.report_type == "monthly":
            return self.end_date.strftime("%Y-%m")
        return f"{self.start_date.isoformat()}_{self.end_date.isoformat()}"


class ReportingCalendarService:
    """Choose periods from the latest available, fully loaded data date."""

    def daily(self, latest_data_date: date, as_of: date | None = None) -> ReportingPeriod:
        cutoff = self._cutoff(latest_data_date, as_of)
        return ReportingPeriod("daily", cutoff, cutoff, cutoff)

    def weekly(self, latest_data_date: date, as_of: date | None = None) -> ReportingPeriod:
        cutoff = self._cutoff(latest_data_date, as_of)
        end = cutoff - timedelta(days=(cutoff.weekday() + 1) % 7)
        return ReportingPeriod("weekly", end - timedelta(days=6), end, cutoff)

    def monthly(self, latest_data_date: date, as_of: date | None = None) -> ReportingPeriod:
        cutoff = self._cutoff(latest_data_date, as_of)
        first_current = cutoff.replace(day=1)
        end = first_current - timedelta(days=1)
        start = end.replace(day=1)
        return ReportingPeriod("monthly", start, end, cutoff)

    @staticmethod
    def _cutoff(latest_data_date: date, as_of: date | None) -> date:
        if not isinstance(latest_data_date, date):
            raise TypeError("latest_data_date must be a date.")
        effective_as_of = as_of or date.today()
        if latest_data_date >= effective_as_of:
            return effective_as_of - timedelta(days=1)
        return latest_data_date
