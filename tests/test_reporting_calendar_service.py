from datetime import date

from src.services.reporting_calendar_service import ReportingCalendarService


def test_periods_use_latest_complete_data() -> None:
    service = ReportingCalendarService()
    assert service.daily(date(2026, 8, 22), date(2026, 8, 23)).period_key == "2026-08-22"
    weekly = service.weekly(date(2026, 8, 22), date(2026, 8, 23))
    assert (weekly.start_date, weekly.end_date) == (date(2026, 8, 10), date(2026, 8, 16))
    assert service.monthly(date(2026, 8, 22), date(2026, 8, 23)).period_key == "2026-07"
