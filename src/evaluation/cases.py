"""Representative evaluation cases for autonomous FP&A behavior."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationCase:
    name: str
    question: str
    expected_status: str
    expected_tools: tuple[str, ...]


EVALUATION_CASES = (
    EvaluationCase("ambiguous annual pnl", "Show P&L 2026", "waiting_for_user", ()),
    EvaluationCase("monthly pnl", "Show P&L for August 2026", "completed", ("calculate_pnl",)),
    EvaluationCase(
        "monthly management pack",
        "August 2026 P&L, revenue variance, GP decomposition and forecast",
        "completed",
        ("calculate_pnl", "calculate_revenue_variance", "calculate_gp_decomposition", "calculate_rolling_forecast"),
    ),
    EvaluationCase(
        "category KPI",
        "Show actual KPI by vehicle category for August 2026",
        "completed",
        ("calculate_kpis",),
    ),
    EvaluationCase(
        "category rolling forecast",
        "Show rolling forecast by vehicle category from August 2026",
        "completed",
        ("calculate_rolling_forecast",),
    ),
    EvaluationCase(
        "cited revenue root cause",
        "Explain why revenue changed in August 2026 using company evidence",
        "completed",
        ("calculate_revenue_variance", "retrieve_company_context"),
    ),
    EvaluationCase("daily trend", "Show daily revenue and order trend for July 2026", "completed", ("analyze_trends",)),
    EvaluationCase("period comparison", "Compare July 2026 versus previous month", "completed", ("compare_periods",)),
    EvaluationCase("anomaly detection", "Show KPI anomalies for July 2026", "completed", ("detect_anomalies",)),
    EvaluationCase("category profitability", "Show category profitability for August 2026", "completed", ("analyze_category_profitability",)),
    EvaluationCase("scenario", "What if orders fall by 10% in August 2026?", "completed", ("analyze_scenario",)),
    EvaluationCase("driver forecast", "Show driver-based forecast from August 2026", "completed", ("forecast_from_drivers",)),
)
