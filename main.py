"""Direct pipeline integration for the Finance Agentic AI System.

This module connects the completed deterministic agents before LangGraph is
introduced. It supports KPI-only, budget-only, forecast-only, and full-analysis
flows while preserving the responsibility of each existing agent.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from src.agents.analytics.anomaly_agent import AnomalyAgent
from src.agents.analytics.operations_analysis_agent import OperationsAnalysisAgent
from src.agents.analytics.recommendation_agent import RecommendationAgent
from src.agents.analytics.root_cause_agent import RootCauseAgent
from src.agents.data_quality.cleaning_agent import CleaningAgent
from src.agents.data_quality.profiling_agent import ProfilingAgent
from src.agents.data_quality.validation_agent import ValidationAgent, ValidationResult
from src.agents.finance.budget_agent import BudgetAgent
from src.agents.finance.finance_rules_agent import FinanceRulesAgent
from src.agents.finance.forecast_agent import ForecastAgent
from src.agents.finance.kpi_agent import KPIAgent
from src.agents.finance.scenario_agent import ScenarioAgent
from src.agents.finance.variance_agent import RevenueVarianceAgent
from src.agents.reporting.commentary_agent import CommentaryAgent
from src.agents.reporting.report_agent import ReportAgent


FlowName = Literal["kpi", "budget", "forecast", "full"]

DEFAULT_KPIS = [
    "total_orders",
    "completed_orders",
    "cancelled_orders",
    "fulfillment_percentage",
    "cancellation_percentage",
    "actual_revenue",
    "actual_aov",
    "budget_orders",
    "budget_revenue",
    "budget_aov",
    "revenue_variance",
    "forecast_orders",
    "forecast_revenue",
    "forecast_aov",
    "scenario_orders",
    "scenario_revenue",
    "finance_rules_status",
]


@dataclass
class PipelineResult:
    """Stores outputs produced by one direct pipeline execution."""

    flow: FlowName
    operations_validation: ValidationResult | None = None
    budget_validation: ValidationResult | None = None
    cleaned_operations_data: pd.DataFrame | None = None
    cleaned_budget_data: pd.DataFrame | None = None
    operations_profile: Any | None = None
    budget_profile: Any | None = None
    operations_result: Any | None = None
    budget_result: Any | None = None
    forecast_result: Any | None = None
    scenario_result: Any | None = None
    variance_result: Any | None = None
    finance_rules_result: Any | None = None
    anomaly_result: Any | None = None
    root_cause_result: Any | None = None
    recommendation_result: Any | None = None
    kpi_result: Any | None = None
    commentary_result: Any | None = None
    report_result: Any | None = None


class FinancePipeline:
    """Coordinate completed agents without adding LangGraph routing logic."""

    def __init__(self) -> None:
        self.operations_agent = OperationsAnalysisAgent()
        self.budget_agent = BudgetAgent()
        self.forecast_agent = ForecastAgent()
        self.scenario_agent = ScenarioAgent()
        self.variance_agent = RevenueVarianceAgent()
        self.finance_rules_agent = FinanceRulesAgent()
        self.anomaly_agent = AnomalyAgent()
        self.root_cause_agent = RootCauseAgent()
        self.recommendation_agent = RecommendationAgent()
        self.kpi_agent = KPIAgent()
        self.commentary_agent = CommentaryAgent()
        self.report_agent = ReportAgent()

    @staticmethod
    def _require_valid(result: ValidationResult, dataset_name: str) -> None:
        if not result.is_valid:
            details = "; ".join(result.errors) or "Unknown validation error."
            raise ValueError(f"{dataset_name} validation failed: {details}")

    @staticmethod
    def _prepare_operations(raw_data: pd.DataFrame) -> tuple[ValidationResult, pd.DataFrame]:
        validation = ValidationAgent.validate_operations_data(raw_data)
        FinancePipeline._require_valid(validation, "Operations data")
        cleaned = CleaningAgent.clean_operations_data(raw_data)
        return validation, cleaned

    @staticmethod
    def _prepare_budget(raw_data: pd.DataFrame) -> tuple[ValidationResult, pd.DataFrame]:
        validation = ValidationAgent.validate_budget_data(raw_data)
        FinancePipeline._require_valid(validation, "Budget data")
        cleaned = CleaningAgent.clean_budget_data(raw_data)
        return validation, cleaned

    def run_kpi_flow(
        self,
        operations_data: pd.DataFrame,
        requested_kpis: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        group_by: str = "month",
        include_commentary: bool = True,
    ) -> PipelineResult:
        """Run validation, cleaning, operations analysis, KPI, and commentary."""

        validation, cleaned = self._prepare_operations(operations_data)
        operations_result = self.operations_agent.analyze(
            data=cleaned,
            start_date=start_date,
            end_date=end_date,
            group_by=group_by,
        )
        kpi_result = self.kpi_agent.analyze(
            requested_kpis=requested_kpis or [
                "total_orders",
                "completed_orders",
                "cancelled_orders",
                "fulfillment_percentage",
                "cancellation_percentage",
                "actual_revenue",
                "actual_aov",
            ],
            operations_result=operations_result,
        )
        commentary_result = (
            self.commentary_agent.analyze(kpi_result=kpi_result)
            if include_commentary
            else None
        )
        return PipelineResult(
            flow="kpi",
            operations_validation=validation,
            cleaned_operations_data=cleaned,
            operations_result=operations_result,
            kpi_result=kpi_result,
            commentary_result=commentary_result,
        )

    def run_budget_flow(
        self,
        budget_data: pd.DataFrame,
        start_month: str | None = None,
        end_month: str | None = None,
        group_by: str = "month",
    ) -> PipelineResult:
        """Run validation, cleaning, and budget analysis."""

        validation, cleaned = self._prepare_budget(budget_data)
        budget_result = self.budget_agent.analyze(
            data=cleaned,
            start_month=start_month,
            end_month=end_month,
            group_by=group_by,
        )
        return PipelineResult(
            flow="budget",
            budget_validation=validation,
            cleaned_budget_data=cleaned,
            budget_result=budget_result,
        )

    def run_forecast_flow(
        self,
        operations_data: pd.DataFrame,
        start_date: str | None = None,
        end_date: str | None = None,
        frequency: str = "month",
        rolling_window: int = 3,
        forecast_periods: int = 3,
        include_commentary: bool = True,
    ) -> PipelineResult:
        """Run validation, cleaning, historical analysis, and forecasting."""

        validation, cleaned = self._prepare_operations(operations_data)
        operations_result = self.operations_agent.analyze(
            data=cleaned,
            start_date=start_date,
            end_date=end_date,
            group_by=frequency,
        )
        forecast_result = self.forecast_agent.analyze(
            period_summary=operations_result.period_summary,
            frequency=frequency,
            rolling_window=rolling_window,
            forecast_periods=forecast_periods,
        )
        forecast_period = forecast_result.forecast_summary[0]["forecast_period"]
        kpi_result = self.kpi_agent.analyze(
            requested_kpis=["forecast_orders", "forecast_revenue", "forecast_aov"],
            forecast_result=forecast_result,
            forecast_period=str(forecast_period),
        )
        commentary_result = (
            self.commentary_agent.analyze(
                kpi_result=kpi_result,
                forecast_result=forecast_result,
            )
            if include_commentary
            else None
        )
        return PipelineResult(
            flow="forecast",
            operations_validation=validation,
            cleaned_operations_data=cleaned,
            operations_result=operations_result,
            forecast_result=forecast_result,
            kpi_result=kpi_result,
            commentary_result=commentary_result,
        )

    def run_full_analysis(
        self,
        operations_data: pd.DataFrame,
        budget_data: pd.DataFrame,
        assumptions: pd.DataFrame | list[dict[str, Any]],
        requested_kpis: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        start_month: str | None = None,
        end_month: str | None = None,
        frequency: str = "month",
        rolling_window: int = 3,
        forecast_periods: int = 6,
        scenario_name: str = "Management Case",
    ) -> PipelineResult:
        """Run the complete deterministic analysis chain before LangGraph."""

        operations_validation, cleaned_operations = self._prepare_operations(
            operations_data
        )
        budget_validation, cleaned_budget = self._prepare_budget(budget_data)

        operations_profile = ProfilingAgent.profile_operations_data(cleaned_operations)
        budget_profile = ProfilingAgent.profile_budget_data(cleaned_budget)

        operations_result = self.operations_agent.analyze(
            data=cleaned_operations,
            start_date=start_date,
            end_date=end_date,
            group_by=frequency,
        )
        budget_result = self.budget_agent.analyze(
            data=cleaned_budget,
            start_month=start_month,
            end_month=end_month,
            group_by=frequency,
        )
        forecast_result = self.forecast_agent.analyze(
            period_summary=operations_result.period_summary,
            frequency=frequency,
            rolling_window=rolling_window,
            forecast_periods=forecast_periods,
        )
        scenario_result = self.scenario_agent.analyze(
            forecast_result=forecast_result,
            assumptions=assumptions,
            scenario_name=scenario_name,
        )
        variance_result = self.variance_agent.analyze(
            actual_result=operations_result,
            budget_result=budget_result,
        )
        finance_rules_result = self.finance_rules_agent.analyze(
            operations_result=operations_result,
            budget_result=budget_result,
            revenue_variance_result=variance_result,
            forecast_result=forecast_result,
            scenario_result=scenario_result,
        )
        anomaly_result = self.anomaly_agent.analyze(operations_result)
        root_cause_result = self.root_cause_agent.analyze(
            anomaly_result=anomaly_result,
            operations_result=operations_result,
            variance_result=variance_result,
        )
        recommendation_result = self.recommendation_agent.analyze(root_cause_result)

        forecast_period = str(forecast_result.forecast_summary[0]["forecast_period"])
        scenario_period = str(scenario_result.adjusted_forecast[0]["forecast_period"])
        kpi_result = self.kpi_agent.analyze(
            requested_kpis=requested_kpis or DEFAULT_KPIS,
            operations_result=operations_result,
            budget_result=budget_result,
            revenue_variance_result=variance_result,
            forecast_result=forecast_result,
            scenario_result=scenario_result,
            finance_rules_result=finance_rules_result,
            forecast_period=forecast_period,
            scenario_period=scenario_period,
        )
        commentary_result = self.commentary_agent.analyze(
            kpi_result=kpi_result,
            revenue_variance_result=variance_result,
            forecast_result=forecast_result,
            scenario_result=scenario_result,
            finance_rules_result=finance_rules_result,
        )
        report_result = self.report_agent.analyze(
            commentary_result=commentary_result,
            operations_result=operations_result,
            kpi_result=kpi_result,
            budget_result=budget_result,
            forecast_result=forecast_result,
            variance_result=variance_result,
            anomaly_result=anomaly_result,
            root_cause_result=root_cause_result,
            recommendation_result=recommendation_result,
            finance_rules_result=finance_rules_result,
            scenario_result=scenario_result,
            report_type="full",
        )

        return PipelineResult(
            flow="full",
            operations_validation=operations_validation,
            budget_validation=budget_validation,
            cleaned_operations_data=cleaned_operations,
            cleaned_budget_data=cleaned_budget,
            operations_profile=operations_profile,
            budget_profile=budget_profile,
            operations_result=operations_result,
            budget_result=budget_result,
            forecast_result=forecast_result,
            scenario_result=scenario_result,
            variance_result=variance_result,
            finance_rules_result=finance_rules_result,
            anomaly_result=anomaly_result,
            root_cause_result=root_cause_result,
            recommendation_result=recommendation_result,
            kpi_result=kpi_result,
            commentary_result=commentary_result,
            report_result=report_result,
        )


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file and raise a clear error when it is unavailable."""

    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    return pd.read_csv(csv_path)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for direct pipeline testing."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow", choices=["kpi", "budget", "forecast", "full"], default="full")
    parser.add_argument("--operations", default="data/operations/sample_orders.csv")
    parser.add_argument("--budget", default="data/planning/sample_budget.csv")
    parser.add_argument("--assumptions", default="data/assumptions/business_assumptions.csv")
    return parser


def main() -> None:
    """Run a selected sample-data flow and print a concise result."""

    args = build_parser().parse_args()
    pipeline = FinancePipeline()

    if args.flow == "kpi":
        result = pipeline.run_kpi_flow(load_csv(args.operations))
        print(result.kpi_result)
    elif args.flow == "budget":
        result = pipeline.run_budget_flow(load_csv(args.budget))
        print(result.budget_result)
    elif args.flow == "forecast":
        result = pipeline.run_forecast_flow(load_csv(args.operations))
        print(result.forecast_result)
    else:
        result = pipeline.run_full_analysis(
            operations_data=load_csv(args.operations),
            budget_data=load_csv(args.budget),
            assumptions=load_csv(args.assumptions),
        )
        print(result.report_result.markdown_report)


if __name__ == "__main__":
    main()