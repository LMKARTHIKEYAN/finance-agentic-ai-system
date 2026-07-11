"""
Finance Agentic AI System entry point.

This module supports two execution modes:

1. Direct mode
   Executes the existing deterministic FinancePipeline.

2. Graph mode
   Executes the LangGraph orchestration workflow using a natural-language
   user request.

Examples:

    python main.py --mode direct --flow full

    python main.py --mode graph --request "Show KPI performance"

    python main.py --mode graph --request "Explain revenue variance"
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from src.agents.analytics.anomaly_agent import AnomalyAgent
from src.agents.analytics.operations_analysis_agent import (
    OperationsAnalysisAgent,
)
from src.agents.analytics.recommendation_agent import RecommendationAgent
from src.agents.analytics.root_cause_agent import RootCauseAgent
from src.agents.data_quality.cleaning_agent import CleaningAgent
from src.agents.data_quality.profiling_agent import ProfilingAgent
from src.agents.data_quality.validation_agent import (
    ValidationAgent,
    ValidationResult,
)
from src.agents.finance.budget_agent import BudgetAgent
from src.agents.finance.finance_rules_agent import FinanceRulesAgent
from src.agents.finance.forecast_agent import ForecastAgent
from src.agents.finance.kpi_agent import KPIAgent
from src.agents.finance.scenario_agent import ScenarioAgent
from src.agents.finance.variance_agent import RevenueVarianceAgent
from src.agents.reporting.commentary_agent import CommentaryAgent
from src.agents.reporting.report_agent import ReportAgent
from src.orchestrator.graph import run_finance_graph
from src.orchestrator.router import identify_flow
from src.orchestrator.state import FinanceGraphState


FlowName = Literal[
    "kpi",
    "budget",
    "forecast",
    "full",
]

ModeName = Literal[
    "direct",
    "graph",
]


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
    """Store outputs produced by one direct pipeline execution."""

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
    """Coordinate completed agents using the direct execution pipeline."""

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
    def _require_valid(
        result: ValidationResult,
        dataset_name: str,
    ) -> None:
        """
        Raise an error when validation fails.

        Args:
            result:
                Validation result returned by ValidationAgent.

            dataset_name:
                Human-readable dataset name.

        Raises:
            ValueError:
                If the validation result is invalid.
        """

        if not result.is_valid:
            details = (
                "; ".join(result.errors)
                or "Unknown validation error."
            )

            raise ValueError(
                f"{dataset_name} validation failed: {details}"
            )

    @staticmethod
    def _prepare_operations(
        raw_data: pd.DataFrame,
    ) -> tuple[ValidationResult, pd.DataFrame]:
        """Validate and clean operations data."""

        validation = ValidationAgent.validate_operations_data(
            raw_data
        )

        FinancePipeline._require_valid(
            validation,
            "Operations data",
        )

        cleaned = CleaningAgent.clean_operations_data(
            raw_data
        )

        return validation, cleaned

    @staticmethod
    def _prepare_budget(
        raw_data: pd.DataFrame,
    ) -> tuple[ValidationResult, pd.DataFrame]:
        """Validate and clean budget data."""

        validation = ValidationAgent.validate_budget_data(
            raw_data
        )

        FinancePipeline._require_valid(
            validation,
            "Budget data",
        )

        cleaned = CleaningAgent.clean_budget_data(
            raw_data
        )

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
        """
        Run validation, cleaning, operations analysis, KPI and commentary.
        """

        validation, cleaned = self._prepare_operations(
            operations_data
        )

        operations_result = self.operations_agent.analyze(
            data=cleaned,
            start_date=start_date,
            end_date=end_date,
            group_by=group_by,
        )

        kpi_result = self.kpi_agent.analyze(
            requested_kpis=requested_kpis
            or [
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
            self.commentary_agent.analyze(
                kpi_result=kpi_result,
            )
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
        """Run validation, cleaning and budget analysis."""

        validation, cleaned = self._prepare_budget(
            budget_data
        )

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
        """
        Run validation, cleaning, historical analysis and forecasting.
        """

        validation, cleaned = self._prepare_operations(
            operations_data
        )

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

        if not forecast_result.forecast_summary:
            raise ValueError(
                "Forecast Agent produced an empty forecast output."
            )

        forecast_period = str(
            forecast_result.forecast_summary[0][
                "forecast_period"
            ]
        )

        kpi_result = self.kpi_agent.analyze(
            requested_kpis=[
                "forecast_orders",
                "forecast_revenue",
                "forecast_aov",
            ],
            forecast_result=forecast_result,
            forecast_period=forecast_period,
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
        """Run the complete deterministic finance analysis pipeline."""

        (
            operations_validation,
            cleaned_operations,
        ) = self._prepare_operations(
            operations_data
        )

        (
            budget_validation,
            cleaned_budget,
        ) = self._prepare_budget(
            budget_data
        )

        operations_profile = (
            ProfilingAgent.profile_operations_data(
                cleaned_operations
            )
        )

        budget_profile = (
            ProfilingAgent.profile_budget_data(
                cleaned_budget
            )
        )

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

        if not forecast_result.forecast_summary:
            raise ValueError(
                "Forecast Agent produced an empty forecast output."
            )

        scenario_result = self.scenario_agent.analyze(
            forecast_result=forecast_result,
            assumptions=assumptions,
            scenario_name=scenario_name,
        )

        if not scenario_result.adjusted_forecast:
            raise ValueError(
                "Scenario Agent produced an empty scenario output."
            )

        variance_result = self.variance_agent.analyze(
            actual_result=operations_result,
            budget_result=budget_result,
        )

        finance_rules_result = (
            self.finance_rules_agent.analyze(
                operations_result=operations_result,
                budget_result=budget_result,
                revenue_variance_result=variance_result,
                forecast_result=forecast_result,
                scenario_result=scenario_result,
            )
        )

        anomaly_result = self.anomaly_agent.analyze(
            operations_result
        )

        root_cause_result = self.root_cause_agent.analyze(
            anomaly_result=anomaly_result,
            operations_result=operations_result,
            variance_result=variance_result,
        )

        recommendation_result = (
            self.recommendation_agent.analyze(
                root_cause_result
            )
        )

        forecast_period = str(
            forecast_result.forecast_summary[0][
                "forecast_period"
            ]
        )

        scenario_period = str(
            scenario_result.adjusted_forecast[0][
                "forecast_period"
            ]
        )

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

        commentary_result = (
            self.commentary_agent.analyze(
                kpi_result=kpi_result,
                revenue_variance_result=variance_result,
                forecast_result=forecast_result,
                scenario_result=scenario_result,
                finance_rules_result=finance_rules_result,
            )
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
    """
    Load a CSV file.

    Args:
        path:
            CSV file path.

    Returns:
        Loaded pandas DataFrame.

    Raises:
        FileNotFoundError:
            If the CSV file does not exist.
    """

    csv_path = Path(path)

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}"
        )

    return pd.read_csv(csv_path)


def build_parser() -> argparse.ArgumentParser:
    """Build the application command-line interface."""

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--mode",
        choices=["direct", "graph"],
        default="direct",
        help=(
            "Execution mode. Use 'direct' for the existing "
            "pipeline or 'graph' for LangGraph."
        ),
    )

    parser.add_argument(
        "--flow",
        choices=[
            "kpi",
            "budget",
            "forecast",
            "full",
        ],
        default="full",
        help="Direct pipeline flow.",
    )

    parser.add_argument(
        "--request",
        default=None,
        help=(
            "Natural-language request used in graph mode. "
            'Example: "Show KPI performance".'
        ),
    )

    parser.add_argument(
        "--operations",
        default="data/operations/sample_orders.csv",
        help="Path to the operations CSV file.",
    )

    parser.add_argument(
        "--budget",
        default="data/planning/sample_budget.csv",
        help="Path to the budget CSV file.",
    )

    parser.add_argument(
        "--assumptions",
        default=(
            "data/assumptions/"
            "business_assumptions.csv"
        ),
        help="Path to the business assumptions CSV file.",
    )

    parser.add_argument(
        "--frequency",
        default="month",
        help="Analysis and forecast frequency.",
    )

    parser.add_argument(
        "--rolling-window",
        type=int,
        default=3,
        help="Forecast rolling-window size.",
    )

    parser.add_argument(
        "--forecast-periods",
        type=int,
        default=6,
        help="Number of future periods to forecast.",
    )

    parser.add_argument(
        "--scenario-name",
        default="Management Case",
        help="Scenario name used by scenario analysis.",
    )

    return parser


def run_direct_mode(
    args: argparse.Namespace,
) -> None:
    """
    Execute the existing deterministic pipeline.

    This function preserves the original main.py behavior.
    """

    pipeline = FinancePipeline()

    if args.flow == "kpi":
        result = pipeline.run_kpi_flow(
            operations_data=load_csv(
                args.operations
            ),
        )

        print(result.kpi_result)
        return

    if args.flow == "budget":
        result = pipeline.run_budget_flow(
            budget_data=load_csv(
                args.budget
            ),
        )

        print(result.budget_result)
        return

    if args.flow == "forecast":
        result = pipeline.run_forecast_flow(
            operations_data=load_csv(
                args.operations
            ),
            frequency=args.frequency,
            rolling_window=args.rolling_window,
            forecast_periods=args.forecast_periods,
        )

        print(result.forecast_result)
        return

    result = pipeline.run_full_analysis(
        operations_data=load_csv(
            args.operations
        ),
        budget_data=load_csv(
            args.budget
        ),
        assumptions=load_csv(
            args.assumptions
        ),
        frequency=args.frequency,
        rolling_window=args.rolling_window,
        forecast_periods=args.forecast_periods,
        scenario_name=args.scenario_name,
    )

    print(result.report_result.markdown_report)


def build_graph_state(
    args: argparse.Namespace,
) -> FinanceGraphState:
    """
    Build the initial LangGraph shared state.

    Only the files required by the selected graph route are loaded.
    """

    if (
        args.request is None
        or not args.request.strip()
    ):
        raise ValueError(
            "--request is required when "
            "--mode graph is used."
        )

    selected_flow = identify_flow(
        args.request
    )

    state: FinanceGraphState = {
        "user_request": args.request,
        "selected_flow": selected_flow,
        "execution_status": "pending",
        "errors": [],
        "error_message": "",
        "failed_node": "",
        "executed_nodes": [],
        "filters": {},
        "group_by": args.frequency,
        "frequency": args.frequency,
        "rolling_window": args.rolling_window,
        "forecast_periods": args.forecast_periods,
        "scenario_name": args.scenario_name,
    }

    operations_flows = {
        "kpi",
        "forecast",
        "variance",
        "scenario",
        "full",
    }

    budget_flows = {
        "budget",
        "variance",
        "full",
    }

    assumption_flows = {
        "scenario",
        "full",
    }

    if selected_flow in operations_flows:
        state["operations_data"] = load_csv(
            args.operations
        )

    if selected_flow in budget_flows:
        state["budget_data"] = load_csv(
            args.budget
        )

    if selected_flow in assumption_flows:
        state["business_assumptions"] = load_csv(
            args.assumptions
        )

    return state


def print_graph_result(
    result: FinanceGraphState,
) -> None:
    """Print the LangGraph execution result."""

    selected_flow = result.get(
        "selected_flow",
        "unknown",
    )

    execution_status = result.get(
        "execution_status",
        "failed",
    )

    print(
        f"Selected flow: {selected_flow}"
    )

    print(
        f"Execution status: {execution_status}"
    )

    executed_nodes = result.get(
        "executed_nodes",
        [],
    )

    if executed_nodes:
        print(
            "Executed nodes: "
            + " -> ".join(executed_nodes)
        )

    if execution_status == "failed":
        failed_node = result.get(
            "failed_node"
        )

        if failed_node:
            print(
                f"Failed node: {failed_node}"
            )

        error_message = result.get(
            "error_message",
            "Graph execution failed.",
        )

        print(
            f"Error: {error_message}"
        )

        errors = result.get(
            "errors",
            [],
        )

        if errors:
            print("Error history:")

            for error in errors:
                print(f"- {error}")

        return

    report_result = result.get(
        "report_result"
    )

    if report_result is not None:
        markdown_report = getattr(
            report_result,
            "markdown_report",
            None,
        )

        if markdown_report is not None:
            print(markdown_report)
        else:
            print(report_result)

        return

    commentary_result = result.get(
        "commentary_result"
    )

    if commentary_result is not None:
        print(commentary_result)
        return

    output_fields = (
        "kpi_result",
        "variance_result",
        "scenario_result",
        "forecast_result",
        "budget_result",
        "operations_result",
    )

    for field_name in output_fields:
        output = result.get(
            field_name
        )

        if output is not None:
            print(output)
            return

    print(
        "Graph completed without a printable result."
    )


def run_graph_mode(
    args: argparse.Namespace,
) -> None:
    """Build the initial state and execute LangGraph."""

    initial_state = build_graph_state(
        args
    )

    result = run_finance_graph(
        initial_state
    )

    print_graph_result(
        result
    )

    if result.get(
        "execution_status"
    ) == "failed":
        raise SystemExit(1)


def main() -> None:
    """Run the Finance Agentic AI System."""

    args = build_parser().parse_args()

    try:
        if args.mode == "graph":
            run_graph_mode(args)
        else:
            run_direct_mode(args)

    except (
        FileNotFoundError,
        ValueError,
        TypeError,
    ) as error:
        print(
            f"Execution failed: {error}"
        )

        raise SystemExit(1) from error


if __name__ == "__main__":
    main()