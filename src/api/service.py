"""
Application service for the Finance Agentic AI API.

This module coordinates:

- Natural-language finance intent parsing
- Clarification handling
- Source-data loading
- Period and category filtering
- Execution-plan creation
- LangGraph workflow execution
- Dashboard response generation
- Finance response-context generation
- RAG and OpenAI response generation

Finance calculation formulas and Streamlit UI logic must not be placed here.
"""

from __future__ import annotations

from dataclasses import (
    asdict,
    dataclass,
    field,
    is_dataclass,
)
from enum import Enum
import logging
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.api.dashboard_response import build_dashboard_response
from src.api.finance_response_context import (
    build_finance_response_context,
)
from src.orchestrator.graph import run_finance_graph
from src.orchestrator.intent_parser import (
    FinanceIntent,
    parse_finance_intent,
)
from src.orchestrator.planner import (
    create_execution_plan,
    validate_plan_against_state,
)
from src.orchestrator.state import FinanceGraphState
from src.rag.prompt_templates import PromptType
from src.rag.rag_agent import FinanceRAGAgent


logger = logging.getLogger(__name__)


class FinanceAskServiceError(RuntimeError):
    """Raised when the service cannot complete a finance request."""


@dataclass(frozen=True)
class FinanceDataPaths:
    """
    Paths to datasets used by the finance workflow.

    These local files support the current GitHub project. They can later be
    replaced by Snowflake, S3 or database sources without changing the public
    service contract.
    """

    operations: Path
    budget: Path
    assumptions: Path


@dataclass(frozen=True)
class AskServiceResult:
    """
    Framework-independent result returned by FinanceAskService.

    Attributes:
        answer:
            Final AI answer or clarification question.

        sources:
            Retrieved RAG source records.

        selected_flow:
            Finance workflow selected from the request.

        execution_status:
            Workflow execution status.

        used_fallback:
            Whether RAG fallback generation was used.

        dashboard:
            Structured dashboard payload.

        clarification_required:
            Whether more information is required from the user.

        intent:
            Serialized parsed finance intent.
    """

    answer: str
    sources: list[dict[str, Any]]
    selected_flow: str
    execution_status: str
    used_fallback: bool
    dashboard: dict[str, Any] = field(default_factory=dict)
    clarification_required: bool = False
    intent: dict[str, Any] = field(default_factory=dict)


GraphExecutor = Callable[
    [FinanceGraphState],
    FinanceGraphState,
]


class FinanceAskService:
    """
    Coordinate intent parsing, filtering, graph execution and RAG generation.

    Processing flow:

        User request
            │
            ▼
        Intent parser
            │
            ├── Incomplete intent
            │       └── Clarification response
            │
            └── Complete intent
                    │
                    ▼
              Load source data
                    │
                    ▼
              Apply period/category filters
                    │
                    ▼
              Create execution plan
                    │
                    ▼
              Execute LangGraph
                    │
                    ├── Dashboard response
                    │
                    └── RAG/OpenAI response
    """

    def __init__(
        self,
        rag_agent: FinanceRAGAgent,
        data_paths: FinanceDataPaths,
        graph_executor: GraphExecutor = run_finance_graph,
    ) -> None:
        """
        Initialize the finance application service.

        Args:
            rag_agent:
                Existing FinanceRAGAgent instance.

            data_paths:
                Paths to operations, budget and assumption files.

            graph_executor:
                LangGraph execution function. This remains injectable so
                tests can provide a fake executor.

        Raises:
            TypeError:
                If a supplied dependency is invalid.
        """

        if not hasattr(rag_agent, "run"):
            raise TypeError(
                "rag_agent must provide a callable run method."
            )

        if not callable(rag_agent.run):
            raise TypeError(
                "rag_agent.run must be callable."
            )

        if not isinstance(
            data_paths,
            FinanceDataPaths,
        ):
            raise TypeError(
                "data_paths must be FinanceDataPaths."
            )

        if not callable(graph_executor):
            raise TypeError(
                "graph_executor must be callable."
            )

        self._rag_agent = rag_agent
        self._data_paths = data_paths
        self._graph_executor = graph_executor

    def ask(
        self,
        question: str,
        *,
        top_k: int | None = None,
        score_threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> AskServiceResult:
        """
        Answer one natural-language finance request.

        If required information is missing, a clarification response is
        returned without loading data, executing LangGraph or calling RAG.

        Args:
            question:
                Natural-language finance request.

            top_k:
                Maximum number of RAG documents to retrieve.

            score_threshold:
                Optional minimum vector similarity score.

            metadata_filter:
                Optional RAG document metadata filter.

        Returns:
            AskServiceResult containing a clarification response or completed
            finance analysis.

        Raises:
            FinanceAskServiceError:
                If parsing, planning, data filtering, graph execution,
                dashboard generation or RAG generation fails.
        """

        cleaned_question = self._validate_question(
            question
        )

        try:
            intent = parse_finance_intent(
                cleaned_question
            )
        except (TypeError, ValueError) as exc:
            raise FinanceAskServiceError(
                f"Unable to understand finance request: {exc}"
            ) from exc

        if not intent.is_complete:
            return self._build_clarification_result(
                intent
            )

        graph_state = self._build_graph_state(
            question=cleaned_question,
            intent=intent,
        )

        selected_flow = str(
            graph_state.get(
                "selected_flow",
                intent.selected_flow,
            )
        )

        try:
            execution_plan = create_execution_plan(
                selected_flow
            )
        except (TypeError, ValueError) as exc:
            raise FinanceAskServiceError(
                "Unable to create finance execution plan: "
                f"{exc}"
            ) from exc

        missing_state_fields = (
            validate_plan_against_state(
                execution_plan,
                graph_state,
            )
        )

        if missing_state_fields:
            missing_text = ", ".join(
                missing_state_fields
            )

            raise FinanceAskServiceError(
                "Finance workflow is missing required state fields: "
                f"{missing_text}."
            )

        try:
            graph_result = self._graph_executor(
                graph_state
            )
        except Exception as exc:
            raise FinanceAskServiceError(
                f"Finance graph execution failed: {exc}"
            ) from exc

        execution_status = str(
            graph_result.get(
                "execution_status",
                "failed",
            )
        )

        if execution_status == "failed":
            error_message = (
                graph_result.get("error_message")
                or "Finance graph execution failed."
            )

            raise FinanceAskServiceError(
                str(error_message)
            )

        selected_flow = str(
            graph_result.get(
                "selected_flow",
                selected_flow,
            )
        )

        raw_finance_analysis = (
            self._extract_finance_analysis(
                graph_result
            )
        )

        dashboard = self._build_dashboard(
            selected_flow=selected_flow,
            finance_analysis=raw_finance_analysis,
            intent=intent,
        )

        finance_analysis = (
            build_finance_response_context(
                selected_flow=selected_flow,
                finance_analysis=raw_finance_analysis,
            )
        )

        prompt_type = self._resolve_prompt_type(
            selected_flow
        )

        try:
            rag_result = self._rag_agent.run(
                user_request=cleaned_question,
                finance_analysis=finance_analysis,
                retrieval_query=cleaned_question,
                top_k=top_k,
                score_threshold=score_threshold,
                metadata_filter=metadata_filter or {},
                prompt_type=prompt_type,
            )
        except Exception as exc:
            raise FinanceAskServiceError(
                f"RAG answer generation failed: {exc}"
            ) from exc

        sources = [
            {
                "id": item.document.id,
                "score": float(item.score),
                "rank": item.rank,
                "metadata": dict(
                    item.document.metadata
                ),
                "excerpt": self._excerpt(
                    item.document.text
                ),
            }
            for item in rag_result.retrieval_result.documents
        ]

        return AskServiceResult(
            answer=rag_result.response,
            sources=sources,
            selected_flow=selected_flow,
            execution_status=execution_status,
            used_fallback=rag_result.used_fallback,
            dashboard=dashboard,
            clarification_required=False,
            intent=self._serialize_intent(
                intent
            ),
        )

    def _build_graph_state(
        self,
        *,
        question: str,
        intent: FinanceIntent,
    ) -> FinanceGraphState:
        """
        Build initial LangGraph state from the parsed finance intent.

        Operations and budget datasets are filtered before they are placed
        into graph state. Therefore, downstream agents receive only records
        belonging to the requested period and category.
        """

        selected_flow = intent.selected_flow
        filters = intent.to_filters()

        state: FinanceGraphState = {
            "user_request": question,
            "selected_flow": selected_flow,
            "execution_status": "pending",
            "errors": [],
            "error_message": "",
            "failed_node": "",
            "executed_nodes": [],
            "filters": filters,
            "group_by": self._resolve_group_by(
                intent
            ),
            "frequency": self._resolve_frequency(
                intent
            ),
            "rolling_window": 3,
            "forecast_periods": 6,
            "scenario_name": (
                intent.scenario_name
                or "Management Case"
            ),
        }

        if intent.requested_kpis:
            state["requested_kpis"] = list(
                intent.requested_kpis
            )

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
            operations_data = self._load_csv(
                self._data_paths.operations
            )

            filtered_operations = self._apply_intent_filters(
                dataframe=operations_data,
                intent=intent,
                dataset_name="operations",
            )

            self._log_filtered_dataset(
                dataframe=filtered_operations,
                dataset_name="operations",
                intent=intent,
            )

            state["operations_data"] = filtered_operations

        if selected_flow in budget_flows:
            budget_data = self._load_csv(
                self._data_paths.budget
            )

            filtered_budget = self._apply_intent_filters(
                dataframe=budget_data,
                intent=intent,
                dataset_name="budget",
            )

            self._log_filtered_dataset(
                dataframe=filtered_budget,
                dataset_name="budget",
                intent=intent,
            )

            state["budget_data"] = filtered_budget

        if selected_flow in assumption_flows:
            state["business_assumptions"] = (
                self._load_csv(
                    self._data_paths.assumptions
                )
            )

        return state

    @staticmethod
    def _build_clarification_result(
        intent: FinanceIntent,
    ) -> AskServiceResult:
        """Return a clarification response without executing the workflow."""

        clarification = (
            intent.clarification_question
            or (
                "Please provide the missing information "
                "required for this finance analysis."
            )
        )

        return AskServiceResult(
            answer=clarification,
            sources=[],
            selected_flow=str(
                intent.selected_flow
            ),
            execution_status="clarification_required",
            used_fallback=False,
            dashboard={},
            clarification_required=True,
            intent=FinanceAskService._serialize_intent(
                intent
            ),
        )

    @staticmethod
    def _build_dashboard(
        *,
        selected_flow: str,
        finance_analysis: dict[str, Any],
        intent: FinanceIntent,
    ) -> dict[str, Any]:
        """
        Build dashboard-ready data from existing agent outputs.

        Raises:
            FinanceAskServiceError:
                If dashboard normalization fails.
        """

        try:
            dashboard_payload = build_dashboard_response(
                selected_flow=selected_flow,
                finance_analysis=finance_analysis,
                period=intent.period.display_value,
                analysis_type=selected_flow,
                comparison=(
                    None
                    if intent.comparison == "none"
                    else intent.comparison
                ),
                category=(
                    intent.category
                    or "All Categories"
                ),
            )
        except (TypeError, ValueError) as exc:
            raise FinanceAskServiceError(
                "Dashboard response generation failed: "
                f"{exc}"
            ) from exc

        return dashboard_payload.model_dump(
            mode="python"
        )

    @staticmethod
    def _resolve_group_by(
        intent: FinanceIntent,
    ) -> str:
        """Resolve the reporting grouping level from parsed period."""

        grouping_by_granularity = {
            "day": "day",
            "month": "month",
            "year": "month",
            "range": "month",
            "unknown": "month",
        }

        return grouping_by_granularity.get(
            intent.period.granularity,
            "month",
        )

    @staticmethod
    def _resolve_frequency(
        intent: FinanceIntent,
    ) -> str:
        """Resolve graph execution frequency from parsed period."""

        frequency_by_granularity = {
            "day": "day",
            "month": "month",
            "year": "month",
            "range": "month",
            "unknown": "month",
        }

        return frequency_by_granularity.get(
            intent.period.granularity,
            "month",
        )

    @classmethod
    def _apply_intent_filters(
        cls,
        *,
        dataframe: pd.DataFrame,
        intent: FinanceIntent,
        dataset_name: str,
    ) -> pd.DataFrame:
        """
        Apply parsed period and category filters before graph execution.

        Args:
            dataframe:
                Raw source dataframe.

            intent:
                Structured finance intent.

            dataset_name:
                Human-readable source name used in validation errors.

        Returns:
            Filtered dataframe with index reset.

        Raises:
            FinanceAskServiceError:
                If data is invalid, a requested filter cannot be applied or
                no matching rows remain after filtering.
        """

        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):
            raise FinanceAskServiceError(
                f"{dataset_name} data must be a pandas DataFrame."
            )

        if dataframe.empty:
            raise FinanceAskServiceError(
                f"{dataset_name} data is empty."
            )

        filtered_data = dataframe.copy()

        filtered_data = cls._filter_by_period(
            dataframe=filtered_data,
            intent=intent,
            dataset_name=dataset_name,
        )

        filtered_data = cls._filter_by_category(
            dataframe=filtered_data,
            intent=intent,
            dataset_name=dataset_name,
        )

        if filtered_data.empty:
            period_label = (
                intent.period.display_value
                or "requested period"
            )

            category_label = (
                intent.category
                or "All Categories"
            )

            raise FinanceAskServiceError(
                "No matching records were found in "
                f"{dataset_name} data for period "
                f"'{period_label}' and category "
                f"'{category_label}'."
            )

        return filtered_data.reset_index(
            drop=True
        )

    @classmethod
    def _filter_by_period(
        cls,
        *,
        dataframe: pd.DataFrame,
        intent: FinanceIntent,
        dataset_name: str,
    ) -> pd.DataFrame:
        """Filter data using the parsed inclusive start and end dates."""

        start_date = intent.period.start_date
        end_date = intent.period.end_date

        if not start_date or not end_date:
            return dataframe

        parsed_dates = cls._extract_dataframe_dates(
            dataframe
        )

        if parsed_dates is None:
            raise FinanceAskServiceError(
                "Unable to apply the requested reporting period because "
                f"{dataset_name} data does not contain a supported date, "
                "month, period or year column."
            )

        try:
            start_timestamp = pd.Timestamp(
                start_date
            ).normalize()

            end_timestamp = pd.Timestamp(
                end_date
            ).normalize()
        except (TypeError, ValueError) as exc:
            raise FinanceAskServiceError(
                "Unable to apply the requested reporting period because "
                "the parsed start or end date is invalid."
            ) from exc

        normalized_dates = pd.to_datetime(
            parsed_dates,
            errors="coerce",
        ).dt.normalize()

        period_mask = (
            normalized_dates.notna()
            & normalized_dates.ge(
                start_timestamp
            )
            & normalized_dates.le(
                end_timestamp
            )
        )

        return dataframe.loc[
            period_mask
        ].copy()

    @classmethod
    def _extract_dataframe_dates(
        cls,
        dataframe: pd.DataFrame,
    ) -> pd.Series | None:
        """
        Extract reporting dates from common source-data structures.

        Supported structures include:

        - A direct date column
        - A month or period column
        - A year-month column
        - Separate year and month columns
        """

        direct_date_column = cls._find_dataframe_column(
            dataframe=dataframe,
            candidates=(
                "date",
                "order_date",
                "transaction_date",
                "report_date",
                "reporting_date",
                "business_date",
                "month_date",
            ),
        )

        if direct_date_column is not None:
            return cls._parse_dataframe_dates(
                dataframe[
                    direct_date_column
                ]
            )

        period_column = cls._find_dataframe_column(
            dataframe=dataframe,
            candidates=(
                "period",
                "reporting_period",
                "month",
                "year_month",
                "yearmonth",
                "month_year",
                "monthyear",
            ),
        )

        if period_column is not None:
            return cls._parse_dataframe_dates(
                dataframe[
                    period_column
                ]
            )

        year_column = cls._find_dataframe_column(
            dataframe=dataframe,
            candidates=(
                "year",
                "reporting_year",
                "financial_year",
                "calendar_year",
            ),
        )

        month_column = cls._find_dataframe_column(
            dataframe=dataframe,
            candidates=(
                "month",
                "month_number",
                "month_name",
                "reporting_month",
                "calendar_month",
            ),
        )

        if (
            year_column is not None
            and month_column is not None
        ):
            return cls._parse_year_and_month_columns(
                years=dataframe[
                    year_column
                ],
                months=dataframe[
                    month_column
                ],
            )

        if year_column is not None:
            year_values = pd.to_numeric(
                dataframe[
                    year_column
                ],
                errors="coerce",
            )

            return pd.to_datetime(
                {
                    "year": year_values,
                    "month": 1,
                    "day": 1,
                },
                errors="coerce",
            )

        return None

    @staticmethod
    def _parse_dataframe_dates(
        values: pd.Series,
    ) -> pd.Series:
        """
        Parse operational dates and monthly reporting periods.

        Current project formats:
        - Operations: DD-MM-YYYY
        - Budget: YYYY-MM
        """

        text_values = (
            values
            .fillna("")
            .astype(str)
            .str.strip()
        )

        parsed_dates = pd.Series(
            pd.NaT,
            index=text_values.index,
            dtype="datetime64[ns]",
        )

        formats = (
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%Y-%m",
            "%Y/%m",
            "%m/%Y",
            "%B %Y",
            "%b %Y",
            "%B-%Y",
            "%b-%Y",
        )

        for date_format in formats:
            unresolved_mask = (
                parsed_dates.isna()
                & text_values.ne("")
            )

            if not unresolved_mask.any():
                break

            converted_values = pd.to_datetime(
                text_values.loc[unresolved_mask],
                format=date_format,
                errors="coerce",
            )

            parsed_dates.loc[
                unresolved_mask
            ] = converted_values

        unresolved_mask = (
            parsed_dates.isna()
            & text_values.ne("")
        )

        if unresolved_mask.any():
            parsed_dates.loc[
                unresolved_mask
            ] = pd.to_datetime(
                text_values.loc[unresolved_mask],
                errors="coerce",
                dayfirst=True,
            )

        return parsed_dates

    @classmethod
    def _parse_year_and_month_columns(
        cls,
        *,
        years: pd.Series,
        months: pd.Series,
    ) -> pd.Series:
        """Convert separate year and month columns into timestamps."""

        parsed_years = pd.to_numeric(
            years,
            errors="coerce",
        )

        parsed_months = months.map(
            cls._parse_month_value
        )

        return pd.to_datetime(
            {
                "year": parsed_years,
                "month": parsed_months,
                "day": 1,
            },
            errors="coerce",
        )

    @staticmethod
    def _parse_month_value(
        value: Any,
    ) -> float:
        """Convert numeric or named month values into month numbers."""

        if pd.isna(value):
            return float("nan")

        cleaned_value = str(
            value
        ).strip()

        if not cleaned_value:
            return float("nan")

        try:
            numeric_month = int(
                float(cleaned_value)
            )

            if 1 <= numeric_month <= 12:
                return float(
                    numeric_month
                )
        except ValueError:
            pass

        month_mapping = {
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

        month_number = month_mapping.get(
            cleaned_value.lower()
        )

        if month_number is None:
            return float("nan")

        return float(
            month_number
        )

    @classmethod
    def _filter_by_category(
        cls,
        *,
        dataframe: pd.DataFrame,
        intent: FinanceIntent,
        dataset_name: str,
    ) -> pd.DataFrame:
        """Filter source data by requested business category."""

        requested_category = intent.category

        if not requested_category:
            return dataframe

        category_column = cls._find_dataframe_column(
            dataframe=dataframe,
            candidates=(
                "category",
                "vehicle_category",
                "service_category",
                "business_category",
                "product",
                "product_category",
                "segment",
                "vehicle_type",
                "service_type",
            ),
        )

        if category_column is None:
            raise FinanceAskServiceError(
                "Unable to apply category filter "
                f"'{requested_category}' because {dataset_name} data "
                "does not contain a supported category column."
            )

        requested_value = cls._normalize_filter_value(
            requested_category
        )

        accepted_aliases = cls._category_aliases(
            requested_value
        )

        normalized_categories = (
            dataframe[
                category_column
            ]
            .fillna("")
            .astype(str)
            .map(
                cls._normalize_filter_value
            )
        )

        category_mask = normalized_categories.isin(
            accepted_aliases
        )

        return dataframe.loc[
            category_mask
        ].copy()

    @staticmethod
    def _find_dataframe_column(
        *,
        dataframe: pd.DataFrame,
        candidates: tuple[str, ...],
    ) -> str | None:
        """Find a dataframe column using normalized column names."""

        normalized_columns = {
            FinanceAskService._normalize_column_name(
                column
            ): str(column)
            for column in dataframe.columns
        }

        for candidate in candidates:
            normalized_candidate = (
                FinanceAskService
                ._normalize_column_name(
                    candidate
                )
            )

            matched_column = normalized_columns.get(
                normalized_candidate
            )

            if matched_column is not None:
                return matched_column

        return None

    @staticmethod
    def _normalize_column_name(
        value: Any,
    ) -> str:
        """Normalize a dataframe column name for matching."""

        return (
            str(value)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
            .replace("%", "percentage")
        )

    @staticmethod
    def _normalize_filter_value(
        value: Any,
    ) -> str:
        """Normalize category values for case-insensitive matching."""

        return " ".join(
            str(value)
            .strip()
            .lower()
            .replace("-", " ")
            .replace("_", " ")
            .replace("&", " and ")
            .split()
        )

    @staticmethod
    def _category_aliases(
        normalized_category: str,
    ) -> set[str]:
        """Return supported aliases for common business categories."""

        alias_groups = (
            {
                "2w",
                "2 wheeler",
                "two wheeler",
                "2 wheelers",
                "two wheelers",
            },
            {
                "3w",
                "3 wheeler",
                "three wheeler",
                "3 wheelers",
                "three wheelers",
            },
            {
                "compact auto",
                "compactauto",
                "compact autos",
            },
            {
                "tata ace open",
                "tata ace open body",
            },
            {
                "tata ace close",
                "tata ace closed",
                "tata ace closed body",
            },
            {
                "tata ace",
            },
            {
                "packer and movers",
                "packers and movers",
                "packer movers",
                "packers movers",
            },
        )

        for alias_group in alias_groups:
            if normalized_category in alias_group:
                return alias_group

        return {
            normalized_category
        }

    @classmethod
    def _log_filtered_dataset(
        cls,
        *,
        dataframe: pd.DataFrame,
        dataset_name: str,
        intent: FinanceIntent,
    ) -> None:
        """Log a compact summary after period/category filtering."""

        summary: dict[str, Any] = {
            "dataset": dataset_name,
            "period": intent.period.display_value,
            "category": intent.category or "All Categories",
            "rows": int(len(dataframe)),
        }

        category_column = cls._find_dataframe_column(
            dataframe=dataframe,
            candidates=(
                "vehicle_category",
                "category",
                "service_category",
                "business_category",
                "product_category",
                "segment",
            ),
        )

        if category_column is not None:
            summary["categories"] = (
                dataframe[category_column]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

        if dataset_name == "operations":
            if (
                "fare" in dataframe.columns
                and "order_status" in dataframe.columns
            ):
                completed_mask = (
                    dataframe["order_status"]
                    .fillna("")
                    .astype(str)
                    .str.casefold()
                    .eq("completed")
                )
                summary["completed_revenue"] = float(
                    pd.to_numeric(
                        dataframe.loc[completed_mask, "fare"],
                        errors="coerce",
                    ).fillna(0).sum()
                )

        if dataset_name == "budget":
            if "budget_orders" in dataframe.columns:
                summary["budget_orders"] = float(
                    pd.to_numeric(
                        dataframe["budget_orders"],
                        errors="coerce",
                    ).fillna(0).sum()
                )

            if "budget_revenue" in dataframe.columns:
                summary["budget_revenue"] = float(
                    pd.to_numeric(
                        dataframe["budget_revenue"],
                        errors="coerce",
                    ).fillna(0).sum()
                )

        logger.info(
            "Finance input filter applied: %s",
            summary,
        )

    @staticmethod
    def _resolve_prompt_type(
        selected_flow: str,
    ) -> PromptType:
        """Map a finance workflow to its corresponding RAG prompt type."""

        normalized_flow = (
            str(selected_flow)
            .strip()
            .lower()
        )

        prompt_type_by_flow = {
            "kpi": PromptType.KPI_EXPLANATION,
            "budget": PromptType.BUDGET_ANALYSIS,
            "forecast": PromptType.FORECAST_ANALYSIS,
            "variance": PromptType.VARIANCE_ANALYSIS,
            "root_cause": PromptType.ROOT_CAUSE_ANALYSIS,
            "scenario": PromptType.SCENARIO_ANALYSIS,
            "recommendation": PromptType.RECOMMENDATION,
            "commentary": PromptType.COMMENTARY,
            "full": PromptType.COMMENTARY,
        }

        return prompt_type_by_flow.get(
            normalized_flow,
            PromptType.FINANCE_QA,
        )

    @staticmethod
    def _load_csv(
        path: Path,
    ) -> pd.DataFrame:
        """
        Load one CSV dataset.

        Raises:
            FinanceAskServiceError:
                If the file is missing or cannot be loaded.
        """

        if not path.exists():
            raise FinanceAskServiceError(
                f"Required data file not found: {path}"
            )

        try:
            return pd.read_csv(
                path
            )
        except Exception as exc:
            raise FinanceAskServiceError(
                f"Unable to load data file "
                f"'{path}': {exc}"
            ) from exc

    @staticmethod
    def _extract_finance_analysis(
        state: FinanceGraphState,
    ) -> dict[str, Any]:
        """
        Extract and serialize agent outputs from graph state.

        No finance calculation occurs in this method.
        """

        result_fields = (
            "operations_result",
            "budget_result",
            "forecast_result",
            "scenario_result",
            "variance_result",
            "finance_rules_result",
            "anomaly_result",
            "root_cause_result",
            "recommendation_result",
            "kpi_result",
            "commentary_result",
            "report_result",
        )

        return {
            field_name: FinanceAskService._serialize(
                state[
                    field_name
                ]
            )
            for field_name in result_fields
            if state.get(
                field_name
            ) is not None
        }

    @staticmethod
    def _serialize_intent(
        intent: FinanceIntent,
    ) -> dict[str, Any]:
        """Serialize FinanceIntent into JSON-compatible values."""

        return {
            "original_request": intent.original_request,
            "selected_flow": str(
                intent.selected_flow
            ),
            "comparison": intent.comparison,
            "period": {
                "start_date": intent.period.start_date,
                "end_date": intent.period.end_date,
                "display_value": intent.period.display_value,
                "granularity": intent.period.granularity,
            },
            "category": intent.category,
            "scenario_name": intent.scenario_name,
            "requested_kpis": list(
                intent.requested_kpis
            ),
            "missing_fields": list(
                intent.missing_fields
            ),
            "clarification_question": (
                intent.clarification_question
            ),
            "is_complete": intent.is_complete,
            "filters": intent.to_filters(),
        }

    @staticmethod
    def _serialize(
        value: Any,
    ) -> Any:
        """Convert agent results into JSON-compatible Python values."""

        if is_dataclass(value):
            return FinanceAskService._serialize(
                asdict(value)
            )

        if isinstance(
            value,
            pd.DataFrame,
        ):
            return value.to_dict(
                orient="records"
            )

        if isinstance(
            value,
            pd.Series,
        ):
            return value.to_dict()

        if isinstance(
            value,
            Enum,
        ):
            return value.value

        if isinstance(
            value,
            dict,
        ):
            return {
                str(key): FinanceAskService._serialize(
                    item
                )
                for key, item in value.items()
            }

        if isinstance(
            value,
            (list, tuple, set),
        ):
            return [
                FinanceAskService._serialize(
                    item
                )
                for item in value
            ]

        if (
            hasattr(
                value,
                "item",
            )
            and callable(
                value.item
            )
        ):
            try:
                return value.item()
            except (ValueError, TypeError):
                pass

        return value

    @staticmethod
    def _validate_question(
        question: str,
    ) -> str:
        """Validate and normalize a submitted finance question."""

        if not isinstance(
            question,
            str,
        ):
            raise TypeError(
                "question must be a string."
            )

        cleaned_question = " ".join(
            question.split()
        )

        if not cleaned_question:
            raise ValueError(
                "question cannot be empty."
            )

        return cleaned_question

    @staticmethod
    def _excerpt(
        text: str,
        limit: int = 300,
    ) -> str:
        """Produce a short single-line source excerpt."""

        cleaned_text = " ".join(
            text.split()
        )

        if len(cleaned_text) <= limit:
            return cleaned_text

        return (
            f"{cleaned_text[: limit - 3]}..."
        )