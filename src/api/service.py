"""
Application service for the Finance Agentic AI API.

This module coordinates:

- Natural-language intent parsing
- Temporary clarification handling
- Deterministic execution planning
- LangGraph workflow execution
- Finance-agent result collection
- Dashboard response building
- Finance response-context generation
- RAG and OpenAI response generation

It must not contain finance calculation formulas or Streamlit UI logic.
"""

from __future__ import annotations

from dataclasses import (
    asdict,
    dataclass,
    field,
    is_dataclass,
)
from enum import Enum
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


class FinanceAskServiceError(RuntimeError):
    """Raised when the service cannot complete a finance request."""


@dataclass(frozen=True)
class FinanceDataPaths:
    """
    Paths to datasets used by the LangGraph workflow.

    These local CSV files support the current GitHub project. They can later
    be replaced with Snowflake, S3 or database sources without changing the
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
            Final AI response or clarification question.

        sources:
            Retrieved RAG source records.

        selected_flow:
            Finance workflow selected from the user request.

        execution_status:
            Workflow status such as completed or clarification_required.

        used_fallback:
            Whether the RAG fallback response was used.

        dashboard:
            Structured dashboard payload.

        clarification_required:
            Whether the user must supply missing information.

        intent:
            Serialized request intent detected from the user question.
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
    Coordinate intent parsing, LangGraph execution and RAG generation.

    Processing flow:

        User question
            │
            ▼
        Intent parser
            │
            ├── Missing information
            │       └── Return clarification question
            │
            └── Complete intent
                    │
                    ▼
              Load required data
                    │
                    ▼
              Create execution plan
                    │
                    ▼
              Execute LangGraph
                    │
                    ├── Dashboard response
                    │
                    └── RAG/OpenAI answer
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
                Paths to local operations, budget and assumption files.

            graph_executor:
                Existing LangGraph execution function. It remains injectable
                so tests can provide a fake executor.

        Raises:
            TypeError:
                If a dependency is invalid.
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
        Answer one natural-language finance question.

        If the intent parser finds missing required information, the method
        returns a clarification question without loading data, executing
        LangGraph or calling the RAG agent.

        Args:
            question:
                Natural-language finance request.

            top_k:
                Maximum number of RAG documents to retrieve.

            score_threshold:
                Optional minimum vector similarity score.

            metadata_filter:
                Optional document metadata filter.

        Returns:
            AskServiceResult containing either:

            - A clarification question, or
            - A completed AI answer and dashboard.

        Raises:
            FinanceAskServiceError:
                If planning, LangGraph, dashboard or RAG execution fails.
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
            for item
            in rag_result.retrieval_result.documents
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
        Build the initial LangGraph state from parsed intent.

        Only datasets required by the selected finance flow are loaded.

        The parsed period, comparison, category and KPI values are preserved
        inside ``filters`` for downstream agents.
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
            state["operations_data"] = (
                self._load_csv(
                    self._data_paths.operations
                )
            )

        if selected_flow in budget_flows:
            state["budget_data"] = (
                self._load_csv(
                    self._data_paths.budget
                )
            )

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
        """
        Return a clarification response without running the workflow.
        """

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
        """
        Resolve the grouping level from the parsed period.

        A date range remains monthly by default for management reporting.
        """

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
        """Resolve graph frequency from the parsed period."""

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

    @staticmethod
    def _resolve_prompt_type(
        selected_flow: str,
    ) -> PromptType:
        """
        Map a finance workflow to the matching RAG prompt type.
        """

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
                If the file is unavailable or cannot be loaded.
        """

        if not path.exists():
            raise FinanceAskServiceError(
                f"Required data file not found: {path}"
            )

        try:
            return pd.read_csv(path)
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
        Extract and serialize existing agent outputs from graph state.

        No financial calculation occurs here.
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
                state[field_name]
            )
            for field_name in result_fields
            if state.get(field_name) is not None
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

        if isinstance(value, pd.DataFrame):
            return value.to_dict(
                orient="records"
            )

        if isinstance(value, pd.Series):
            return value.to_dict()

        if isinstance(value, Enum):
            return value.value

        if isinstance(value, dict):
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
                FinanceAskService._serialize(item)
                for item in value
            ]

        if (
            hasattr(value, "item")
            and callable(value.item)
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
        """Validate and clean a submitted finance question."""

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