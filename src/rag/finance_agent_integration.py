"""
RAG integration for the Commentary and Recommendation agents.

This module enriches existing deterministic finance-agent outputs with
retrieved company context.

It does not:

- Replace existing finance agents
- Recalculate finance results
- Duplicate commentary or recommendation logic
- Modify LangGraph nodes
- Require an LLM
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Mapping

from src.agents.analytics.recommendation_agent import (
    RecommendationAgent,
    RecommendationResult,
)
from src.agents.reporting.commentary_agent import (
    CommentaryAgent,
    CommentaryResult,
)
from src.rag.prompt_templates import PromptType
from src.rag.rag_agent import (
    FinanceRAGAgent,
    RAGAgentError,
    RAGResult,
)


class FinanceAgentRAGIntegrationError(RuntimeError):
    """Raised when finance-agent RAG enrichment fails."""


@dataclass(frozen=True)
class RAGEnrichedCommentaryResult:
    """
    Combined deterministic commentary and RAG enrichment.

    Attributes:
        commentary_result:
            Original deterministic CommentaryAgent result.
        rag_result:
            Complete RAG execution result.
        enriched_commentary:
            RAG-generated management commentary.
    """

    commentary_result: CommentaryResult
    rag_result: RAGResult
    enriched_commentary: str

    def __post_init__(self) -> None:
        if not isinstance(
            self.commentary_result,
            CommentaryResult,
        ):
            raise TypeError(
                "commentary_result must be a CommentaryResult."
            )

        if not isinstance(self.rag_result, RAGResult):
            raise TypeError(
                "rag_result must be a RAGResult."
            )

        if not isinstance(self.enriched_commentary, str):
            raise TypeError(
                "enriched_commentary must be a string."
            )

        cleaned_commentary = self.enriched_commentary.strip()

        if not cleaned_commentary:
            raise ValueError(
                "enriched_commentary cannot be empty."
            )

        object.__setattr__(
            self,
            "enriched_commentary",
            cleaned_commentary,
        )

    @property
    def has_context(self) -> bool:
        """Return whether RAG retrieved supporting context."""

        return self.rag_result.has_context

    @property
    def used_fallback(self) -> bool:
        """Return whether deterministic RAG fallback was used."""

        return self.rag_result.used_fallback


@dataclass(frozen=True)
class RAGEnrichedRecommendationResult:
    """
    Combined deterministic recommendations and RAG enrichment.

    Attributes:
        recommendation_result:
            Original deterministic RecommendationAgent result.
        rag_result:
            Complete RAG execution result.
        enriched_recommendations:
            RAG-generated management recommendation text.
    """

    recommendation_result: RecommendationResult
    rag_result: RAGResult
    enriched_recommendations: str

    def __post_init__(self) -> None:
        if not isinstance(
            self.recommendation_result,
            RecommendationResult,
        ):
            raise TypeError(
                "recommendation_result must be a "
                "RecommendationResult."
            )

        if not isinstance(self.rag_result, RAGResult):
            raise TypeError(
                "rag_result must be a RAGResult."
            )

        if not isinstance(
            self.enriched_recommendations,
            str,
        ):
            raise TypeError(
                "enriched_recommendations must be a string."
            )

        cleaned_recommendations = (
            self.enriched_recommendations.strip()
        )

        if not cleaned_recommendations:
            raise ValueError(
                "enriched_recommendations cannot be empty."
            )

        object.__setattr__(
            self,
            "enriched_recommendations",
            cleaned_recommendations,
        )

    @property
    def has_context(self) -> bool:
        """Return whether RAG retrieved supporting context."""

        return self.rag_result.has_context

    @property
    def used_fallback(self) -> bool:
        """Return whether deterministic RAG fallback was used."""

        return self.rag_result.used_fallback


class RAGCommentaryAgent:
    """
    Run the existing CommentaryAgent and optionally enrich its output.

    The original CommentaryResult is always preserved.
    """

    def __init__(
        self,
        commentary_agent: CommentaryAgent,
        rag_agent: FinanceRAGAgent,
    ) -> None:
        if not isinstance(
            commentary_agent,
            CommentaryAgent,
        ):
            raise TypeError(
                "commentary_agent must be a CommentaryAgent."
            )

        if not isinstance(rag_agent, FinanceRAGAgent):
            raise TypeError(
                "rag_agent must be a FinanceRAGAgent."
            )

        self._commentary_agent = commentary_agent
        self._rag_agent = rag_agent

    @property
    def commentary_agent(self) -> CommentaryAgent:
        """Return the existing deterministic commentary agent."""

        return self._commentary_agent

    @property
    def rag_agent(self) -> FinanceRAGAgent:
        """Return the configured RAG agent."""

        return self._rag_agent

    def analyze(
        self,
        kpi_result: Any,
        revenue_variance_result: Any | None = None,
        forecast_result: Any | None = None,
        scenario_result: Any | None = None,
        finance_rules_result: Any | None = None,
        *,
        user_request: str = (
            "Prepare management commentary using the supplied "
            "finance results and relevant company context."
        ),
        retrieval_query: str | None = None,
        metadata_filter: Mapping[str, Any] | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
        additional_instructions: str | None = None,
    ) -> RAGEnrichedCommentaryResult:
        """
        Generate deterministic commentary and then enrich it with RAG.

        Existing CommentaryAgent behaviour remains unchanged.
        """

        commentary_result = self._commentary_agent.analyze(
            kpi_result=kpi_result,
            revenue_variance_result=revenue_variance_result,
            forecast_result=forecast_result,
            scenario_result=scenario_result,
            finance_rules_result=finance_rules_result,
        )

        return self.enrich(
            commentary_result=commentary_result,
            user_request=user_request,
            retrieval_query=retrieval_query,
            metadata_filter=metadata_filter,
            top_k=top_k,
            score_threshold=score_threshold,
            additional_instructions=additional_instructions,
        )

    def enrich(
        self,
        commentary_result: CommentaryResult,
        *,
        user_request: str = (
            "Enrich the management commentary using relevant "
            "company policies, assumptions, and historical guidance."
        ),
        retrieval_query: str | None = None,
        metadata_filter: Mapping[str, Any] | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
        additional_instructions: str | None = None,
    ) -> RAGEnrichedCommentaryResult:
        """
        Enrich an already-created CommentaryResult.

        This method is useful when commentary was generated earlier in the
        existing direct or LangGraph pipeline.
        """

        if not isinstance(
            commentary_result,
            CommentaryResult,
        ):
            raise TypeError(
                "commentary_result must be a CommentaryResult."
            )

        analysis_payload = _serialize_result(
            commentary_result
        )

        try:
            rag_result = self._rag_agent.run(
                user_request=user_request,
                finance_analysis=analysis_payload,
                prompt_type=PromptType.COMMENTARY,
                retrieval_query=retrieval_query,
                metadata_filter=metadata_filter,
                top_k=top_k,
                score_threshold=score_threshold,
                additional_instructions=(
                    additional_instructions
                ),
            )
        except RAGAgentError as exc:
            raise FinanceAgentRAGIntegrationError(
                f"Commentary RAG enrichment failed: {exc}"
            ) from exc

        return RAGEnrichedCommentaryResult(
            commentary_result=commentary_result,
            rag_result=rag_result,
            enriched_commentary=rag_result.response,
        )


class RAGRecommendationAgent:
    """
    Run the existing RecommendationAgent and enrich its output with RAG.

    The deterministic RecommendationResult is always preserved.
    """

    def __init__(
        self,
        recommendation_agent: RecommendationAgent,
        rag_agent: FinanceRAGAgent,
    ) -> None:
        if not isinstance(
            recommendation_agent,
            RecommendationAgent,
        ):
            raise TypeError(
                "recommendation_agent must be a "
                "RecommendationAgent."
            )

        if not isinstance(rag_agent, FinanceRAGAgent):
            raise TypeError(
                "rag_agent must be a FinanceRAGAgent."
            )

        self._recommendation_agent = recommendation_agent
        self._rag_agent = rag_agent

    @property
    def recommendation_agent(
        self,
    ) -> RecommendationAgent:
        """Return the deterministic recommendation agent."""

        return self._recommendation_agent

    @property
    def rag_agent(self) -> FinanceRAGAgent:
        """Return the configured RAG agent."""

        return self._rag_agent

    def analyze(
        self,
        root_cause_result: Any,
        *,
        user_request: str = (
            "Prepare evidence-based management recommendations "
            "using the root-cause analysis and relevant company "
            "policies."
        ),
        retrieval_query: str | None = None,
        metadata_filter: Mapping[str, Any] | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
        additional_instructions: str | None = None,
    ) -> RAGEnrichedRecommendationResult:
        """
        Generate deterministic recommendations and enrich them with RAG.

        Existing RecommendationAgent behaviour remains unchanged.
        """

        recommendation_result = (
            self._recommendation_agent.analyze(
                root_cause_result=root_cause_result
            )
        )

        return self.enrich(
            recommendation_result=recommendation_result,
            user_request=user_request,
            retrieval_query=retrieval_query,
            metadata_filter=metadata_filter,
            top_k=top_k,
            score_threshold=score_threshold,
            additional_instructions=additional_instructions,
        )

    def enrich(
        self,
        recommendation_result: RecommendationResult,
        *,
        user_request: str = (
            "Enrich the existing management recommendations using "
            "relevant company policies, operational constraints, "
            "and historical guidance."
        ),
        retrieval_query: str | None = None,
        metadata_filter: Mapping[str, Any] | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
        additional_instructions: str | None = None,
    ) -> RAGEnrichedRecommendationResult:
        """
        Enrich an already-created RecommendationResult.
        """

        if not isinstance(
            recommendation_result,
            RecommendationResult,
        ):
            raise TypeError(
                "recommendation_result must be a "
                "RecommendationResult."
            )

        analysis_payload = _serialize_result(
            recommendation_result
        )

        try:
            rag_result = self._rag_agent.run(
                user_request=user_request,
                finance_analysis=analysis_payload,
                prompt_type=PromptType.RECOMMENDATION,
                retrieval_query=retrieval_query,
                metadata_filter=metadata_filter,
                top_k=top_k,
                score_threshold=score_threshold,
                additional_instructions=(
                    additional_instructions
                ),
            )
        except RAGAgentError as exc:
            raise FinanceAgentRAGIntegrationError(
                f"Recommendation RAG enrichment failed: {exc}"
            ) from exc

        return RAGEnrichedRecommendationResult(
            recommendation_result=recommendation_result,
            rag_result=rag_result,
            enriched_recommendations=rag_result.response,
        )


def create_rag_commentary_agent(
    commentary_agent: CommentaryAgent,
    rag_agent: FinanceRAGAgent,
) -> RAGCommentaryAgent:
    """Create a RAG-enabled commentary wrapper."""

    return RAGCommentaryAgent(
        commentary_agent=commentary_agent,
        rag_agent=rag_agent,
    )


def create_rag_recommendation_agent(
    recommendation_agent: RecommendationAgent,
    rag_agent: FinanceRAGAgent,
) -> RAGRecommendationAgent:
    """Create a RAG-enabled recommendation wrapper."""

    return RAGRecommendationAgent(
        recommendation_agent=recommendation_agent,
        rag_agent=rag_agent,
    )


def _serialize_result(result: Any) -> dict[str, Any]:
    """
    Convert an existing finance-agent result into prompt-safe data.

    Dataclass results are converted recursively without changing the
    original result.
    """

    if is_dataclass(result) and not isinstance(
        result,
        type,
    ):
        serialized = asdict(result)

        if not isinstance(serialized, dict):
            raise FinanceAgentRAGIntegrationError(
                "Finance-agent result did not serialize to a "
                "dictionary."
            )

        return copy.deepcopy(serialized)

    to_dict_method = getattr(result, "to_dict", None)

    if callable(to_dict_method):
        try:
            serialized = to_dict_method()
        except Exception as exc:
            raise FinanceAgentRAGIntegrationError(
                f"Unable to serialize finance-agent result: {exc}"
            ) from exc

        if not isinstance(serialized, Mapping):
            raise FinanceAgentRAGIntegrationError(
                "to_dict() must return a mapping."
            )

        return copy.deepcopy(dict(serialized))

    if hasattr(result, "__dict__"):
        return {
            key: copy.deepcopy(value)
            for key, value in vars(result).items()
            if not key.startswith("_")
        }

    raise TypeError(
        "result must be a dataclass, expose to_dict(), "
        "or contain public attributes."
    )