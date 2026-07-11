"""
Tests for RAG integration with CommentaryAgent and RecommendationAgent.

The integration tests verify that:

- Existing deterministic finance-agent results are preserved
- RAG enrichment uses the correct prompt type
- Retrieval parameters are forwarded correctly
- Existing finance-agent methods are called only once
- RAG failures are converted into integration errors
- Factory helpers work
- Finance results are serialized without mutation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
from unittest.mock import MagicMock

import pytest

from src.agents.analytics.recommendation_agent import (
    RecommendationAgent,
    RecommendationResult,
)
from src.agents.reporting.commentary_agent import (
    CommentaryAgent,
    CommentaryResult,
)
from src.rag.finance_agent_integration import (
    FinanceAgentRAGIntegrationError,
    RAGCommentaryAgent,
    RAGEnrichedCommentaryResult,
    RAGEnrichedRecommendationResult,
    RAGRecommendationAgent,
    _serialize_result,
    create_rag_commentary_agent,
    create_rag_recommendation_agent,
)
from src.rag.prompt_templates import PromptMessages, PromptType
from src.rag.rag_agent import (
    ExecutionMode,
    FinanceRAGAgent,
    RAGAgentError,
    RAGResult,
)
from src.rag.retriever import RetrievalResult


def create_instance_without_init(
    object_type: type[Any],
    **attributes: Any,
) -> Any:
    """
    Create an instance without calling its constructor.

    This keeps the integration tests independent of the completed finance
    agents' internal constructor signatures.
    """

    instance = object.__new__(object_type)

    for name, value in attributes.items():
        object.__setattr__(instance, name, value)

    return instance


def build_commentary_result() -> CommentaryResult:
    """Create a valid CommentaryResult instance."""

    return CommentaryResult(
        executive_summary="Revenue exceeded budget.",
        kpi_commentary=[
            "Revenue KPI performed above budget.",
        ],
        variance_commentary=[
            "Higher volume supported revenue performance.",
        ],
        forecast_commentary=[
            "Forecast performance remains stable.",
        ],
        scenario_commentary=[
            "The base scenario remains achievable.",
        ],
        control_commentary=[
            "No material finance-control exceptions were identified.",
        ],
        positive_drivers=[
            "Higher volume",
            "Improved customer mix",
        ],
        risks=[
            "Cost inflation",
        ],
        management_attention=[
            "Monitor revenue and cost performance.",
        ],
        source_kpis=[
            {
                "name": "revenue",
                "actual": 1200,
                "budget": 1000,
            }
        ],
    )


def build_recommendation_result() -> RecommendationResult:
    """Create a valid RecommendationResult instance."""

    return RecommendationResult(
        overall_status="action_required",
        root_causes_received=2,
        root_causes_processed=2,
        recommendation_count=2,
        critical_priority_count=0,
        high_priority_count=1,
        monitor_count=1,
        recommendations=[],
        unresolved_root_causes=[],
    )


def build_retrieval_result(
    *,
    has_context: bool = True,
) -> RetrievalResult:
    """Create a valid retrieval result for integration tests."""

    if has_context:
        context = (
            "Source 1\n"
            "Document ID: policy-1\n"
            "Content:\n"
            "Management must investigate material variances."
        )
    else:
        context = ""

    return RetrievalResult(
        query="finance management policy",
        documents=(),
        context=context,
        metadata_filter={},
        total_results=0,
    )


def build_rag_result(
    *,
    prompt_type: PromptType,
    response: str,
    has_context: bool = True,
    used_fallback: bool = False,
) -> RAGResult:
    """Create a lightweight valid RAGResult."""

    return RAGResult(
        response=response,
        user_request="Prepare management output",
        prompt_type=prompt_type,
        execution_mode=ExecutionMode.DETERMINISTIC,
        retrieval_result=build_retrieval_result(
            has_context=has_context
        ),
        prompt_messages=PromptMessages(
            system="Finance system instruction",
            user="Finance user prompt",
        ),
        used_fallback=used_fallback,
        success=True,
        errors=(),
    )


def build_commentary_agent_mock() -> CommentaryAgent:
    """Create a CommentaryAgent-compatible mock."""

    return MagicMock(spec=CommentaryAgent)


def build_recommendation_agent_mock() -> RecommendationAgent:
    """Create a RecommendationAgent-compatible mock."""

    return MagicMock(spec=RecommendationAgent)


def build_rag_agent_mock() -> FinanceRAGAgent:
    """Create a FinanceRAGAgent-compatible mock."""

    return MagicMock(spec=FinanceRAGAgent)


def test_enriched_commentary_result_creation() -> None:
    """Combined commentary result should retain all outputs."""

    commentary_result = build_commentary_result()
    rag_result = build_rag_result(
        prompt_type=PromptType.COMMENTARY,
        response="Enriched management commentary.",
    )

    result = RAGEnrichedCommentaryResult(
        commentary_result=commentary_result,
        rag_result=rag_result,
        enriched_commentary=(
            "  Enriched management commentary.  "
        ),
    )

    assert result.commentary_result is commentary_result
    assert result.rag_result is rag_result
    assert (
        result.enriched_commentary
        == "Enriched management commentary."
    )


def test_enriched_commentary_result_properties() -> None:
    """Commentary wrapper should expose RAG status properties."""

    result = RAGEnrichedCommentaryResult(
        commentary_result=build_commentary_result(),
        rag_result=build_rag_result(
            prompt_type=PromptType.COMMENTARY,
            response="Enriched commentary.",
            used_fallback=True,
        ),
        enriched_commentary="Enriched commentary.",
    )

    assert result.used_fallback is True
    assert result.has_context is False


def test_enriched_commentary_rejects_invalid_result() -> None:
    """Commentary result must use the completed result type."""

    with pytest.raises(
        TypeError,
        match="CommentaryResult",
    ):
        RAGEnrichedCommentaryResult(
            commentary_result="invalid",  # type: ignore[arg-type]
            rag_result=build_rag_result(
                prompt_type=PromptType.COMMENTARY,
                response="Commentary",
            ),
            enriched_commentary="Commentary",
        )


@pytest.mark.parametrize("value", ["", " ", "   "])
def test_enriched_commentary_rejects_empty_text(
    value: str,
) -> None:
    """Enriched commentary cannot be blank."""

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        RAGEnrichedCommentaryResult(
            commentary_result=build_commentary_result(),
            rag_result=build_rag_result(
                prompt_type=PromptType.COMMENTARY,
                response="Valid response",
            ),
            enriched_commentary=value,
        )


def test_enriched_recommendation_result_creation() -> None:
    """Combined recommendation result should retain all outputs."""

    recommendation_result = build_recommendation_result()
    rag_result = build_rag_result(
        prompt_type=PromptType.RECOMMENDATION,
        response="Enriched recommendations.",
    )

    result = RAGEnrichedRecommendationResult(
        recommendation_result=recommendation_result,
        rag_result=rag_result,
        enriched_recommendations=(
            "  Enriched recommendations.  "
        ),
    )

    assert result.recommendation_result is recommendation_result
    assert result.rag_result is rag_result
    assert (
        result.enriched_recommendations
        == "Enriched recommendations."
    )


def test_enriched_recommendation_result_properties() -> None:
    """Recommendation wrapper should expose RAG properties."""

    result = RAGEnrichedRecommendationResult(
        recommendation_result=build_recommendation_result(),
        rag_result=build_rag_result(
            prompt_type=PromptType.RECOMMENDATION,
            response="Enriched recommendations.",
            used_fallback=True,
        ),
        enriched_recommendations="Enriched recommendations.",
    )

    assert result.used_fallback is True
    assert result.has_context is False


def test_enriched_recommendation_rejects_invalid_result() -> None:
    """Recommendation result must use the completed result type."""

    with pytest.raises(
        TypeError,
        match="RecommendationResult",
    ):
        RAGEnrichedRecommendationResult(
            recommendation_result="invalid",  # type: ignore[arg-type]
            rag_result=build_rag_result(
                prompt_type=PromptType.RECOMMENDATION,
                response="Recommendations",
            ),
            enriched_recommendations="Recommendations",
        )


@pytest.mark.parametrize("value", ["", " ", "   "])
def test_enriched_recommendations_reject_empty_text(
    value: str,
) -> None:
    """Enriched recommendation text cannot be blank."""

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        RAGEnrichedRecommendationResult(
            recommendation_result=build_recommendation_result(),
            rag_result=build_rag_result(
                prompt_type=PromptType.RECOMMENDATION,
                response="Valid response",
            ),
            enriched_recommendations=value,
        )


def test_rag_commentary_agent_creation() -> None:
    """Commentary wrapper should retain its dependencies."""

    commentary_agent = build_commentary_agent_mock()
    rag_agent = build_rag_agent_mock()

    integration = RAGCommentaryAgent(
        commentary_agent=commentary_agent,
        rag_agent=rag_agent,
    )

    assert integration.commentary_agent is commentary_agent
    assert integration.rag_agent is rag_agent


def test_rag_commentary_agent_rejects_invalid_agent() -> None:
    """Commentary wrapper requires CommentaryAgent."""

    with pytest.raises(
        TypeError,
        match="CommentaryAgent",
    ):
        RAGCommentaryAgent(
            commentary_agent="invalid",  # type: ignore[arg-type]
            rag_agent=build_rag_agent_mock(),
        )


def test_rag_commentary_agent_rejects_invalid_rag_agent() -> None:
    """Commentary wrapper requires FinanceRAGAgent."""

    with pytest.raises(
        TypeError,
        match="FinanceRAGAgent",
    ):
        RAGCommentaryAgent(
            commentary_agent=build_commentary_agent_mock(),
            rag_agent="invalid",  # type: ignore[arg-type]
        )


def test_commentary_enrich_uses_commentary_prompt() -> None:
    """Commentary enrichment should use the commentary prompt type."""

    commentary_result = build_commentary_result()
    rag_agent = build_rag_agent_mock()

    rag_result = build_rag_result(
        prompt_type=PromptType.COMMENTARY,
        response="RAG-enriched commentary.",
    )
    rag_agent.run.return_value = rag_result

    integration = RAGCommentaryAgent(
        commentary_agent=build_commentary_agent_mock(),
        rag_agent=rag_agent,
    )

    result = integration.enrich(
        commentary_result=commentary_result,
        user_request="Prepare management commentary",
        retrieval_query="revenue policy",
        metadata_filter={"document_type": "policy"},
        top_k=3,
        score_threshold=0.4,
        additional_instructions="Keep it concise.",
    )

    assert result.commentary_result is commentary_result
    assert (
        result.enriched_commentary
        == "RAG-enriched commentary."
    )

    call_arguments = rag_agent.run.call_args.kwargs

    assert (
        call_arguments["prompt_type"]
        is PromptType.COMMENTARY
    )
    assert (
        call_arguments["user_request"]
        == "Prepare management commentary"
    )
    assert (
        call_arguments["retrieval_query"]
        == "revenue policy"
    )
    assert call_arguments["metadata_filter"] == {
        "document_type": "policy"
    }
    assert call_arguments["top_k"] == 3
    assert call_arguments["score_threshold"] == 0.4
    assert (
        call_arguments["additional_instructions"]
        == "Keep it concise."
    )


def test_commentary_enrich_serializes_result() -> None:
    """Existing commentary output should become finance-analysis data."""

    commentary_result = build_commentary_result()
    rag_agent = build_rag_agent_mock()
    rag_agent.run.return_value = build_rag_result(
        prompt_type=PromptType.COMMENTARY,
        response="Enriched commentary.",
    )

    integration = RAGCommentaryAgent(
        commentary_agent=build_commentary_agent_mock(),
        rag_agent=rag_agent,
    )

    integration.enrich(commentary_result)

    finance_analysis = (
        rag_agent.run.call_args.kwargs["finance_analysis"]
    )

    assert isinstance(finance_analysis, dict)
    assert finance_analysis
    assert finance_analysis is not commentary_result


def test_commentary_analyze_preserves_original_result() -> None:
    """Existing CommentaryAgent result should remain available."""

    commentary_agent = build_commentary_agent_mock()
    commentary_result = build_commentary_result()
    commentary_agent.analyze.return_value = commentary_result

    rag_agent = build_rag_agent_mock()
    rag_agent.run.return_value = build_rag_result(
        prompt_type=PromptType.COMMENTARY,
        response="Enriched commentary.",
    )

    integration = RAGCommentaryAgent(
        commentary_agent=commentary_agent,
        rag_agent=rag_agent,
    )

    kpi_result = object()
    variance_result = object()
    forecast_result = object()
    scenario_result = object()
    rules_result = object()

    result = integration.analyze(
        kpi_result=kpi_result,
        revenue_variance_result=variance_result,
        forecast_result=forecast_result,
        scenario_result=scenario_result,
        finance_rules_result=rules_result,
    )

    commentary_agent.analyze.assert_called_once_with(
        kpi_result=kpi_result,
        revenue_variance_result=variance_result,
        forecast_result=forecast_result,
        scenario_result=scenario_result,
        finance_rules_result=rules_result,
    )

    assert result.commentary_result is commentary_result
    assert result.enriched_commentary == "Enriched commentary."


def test_commentary_enrich_rejects_invalid_result() -> None:
    """Commentary enrichment requires CommentaryResult."""

    integration = RAGCommentaryAgent(
        commentary_agent=build_commentary_agent_mock(),
        rag_agent=build_rag_agent_mock(),
    )

    with pytest.raises(
        TypeError,
        match="CommentaryResult",
    ):
        integration.enrich(
            "invalid"  # type: ignore[arg-type]
        )


def test_commentary_enrich_wraps_rag_failure() -> None:
    """RAG failures should become integration errors."""

    rag_agent = build_rag_agent_mock()
    rag_agent.run.side_effect = RAGAgentError(
        "Mock RAG failure"
    )

    integration = RAGCommentaryAgent(
        commentary_agent=build_commentary_agent_mock(),
        rag_agent=rag_agent,
    )

    with pytest.raises(
        FinanceAgentRAGIntegrationError,
        match="Commentary RAG enrichment failed",
    ):
        integration.enrich(
            build_commentary_result()
        )


def test_rag_recommendation_agent_creation() -> None:
    """Recommendation wrapper should retain its dependencies."""

    recommendation_agent = build_recommendation_agent_mock()
    rag_agent = build_rag_agent_mock()

    integration = RAGRecommendationAgent(
        recommendation_agent=recommendation_agent,
        rag_agent=rag_agent,
    )

    assert (
        integration.recommendation_agent
        is recommendation_agent
    )
    assert integration.rag_agent is rag_agent


def test_rag_recommendation_agent_rejects_invalid_agent() -> None:
    """Recommendation wrapper requires RecommendationAgent."""

    with pytest.raises(
        TypeError,
        match="RecommendationAgent",
    ):
        RAGRecommendationAgent(
            recommendation_agent="invalid",  # type: ignore[arg-type]
            rag_agent=build_rag_agent_mock(),
        )


def test_rag_recommendation_agent_rejects_invalid_rag_agent() -> None:
    """Recommendation wrapper requires FinanceRAGAgent."""

    with pytest.raises(
        TypeError,
        match="FinanceRAGAgent",
    ):
        RAGRecommendationAgent(
            recommendation_agent=(
                build_recommendation_agent_mock()
            ),
            rag_agent="invalid",  # type: ignore[arg-type]
        )


def test_recommendation_enrich_uses_recommendation_prompt() -> None:
    """Recommendation enrichment should use recommendation prompt."""

    recommendation_result = build_recommendation_result()
    rag_agent = build_rag_agent_mock()
    rag_agent.run.return_value = build_rag_result(
        prompt_type=PromptType.RECOMMENDATION,
        response="RAG-enriched recommendations.",
    )

    integration = RAGRecommendationAgent(
        recommendation_agent=build_recommendation_agent_mock(),
        rag_agent=rag_agent,
    )

    result = integration.enrich(
        recommendation_result=recommendation_result,
        user_request="Prepare management recommendations",
        retrieval_query="cost control policy",
        metadata_filter={"document_type": "policy"},
        top_k=2,
        score_threshold=0.3,
        additional_instructions=(
            "Return three recommendations."
        ),
    )

    assert result.recommendation_result is recommendation_result
    assert (
        result.enriched_recommendations
        == "RAG-enriched recommendations."
    )

    call_arguments = rag_agent.run.call_args.kwargs

    assert (
        call_arguments["prompt_type"]
        is PromptType.RECOMMENDATION
    )
    assert (
        call_arguments["retrieval_query"]
        == "cost control policy"
    )
    assert call_arguments["top_k"] == 2
    assert call_arguments["score_threshold"] == 0.3


def test_recommendation_analyze_preserves_original_result() -> None:
    """Existing RecommendationAgent result should remain available."""

    recommendation_agent = build_recommendation_agent_mock()
    recommendation_result = build_recommendation_result()

    recommendation_agent.analyze.return_value = (
        recommendation_result
    )

    rag_agent = build_rag_agent_mock()
    rag_agent.run.return_value = build_rag_result(
        prompt_type=PromptType.RECOMMENDATION,
        response="Enriched recommendations.",
    )

    integration = RAGRecommendationAgent(
        recommendation_agent=recommendation_agent,
        rag_agent=rag_agent,
    )

    root_cause_result = object()

    result = integration.analyze(
        root_cause_result=root_cause_result
    )

    recommendation_agent.analyze.assert_called_once_with(
        root_cause_result=root_cause_result
    )

    assert (
        result.recommendation_result
        is recommendation_result
    )
    assert (
        result.enriched_recommendations
        == "Enriched recommendations."
    )


def test_recommendation_enrich_rejects_invalid_result() -> None:
    """Recommendation enrichment requires RecommendationResult."""

    integration = RAGRecommendationAgent(
        recommendation_agent=build_recommendation_agent_mock(),
        rag_agent=build_rag_agent_mock(),
    )

    with pytest.raises(
        TypeError,
        match="RecommendationResult",
    ):
        integration.enrich(
            "invalid"  # type: ignore[arg-type]
        )


def test_recommendation_enrich_wraps_rag_failure() -> None:
    """Recommendation RAG failures should be wrapped."""

    rag_agent = build_rag_agent_mock()
    rag_agent.run.side_effect = RAGAgentError(
        "Mock RAG failure"
    )

    integration = RAGRecommendationAgent(
        recommendation_agent=build_recommendation_agent_mock(),
        rag_agent=rag_agent,
    )

    with pytest.raises(
        FinanceAgentRAGIntegrationError,
        match="Recommendation RAG enrichment failed",
    ):
        integration.enrich(
            build_recommendation_result()
        )


def test_create_rag_commentary_agent() -> None:
    """Commentary factory should create the wrapper."""

    commentary_agent = build_commentary_agent_mock()
    rag_agent = build_rag_agent_mock()

    integration = create_rag_commentary_agent(
        commentary_agent=commentary_agent,
        rag_agent=rag_agent,
    )

    assert isinstance(
        integration,
        RAGCommentaryAgent,
    )
    assert integration.commentary_agent is commentary_agent


def test_create_rag_recommendation_agent() -> None:
    """Recommendation factory should create the wrapper."""

    recommendation_agent = build_recommendation_agent_mock()
    rag_agent = build_rag_agent_mock()

    integration = create_rag_recommendation_agent(
        recommendation_agent=recommendation_agent,
        rag_agent=rag_agent,
    )

    assert isinstance(
        integration,
        RAGRecommendationAgent,
    )
    assert (
        integration.recommendation_agent
        is recommendation_agent
    )


@dataclass
class SampleDataclassResult:
    """Sample finance result for serialization."""

    revenue: float
    budget: float
    metadata: dict[str, Any]


def test_serialize_dataclass_result() -> None:
    """Dataclass finance results should serialize recursively."""

    original_metadata = {
        "year": 2026,
    }

    result = SampleDataclassResult(
        revenue=1200.0,
        budget=1000.0,
        metadata=original_metadata,
    )

    serialized = _serialize_result(result)

    assert serialized == {
        "revenue": 1200.0,
        "budget": 1000.0,
        "metadata": {
            "year": 2026,
        },
    }

    original_metadata["year"] = 2030

    assert serialized["metadata"]["year"] == 2026


class ToDictResult:
    """Sample finance result exposing to_dict."""

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "revenue": 1200,
            "budget": 1000,
        }


def test_serialize_to_dict_result() -> None:
    """to_dict finance results should serialize."""

    serialized = _serialize_result(
        ToDictResult()
    )

    assert serialized == {
        "revenue": 1200,
        "budget": 1000,
    }


class PublicAttributeResult:
    """Sample finance result with public attributes."""

    def __init__(self) -> None:
        self.revenue = 1200
        self.budget = 1000
        self._private_value = "hidden"


def test_serialize_public_attributes() -> None:
    """Public object attributes should serialize."""

    serialized = _serialize_result(
        PublicAttributeResult()
    )

    assert serialized == {
        "revenue": 1200,
        "budget": 1000,
    }


def test_serialize_rejects_unsupported_result() -> None:
    """Unsupported values should fail clearly."""

    with pytest.raises(
        TypeError,
        match="dataclass.*to_dict",
    ):
        _serialize_result(100)