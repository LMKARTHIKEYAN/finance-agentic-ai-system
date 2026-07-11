"""
Tests for RAG prompt templates and formatting helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from src.rag.prompt_templates import (
    BUDGET_ANALYSIS_TEMPLATE,
    COMMENTARY_TEMPLATE,
    DEFAULT_SYSTEM_PROMPT,
    FINANCE_QA_TEMPLATE,
    FORECAST_ANALYSIS_TEMPLATE,
    KPI_EXPLANATION_TEMPLATE,
    PROMPT_TEMPLATES,
    RECOMMENDATION_TEMPLATE,
    ROOT_CAUSE_TEMPLATE,
    SCENARIO_ANALYSIS_TEMPLATE,
    VARIANCE_ANALYSIS_TEMPLATE,
    PromptMessages,
    PromptTemplate,
    PromptTemplateError,
    PromptType,
    build_commentary_prompt,
    build_finance_qa_prompt,
    build_prompt_messages,
    build_recommendation_prompt,
    format_finance_analysis,
    format_reference_context,
    get_prompt_template,
    list_prompt_types,
)


@dataclass
class SampleAnalysis:
    """Sample dataclass used for serialization tests."""

    revenue: float
    budget: float
    status: str


class ObjectWithToDict:
    """Sample object exposing a to_dict method."""

    def to_dict(self) -> dict[str, object]:
        return {
            "revenue": 1200,
            "budget": 1000,
        }


class FailingToDictObject:
    """Sample object whose to_dict method fails."""

    def to_dict(self) -> dict[str, object]:
        raise RuntimeError("Mock conversion failure")


def test_prompt_type_values() -> None:
    """PromptType should expose all supported values."""

    assert PromptType.FINANCE_QA.value == "finance_qa"
    assert PromptType.COMMENTARY.value == "commentary"
    assert PromptType.RECOMMENDATION.value == "recommendation"
    assert PromptType.KPI_EXPLANATION.value == "kpi_explanation"
    assert PromptType.BUDGET_ANALYSIS.value == "budget_analysis"
    assert PromptType.FORECAST_ANALYSIS.value == "forecast_analysis"
    assert PromptType.VARIANCE_ANALYSIS.value == "variance_analysis"
    assert PromptType.ROOT_CAUSE_ANALYSIS.value == "root_cause_analysis"
    assert PromptType.SCENARIO_ANALYSIS.value == "scenario_analysis"


def test_default_system_prompt_is_not_empty() -> None:
    """Default system prompt should contain enterprise FP&A rules."""

    assert DEFAULT_SYSTEM_PROMPT
    assert "enterprise FP&A assistant" in DEFAULT_SYSTEM_PROMPT
    assert "Do not invent" in DEFAULT_SYSTEM_PROMPT
    assert "Do not perform calculations" in DEFAULT_SYSTEM_PROMPT


def test_prompt_messages_creation() -> None:
    """PromptMessages should retain validated content."""

    messages = PromptMessages(
        system="  System instruction  ",
        user="  User request  ",
    )

    assert messages.system == "System instruction"
    assert messages.user == "User request"


def test_prompt_messages_as_dicts() -> None:
    """PromptMessages should return OpenAI-compatible dictionaries."""

    messages = PromptMessages(
        system="System instruction",
        user="User request",
    )

    assert messages.as_dicts() == [
        {
            "role": "system",
            "content": "System instruction",
        },
        {
            "role": "user",
            "content": "User request",
        },
    ]


@pytest.mark.parametrize("field_name", ["system", "user"])
def test_prompt_messages_rejects_empty_text(
    field_name: str,
) -> None:
    """PromptMessages should reject blank required fields."""

    kwargs = {
        "system": "Valid system",
        "user": "Valid user",
    }
    kwargs[field_name] = "   "

    with pytest.raises(
        PromptTemplateError,
        match=f"{field_name} cannot be empty",
    ):
        PromptMessages(**kwargs)


@pytest.mark.parametrize("field_name", ["system", "user"])
def test_prompt_messages_rejects_non_string_text(
    field_name: str,
) -> None:
    """PromptMessages should reject non-string fields."""

    kwargs = {
        "system": "Valid system",
        "user": "Valid user",
    }
    kwargs[field_name] = 100

    with pytest.raises(
        TypeError,
        match=f"{field_name} must be a string",
    ):
        PromptMessages(**kwargs)


def test_prompt_template_creation() -> None:
    """PromptTemplate should validate and strip text fields."""

    template = PromptTemplate(
        prompt_type=PromptType.FINANCE_QA,
        title="  Finance Q&A  ",
        instructions="  Explain results  ",
        output_format="  Return summary  ",
    )

    assert template.prompt_type is PromptType.FINANCE_QA
    assert template.title == "Finance Q&A"
    assert template.instructions == "Explain results"
    assert template.output_format == "Return summary"


def test_prompt_template_rejects_invalid_type() -> None:
    """PromptTemplate should require PromptType."""

    with pytest.raises(
        TypeError,
        match="prompt_type must be a PromptType",
    ):
        PromptTemplate(
            prompt_type="finance_qa",  # type: ignore[arg-type]
            title="Finance Q&A",
            instructions="Explain results",
            output_format="Return summary",
        )


@pytest.mark.parametrize(
    "field_name",
    ["title", "instructions", "output_format"],
)
def test_prompt_template_rejects_empty_fields(
    field_name: str,
) -> None:
    """PromptTemplate should reject blank required text."""

    kwargs = {
        "prompt_type": PromptType.FINANCE_QA,
        "title": "Finance Q&A",
        "instructions": "Explain results",
        "output_format": "Return summary",
    }
    kwargs[field_name] = " "

    with pytest.raises(
        PromptTemplateError,
        match=f"{field_name} cannot be empty",
    ):
        PromptTemplate(**kwargs)


def test_registered_templates_are_complete() -> None:
    """Every PromptType should have a registered template."""

    assert set(PROMPT_TEMPLATES) == set(PromptType)


@pytest.mark.parametrize(
    ("prompt_type", "expected_template"),
    [
        (PromptType.FINANCE_QA, FINANCE_QA_TEMPLATE),
        (PromptType.COMMENTARY, COMMENTARY_TEMPLATE),
        (PromptType.RECOMMENDATION, RECOMMENDATION_TEMPLATE),
        (PromptType.KPI_EXPLANATION, KPI_EXPLANATION_TEMPLATE),
        (PromptType.BUDGET_ANALYSIS, BUDGET_ANALYSIS_TEMPLATE),
        (PromptType.FORECAST_ANALYSIS, FORECAST_ANALYSIS_TEMPLATE),
        (PromptType.VARIANCE_ANALYSIS, VARIANCE_ANALYSIS_TEMPLATE),
        (PromptType.ROOT_CAUSE_ANALYSIS, ROOT_CAUSE_TEMPLATE),
        (PromptType.SCENARIO_ANALYSIS, SCENARIO_ANALYSIS_TEMPLATE),
    ],
)
def test_get_prompt_template_with_enum(
    prompt_type: PromptType,
    expected_template: PromptTemplate,
) -> None:
    """Templates should be retrievable by enum."""

    assert get_prompt_template(prompt_type) is expected_template


@pytest.mark.parametrize(
    ("prompt_type", "expected_template"),
    [
        ("finance_qa", FINANCE_QA_TEMPLATE),
        (" COMMENTARY ", COMMENTARY_TEMPLATE),
        ("Recommendation", RECOMMENDATION_TEMPLATE),
        ("kpi_explanation", KPI_EXPLANATION_TEMPLATE),
    ],
)
def test_get_prompt_template_with_string(
    prompt_type: str,
    expected_template: PromptTemplate,
) -> None:
    """Templates should be retrievable by normalized string."""

    assert get_prompt_template(prompt_type) is expected_template


def test_get_prompt_template_rejects_empty_value() -> None:
    """Blank prompt types should fail."""

    with pytest.raises(
        PromptTemplateError,
        match="prompt_type cannot be empty",
    ):
        get_prompt_template("   ")


def test_get_prompt_template_rejects_unknown_value() -> None:
    """Unknown prompt types should fail clearly."""

    with pytest.raises(
        PromptTemplateError,
        match="Unsupported prompt type",
    ):
        get_prompt_template("unknown")


def test_get_prompt_template_rejects_non_string_or_enum() -> None:
    """Prompt type must be a string or PromptType."""

    with pytest.raises(
        TypeError,
        match="PromptType or string",
    ):
        get_prompt_template(100)  # type: ignore[arg-type]


def test_list_prompt_types() -> None:
    """Prompt type listing should include all supported values."""

    assert list_prompt_types() == [
        "finance_qa",
        "commentary",
        "recommendation",
        "kpi_explanation",
        "budget_analysis",
        "forecast_analysis",
        "variance_analysis",
        "root_cause_analysis",
        "scenario_analysis",
    ]


def test_format_finance_analysis_none() -> None:
    """None should produce an explicit fallback message."""

    assert format_finance_analysis(None) == (
        "No finance analysis was supplied."
    )


def test_format_finance_analysis_string() -> None:
    """String analysis should be stripped and returned."""

    assert format_finance_analysis(
        "  Revenue increased  "
    ) == "Revenue increased"


def test_format_finance_analysis_empty_string() -> None:
    """Blank strings should produce a fallback message."""

    assert format_finance_analysis("   ") == (
        "No finance analysis was supplied."
    )


def test_format_finance_analysis_mapping() -> None:
    """Mappings should be rendered as deterministic JSON."""

    result = format_finance_analysis(
        {
            "revenue": 1200,
            "budget": 1000,
        }
    )

    assert '"budget": 1000' in result
    assert '"revenue": 1200' in result
    assert result.index('"budget"') < result.index('"revenue"')


def test_format_finance_analysis_sequence() -> None:
    """Sequences should be converted to JSON arrays."""

    result = format_finance_analysis(
        [
            {"category": "revenue"},
            {"category": "cost"},
        ]
    )

    assert '"category": "revenue"' in result
    assert '"category": "cost"' in result


def test_format_finance_analysis_dataclass() -> None:
    """Dataclass-like objects should be converted through attributes."""

    result = format_finance_analysis(
        SampleAnalysis(
            revenue=1200.0,
            budget=1000.0,
            status="favourable",
        )
    )

    assert '"budget": 1000.0' in result
    assert '"revenue": 1200.0' in result
    assert '"status": "favourable"' in result


def test_format_finance_analysis_to_dict_object() -> None:
    """Objects exposing to_dict should use it."""

    result = format_finance_analysis(
        ObjectWithToDict()
    )

    assert '"budget": 1000' in result
    assert '"revenue": 1200' in result


def test_format_finance_analysis_namespace() -> None:
    """Objects with public attributes should be serialized."""

    result = format_finance_analysis(
        SimpleNamespace(
            revenue=1200,
            budget=1000,
        )
    )

    assert '"budget": 1000' in result
    assert '"revenue": 1200' in result


def test_format_finance_analysis_ignores_private_attributes() -> None:
    """Private object attributes should not be included."""

    class SampleObject:
        def __init__(self) -> None:
            self.public_value = 10
            self._private_value = 20

    result = format_finance_analysis(
        SampleObject()
    )

    assert '"public_value": 10' in result
    assert "_private_value" not in result


def test_format_finance_analysis_failing_to_dict() -> None:
    """to_dict errors should become PromptTemplateError."""

    with pytest.raises(
        PromptTemplateError,
        match="Unable to convert object using to_dict",
    ):
        format_finance_analysis(
            FailingToDictObject()
        )


def test_format_reference_context_with_text() -> None:
    """Context should be stripped and returned."""

    assert format_reference_context(
        "  Source 1 context  "
    ) == "Source 1 context"


def test_format_reference_context_empty() -> None:
    """Blank context should produce a clear fallback statement."""

    result = format_reference_context("   ")

    assert "No retrieved reference context was available" in result
    assert "Use only the supplied finance analysis" in result


def test_format_reference_context_rejects_non_string() -> None:
    """Context should be textual."""

    with pytest.raises(
        TypeError,
        match="context must be a string",
    ):
        format_reference_context(100)  # type: ignore[arg-type]


def test_build_prompt_messages() -> None:
    """Generic prompt builder should return structured messages."""

    messages = build_prompt_messages(
        PromptType.VARIANCE_ANALYSIS,
        user_request="Explain the revenue variance",
        finance_analysis={
            "actual_revenue": 1200,
            "budget_revenue": 1000,
        },
        retrieved_context="Source 1: Revenue policy",
    )

    assert isinstance(messages, PromptMessages)
    assert messages.system == DEFAULT_SYSTEM_PROMPT
    assert "Task: Variance Analysis" in messages.user
    assert "Explain the revenue variance" in messages.user
    assert '"actual_revenue": 1200' in messages.user
    assert "Source 1: Revenue policy" in messages.user
    assert "Evidence Rules:" in messages.user


def test_build_prompt_messages_with_string_type() -> None:
    """Generic prompt builder should accept string prompt types."""

    messages = build_prompt_messages(
        "budget_analysis",
        user_request="Explain budget risks",
        finance_analysis={"budget": 1000},
    )

    assert "Task: Budget Analysis" in messages.user


def test_build_prompt_messages_additional_instructions() -> None:
    """Additional instructions should be included when supplied."""

    messages = build_prompt_messages(
        PromptType.COMMENTARY,
        user_request="Prepare commentary",
        finance_analysis={"revenue": 1200},
        additional_instructions=(
            "Keep the response under 200 words."
        ),
    )

    assert "Additional Instructions:" in messages.user
    assert "Keep the response under 200 words." in messages.user


def test_build_prompt_messages_without_context() -> None:
    """Missing retrieved context should be explicitly stated."""

    messages = build_prompt_messages(
        PromptType.FINANCE_QA,
        user_request="Explain performance",
        finance_analysis={"revenue": 1200},
    )

    assert (
        "No retrieved reference context was available"
        in messages.user
    )


def test_build_prompt_messages_custom_system_prompt() -> None:
    """A custom system prompt should override the default."""

    messages = build_prompt_messages(
        PromptType.FINANCE_QA,
        user_request="Explain revenue",
        finance_analysis={"revenue": 1200},
        system_prompt="Custom finance system prompt",
    )

    assert messages.system == "Custom finance system prompt"


def test_build_prompt_messages_rejects_empty_request() -> None:
    """User request is required."""

    with pytest.raises(
        PromptTemplateError,
        match="user_request cannot be empty",
    ):
        build_prompt_messages(
            PromptType.FINANCE_QA,
            user_request="   ",
            finance_analysis={},
        )


def test_build_prompt_messages_rejects_empty_system_prompt() -> None:
    """System prompt is required."""

    with pytest.raises(
        PromptTemplateError,
        match="system_prompt cannot be empty",
    ):
        build_prompt_messages(
            PromptType.FINANCE_QA,
            user_request="Explain revenue",
            finance_analysis={},
            system_prompt="   ",
        )


def test_build_prompt_messages_rejects_invalid_context() -> None:
    """Retrieved context must be textual."""

    with pytest.raises(
        TypeError,
        match="retrieved_context must be a string",
    ):
        build_prompt_messages(
            PromptType.FINANCE_QA,
            user_request="Explain revenue",
            finance_analysis={},
            retrieved_context=100,  # type: ignore[arg-type]
        )


def test_build_prompt_messages_rejects_invalid_additional_instructions() -> None:
    """Additional instructions must be textual."""

    with pytest.raises(
        TypeError,
        match="additional_instructions must be a string",
    ):
        build_prompt_messages(
            PromptType.FINANCE_QA,
            user_request="Explain revenue",
            finance_analysis={},
            additional_instructions=100,  # type: ignore[arg-type]
        )


def test_build_finance_qa_prompt() -> None:
    """Finance Q&A helper should use the correct template."""

    messages = build_finance_qa_prompt(
        user_request="What caused the revenue increase?",
        finance_analysis={
            "revenue_variance": 200,
        },
        retrieved_context="Source 1: Revenue assumption",
    )

    assert "Task: Finance Question Answering" in messages.user
    assert "Direct answer" in messages.user
    assert "What caused the revenue increase?" in messages.user


def test_build_commentary_prompt() -> None:
    """Commentary helper should use the commentary template."""

    messages = build_commentary_prompt(
        user_request="Prepare monthly commentary",
        finance_analysis={
            "revenue": 1200,
            "budget": 1000,
        },
        retrieved_context="Source 1: Budget assumptions",
    )

    assert "Task: Management Commentary" in messages.user
    assert "Executive Summary" in messages.user
    assert "Key Performance Drivers" in messages.user


def test_build_commentary_prompt_with_additional_instructions() -> None:
    """Commentary helper should pass additional instructions."""

    messages = build_commentary_prompt(
        user_request="Prepare monthly commentary",
        finance_analysis={"revenue": 1200},
        additional_instructions="Use concise bullet points.",
    )

    assert "Use concise bullet points." in messages.user


def test_build_recommendation_prompt() -> None:
    """Recommendation helper should use recommendation template."""

    messages = build_recommendation_prompt(
        user_request="Recommend management actions",
        finance_analysis={
            "cost_variance": -100,
        },
        retrieved_context="Source 1: Cost policy",
    )

    assert "Task: Management Recommendations" in messages.user
    assert "Expected Outcome" in messages.user
    assert "Priority: High, Medium, or Low" in messages.user


def test_build_recommendation_prompt_with_additional_instructions() -> None:
    """Recommendation helper should pass additional instructions."""

    messages = build_recommendation_prompt(
        user_request="Recommend management actions",
        finance_analysis={"cost_variance": -100},
        additional_instructions=(
            "Return no more than three recommendations."
        ),
    )

    assert (
        "Return no more than three recommendations."
        in messages.user
    )


@pytest.mark.parametrize(
    ("prompt_type", "expected_text"),
    [
        (
            PromptType.KPI_EXPLANATION,
            "Task: KPI Performance Explanation",
        ),
        (
            PromptType.BUDGET_ANALYSIS,
            "Task: Budget Analysis",
        ),
        (
            PromptType.FORECAST_ANALYSIS,
            "Task: Forecast Analysis",
        ),
        (
            PromptType.VARIANCE_ANALYSIS,
            "Task: Variance Analysis",
        ),
        (
            PromptType.ROOT_CAUSE_ANALYSIS,
            "Task: Root Cause Analysis",
        ),
        (
            PromptType.SCENARIO_ANALYSIS,
            "Task: Scenario Analysis",
        ),
    ],
)
def test_build_each_registered_prompt(
    prompt_type: PromptType,
    expected_text: str,
) -> None:
    """Every registered prompt type should build successfully."""

    messages = build_prompt_messages(
        prompt_type,
        user_request="Explain the result",
        finance_analysis={"value": 100},
    )

    assert expected_text in messages.user


def test_prompt_contains_no_new_calculation_instruction() -> None:
    """Built prompts should explicitly prevent new finance calculations."""

    messages = build_prompt_messages(
        PromptType.VARIANCE_ANALYSIS,
        user_request="Explain variance",
        finance_analysis={"variance": 100},
    )

    assert (
        "Do not perform new finance calculations."
        in messages.user
    )


def test_prompt_contains_evidence_only_instruction() -> None:
    """Built prompts should constrain responses to supplied evidence."""

    messages = build_prompt_messages(
        PromptType.COMMENTARY,
        user_request="Prepare commentary",
        finance_analysis={"revenue": 1200},
    )

    assert (
        "Use only the finance analysis and retrieved "
        "reference context above."
        in messages.user
    )