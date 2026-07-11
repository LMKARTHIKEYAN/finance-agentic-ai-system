"""
Prompt templates for the Finance Agentic AI System.

This module contains reusable prompt templates and formatting helpers.
It does not call an LLM and does not perform finance calculations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence


DEFAULT_SYSTEM_PROMPT = """
You are an enterprise FP&A assistant.

Your role is to explain finance results clearly, accurately, and concisely
for management users.

Follow these rules:

1. Use only the supplied finance analysis and retrieved context.
2. Do not invent financial values, assumptions, policies, or explanations.
3. Clearly distinguish facts from interpretations.
4. Highlight material business drivers.
5. Explain favourable and unfavourable performance.
6. Mention data limitations when evidence is incomplete.
7. Do not perform calculations that are not already present in the analysis.
8. Do not provide investment advice.
9. Use professional management-reporting language.
10. Keep conclusions traceable to the supplied evidence.
""".strip()


class PromptTemplateError(ValueError):
    """Raised when prompt-template input is invalid."""


def _validate_required_text(
    value: Any,
    *,
    field_name: str,
) -> str:
    """Validate required text input."""

    if not isinstance(value, str):
        raise TypeError(
            f"{field_name} must be a string."
        )

    cleaned_value = value.strip()

    if not cleaned_value:
        raise PromptTemplateError(
            f"{field_name} cannot be empty."
        )

    return cleaned_value


def _validate_optional_text(
    value: Any,
    *,
    field_name: str,
) -> str:
    """Validate optional text input."""

    if value is None:
        return ""

    if not isinstance(value, str):
        raise TypeError(
            f"{field_name} must be a string."
        )

    return value.strip()


class PromptType(str, Enum):
    """Supported finance prompt categories."""

    FINANCE_QA = "finance_qa"
    COMMENTARY = "commentary"
    RECOMMENDATION = "recommendation"
    KPI_EXPLANATION = "kpi_explanation"
    BUDGET_ANALYSIS = "budget_analysis"
    FORECAST_ANALYSIS = "forecast_analysis"
    VARIANCE_ANALYSIS = "variance_analysis"
    ROOT_CAUSE_ANALYSIS = "root_cause_analysis"
    SCENARIO_ANALYSIS = "scenario_analysis"


@dataclass(frozen=True)
class PromptMessages:
    """Structured system and user prompt messages."""

    system: str
    user: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "system",
            _validate_required_text(
                self.system,
                field_name="system",
            ),
        )

        object.__setattr__(
            self,
            "user",
            _validate_required_text(
                self.user,
                field_name="user",
            ),
        )

    def as_dicts(self) -> list[dict[str, str]]:
        """Return OpenAI-compatible message dictionaries."""

        return [
            {
                "role": "system",
                "content": self.system,
            },
            {
                "role": "user",
                "content": self.user,
            },
        ]


@dataclass(frozen=True)
class PromptTemplate:
    """Reusable prompt template definition."""

    prompt_type: PromptType
    title: str
    instructions: str
    output_format: str

    def __post_init__(self) -> None:
        if not isinstance(self.prompt_type, PromptType):
            raise TypeError(
                "prompt_type must be a PromptType value."
            )

        object.__setattr__(
            self,
            "title",
            _validate_required_text(
                self.title,
                field_name="title",
            ),
        )

        object.__setattr__(
            self,
            "instructions",
            _validate_required_text(
                self.instructions,
                field_name="instructions",
            ),
        )

        object.__setattr__(
            self,
            "output_format",
            _validate_required_text(
                self.output_format,
                field_name="output_format",
            ),
        )


FINANCE_QA_TEMPLATE = PromptTemplate(
    prompt_type=PromptType.FINANCE_QA,
    title="Finance Question Answering",
    instructions="""
Answer the user's finance question using only the supplied finance analysis
and retrieved reference context.

Explain the answer in business language. Cite the relevant source document
identifiers when retrieved context is used.

When the answer is not supported by the supplied evidence, state that there
is insufficient information.
""".strip(),
    output_format="""
Return:

1. Direct answer
2. Supporting evidence
3. Business interpretation
4. Data limitations, if any
""".strip(),
)


COMMENTARY_TEMPLATE = PromptTemplate(
    prompt_type=PromptType.COMMENTARY,
    title="Management Commentary",
    instructions="""
Create management commentary for the supplied finance analysis.

Focus on:

- Overall performance
- Material favourable and unfavourable movements
- Main price, volume, mix, cost, margin, budget, or forecast drivers
- Relevant anomalies and root causes
- Business impact
- Management attention areas

Use retrieved context only when it directly supports the commentary.

Do not repeat every number. Prioritize material insights.
""".strip(),
    output_format="""
Return the commentary using these sections:

Executive Summary
Key Performance Drivers
Risks and Exceptions
Management Outlook
""".strip(),
)


RECOMMENDATION_TEMPLATE = PromptTemplate(
    prompt_type=PromptType.RECOMMENDATION,
    title="Management Recommendations",
    instructions="""
Generate practical management recommendations from the supplied finance
analysis.

Each recommendation must:

- Be supported by the analysis or retrieved context
- Address a specific business driver, risk, or opportunity
- Be actionable
- Avoid unsupported financial claims
- Include the expected business outcome where evidence allows

Do not generate generic recommendations unrelated to the evidence.
""".strip(),
    output_format="""
Return a numbered list.

For each recommendation include:

Action
Reason
Expected Outcome
Priority: High, Medium, or Low
Supporting Evidence
""".strip(),
)


KPI_EXPLANATION_TEMPLATE = PromptTemplate(
    prompt_type=PromptType.KPI_EXPLANATION,
    title="KPI Performance Explanation",
    instructions="""
Explain the supplied KPI results.

Identify:

- KPI performance against target, budget, forecast, or prior period
- Strongest and weakest KPIs
- Material trends
- Potential business drivers
- KPI relationships
- Areas requiring management attention

Do not calculate new KPI values.
""".strip(),
    output_format="""
Return:

KPI Summary
Positive Performance
Negative Performance
Key Drivers
Management Attention
""".strip(),
)


BUDGET_ANALYSIS_TEMPLATE = PromptTemplate(
    prompt_type=PromptType.BUDGET_ANALYSIS,
    title="Budget Analysis",
    instructions="""
Explain the supplied budget analysis.

Focus on:

- Major budget allocations
- Key assumptions
- Budget risks
- Cost or revenue concentration
- Potential operational constraints
- Areas requiring validation or approval

Do not modify the budget or create unsupported assumptions.
""".strip(),
    output_format="""
Return:

Budget Summary
Key Assumptions
Risks
Opportunities
Management Actions
""".strip(),
)


FORECAST_ANALYSIS_TEMPLATE = PromptTemplate(
    prompt_type=PromptType.FORECAST_ANALYSIS,
    title="Forecast Analysis",
    instructions="""
Explain the supplied forecast result.

Focus on:

- Expected performance
- Major forecast drivers
- Differences from budget or prior forecast
- Risks to forecast delivery
- Sensitivity to assumptions
- Areas requiring monitoring

Clearly identify whether conclusions come from forecast data or retrieved
reference context.
""".strip(),
    output_format="""
Return:

Forecast Summary
Key Drivers
Risks and Uncertainty
Monitoring Indicators
Management Outlook
""".strip(),
)


VARIANCE_ANALYSIS_TEMPLATE = PromptTemplate(
    prompt_type=PromptType.VARIANCE_ANALYSIS,
    title="Variance Analysis",
    instructions="""
Explain the supplied actual-versus-budget, actual-versus-forecast, or
actual-versus-prior-period variance analysis.

Focus on:

- Total variance
- Price effect
- Volume effect
- Mix effect
- Cost effect
- Margin effect
- Main favourable and unfavourable contributors
- Reconciliation or validation issues

Use only variance calculations already supplied.
""".strip(),
    output_format="""
Return:

Variance Summary
Favourable Drivers
Unfavourable Drivers
Reconciliation Status
Management Interpretation
""".strip(),
)


ROOT_CAUSE_TEMPLATE = PromptTemplate(
    prompt_type=PromptType.ROOT_CAUSE_ANALYSIS,
    title="Root Cause Analysis",
    instructions="""
Explain the likely root causes already identified by the finance analysis.

Separate:

- Direct causes
- Contributing factors
- Symptoms
- Evidence gaps

Do not describe correlation as confirmed causation unless the evidence
supports it.
""".strip(),
    output_format="""
Return:

Observed Issue
Primary Root Causes
Contributing Factors
Evidence
Data Gaps
""".strip(),
)


SCENARIO_ANALYSIS_TEMPLATE = PromptTemplate(
    prompt_type=PromptType.SCENARIO_ANALYSIS,
    title="Scenario Analysis",
    instructions="""
Explain the supplied scenario-analysis results.

Compare scenarios using the supplied outputs and assumptions.

Focus on:

- Base, upside, and downside outcomes
- Main changing assumptions
- Revenue, cost, profit, and margin impact
- Key risks
- Decision implications

Do not create new scenarios or recalculate existing scenarios.
""".strip(),
    output_format="""
Return:

Scenario Comparison
Key Assumption Changes
Financial Impact
Risks
Decision Implications
""".strip(),
)


PROMPT_TEMPLATES: dict[PromptType, PromptTemplate] = {
    PromptType.FINANCE_QA: FINANCE_QA_TEMPLATE,
    PromptType.COMMENTARY: COMMENTARY_TEMPLATE,
    PromptType.RECOMMENDATION: RECOMMENDATION_TEMPLATE,
    PromptType.KPI_EXPLANATION: KPI_EXPLANATION_TEMPLATE,
    PromptType.BUDGET_ANALYSIS: BUDGET_ANALYSIS_TEMPLATE,
    PromptType.FORECAST_ANALYSIS: FORECAST_ANALYSIS_TEMPLATE,
    PromptType.VARIANCE_ANALYSIS: VARIANCE_ANALYSIS_TEMPLATE,
    PromptType.ROOT_CAUSE_ANALYSIS: ROOT_CAUSE_TEMPLATE,
    PromptType.SCENARIO_ANALYSIS: SCENARIO_ANALYSIS_TEMPLATE,
}


def get_prompt_template(
    prompt_type: PromptType | str,
) -> PromptTemplate:
    """Return a registered prompt template."""

    resolved_type = _resolve_prompt_type(prompt_type)

    try:
        return PROMPT_TEMPLATES[resolved_type]
    except KeyError as exc:
        raise PromptTemplateError(
            f"No prompt template registered for "
            f"'{resolved_type.value}'."
        ) from exc


def build_prompt_messages(
    prompt_type: PromptType | str,
    *,
    user_request: str,
    finance_analysis: Any,
    retrieved_context: str = "",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    additional_instructions: str | None = None,
) -> PromptMessages:
    """Build structured system and user messages."""

    template = get_prompt_template(prompt_type)

    validated_request = _validate_required_text(
        user_request,
        field_name="user_request",
    )

    validated_system_prompt = _validate_required_text(
        system_prompt,
        field_name="system_prompt",
    )

    validated_context = _validate_optional_text(
        retrieved_context,
        field_name="retrieved_context",
    )

    validated_additional_instructions = _validate_optional_text(
        additional_instructions,
        field_name="additional_instructions",
    )

    analysis_text = format_finance_analysis(
        finance_analysis
    )

    user_prompt = _compose_user_prompt(
        template=template,
        user_request=validated_request,
        finance_analysis=analysis_text,
        retrieved_context=validated_context,
        additional_instructions=(
            validated_additional_instructions
        ),
    )

    return PromptMessages(
        system=validated_system_prompt,
        user=user_prompt,
    )


def build_finance_qa_prompt(
    *,
    user_request: str,
    finance_analysis: Any,
    retrieved_context: str = "",
) -> PromptMessages:
    """Build a finance question-answering prompt."""

    return build_prompt_messages(
        PromptType.FINANCE_QA,
        user_request=user_request,
        finance_analysis=finance_analysis,
        retrieved_context=retrieved_context,
    )


def build_commentary_prompt(
    *,
    user_request: str,
    finance_analysis: Any,
    retrieved_context: str = "",
    additional_instructions: str | None = None,
) -> PromptMessages:
    """Build a management-commentary prompt."""

    return build_prompt_messages(
        PromptType.COMMENTARY,
        user_request=user_request,
        finance_analysis=finance_analysis,
        retrieved_context=retrieved_context,
        additional_instructions=additional_instructions,
    )


def build_recommendation_prompt(
    *,
    user_request: str,
    finance_analysis: Any,
    retrieved_context: str = "",
    additional_instructions: str | None = None,
) -> PromptMessages:
    """Build a management-recommendation prompt."""

    return build_prompt_messages(
        PromptType.RECOMMENDATION,
        user_request=user_request,
        finance_analysis=finance_analysis,
        retrieved_context=retrieved_context,
        additional_instructions=additional_instructions,
    )


def format_finance_analysis(
    finance_analysis: Any,
) -> str:
    """Convert finance-agent output into deterministic prompt text."""

    if finance_analysis is None:
        return "No finance analysis was supplied."

    if isinstance(finance_analysis, str):
        cleaned_text = finance_analysis.strip()

        return (
            cleaned_text
            if cleaned_text
            else "No finance analysis was supplied."
        )

    serializable_value = _to_serializable(
        finance_analysis
    )

    try:
        return json.dumps(
            serializable_value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
    except (TypeError, ValueError) as exc:
        raise PromptTemplateError(
            f"Unable to format finance analysis: {exc}"
        ) from exc


def format_reference_context(
    context: str,
) -> str:
    """Format retrieved context for insertion into a prompt."""

    validated_context = _validate_optional_text(
        context,
        field_name="context",
    )

    if not validated_context:
        return (
            "No retrieved reference context was available. "
            "Use only the supplied finance analysis."
        )

    return validated_context


def list_prompt_types() -> list[str]:
    """Return supported prompt-type names."""

    return [
        prompt_type.value
        for prompt_type in PromptType
    ]


def _compose_user_prompt(
    *,
    template: PromptTemplate,
    user_request: str,
    finance_analysis: str,
    retrieved_context: str,
    additional_instructions: str,
) -> str:
    """Create the complete user prompt."""

    sections = [
        f"Task: {template.title}",
        "",
        "Task Instructions:",
        template.instructions,
        "",
        "Required Output Format:",
        template.output_format,
        "",
        "User Request:",
        user_request,
        "",
        "Finance Analysis:",
        finance_analysis,
        "",
        "Retrieved Reference Context:",
        format_reference_context(
            retrieved_context
        ),
    ]

    if additional_instructions:
        sections.extend(
            [
                "",
                "Additional Instructions:",
                additional_instructions,
            ]
        )

    sections.extend(
        [
            "",
            "Evidence Rules:",
            (
                "- Use only the finance analysis and retrieved "
                "reference context above.\n"
                "- Do not invent values or business facts.\n"
                "- State clearly when evidence is insufficient.\n"
                "- Do not perform new finance calculations."
            ),
        ]
    )

    return "\n".join(sections).strip()


def _resolve_prompt_type(
    prompt_type: PromptType | str,
) -> PromptType:
    """Convert a supported value into PromptType."""

    if isinstance(prompt_type, PromptType):
        return prompt_type

    if not isinstance(prompt_type, str):
        raise TypeError(
            "prompt_type must be a PromptType or string."
        )

    cleaned_type = prompt_type.strip().lower()

    if not cleaned_type:
        raise PromptTemplateError(
            "prompt_type cannot be empty."
        )

    try:
        return PromptType(cleaned_type)
    except ValueError as exc:
        supported_types = ", ".join(
            list_prompt_types()
        )

        raise PromptTemplateError(
            f"Unsupported prompt type '{cleaned_type}'. "
            f"Supported types: {supported_types}."
        ) from exc


def _to_serializable(
    value: Any,
) -> Any:
    """Convert common project objects into JSON-safe values."""

    if value is None or isinstance(
        value,
        (str, int, float, bool),
    ):
        return value

    if isinstance(value, Mapping):
        return {
            str(key): _to_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [
            _to_serializable(item)
            for item in value
        ]

    to_dict_method = getattr(
        value,
        "to_dict",
        None,
    )

    if callable(to_dict_method):
        try:
            return _to_serializable(
                to_dict_method()
            )
        except Exception as exc:
            raise PromptTemplateError(
                f"Unable to convert object using to_dict: {exc}"
            ) from exc

    if hasattr(value, "__dict__"):
        return {
            key: _to_serializable(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }

    return str(value)