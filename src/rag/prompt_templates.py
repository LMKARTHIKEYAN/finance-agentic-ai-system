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
You are an enterprise FP&A assistant supporting CFOs, Finance Directors,
and FP&A Managers.

Your role is to convert validated finance-agent outputs into concise,
decision-ready management analysis.

Follow these rules:

1. Use only the supplied finance analysis and retrieved reference context.
2. Treat finance-agent outputs as the primary source of financial truth.
3. Use retrieved context only to explain policies, assumptions, definitions,
   or other reference information that directly supports the answer.
4. Do not invent financial values, assumptions, causes, risks, policies, or
   recommendations.
5. Do not perform calculations or recalculate metrics, effects, percentages,
   forecasts, scenarios, ratios, currency conversions, or unit conversions.
6. Preserve every supplied financial value exactly as provided, including its
   currency, unit, scale, sign, decimal precision, period, and label.
7. Never convert rupees into lakhs or crores, or values into thousands,
   millions, or billions, unless the supplied finance analysis explicitly
   contains that converted value.
8. Never infer a financial unit from the size of a number. A raw numeric value
   without an explicit unit must remain in its supplied scale and must not be
   labelled as rupees, lakhs, crores, thousands, millions, or billions.
9. When both a raw value and a preformatted display value are supplied, use the
   supplied display value. Do not create a new formatted or abbreviated value.
10. Use the same exact value consistently in the executive summary, detailed
    sections, tables, commentary, recommendations, and conclusions.
11. If two supplied values conflict, report the conflict as a data-quality
    issue. Do not choose one value or reconcile them yourself.
12. Clearly distinguish reported facts, supported interpretations, and data
    limitations.
13. Prioritize material drivers, exceptions, risks, and management actions.
14. Explain favourable and unfavourable performance in business language.
15. Do not describe correlation as confirmed causation without evidence.
16. Do not provide investment advice.
17. Use professional, concise, enterprise management-reporting language.
18. Avoid generic filler, repetition, and unsupported conclusions.
19. Keep every conclusion traceable to the supplied evidence.
20. When a requested value or section is unavailable, write "Not available in
    the supplied analysis" rather than estimating it.
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
    PNL_ANALYSIS = "pnl_analysis"
    GP_VARIANCE_ANALYSIS = "gp_variance_analysis"


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
Answer the user's finance question using the supplied finance-agent outputs
as the primary evidence.

Identify the exact result, period, entity, comparison basis, and business
context relevant to the question. Use retrieved reference context only when
it directly supports the explanation, and cite its source identifiers.

Prefer specific values and named drivers over generic descriptions. Where
multiple agent outputs are relevant, combine them without duplicating the
same point. When evidence is missing, state precisely what is unavailable.
""".strip(),
    output_format="""
Return these sections when supported by the evidence:

Direct answer / Executive Answer
Key Financial Evidence
Business Interpretation
Risks or Exceptions
Recommended Management Attention
Source References
Data Limitations
""".strip(),
)


COMMENTARY_TEMPLATE = PromptTemplate(
    prompt_type=PromptType.COMMENTARY,
    title="Management Commentary",
    instructions="""
Create concise management commentary from the supplied finance-agent outputs.

Synthesize overall performance, material favourable and unfavourable
movements, price-volume-mix or cost-margin drivers, anomalies, supported root
causes, recommendations, risks, and outlook. Prioritize material insights and
do not repeat every number.

Use retrieved context only when it directly supports a policy, assumption, or
business explanation. Clearly label evidence gaps and avoid unsupported
causal claims.
""".strip(),
    output_format="""
Return:

Executive Summary
Financial Performance
Key Performance Drivers
Material Favourable Drivers
Material Unfavourable Drivers
Root Causes and Business Impact
Risks and Exceptions
Management Actions
Outlook
Source References
Data Limitations
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
Explain the supplied KPI results for management decision-making.

Identify the reporting period, comparison basis, actual value, target or
benchmark, variance or status, strongest and weakest KPIs, material trends,
relationships between KPIs, supported drivers, anomalies, and management
attention areas.

Use KPI results as the primary evidence and supporting agent outputs only when
they directly explain a KPI movement. Do not calculate missing KPI values.
""".strip(),
    output_format="""
Return:

Executive KPI Summary
KPI Scorecard
Positive Performance
Underperformance and Exceptions
Key Drivers and KPI Relationships
Business Impact
Management Actions
Source References
Data Limitations
""".strip(),
)


BUDGET_ANALYSIS_TEMPLATE = PromptTemplate(
    prompt_type=PromptType.BUDGET_ANALYSIS,
    title="Budget Analysis",
    instructions="""
Explain the supplied budget outputs for management review.

Identify the budget period, major revenue and cost allocations, profitability
or margin expectations, key assumptions, material concentrations, validation
or approval issues, operational constraints, risks, opportunities, and
required management actions.

Use budget-agent and finance-rules outputs as primary evidence. Do not alter
the budget, create assumptions, or estimate missing values.
""".strip(),
    output_format="""
Return:

Executive Budget Summary
Budget Profile and Major Allocations
Revenue, Cost, Profit, and Margin View
Key Assumptions
Concentration and Constraint Analysis
Risks and Opportunities
Validation or Approval Matters
Management Actions
Source References
Data Limitations
""".strip(),
)


FORECAST_ANALYSIS_TEMPLATE = PromptTemplate(
    prompt_type=PromptType.FORECAST_ANALYSIS,
    title="Forecast Analysis",
    instructions="""
Explain the supplied forecast outputs for management decision-making.

Identify the forecast horizon, expected revenue, cost, profit, margin, or KPI
outcomes, trend direction, major forecast drivers, differences from budget or
prior forecast, assumption sensitivity, uncertainty, risks to delivery,
anomalies, and monitoring indicators.

Clearly distinguish forecast-agent outputs from retrieved reference context.
Do not extend the forecast, create new assumptions, or calculate missing
values.
""".strip(),
    output_format="""
Return:

Executive Forecast Summary
Forecast Horizon and Expected Performance
Budget or Prior-Forecast Comparison
Key Forecast Drivers
Assumptions and Sensitivities
Risks and Uncertainty
Monitoring Indicators
Management Actions and Outlook
Source References
Data Limitations
""".strip(),
)


VARIANCE_ANALYSIS_TEMPLATE = PromptTemplate(
    prompt_type=PromptType.VARIANCE_ANALYSIS,
    title="Variance Analysis",
    instructions="""
Explain the supplied actual-versus-budget, actual-versus-forecast, or
actual-versus-prior-period variance outputs for CFO and FP&A review.

Identify the metric, entity, period, comparison basis, actual value, comparison
value, absolute variance, percentage variance, and favourable or unfavourable
status when supplied.

Explain the supplied price, volume, mix, quantity, cost, margin, rate,
efficiency, or other bridge effects. Use anomaly, root-cause,
recommendation, KPI, commentary, and report outputs when they directly support
the variance explanation.

Confirm whether the bridge reconciles to the reported total variance and
surface any validation issue. Use only calculations already present in the
analysis. Do not infer missing effects or force a reconciliation.

Numerical presentation rules:

- Copy actual, comparison, variance, percentage, and bridge-effect values
  exactly from the supplied finance analysis.
- Do not independently convert values into lakhs, crores, thousands,
  millions, or billions.
- Do not add a currency symbol or financial unit unless that symbol or unit is
  explicitly supplied by the finance analysis.
- If a preformatted display value is available, use it instead of formatting
  the raw number.
- Ensure the Executive Summary uses the same exact financial values and units
  shown in the Financial Performance and Variance Bridge sections.
- If the supplied context contains conflicting values or units, identify the
  conflict under Data Limitations rather than selecting or recalculating one.
""".strip(),
    output_format="""
Return:

Executive Summary
Financial Performance
- Actual value
- Budget, forecast, or prior-period value
- Absolute variance
- Variance percentage
- Favourable or unfavourable status

Variance Bridge
- Price effect
- Volume or quantity effect
- Mix effect
- Cost, margin, rate, efficiency, or other supplied effects
- Reconciliation status

Favourable Drivers
Unfavourable Drivers
Root-Cause Assessment
Business Impact
Recommendations and Management Actions
Risks and Monitoring Indicators
Management Commentary
Source References
Data Limitations

For any unavailable requested item, write "Not available in the supplied
analysis". Do not omit the limitation or invent a value.
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
Explain and compare the supplied scenario-analysis outputs.

Identify each scenario, the assumptions that change, and the supplied impact
on revenue, cost, profit, margin, cash, or KPIs. Compare base, upside,
downside, management, or other named scenarios without assuming that all
three standard scenarios exist.

Highlight sensitivities, breakpoints, risks, opportunities, decision
implications, and monitoring triggers supported by the analysis. Do not create
new scenarios or recalculate existing scenarios.
""".strip(),
    output_format="""
Return:

Executive Scenario Summary
Scenario Comparison
Key Assumption Changes
Revenue, Cost, Profit, Margin, and KPI Impact
Sensitivity and Key Breakpoints
Risks and Opportunities
Decision Implications
Recommended Management Actions
Monitoring Triggers
Source References
Data Limitations
""".strip(),
)


PNL_ANALYSIS_TEMPLATE = PromptTemplate(
    prompt_type=PromptType.PNL_ANALYSIS,
    title="Profit and Loss Analysis",
    instructions="""
Present the supplied P&L agent results as a professional FP&A Profit and Loss
statement for CFO, Finance Director, and management review.

Use pnl_result as the primary source for all P&L amounts, margins, variance
amounts, variance percentages, monthly results, consolidated results,
available months, and excluded-month information.

Use pnl_commentary_result only for management commentary and supported
business interpretation.

Do not recalculate, derive, estimate, reconcile, aggregate, or modify any P&L
value. Use the values already calculated and supplied by the P&L Agent.

P&L presentation rules:

1. Present P&L line items in this order when the corresponding values are
   supplied:

   - Revenue
   - Less: Direct Cost
   - Gross Profit
   - Gross Margin %
   - Less: Sales & Marketing
   - Less: Other OPEX
   - EBITDA
   - Less: Depreciation
   - EBIT
   - Less: Interest
   - EBT
   - Less: Income Tax
   - Net Profit

2. Use a Markdown table with these columns:

   | Particulars | Actual | Budget | Variance | Variance % |

3. Copy Actual, Budget, Variance, and Variance % values exactly from the
   supplied P&L agent output.

4. Do not calculate variance amount or variance percentage in the prompt
   layer.

5. For Gross Margin %, use the supplied:

   - gross_margin_percentage_actual
   - gross_margin_percentage_budget
   - gross_margin_percentage_point_variance

6. Display Gross Margin % variance as a percentage-point variance using the
   suffix "pp".

7. Do not display Gross Margin % variance as a normal monetary variance
   percentage.

8. When the supplied percentage-point variance is negative, preserve the
   negative sign. When positive, preserve the positive value as supplied.

9. For the Gross Margin % row:

   - Actual column must show actual gross margin percentage.
   - Budget column must show budget gross margin percentage.
   - Variance column must show the supplied percentage-point variance with
     "pp".
   - Variance % column must show an em dash because a second percentage
     variance is not applicable.

10. Do not insert a currency symbol unless the supplied P&L result contains
    that currency symbol or a preformatted display value.

11. Do not convert values into lakhs, crores, thousands, millions, or
    billions unless the supplied P&L result already contains those converted
    values.

12. For a period-specific request, such as "Generate P&L for April 2026",
    present the P&L table for that requested period.

13. For a general request such as "pnl", present:

    - Consolidated Actual vs Budget P&L
    - Monthly P&L for all available months
    - Variance amount
    - Variance percentage
    - Actual Gross Margin %
    - Budget Gross Margin %
    - Gross Margin % variance in percentage points
    - Management commentary after all tables

14. When monthly P&L results are supplied, present one monthly comparison
    table per month or one clearly structured monthly table that preserves all
    supplied values.

15. Do not omit available P&L lines only to shorten the response.

16. If a P&L line is not present in the supplied analysis, do not invent it.
    State the limitation only when the missing line is relevant to the user's
    request.

Variance interpretation rules:

- For revenue and profit lines, a positive variance is normally favourable
  and a negative variance is normally unfavourable.
- For cost and expense lines, a negative variance is normally favourable and
  a positive variance is normally unfavourable.
- For Gross Margin %, a positive percentage-point variance is favourable and
  a negative percentage-point variance is unfavourable.
- Apply favourability according to the nature of the line item, not only the
  mathematical sign.
- Use supplied commentary and supporting evidence when available.
- Do not invent a business cause for any variance.
- Clearly separate factual variance results from supported interpretation.

Commentary rules:

- Place management commentary after the P&L tables.
- Identify the most material favourable variances.
- Identify the most material unfavourable variances.
- Explain profitability and margin performance using only supplied evidence.
- Include risks, exceptions, and management attention only when supported.
- Avoid repeating every value already displayed in the tables.
- Mention data limitations only when information is genuinely missing,
  excluded, inconsistent, or unavailable.
""".strip(),
    output_format="""
Return the response in the following order:

# Profit and Loss Statement

## Reporting Period

State the requested period or consolidated period using the supplied analysis.

## Actual vs Budget P&L

Use this Markdown table structure:

| Particulars | Actual | Budget | Variance | Variance % |
|---|---:|---:|---:|---:|
| Revenue | supplied value | supplied value | supplied value | supplied value |
| Less: Direct Cost | supplied value | supplied value | supplied value | supplied value |
| Gross Profit | supplied value | supplied value | supplied value | supplied value |
| Gross Margin % | supplied value | supplied value | supplied pp variance | — |
| Less: Sales & Marketing | supplied value | supplied value | supplied value | supplied value |
| Less: Other OPEX | supplied value | supplied value | supplied value | supplied value |
| EBITDA | supplied value | supplied value | supplied value | supplied value |
| Less: Depreciation | supplied value | supplied value | supplied value | supplied value |
| EBIT | supplied value | supplied value | supplied value | supplied value |
| Less: Interest | supplied value | supplied value | supplied value | supplied value |
| EBT | supplied value | supplied value | supplied value | supplied value |
| Less: Income Tax | supplied value | supplied value | supplied value | supplied value |
| Net Profit | supplied value | supplied value | supplied value | supplied value |

Include only line items supported by the supplied analysis.

For Gross Margin %:

- Actual = supplied actual Gross Margin %
- Budget = supplied budget Gross Margin %
- Variance = supplied percentage-point variance followed by "pp"
- Variance % = —

## Monthly P&L

Include this section when monthly P&L results are supplied or when the user
requests a general consolidated P&L.

Present all available months using the supplied monthly P&L results. Preserve
the supplied month labels and values.

## Management Commentary

### Executive Summary

Provide a concise summary of revenue, gross profit, Gross Margin %, EBITDA,
EBIT, EBT, income tax, and Net Profit performance when supplied.

### Material Favourable Variances

List the most material favourable movements and explain why they are
favourable based on the P&L line type.

### Material Unfavourable Variances

List the most material unfavourable movements and explain why they are
unfavourable based on the P&L line type.

### Margin and Profitability Assessment

Explain actual Gross Margin %, budget Gross Margin %, and the supplied
percentage-point variance. Do not calculate a new margin variance.

### Risks, Exceptions, and Management Attention

Include only supported risks, exceptions, and actions.

## Data Limitations

Include this section only when required data is genuinely unavailable,
excluded, inconsistent, or missing.

For any requested value that is unavailable, write:
"Not available in the supplied analysis".

Do not replace unavailable values with estimates.
""".strip(),
)

GP_VARIANCE_ANALYSIS_TEMPLATE = PromptTemplate(
    prompt_type=PromptType.GP_VARIANCE_ANALYSIS,
    title="Gross Margin Percentage Variance Decomposition",
    instructions="""
Present the supplied gp_variance_result as an FP&A Gross Margin % bridge.
Use only agent-calculated values. Do not recalculate or invent causes.
Show Budget GP%, Mix-only GP%, Price-only GP%, Actual GP%, and the Mix,
Price, and Cost effects in both percentage points and basis points.
Classify positive effects as favourable and negative effects as
unfavourable. Show the reconciliation difference and PASS/FAIL status.
Report unmatched categories and data limitations. Explain mathematical
drivers without claiming unsupported operational causes.
""".strip(),
    output_format="""
# Gross Margin % Variance Decomposition

## Reporting Period

## GP% Bridge

| Step | GP% | Effect (pp) |
|---|---:|---:|
| Budget GP% | supplied value | — |
| Mix-only GP% | supplied value | supplied mix effect |
| Price-only GP% | supplied value | supplied price effect |
| Actual GP% | supplied value | supplied cost effect |

## Driver Summary

| Driver | Percentage Points | Basis Points | Classification |
|---|---:|---:|---|
| Mix | supplied value | supplied value | supplied interpretation |
| Price | supplied value | supplied value | supplied interpretation |
| Cost | supplied value | supplied value | supplied interpretation |
| Total | supplied value | supplied value | supplied interpretation |

## Category Unit Economics

## Reconciliation Check

## Management Commentary

## Data Limitations
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
    PromptType.PNL_ANALYSIS: PNL_ANALYSIS_TEMPLATE,
    PromptType.GP_VARIANCE_ANALYSIS: GP_VARIANCE_ANALYSIS_TEMPLATE,
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


def build_pnl_analysis_prompt(
    *,
    user_request: str,
    finance_analysis: Any,
    retrieved_context: str = "",
    additional_instructions: str | None = None,
) -> PromptMessages:
    """Build a specialist Profit and Loss analysis prompt."""

    return build_prompt_messages(
        PromptType.PNL_ANALYSIS,
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
