"""Deterministic-first classifier for hybrid finance request routing."""

from __future__ import annotations

import re

from src.autonomous.schemas import ComplexityDecision
from src.llm.client import StructuredLLMClient


_COMPLEX_PATTERNS: tuple[str, ...] = (
    r"\bwhy\b",
    r"\bwhat caused\b",
    r"\broot cause\b",
    r"\bidentify (?:the )?risks?\b",
    r"\brecommend (?:an )?actions?\b",
    r"\bwhat should management do\b",
    r"\bmanagement actions?\b",
    r"\bexplain (?:the )?(?:drivers?|change|movement)\b",
    r"\banaly[sz]e performance\b.*\b(?:risks?|recommend)\b",
)

_SIMPLE_ACTION_PATTERNS: tuple[str, ...] = (
    r"^\s*show\b",
    r"^\s*generate\b",
    r"^\s*prepare\b",
    r"^\s*calculate\b",
    r"^\s*compare\b",
    r"^\s*display\b",
)

_SUPPORTED_SIMPLE_SUBJECTS: tuple[str, ...] = (
    "kpi",
    "pnl",
    "p&l",
    "profit and loss",
    "actual vs budget",
    "actual versus budget",
    "revenue variance",
    "gp% decomposition",
    "gp percentage decomposition",
    "gross margin decomposition",
    "budget",
    "forecast",
    "scenario",
)


class ComplexityClassifier:
    """Classify requests without using an LLM when rules are sufficient."""

    def __init__(
        self,
        *,
        llm_client: StructuredLLMClient | None = None,
        llm_enabled: bool = False,
        confidence_threshold: float = 0.8,
    ) -> None:
        if not isinstance(llm_enabled, bool):
            raise TypeError("llm_enabled must be a boolean.")
        if isinstance(confidence_threshold, bool) or not isinstance(
            confidence_threshold,
            (int, float),
        ):
            raise TypeError("confidence_threshold must be numeric.")
        if not 0.0 <= float(confidence_threshold) <= 1.0:
            raise ValueError(
                "confidence_threshold must be between 0 and 1."
            )

        self._llm_client = llm_client
        self._llm_enabled = llm_enabled
        self._confidence_threshold = float(confidence_threshold)

    def classify(self, request: str) -> ComplexityDecision:
        """Return a safe structured complexity decision."""

        cleaned_request = _validate_request(request)
        normalized = " ".join(cleaned_request.lower().split())

        if any(
            re.search(pattern, normalized)
            for pattern in _COMPLEX_PATTERNS
        ):
            return ComplexityDecision(
                execution_mode="autonomous",
                request_type=_resolve_complex_request_type(normalized),
                confidence=1.0,
                reasons=(
                    "The request asks for diagnostic or decision-support "
                    "reasoning.",
                ),
                fallback_flow=_infer_fallback_flow(normalized),
            )

        if _is_clear_simple_request(normalized):
            return ComplexityDecision(
                execution_mode="deterministic",
                request_type="simple_report",
                confidence=1.0,
                reasons=(
                    "The request maps directly to an existing deterministic "
                    "finance workflow.",
                ),
                fallback_flow=_infer_fallback_flow(normalized),
            )

        return self._classify_ambiguous(cleaned_request)

    def _classify_ambiguous(self, request: str) -> ComplexityDecision:
        fallback = ComplexityDecision(
            execution_mode="deterministic",
            request_type="unknown",
            confidence=0.5,
            reasons=(
                "The request is ambiguous, so deterministic fallback was "
                "selected.",
            ),
            fallback_flow=_infer_fallback_flow(request.lower()),
        )

        if not self._llm_enabled or self._llm_client is None:
            return fallback

        try:
            result = self._llm_client.generate_structured(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Classify the finance request as deterministic "
                            "or autonomous. Direct reports and calculations "
                            "are deterministic. Diagnostics, causal analysis, "
                            "risk analysis, and recommendations are autonomous."
                        ),
                    },
                    {
                        "role": "user",
                        "content": request,
                    },
                ],
                response_model=ComplexityDecision,
                max_output_tokens=250,
            )
        except Exception:
            return fallback

        decision = result.output

        if decision.confidence < self._confidence_threshold:
            return fallback

        return decision


def _validate_request(request: str) -> str:
    if not isinstance(request, str):
        raise TypeError("request must be a string.")

    cleaned = " ".join(request.split())

    if not cleaned:
        raise ValueError("request cannot be empty.")

    return cleaned


def _is_clear_simple_request(normalized: str) -> bool:
    has_simple_action = any(
        re.search(pattern, normalized)
        for pattern in _SIMPLE_ACTION_PATTERNS
    )
    has_supported_subject = any(
        subject in normalized
        for subject in _SUPPORTED_SIMPLE_SUBJECTS
    )
    return has_simple_action and has_supported_subject


def _resolve_complex_request_type(normalized: str) -> str:
    has_risk_or_action = any(
        phrase in normalized
        for phrase in (
            "risk",
            "recommend",
            "what should management do",
            "management action",
        )
    )
    has_diagnostic = any(
        phrase in normalized
        for phrase in ("why", "what caused", "root cause", "driver")
    )

    if has_risk_or_action and has_diagnostic:
        return "multi_analysis"
    if has_risk_or_action:
        return "decision_support"
    return "diagnostic"


def _infer_fallback_flow(normalized: str) -> str:
    if any(
        value in normalized
        for value in (
            "gp%",
            "gp percentage",
            "gross margin",
            "margin change",
            "margin movement",
        )
    ):
        return "gp_variance"
    if any(
        value in normalized
        for value in ("pnl", "p&l", "profit and loss", "profit")
    ):
        return "pnl"
    if any(
        value in normalized
        for value in ("variance", "actual vs budget", "revenue")
    ):
        return "variance"
    if "forecast" in normalized:
        return "forecast"
    if "scenario" in normalized:
        return "scenario"
    if "budget" in normalized:
        return "budget"
    if "kpi" in normalized or "performance" in normalized:
        return "kpi"
    return "unknown"
