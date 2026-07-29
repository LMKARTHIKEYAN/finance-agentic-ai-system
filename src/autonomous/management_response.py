"""Deterministic management responses grounded in diagnostic evidence."""

from __future__ import annotations

from typing import Any

from src.autonomous.agents.root_cause_recommendation_agent import (
    DiagnosticAnalysisResult,
)


class ManagementResponseComposer:
    """Verbalize trusted diagnostics without calculating finance values."""

    def compose(
        self,
        diagnostics: DiagnosticAnalysisResult,
        *,
        fallback_answer: str,
    ) -> str:
        """Return a grounded P&L response or the existing safe fallback."""

        if not isinstance(diagnostics, DiagnosticAnalysisResult):
            raise TypeError(
                "diagnostics must be a DiagnosticAnalysisResult."
            )
        cleaned_fallback = _required_text(
            fallback_answer,
            "fallback_answer",
        )
        root_cause = diagnostics.root_cause_result.payload
        if (
            root_cause.get("analysis_type")
            == "multi_evidence_performance"
        ):
            return _compose_multi_evidence_response(
                diagnostics
            )
        if root_cause.get("analysis_type") != "pnl_period_change":
            return cleaned_fallback

        base_month = _required_payload_text(
            root_cause,
            "base_month",
        )
        current_month = _required_payload_text(
            root_cause,
            "current_month",
        )
        base_profit = _required_number(
            root_cause,
            "base_net_profit",
        )
        current_profit = _required_number(
            root_cause,
            "current_net_profit",
        )
        profit_change = _required_number(
            root_cause,
            "net_profit_change",
        )
        movement = (
            "increased"
            if profit_change > 0
            else "decreased"
            if profit_change < 0
            else "was unchanged"
        )
        response_parts = [
            (
                f"Net profit {movement} by "
                f"{_currency(profit_change)} from "
                f"{_currency(base_profit)} in {base_month} to "
                f"{_currency(current_profit)} in {current_month}."
            )
        ]

        driver_sentences = _driver_sentences(
            root_cause.get("drivers")
        )
        if driver_sentences:
            response_parts.append(
                "Primary profit drivers: "
                + " ".join(driver_sentences)
            )

        recommendation_sentences = _recommendation_sentences(
            diagnostics.recommendation_result.payload.get(
                "recommendations"
            )
        )
        if recommendation_sentences:
            response_parts.append(
                "Management actions: "
                + " ".join(recommendation_sentences)
            )

        response_parts.append(
            "Evidence: "
            + ", ".join(diagnostics.evidence_ids)
            + "."
        )
        return "\n\n".join(response_parts)


def _compose_multi_evidence_response(
    diagnostics: DiagnosticAnalysisResult,
) -> str:
    """Verbalize existing multi-specialist findings and linked actions."""

    root_cause = diagnostics.root_cause_result.payload
    findings = root_cause.get("findings")
    if not isinstance(findings, list) or not findings:
        raise ValueError(
            "Multi-evidence diagnostics require supported findings."
        )

    finding_sentences = [
        sentence
        for sentence in (
            _finding_sentence(item)
            for item in findings[:12]
        )
        if sentence is not None
    ]
    if not finding_sentences:
        raise ValueError(
            "Multi-evidence diagnostics contain no displayable findings."
        )

    risk_findings = root_cause.get("risk_findings")
    risk_sentences = (
        [
            sentence
            for sentence in (
                _finding_sentence(item)
                for item in risk_findings[:5]
            )
            if sentence is not None
        ]
        if isinstance(risk_findings, list)
        else []
    )
    recommendation_sentences = _recommendation_sentences(
        diagnostics.recommendation_result.payload.get(
            "recommendations"
        )
    )

    sections = [
        "Performance findings: " + " ".join(finding_sentences)
    ]
    sections.append(
        (
            "Financial risk indicators: "
            + " ".join(risk_sentences)
        )
        if risk_sentences
        else (
            "Financial risk indicators: No adverse indicator was "
            "identified in the supplied reconciled evidence."
        )
    )
    if recommendation_sentences:
        sections.append(
            "Management actions: "
            + " ".join(recommendation_sentences)
        )
    sections.append(
        "Evidence: "
        + ", ".join(diagnostics.evidence_ids)
        + "."
    )
    return "\n\n".join(sections)


def _finding_sentence(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    metric = value.get("metric")
    finding_value = value.get("value")
    unit = value.get("unit")
    if not isinstance(metric, str):
        return None
    if (
        isinstance(finding_value, bool)
        or not isinstance(finding_value, (int, float, str))
    ):
        return None
    return (
        f"{metric.replace('_', ' ').title()}: "
        f"{_display_value(finding_value, unit)} "
        f"[{value.get('evidence_id', 'verified evidence')}]."
    )


def _display_value(value: int | float | str, unit: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return str(value)
    if unit == "currency":
        return _currency(value)
    if unit in {"percentage", "percentage_points"}:
        suffix = "%" if unit == "percentage" else " pp"
        return f"{value:,.2f}{suffix}"
    return f"{value:,.2f}"


def _driver_sentences(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    sentences: list[str] = []
    for item in value[:5]:
        if not isinstance(item, dict):
            continue
        metric = item.get("metric")
        change = item.get("change")
        profit_effect = item.get("profit_effect")
        impact = item.get("impact")
        if (
            not isinstance(metric, str)
            or not _is_number(change)
            or not _is_number(profit_effect)
            or impact not in {"favorable", "unfavorable"}
        ):
            continue
        sentences.append(
            f"{metric.replace('_', ' ').title()} changed by "
            f"{_currency(change)}, with a "
            f"{_currency(profit_effect)} profit effect "
            f"({impact})."
        )
    return sentences


def _recommendation_sentences(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    sentences: list[str] = []
    for item in value[:5]:
        if not isinstance(item, dict):
            continue
        action = item.get("action") or item.get("recommended_action")
        if isinstance(action, str) and action.strip():
            sentences.append(action.strip().rstrip(".") + ".")
    return sentences


def _required_payload_text(
    payload: dict[str, Any],
    field_name: str,
) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Diagnostic field {field_name!r} must be non-empty text."
        )
    return value.strip()


def _required_number(
    payload: dict[str, Any],
    field_name: str,
) -> float:
    value = payload.get(field_name)
    if not _is_number(value):
        raise ValueError(
            f"Diagnostic field {field_name!r} must be numeric."
        )
    return float(value)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _currency(value: int | float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}₹{abs(value):,.2f}"


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty.")
    return cleaned
