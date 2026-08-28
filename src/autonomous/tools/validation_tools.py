"""Deterministic validation tools for autonomous finance outputs."""

from __future__ import annotations

from typing import Any, Iterable


def validate_revenue_definition(payloads: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Reject evidence that treats fare as company revenue."""

    violations: list[str] = []
    for index, payload in enumerate(payloads):
        definition = payload.get("revenue_definition")
        if definition is not None and definition != "commission_amount":
            violations.append(f"payload[{index}] uses {definition!r}")
    return {
        "passed": not violations,
        "revenue_definition": "commission_amount",
        "violations": violations,
        "summary": (
            "Revenue-definition validation passed."
            if not violations
            else "Revenue-definition validation failed."
        ),
    }


def validate_citations(
    *, answer: str, evidence_ids: Iterable[str]
) -> dict[str, Any]:
    ids = tuple(dict.fromkeys(str(item).strip() for item in evidence_ids if str(item).strip()))
    missing = tuple(item for item in ids if f"[{item}]" not in answer)
    return {
        "passed": bool(ids) and not missing,
        "evidence_ids": ids,
        "missing_citations": missing,
        "summary": "Citation validation passed." if ids and not missing else "Citation validation failed.",
    }
