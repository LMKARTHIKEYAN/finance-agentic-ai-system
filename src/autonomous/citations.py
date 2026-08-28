"""Citation formatting for finance and RAG evidence."""

from __future__ import annotations

from collections.abc import Iterable

from src.autonomous.schemas import EvidenceRecord


def append_evidence_citations(
    answer: str, records: Iterable[EvidenceRecord]
) -> str:
    """Append stable evidence IDs and source labels to a final answer."""

    citations = []
    for record in records:
        label = record.source_tool
        if record.period:
            label += f", {record.period}"
        if record.category:
            label += f", vehicle category: {record.category}"
        citations.append(f"[{record.evidence_id}] {label}")
    if not citations:
        return answer.strip()
    return answer.strip() + "\n\nEvidence:\n" + "\n".join(f"- {item}" for item in citations)
