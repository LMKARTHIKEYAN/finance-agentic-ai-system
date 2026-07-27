"""Tests for reviewed autonomous long-term summaries."""

from pathlib import Path

import pandas as pd
import pytest

from src.memory.long_term_memory import LongTermMemory


def _summary() -> dict:
    return {
        "execution_mode": "autonomous",
        "autonomous_status": "completed",
        "review_decision": "approved_with_caveats",
        "required_caveats": ["Limited period coverage."],
        "reconciliation_passed": True,
        "evidence_ids": ["pnl-001"],
        "usage": {"total_tokens": 500, "estimated_cost_usd": 0.01},
        "management_answer": "Reviewed answer.",
    }


def test_reviewed_summary_persists_across_instances(
    tmp_path: Path,
) -> None:
    path = tmp_path / "autonomous.db"
    first = LongTermMemory(path)
    first.save_autonomous_summary(key="april", summary=_summary())
    first.close()

    second = LongTermMemory(path)
    restored = second.get_autonomous_summary("april")
    second.close()

    assert restored == _summary()


@pytest.mark.parametrize("decision", [None, "failed", "replan_required"])
def test_unapproved_summary_is_not_persisted(decision: str | None) -> None:
    memory = LongTermMemory(":memory:")
    summary = {**_summary(), "review_decision": decision}

    with pytest.raises(ValueError, match="reviewed"):
        memory.save_autonomous_summary(key="unsafe", summary=summary)


def test_long_term_summary_rejects_dataframe() -> None:
    memory = LongTermMemory(":memory:")

    with pytest.raises(TypeError, match="DataFrames"):
        memory.save_autonomous_summary(
            key="unsafe",
            summary={**_summary(), "raw": pd.DataFrame({"x": [1]})},
        )


@pytest.mark.parametrize("field", ["api_key", "prompt", "exception"])
def test_long_term_summary_rejects_sensitive_fields(field: str) -> None:
    memory = LongTermMemory(":memory:")

    with pytest.raises(ValueError, match="sensitive"):
        memory.save_autonomous_summary(
            key="unsafe",
            summary={**_summary(), field: "secret"},
        )
