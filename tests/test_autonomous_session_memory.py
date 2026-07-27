"""Tests for safe autonomous session context."""

import pandas as pd
import pytest

from src.memory.session_memory import SessionMemory


def test_autonomous_session_context_round_trips_and_is_isolated() -> None:
    memory = SessionMemory()
    first = memory.create_session("first")
    second = memory.create_session("second")
    context = {
        "execution_mode": "autonomous",
        "reporting_scope": {"period": "2026-04", "category": "A"},
        "evidence_ids": ["pnl-001"],
        "review_decision": "approved",
    }

    saved = memory.set_autonomous_context(first, context)
    saved["evidence_ids"].append("changed")

    assert memory.get_autonomous_context(first) == context
    assert memory.get_autonomous_context(second) == {}


def test_autonomous_session_rejects_dataframe() -> None:
    memory = SessionMemory()
    session_id = memory.create_session()

    with pytest.raises(TypeError, match="DataFrames"):
        memory.set_autonomous_context(
            session_id,
            {"raw": pd.DataFrame({"revenue": [100]})},
        )


@pytest.mark.parametrize(
    "field",
    ["api_key", "prompt", "traceback", "raw_dataframe"],
)
def test_autonomous_session_rejects_sensitive_fields(field: str) -> None:
    memory = SessionMemory()
    session_id = memory.create_session()

    with pytest.raises(ValueError, match="sensitive"):
        memory.set_autonomous_context(
            session_id,
            {field: "secret"},
        )
