"""Small Streamlit presentation adapter for autonomous responses."""

from __future__ import annotations

from typing import Any


def render_autonomous_response(st: Any, response: dict[str, Any]) -> None:
    status = response.get("status", "unknown")
    st.caption(f"Autonomous status: {status}")
    if response.get("question"):
        st.warning(response["question"])
    if response.get("answer"):
        st.markdown(response["answer"])
    evidence = response.get("evidence") or []
    if evidence:
        with st.expander(f"Verified evidence ({len(evidence)})"):
            st.json(evidence)
