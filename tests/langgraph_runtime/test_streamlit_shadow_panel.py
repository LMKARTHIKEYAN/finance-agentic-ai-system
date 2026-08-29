from pathlib import Path


def test_streamlit_retains_and_renders_shadow_metadata():
    source = Path("src/ui/autonomous_app.py").read_text(encoding="utf-8")
    assert '"langgraph_shadow": response.get("langgraph_shadow") or {}' in source
    assert "def _render_langgraph_shadow" in source
    assert "LLM-selected tools" in source
    assert "Execution trace" in source
