from src.autonomous.langgraph_runtime.tool_catalog import LangGraphToolCatalog


def test_catalog_exposes_existing_approved_tools():
    catalog = LangGraphToolCatalog()
    assert catalog.contains("generate_validated_pnl_analysis")
    assert all("required_inputs" in item for item in catalog.describe())
