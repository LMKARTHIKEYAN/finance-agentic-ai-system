import pytest

from src.autonomous.langgraph_runtime.runtime_service import LangGraphRuntimeService


def test_runtime_rejects_empty_request_without_calling_graph():
    service = object.__new__(LangGraphRuntimeService)
    with pytest.raises(ValueError, match="request cannot be empty"):
        service.run("  ")
