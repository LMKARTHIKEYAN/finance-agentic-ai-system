from src.autonomous.langgraph_runtime.validation_node import ValidationNode


def test_default_validation_requires_evidence():
    assert ValidationNode()({"evidence": []})["next_action"] == "supervise"
    assert ValidationNode()({"evidence": [{"evidence_id": "e-1"}]})["next_action"] == "review"
