from langgraph.checkpoint.memory import MemorySaver

from src.autonomous.langgraph_runtime.workflow import build_workflow


def test_graph_can_finish_after_validation_and_review():
    def supervisor(state):
        return {"next_action": "validate", "decision": {"action": "validate"}}
    def validator(state):
        return {"validation": {"passed": True}, "next_action": "review"}
    def reviewer(state):
        return {"review": {"decision": "approved"}, "next_action": "finalize"}
    graph = build_workflow(supervisor=supervisor, tool_executor=lambda state: {}, validator=validator, reviewer=reviewer, checkpointer=MemorySaver())
    result = graph.invoke({"request": "Investigate revenue", "evidence": [{"evidence_id": "e-1"}], "step_count": 0, "max_steps": 8}, config={"configurable": {"thread_id": "test"}})
    assert result["completed"] is True
    assert "e-1" in result["final_answer"]
