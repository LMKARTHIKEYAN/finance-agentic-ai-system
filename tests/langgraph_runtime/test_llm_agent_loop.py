from types import SimpleNamespace

import pandas as pd

from src.autonomous.langgraph_runtime.runtime_service import LangGraphRuntimeService
from src.autonomous.langgraph_runtime.reviewer_node import GraphReview
from src.autonomous.langgraph_runtime.supervisor_node import SupervisorDecision
from src.autonomous.tools.data_tools import FinanceDataContext
from src.llm.client import StructuredLLMClient


class SequencedLLM(StructuredLLMClient):
    def __init__(self):
        self.supervisor_calls = 0

    @property
    def provider(self):
        return "test"

    @property
    def model(self):
        return "test-model"

    def generate_structured(self, *, messages, response_model, **kwargs):
        if response_model is SupervisorDecision:
            self.supervisor_calls += 1
            output = SupervisorDecision(
                action="call_tool", rationale="Need period evidence.",
                tool_name="compare_periods",
                arguments={"current_start": "2026-07-01", "current_end": "2026-07-31",
                           "comparison_start": "2026-06-01", "comparison_end": "2026-06-30"},
            ) if self.supervisor_calls == 1 else SupervisorDecision(
                action="validate", rationale="Evidence is sufficient."
            )
        elif response_model is GraphReview:
            output = GraphReview(decision="approved", rationale="Evidence is validated.")
        else:
            raise AssertionError(response_model)
        return SimpleNamespace(output=output)


def test_llm_decide_act_observe_decide_validate_review_finish():
    orders = pd.DataFrame({
        "order_date": pd.to_datetime(["2026-06-10", "2026-07-10"]),
        "order_id": ["a", "b"], "order_status": ["Completed", "Completed"],
        "commission_amount": [100.0, 120.0], "incentive": [1.0, 1.0],
        "goodwill": [0.0, 0.0], "dry_run": [0.0, 0.0], "surge": [0.0, 0.0],
    })
    llm = SequencedLLM()
    service = LangGraphRuntimeService(llm=llm)
    result = service.run(
        "Why did July revenue change versus June?",
        context={"finance_context": FinanceDataContext(operations_data=orders)},
        thread_id="agent-loop-test",
    )
    nodes = [item["node"] for item in result["execution_trace"]]
    assert llm.supervisor_calls == 2
    assert nodes == ["supervisor", "tool_executor", "observation", "supervisor", "validation", "reviewer", "final_answer"]
    assert result["completed"] is True
    assert result["evidence"][0]["evidence_id"].startswith("graph-evidence-")
