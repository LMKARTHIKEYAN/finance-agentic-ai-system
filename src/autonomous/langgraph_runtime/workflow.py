"""Build and compile the complex autonomous LangGraph workflow."""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.autonomous.langgraph_runtime.approval_node import approval_node
from src.autonomous.langgraph_runtime.clarification_node import clarification_node
from src.autonomous.langgraph_runtime.final_answer_node import final_answer_node
from src.autonomous.langgraph_runtime.graph_routes import *
from src.autonomous.langgraph_runtime.graph_state import FinanceGraphState
from src.autonomous.langgraph_runtime.observation_node import observation_node
from src.autonomous.langgraph_runtime.recovery_node import recovery_node


def build_workflow(*, supervisor, tool_executor, validator, reviewer, checkpointer=None):
    graph = StateGraph(FinanceGraphState)
    graph.add_node("supervisor", supervisor)
    graph.add_node("execute_tool", tool_executor)
    graph.add_node("observe", observation_node)
    graph.add_node("validate", validator)
    graph.add_node("reviewer", reviewer)
    graph.add_node("clarify", clarification_node)
    graph.add_node("approve", approval_node)
    graph.add_node("recover", recovery_node)
    graph.add_node("final_answer", final_answer_node)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route_after_supervisor)
    graph.add_edge("execute_tool", "observe")
    graph.add_conditional_edges("observe", route_after_observation)
    graph.add_conditional_edges("validate", route_after_validation)
    graph.add_conditional_edges("reviewer", route_after_reviewer)
    graph.add_conditional_edges("recover", route_after_recovery)
    graph.add_edge("clarify", END)
    graph.add_edge("approve", END)
    graph.add_edge("final_answer", END)
    return graph.compile(checkpointer=checkpointer or MemorySaver())
