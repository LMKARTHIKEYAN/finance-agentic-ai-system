"""Reuse the project's provider-neutral structured LLM client."""

from src.llm.client import StructuredLLMClient
from src.llm.factory import create_llm_client


def build_langgraph_llm() -> StructuredLLMClient:
    return create_llm_client()
