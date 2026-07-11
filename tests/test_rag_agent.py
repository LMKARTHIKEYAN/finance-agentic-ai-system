"""
Tests for the Finance Agentic AI RAG agent.

These tests cover:

- RAG agent configuration
- RAG request validation
- RAG result validation
- Deterministic response generation
- Optional LLM generation
- Deterministic fallback
- Retrieval integration
- Prompt integration
- Helper functions
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Sequence

import pytest

from src.rag.embeddings import BaseEmbeddingService
from src.rag.prompt_templates import (
    PromptMessages,
    PromptType,
)
from src.rag.rag_agent import (
    DEFAULT_MODEL,
    BaseResponseGenerator,
    DeterministicResponseGenerator,
    ExecutionMode,
    FinanceRAGAgent,
    GenerationError,
    OpenAIResponseGenerator,
    RAGAgentConfig,
    RAGAgentError,
    RAGRequest,
    RAGResult,
    build_deterministic_rag_response,
    create_rag_agent,
    summarize_rag_result,
)
from src.rag.retriever import (
    FinanceRetriever,
    RetrievalConfig,
    RetrievalResult,
)
from src.rag.vector_store import (
    Document,
    InMemoryVectorStore,
)


class FixedEmbeddingService(BaseEmbeddingService):
    """Predictable embedding service for RAG tests."""

    def __init__(
        self,
        vectors: dict[str, list[float]] | None = None,
        dimension: int = 3,
    ) -> None:
        self._dimension = dimension
        self.vectors = vectors or {}

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> list[float]:
        return list(
            self.vectors.get(
                text,
                [1.0] + [0.0] * (self._dimension - 1),
            )
        )

    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        return [
            self.embed_text(text)
            for text in texts
        ]

    def embed_query(self, query: str) -> list[float]:
        return list(
            self.vectors.get(
                query,
                [1.0] + [0.0] * (self._dimension - 1),
            )
        )


class FixedResponseGenerator(BaseResponseGenerator):
    """Response generator returning fixed text."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[PromptMessages] = []

    def generate(
        self,
        messages: PromptMessages,
    ) -> str:
        self.calls.append(messages)
        return self.response


class FailingResponseGenerator(BaseResponseGenerator):
    """Response generator that always fails."""

    def generate(
        self,
        messages: PromptMessages,
    ) -> str:
        raise RuntimeError("Mock generation failure")


class EmptyResponseGenerator(BaseResponseGenerator):
    """Response generator returning blank content."""

    def generate(
        self,
        messages: PromptMessages,
    ) -> str:
        return "   "


class NonStringResponseGenerator(BaseResponseGenerator):
    """Response generator returning a non-string value."""

    def generate(
        self,
        messages: PromptMessages,
    ) -> str:
        return 100  # type: ignore[return-value]


class FakeCompletionsAPI:
    """Mock OpenAI chat completions API."""

    def __init__(
        self,
        content: object = "Generated response",
        should_fail: bool = False,
    ) -> None:
        self.content = content
        self.should_fail = should_fail
        self.calls: list[dict[str, object]] = []

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
    ) -> SimpleNamespace:
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
        )

        if self.should_fail:
            raise RuntimeError("Mock OpenAI failure")

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=self.content
                    )
                )
            ]
        )


class FakeChatAPI:
    """Mock OpenAI chat API."""

    def __init__(
        self,
        content: object = "Generated response",
        should_fail: bool = False,
    ) -> None:
        self.completions = FakeCompletionsAPI(
            content=content,
            should_fail=should_fail,
        )


class FakeOpenAIClient:
    """Mock OpenAI client."""

    def __init__(
        self,
        content: object = "Generated response",
        should_fail: bool = False,
    ) -> None:
        self.chat = FakeChatAPI(
            content=content,
            should_fail=should_fail,
        )


def build_test_retriever() -> FinanceRetriever:
    """Create a predictable populated retriever."""

    vectors = {
        "Explain revenue performance": [
            1.0,
            0.0,
            0.0,
        ],
        "revenue query": [
            1.0,
            0.0,
            0.0,
        ],
        (
            "Revenue increased because actual volume "
            "exceeded budget volume."
        ): [
            1.0,
            0.0,
            0.0,
        ],
        (
            "Operating cost increased because fuel "
            "expense exceeded forecast."
        ): [
            0.0,
            1.0,
            0.0,
        ],
    }

    store = InMemoryVectorStore(
        FixedEmbeddingService(vectors=vectors)
    )

    store.add_documents(
        [
            {
                "id": "revenue-doc",
                "text": (
                    "Revenue increased because actual volume "
                    "exceeded budget volume."
                ),
                "metadata": {
                    "category": "revenue",
                    "year": 2026,
                },
            },
            {
                "id": "cost-doc",
                "text": (
                    "Operating cost increased because fuel "
                    "expense exceeded forecast."
                ),
                "metadata": {
                    "category": "cost",
                    "year": 2026,
                },
            },
        ]
    )

    return FinanceRetriever(
        vector_store=store,
        config=RetrievalConfig(
            top_k=5,
        ),
    )


def build_empty_retriever() -> FinanceRetriever:
    """Create a retriever with no stored documents."""

    store = InMemoryVectorStore(
        FixedEmbeddingService()
    )

    return FinanceRetriever(store)


def build_empty_retrieval_result() -> RetrievalResult:
    """Create a valid retrieval result with no context."""

    return RetrievalResult(
        query="revenue query",
        documents=(),
        context="",
        metadata_filter={},
        total_results=0,
    )


def test_execution_mode_values() -> None:
    """ExecutionMode should expose supported modes."""

    assert (
        ExecutionMode.DETERMINISTIC.value
        == "deterministic"
    )
    assert ExecutionMode.LLM.value == "llm"


def test_default_rag_agent_config() -> None:
    """Default configuration should use deterministic mode."""

    config = RAGAgentConfig()

    assert (
        config.execution_mode
        is ExecutionMode.DETERMINISTIC
    )
    assert (
        config.prompt_type
        is PromptType.FINANCE_QA
    )
    assert config.top_k == 5
    assert config.score_threshold is None
    assert config.require_context is False
    assert config.deterministic_fallback is True
    assert config.include_prompt_messages is True


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (
            "deterministic",
            ExecutionMode.DETERMINISTIC,
        ),
        (
            " DETERMINISTIC ",
            ExecutionMode.DETERMINISTIC,
        ),
        (
            "llm",
            ExecutionMode.LLM,
        ),
        (
            " LLM ",
            ExecutionMode.LLM,
        ),
    ],
)
def test_rag_agent_config_normalizes_execution_mode(
    mode: str,
    expected: ExecutionMode,
) -> None:
    """Execution-mode strings should be normalized."""

    config = RAGAgentConfig(
        execution_mode=mode
    )

    assert config.execution_mode is expected


@pytest.mark.parametrize(
    ("prompt_type", "expected"),
    [
        (
            "finance_qa",
            PromptType.FINANCE_QA,
        ),
        (
            " COMMENTARY ",
            PromptType.COMMENTARY,
        ),
        (
            PromptType.RECOMMENDATION,
            PromptType.RECOMMENDATION,
        ),
    ],
)
def test_rag_agent_config_normalizes_prompt_type(
    prompt_type: PromptType | str,
    expected: PromptType,
) -> None:
    """Prompt-type values should be normalized."""

    config = RAGAgentConfig(
        prompt_type=prompt_type
    )

    assert config.prompt_type is expected


def test_rag_agent_config_rejects_invalid_mode() -> None:
    """Unsupported execution modes should fail."""

    with pytest.raises(
        ValueError,
        match="deterministic.*llm",
    ):
        RAGAgentConfig(
            execution_mode="unknown"
        )


def test_rag_agent_config_rejects_invalid_prompt_type() -> None:
    """Unsupported prompt types should fail."""

    with pytest.raises(
        ValueError,
        match="Unsupported prompt type",
    ):
        RAGAgentConfig(
            prompt_type="unknown"
        )


@pytest.mark.parametrize("top_k", [0, -1, -10])
def test_rag_agent_config_rejects_invalid_top_k(
    top_k: int,
) -> None:
    """top_k must be positive."""

    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        RAGAgentConfig(top_k=top_k)


@pytest.mark.parametrize(
    "top_k",
    [1.5, "5", True],
)
def test_rag_agent_config_rejects_non_integer_top_k(
    top_k: object,
) -> None:
    """top_k must be an integer."""

    with pytest.raises(
        TypeError,
        match="top_k must be an integer",
    ):
        RAGAgentConfig(
            top_k=top_k  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "threshold",
    [-1.1, 1.1, -5.0, 5.0],
)
def test_rag_agent_config_rejects_invalid_threshold(
    threshold: float,
) -> None:
    """Score threshold must remain within cosine bounds."""

    with pytest.raises(
        ValueError,
        match="between -1.0 and 1.0",
    ):
        RAGAgentConfig(
            score_threshold=threshold
        )


@pytest.mark.parametrize(
    "threshold",
    ["0.5", True, []],
)
def test_rag_agent_config_rejects_non_numeric_threshold(
    threshold: object,
) -> None:
    """Score threshold must be numeric."""

    with pytest.raises(
        TypeError,
        match="score_threshold must be numeric",
    ):
        RAGAgentConfig(
            score_threshold=threshold  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "require_context",
        "deterministic_fallback",
        "include_prompt_messages",
    ],
)
def test_rag_agent_config_rejects_non_boolean_fields(
    field_name: str,
) -> None:
    """Boolean configuration fields must be booleans."""

    kwargs = {
        field_name: "yes",
    }

    with pytest.raises(
        TypeError,
        match="must be a boolean",
    ):
        RAGAgentConfig(
            **kwargs  # type: ignore[arg-type]
        )


def test_rag_request_creation() -> None:
    """RAGRequest should validate and retain input."""

    request = RAGRequest(
        user_request="  Explain revenue  ",
        finance_analysis={
            "revenue_variance": 100,
        },
        prompt_type="commentary",
        retrieval_query="  revenue query  ",
        metadata_filter={
            "category": "revenue",
        },
        top_k=3,
        score_threshold=0.5,
        additional_instructions=(
            "  Keep it concise.  "
        ),
    )

    assert (
        request.user_request
        == "Explain revenue"
    )
    assert (
        request.prompt_type
        is PromptType.COMMENTARY
    )
    assert request.retrieval_query == "revenue query"
    assert request.top_k == 3
    assert request.score_threshold == 0.5
    assert (
        request.additional_instructions
        == "Keep it concise."
    )


def test_rag_request_defaults() -> None:
    """Optional RAGRequest values should have safe defaults."""

    request = RAGRequest(
        user_request="Explain revenue",
        finance_analysis={},
    )

    assert request.prompt_type is None
    assert request.retrieval_query is None
    assert request.metadata_filter == {}
    assert request.top_k is None
    assert request.score_threshold is None
    assert request.additional_instructions is None


@pytest.mark.parametrize(
    "user_request",
    ["", " ", "   "],
)
def test_rag_request_rejects_empty_user_request(
    user_request: str,
) -> None:
    """User request cannot be blank."""

    with pytest.raises(
        ValueError,
        match="user_request cannot be empty",
    ):
        RAGRequest(
            user_request=user_request,
            finance_analysis={},
        )


def test_rag_request_rejects_non_string_user_request() -> None:
    """User request must be text."""

    with pytest.raises(
        TypeError,
        match="user_request must be a string",
    ):
        RAGRequest(
            user_request=100,  # type: ignore[arg-type]
            finance_analysis={},
        )


def test_rag_request_rejects_invalid_metadata_filter() -> None:
    """RAGRequest metadata must be a dictionary."""

    with pytest.raises(
        TypeError,
        match="metadata_filter must be a dictionary",
    ):
        RAGRequest(
            user_request="Explain revenue",
            finance_analysis={},
            metadata_filter=["revenue"],  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("top_k", [0, -1])
def test_rag_request_rejects_invalid_top_k(
    top_k: int,
) -> None:
    """Request-level top_k must be positive."""

    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        RAGRequest(
            user_request="Explain revenue",
            finance_analysis={},
            top_k=top_k,
        )


def test_rag_request_deep_copies_metadata() -> None:
    """External metadata changes should not alter request data."""

    metadata_filter = {
        "details": {
            "year": 2026,
        }
    }

    request = RAGRequest(
        user_request="Explain revenue",
        finance_analysis={},
        metadata_filter=metadata_filter,
    )

    metadata_filter["details"]["year"] = 2030

    assert (
        request.metadata_filter["details"]["year"]
        == 2026
    )


def test_deterministic_generator() -> None:
    """Deterministic generator should include user prompt."""

    generator = DeterministicResponseGenerator()

    messages = PromptMessages(
        system="System",
        user="Finance analysis content",
    )

    response = generator.generate(messages)

    assert "Deterministic RAG response" in response
    assert "Finance analysis content" in response


def test_deterministic_generator_rejects_invalid_messages() -> None:
    """Deterministic generator requires PromptMessages."""

    generator = DeterministicResponseGenerator()

    with pytest.raises(
        TypeError,
        match="PromptMessages",
    ):
        generator.generate(
            "invalid"  # type: ignore[arg-type]
        )


def test_openai_generator_defaults() -> None:
    """OpenAI generator should retain default settings."""

    generator = OpenAIResponseGenerator(
        client=FakeOpenAIClient()
    )

    assert generator.model == DEFAULT_MODEL
    assert generator.temperature == 0.0


def test_openai_generator_generates_response() -> None:
    """OpenAI generator should return response content."""

    client = FakeOpenAIClient(
        content="Management response"
    )

    generator = OpenAIResponseGenerator(
        model="test-model",
        temperature=0.2,
        client=client,
    )

    messages = PromptMessages(
        system="System",
        user="User prompt",
    )

    response = generator.generate(messages)

    assert response == "Management response"

    assert client.chat.completions.calls == [
        {
            "model": "test-model",
            "messages": messages.as_dicts(),
            "temperature": 0.2,
        }
    ]


def test_openai_generator_strips_response() -> None:
    """Generated OpenAI text should be stripped."""

    generator = OpenAIResponseGenerator(
        client=FakeOpenAIClient(
            content="  Management response  "
        )
    )

    response = generator.generate(
        PromptMessages(
            system="System",
            user="User",
        )
    )

    assert response == "Management response"


def test_openai_generator_wraps_api_failure() -> None:
    """OpenAI API errors should become GenerationError."""

    generator = OpenAIResponseGenerator(
        client=FakeOpenAIClient(
            should_fail=True
        )
    )

    with pytest.raises(
        GenerationError,
        match="OpenAI response generation failed",
    ):
        generator.generate(
            PromptMessages(
                system="System",
                user="User",
            )
        )


def test_openai_generator_rejects_empty_response() -> None:
    """Blank OpenAI responses should fail."""

    generator = OpenAIResponseGenerator(
        client=FakeOpenAIClient(
            content="   "
        )
    )

    with pytest.raises(
        GenerationError,
        match="empty response",
    ):
        generator.generate(
            PromptMessages(
                system="System",
                user="User",
            )
        )


def test_openai_generator_rejects_non_text_response() -> None:
    """Non-string OpenAI responses should fail."""

    generator = OpenAIResponseGenerator(
        client=FakeOpenAIClient(
            content=100
        )
    )

    with pytest.raises(
        GenerationError,
        match="non-text response",
    ):
        generator.generate(
            PromptMessages(
                system="System",
                user="User",
            )
        )


@pytest.mark.parametrize("temperature", [-0.1, 2.1])
def test_openai_generator_rejects_invalid_temperature(
    temperature: float,
) -> None:
    """Temperature must remain in the supported range."""

    with pytest.raises(
        ValueError,
        match="between 0.0 and 2.0",
    ):
        OpenAIResponseGenerator(
            temperature=temperature,
            client=FakeOpenAIClient(),
        )


def test_openai_generator_rejects_empty_model() -> None:
    """Model name cannot be empty."""

    with pytest.raises(
        ValueError,
        match="model cannot be empty",
    ):
        OpenAIResponseGenerator(
            model="   ",
            client=FakeOpenAIClient(),
        )


def test_finance_rag_agent_creation() -> None:
    """FinanceRAGAgent should retain dependencies."""

    retriever = build_test_retriever()
    config = RAGAgentConfig()

    agent = FinanceRAGAgent(
        retriever=retriever,
        config=config,
    )

    assert agent.retriever is retriever
    assert agent.config is config


def test_finance_rag_agent_rejects_invalid_retriever() -> None:
    """Agent requires FinanceRetriever."""

    with pytest.raises(
        TypeError,
        match="FinanceRetriever",
    ):
        FinanceRAGAgent(
            retriever="invalid"  # type: ignore[arg-type]
        )


def test_llm_mode_requires_response_generator() -> None:
    """LLM mode requires a configured generator."""

    with pytest.raises(
        ValueError,
        match="response_generator is required",
    ):
        FinanceRAGAgent(
            retriever=build_test_retriever(),
            config=RAGAgentConfig(
                execution_mode=ExecutionMode.LLM
            ),
        )


def test_run_deterministic_mode() -> None:
    """Deterministic mode should complete the full workflow."""

    agent = FinanceRAGAgent(
        retriever=build_test_retriever()
    )

    result = agent.run(
        user_request="Explain revenue performance",
        finance_analysis={
            "actual_revenue": 1200,
            "budget_revenue": 1000,
        },
    )

    assert isinstance(result, RAGResult)
    assert result.success is True
    assert (
        result.execution_mode
        is ExecutionMode.DETERMINISTIC
    )
    assert result.used_fallback is False
    assert result.has_context is True
    assert result.retrieval_result.total_results == 2
    assert "Deterministic RAG response" in result.response
    assert '"actual_revenue": 1200' in result.response


def test_run_with_request_object() -> None:
    """Agent should accept a complete RAGRequest."""

    agent = FinanceRAGAgent(
        retriever=build_test_retriever()
    )

    request = RAGRequest(
        user_request="Explain revenue performance",
        finance_analysis={
            "revenue_variance": 100,
        },
        prompt_type=PromptType.COMMENTARY,
        retrieval_query="revenue query",
        top_k=1,
    )

    result = agent.run(request=request)

    assert (
        result.prompt_type
        is PromptType.COMMENTARY
    )
    assert result.retrieval_result.total_results == 1


def test_run_rejects_mixed_request_inputs() -> None:
    """Request object and keyword inputs cannot be mixed."""

    agent = FinanceRAGAgent(
        retriever=build_test_retriever()
    )

    request = RAGRequest(
        user_request="Explain revenue",
        finance_analysis={},
    )

    with pytest.raises(
        ValueError,
        match="not both",
    ):
        agent.run(
            request=request,
            user_request="Another request",
        )


def test_run_requires_user_request() -> None:
    """Keyword execution requires user_request."""

    agent = FinanceRAGAgent(
        retriever=build_test_retriever()
    )

    with pytest.raises(
        ValueError,
        match="user_request is required",
    ):
        agent.run(
            finance_analysis={}
        )


def test_run_uses_request_prompt_type_override() -> None:
    """Request prompt type should override agent default."""

    agent = FinanceRAGAgent(
        retriever=build_test_retriever(),
        config=RAGAgentConfig(
            prompt_type=PromptType.FINANCE_QA
        ),
    )

    result = agent.run(
        user_request="Prepare commentary",
        finance_analysis={
            "revenue": 1200,
        },
        prompt_type=PromptType.COMMENTARY,
    )

    assert (
        result.prompt_type
        is PromptType.COMMENTARY
    )
    assert (
        "Task: Management Commentary"
        in result.response
    )


def test_run_uses_metadata_filter() -> None:
    """Agent should forward metadata filters to retrieval."""

    agent = FinanceRAGAgent(
        retriever=build_test_retriever()
    )

    result = agent.run(
        user_request="Explain finance performance",
        finance_analysis={},
        metadata_filter={
            "category": "cost",
        },
    )

    assert result.retrieval_result.total_results == 1
    assert (
        result.retrieved_documents[0].id
        == "cost-doc"
    )


def test_run_require_context_success() -> None:
    """require_context should allow execution when evidence exists."""

    agent = FinanceRAGAgent(
        retriever=build_test_retriever(),
        config=RAGAgentConfig(
            require_context=True
        ),
    )

    result = agent.run(
        user_request="Explain revenue performance",
        finance_analysis={},
    )

    assert result.has_context is True


def test_run_require_context_failure() -> None:
    """require_context should fail when no evidence is found."""

    agent = FinanceRAGAgent(
        retriever=build_empty_retriever(),
        config=RAGAgentConfig(
            require_context=True
        ),
    )

    with pytest.raises(
        RAGAgentError,
        match="No reference context",
    ):
        agent.run(
            user_request="Explain revenue",
            finance_analysis={},
        )


def test_run_without_context_allowed() -> None:
    """Agent should work without context by default."""

    agent = FinanceRAGAgent(
        retriever=build_empty_retriever()
    )

    result = agent.run(
        user_request="Explain revenue",
        finance_analysis={
            "revenue": 1200,
        },
    )

    assert result.has_context is False
    assert result.success is True
    assert (
        "No retrieved reference context"
        in result.response
    )


def test_run_llm_mode() -> None:
    """LLM mode should use the configured generator."""

    generator = FixedResponseGenerator(
        "LLM management response"
    )

    agent = FinanceRAGAgent(
        retriever=build_test_retriever(),
        config=RAGAgentConfig(
            execution_mode=ExecutionMode.LLM
        ),
        response_generator=generator,
    )

    result = agent.run(
        user_request="Explain revenue performance",
        finance_analysis={
            "revenue": 1200,
        },
    )

    assert result.response == "LLM management response"
    assert result.execution_mode is ExecutionMode.LLM
    assert result.used_fallback is False
    assert len(generator.calls) == 1


def test_run_llm_failure_uses_fallback() -> None:
    """LLM failure should use deterministic fallback."""

    agent = FinanceRAGAgent(
        retriever=build_test_retriever(),
        config=RAGAgentConfig(
            execution_mode=ExecutionMode.LLM,
            deterministic_fallback=True,
        ),
        response_generator=FailingResponseGenerator(),
    )

    result = agent.run(
        user_request="Explain revenue performance",
        finance_analysis={
            "revenue": 1200,
        },
    )

    assert result.success is True
    assert result.used_fallback is True
    assert (
        result.execution_mode
        is ExecutionMode.DETERMINISTIC
    )
    assert "Deterministic RAG response" in result.response
    assert len(result.errors) == 1
    assert "Mock generation failure" in result.errors[0]


def test_run_llm_failure_without_fallback() -> None:
    """LLM failure should raise when fallback is disabled."""

    agent = FinanceRAGAgent(
        retriever=build_test_retriever(),
        config=RAGAgentConfig(
            execution_mode=ExecutionMode.LLM,
            deterministic_fallback=False,
        ),
        response_generator=FailingResponseGenerator(),
    )

    with pytest.raises(
        GenerationError,
        match="LLM response generation failed",
    ):
        agent.run(
            user_request="Explain revenue",
            finance_analysis={},
        )


def test_run_rejects_empty_generated_response() -> None:
    """Blank generated text should fail validation."""

    agent = FinanceRAGAgent(
        retriever=build_test_retriever(),
        config=RAGAgentConfig(
            execution_mode=ExecutionMode.LLM,
            deterministic_fallback=False,
        ),
        response_generator=EmptyResponseGenerator(),
    )

    with pytest.raises(GenerationError):
        agent.run(
            user_request="Explain revenue",
            finance_analysis={},
        )


def test_run_can_hide_prompt_messages() -> None:
    """Prompt storage should be configurable."""

    agent = FinanceRAGAgent(
        retriever=build_test_retriever(),
        config=RAGAgentConfig(
            include_prompt_messages=False
        ),
    )

    result = agent.run(
        user_request="Explain revenue",
        finance_analysis={},
    )

    assert result.prompt_messages is None


def test_run_retains_prompt_messages_by_default() -> None:
    """Prompt messages should be retained by default."""

    agent = FinanceRAGAgent(
        retriever=build_test_retriever()
    )

    result = agent.run(
        user_request="Explain revenue",
        finance_analysis={},
    )

    assert isinstance(
        result.prompt_messages,
        PromptMessages,
    )


def test_ask_returns_response_only() -> None:
    """ask should return only generated text."""

    agent = FinanceRAGAgent(
        retriever=build_test_retriever()
    )

    response = agent.ask(
        user_request="Explain revenue",
        finance_analysis={
            "revenue": 1200,
        },
    )

    assert isinstance(response, str)
    assert "Deterministic RAG response" in response


def test_create_rag_agent_helper() -> None:
    """Factory helper should create FinanceRAGAgent."""

    retriever = build_test_retriever()
    config = RAGAgentConfig()

    agent = create_rag_agent(
        retriever=retriever,
        config=config,
    )

    assert isinstance(agent, FinanceRAGAgent)
    assert agent.retriever is retriever
    assert agent.config is config


def test_build_deterministic_rag_response() -> None:
    """Standalone deterministic helper should build output."""

    retrieval_result = build_empty_retrieval_result()

    response = build_deterministic_rag_response(
        user_request="Explain revenue",
        finance_analysis={
            "revenue": 1200,
        },
        retrieval_result=retrieval_result,
        prompt_type=PromptType.FINANCE_QA,
    )

    assert "Deterministic RAG response" in response
    assert '"revenue": 1200' in response
    assert (
        "No retrieved reference context"
        in response
    )


def test_build_deterministic_rag_response_rejects_invalid_result() -> None:
    """Standalone helper requires RetrievalResult."""

    with pytest.raises(
        TypeError,
        match="RetrievalResult",
    ):
        build_deterministic_rag_response(
            user_request="Explain revenue",
            finance_analysis={},
            retrieval_result="invalid",  # type: ignore[arg-type]
        )


def test_summarize_rag_result() -> None:
    """RAG results should convert into serializable summaries."""

    agent = FinanceRAGAgent(
        retriever=build_test_retriever()
    )

    result = agent.run(
        user_request="Explain revenue performance",
        finance_analysis={
            "revenue": 1200,
        },
        top_k=1,
    )

    summary = summarize_rag_result(result)

    assert summary["success"] is True
    assert (
        summary["user_request"]
        == "Explain revenue performance"
    )
    assert summary["prompt_type"] == "finance_qa"
    assert summary["execution_mode"] == "deterministic"
    assert summary["used_fallback"] is False
    assert summary["has_context"] is True
    assert summary["retrieved_document_count"] == 1
    assert summary["retrieved_document_ids"] == [
        "revenue-doc"
    ]
    assert isinstance(summary["response"], str)
    assert summary["errors"] == []


def test_summarize_rag_result_rejects_invalid_input() -> None:
    """Summary helper requires RAGResult."""

    with pytest.raises(
        TypeError,
        match="RAGResult",
    ):
        summarize_rag_result(
            "invalid"  # type: ignore[arg-type]
        )


def test_rag_result_properties() -> None:
    """RAGResult should expose context and sources."""

    agent = FinanceRAGAgent(
        retriever=build_test_retriever()
    )

    result = agent.run(
        user_request="Explain revenue performance",
        finance_analysis={},
        top_k=1,
    )

    assert result.has_context is True
    assert result.context
    assert len(result.retrieved_documents) == 1
    assert (
        result.retrieved_documents[0].id
        == "revenue-doc"
    )


def test_rag_result_retrieved_documents_are_copies() -> None:
    """Returned source metadata should not expose internal data."""

    agent = FinanceRAGAgent(
        retriever=build_test_retriever()
    )

    result = agent.run(
        user_request="Explain revenue performance",
        finance_analysis={},
        top_k=1,
    )

    documents = result.retrieved_documents
    documents[0].metadata["year"] = 2030

    second_documents = result.retrieved_documents

    assert second_documents[0].metadata["year"] == 2026