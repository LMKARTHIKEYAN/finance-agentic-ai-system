"""
RAG agent for the Finance Agentic AI System.

This module connects the existing RAG infrastructure:

- FinanceRetriever
- Prompt templates
- Optional language-model generation
- Deterministic fallback generation

The RAG agent does not perform finance calculations. It uses finance results
already produced by existing finance agents and enriches them with retrieved
reference context.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence

from src.rag.prompt_templates import (
    PromptMessages,
    PromptTemplateError,
    PromptType,
    build_prompt_messages,
    format_finance_analysis,
)
from src.rag.retriever import (
    FinanceRetriever,
    RetrievalResult,
    RetrieverError,
)
from src.rag.vector_store import Document


DEFAULT_MODEL = "gpt-4o-mini"


class RAGAgentError(RuntimeError):
    """Raised when the RAG agent cannot complete execution."""


class GenerationError(RAGAgentError):
    """Raised when response generation fails."""


class ExecutionMode(str, Enum):
    """Supported RAG response-generation modes."""

    DETERMINISTIC = "deterministic"
    LLM = "llm"


class ResponseGenerator(Protocol):
    """
    Protocol for optional response generators.

    Any LLM provider can be used if it implements this interface.
    """

    def generate(self, messages: PromptMessages) -> str:
        """Generate a response from structured prompt messages."""


class BaseResponseGenerator(ABC):
    """Abstract response-generator interface."""

    @abstractmethod
    def generate(self, messages: PromptMessages) -> str:
        """Generate a response from prompt messages."""


@dataclass(frozen=True)
class RAGAgentConfig:
    """
    Configuration for the RAG agent.

    Attributes:
        execution_mode:
            Response mode. ``deterministic`` is the default.
        prompt_type:
            Default prompt category.
        top_k:
            Maximum number of retrieved documents.
        score_threshold:
            Optional minimum retrieval similarity.
        require_context:
            Whether execution should fail when no documents are retrieved.
        deterministic_fallback:
            Whether LLM failures should use the deterministic response.
        include_prompt_messages:
            Whether prompt messages should be retained in the result.
    """

    execution_mode: ExecutionMode | str = ExecutionMode.DETERMINISTIC
    prompt_type: PromptType | str = PromptType.FINANCE_QA
    top_k: int = 5
    score_threshold: float | None = None
    require_context: bool = False
    deterministic_fallback: bool = True
    include_prompt_messages: bool = True

    def __post_init__(self) -> None:
        execution_mode = _resolve_execution_mode(
            self.execution_mode
        )
        prompt_type = _resolve_prompt_type(
            self.prompt_type
        )

        if isinstance(self.top_k, bool) or not isinstance(
            self.top_k,
            int,
        ):
            raise TypeError("top_k must be an integer.")

        if self.top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        if self.score_threshold is not None:
            if isinstance(
                self.score_threshold,
                bool,
            ) or not isinstance(
                self.score_threshold,
                (int, float),
            ):
                raise TypeError(
                    "score_threshold must be numeric."
                )

            if not -1.0 <= float(self.score_threshold) <= 1.0:
                raise ValueError(
                    "score_threshold must be between "
                    "-1.0 and 1.0."
                )

        if not isinstance(self.require_context, bool):
            raise TypeError(
                "require_context must be a boolean."
            )

        if not isinstance(
            self.deterministic_fallback,
            bool,
        ):
            raise TypeError(
                "deterministic_fallback must be a boolean."
            )

        if not isinstance(
            self.include_prompt_messages,
            bool,
        ):
            raise TypeError(
                "include_prompt_messages must be a boolean."
            )

        object.__setattr__(
            self,
            "execution_mode",
            execution_mode,
        )
        object.__setattr__(
            self,
            "prompt_type",
            prompt_type,
        )


@dataclass(frozen=True)
class RAGRequest:
    """
    One request submitted to the RAG agent.

    Attributes:
        user_request:
            Original user request.
        finance_analysis:
            Existing output from finance agents.
        prompt_type:
            Optional prompt-type override.
        retrieval_query:
            Optional query used for document retrieval.
        metadata_filter:
            Optional metadata filter.
        top_k:
            Optional retrieval-count override.
        score_threshold:
            Optional similarity-threshold override.
        additional_instructions:
            Optional prompt instructions.
    """

    user_request: str
    finance_analysis: Any
    prompt_type: PromptType | str | None = None
    retrieval_query: str | None = None
    metadata_filter: dict[str, Any] = field(
        default_factory=dict
    )
    top_k: int | None = None
    score_threshold: float | None = None
    additional_instructions: str | None = None

    def __post_init__(self) -> None:
        user_request = _validate_required_text(
            self.user_request,
            field_name="user_request",
        )

        retrieval_query = _validate_optional_text(
            self.retrieval_query,
            field_name="retrieval_query",
        )

        additional_instructions = _validate_optional_text(
            self.additional_instructions,
            field_name="additional_instructions",
        )

        if self.prompt_type is not None:
            resolved_prompt_type = _resolve_prompt_type(
                self.prompt_type
            )
            object.__setattr__(
                self,
                "prompt_type",
                resolved_prompt_type,
            )

        if not isinstance(self.metadata_filter, dict):
            raise TypeError(
                "metadata_filter must be a dictionary."
            )

        if self.top_k is not None:
            if isinstance(self.top_k, bool) or not isinstance(
                self.top_k,
                int,
            ):
                raise TypeError(
                    "top_k must be an integer."
                )

            if self.top_k <= 0:
                raise ValueError(
                    "top_k must be greater than zero."
                )

        if self.score_threshold is not None:
            if isinstance(
                self.score_threshold,
                bool,
            ) or not isinstance(
                self.score_threshold,
                (int, float),
            ):
                raise TypeError(
                    "score_threshold must be numeric."
                )

            if not -1.0 <= float(
                self.score_threshold
            ) <= 1.0:
                raise ValueError(
                    "score_threshold must be between "
                    "-1.0 and 1.0."
                )

        object.__setattr__(
            self,
            "user_request",
            user_request,
        )
        object.__setattr__(
            self,
            "retrieval_query",
            retrieval_query or None,
        )
        object.__setattr__(
            self,
            "metadata_filter",
            copy.deepcopy(self.metadata_filter),
        )
        object.__setattr__(
            self,
            "additional_instructions",
            additional_instructions or None,
        )


@dataclass(frozen=True)
class RAGResult:
    """
    Structured result returned by the RAG agent.

    Attributes:
        response:
            Generated grounded response.
        user_request:
            Original validated request.
        prompt_type:
            Prompt type used.
        execution_mode:
            Actual generation mode used.
        retrieval_result:
            Document-retrieval result.
        prompt_messages:
            Prompt messages used for generation.
        used_fallback:
            Whether deterministic fallback was used.
        success:
            Whether execution completed successfully.
        errors:
            Non-fatal execution errors.
    """

    response: str
    user_request: str
    prompt_type: PromptType
    execution_mode: ExecutionMode
    retrieval_result: RetrievalResult
    prompt_messages: PromptMessages | None
    used_fallback: bool = False
    success: bool = True
    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        response = _validate_required_text(
            self.response,
            field_name="response",
        )
        user_request = _validate_required_text(
            self.user_request,
            field_name="user_request",
        )

        if not isinstance(self.prompt_type, PromptType):
            raise TypeError(
                "prompt_type must be a PromptType value."
            )

        if not isinstance(
            self.execution_mode,
            ExecutionMode,
        ):
            raise TypeError(
                "execution_mode must be an ExecutionMode value."
            )

        if not isinstance(
            self.retrieval_result,
            RetrievalResult,
        ):
            raise TypeError(
                "retrieval_result must be a RetrievalResult."
            )

        if (
            self.prompt_messages is not None
            and not isinstance(
                self.prompt_messages,
                PromptMessages,
            )
        ):
            raise TypeError(
                "prompt_messages must be PromptMessages or None."
            )

        if not isinstance(self.used_fallback, bool):
            raise TypeError(
                "used_fallback must be a boolean."
            )

        if not isinstance(self.success, bool):
            raise TypeError(
                "success must be a boolean."
            )

        if not isinstance(self.errors, tuple):
            raise TypeError(
                "errors must be a tuple."
            )

        if not all(
            isinstance(error, str)
            for error in self.errors
        ):
            raise TypeError(
                "errors must contain strings."
            )

        object.__setattr__(
            self,
            "response",
            response,
        )
        object.__setattr__(
            self,
            "user_request",
            user_request,
        )

    @property
    def retrieved_documents(self) -> list[Document]:
        """Return independent copies of retrieved documents."""

        return self.retrieval_result.source_documents

    @property
    def context(self) -> str:
        """Return retrieved prompt context."""

        return self.retrieval_result.context

    @property
    def has_context(self) -> bool:
        """Return whether retrieval returned evidence."""

        return self.retrieval_result.has_results


class DeterministicResponseGenerator(BaseResponseGenerator):
    """
    Generate a stable response without an LLM.

    This generator does not create new financial conclusions. It returns a
    structured evidence summary containing the existing finance analysis and
    the documents retrieved by the RAG layer.
    """

    def generate(self, messages: PromptMessages) -> str:
        """Generate a deterministic prompt-based response."""

        if not isinstance(messages, PromptMessages):
            raise TypeError(
                "messages must be PromptMessages."
            )

        return (
            "Deterministic RAG response\n\n"
            "The finance analysis and retrieved reference context "
            "have been prepared successfully.\n\n"
            f"{messages.user}"
        )


class OpenAIResponseGenerator(BaseResponseGenerator):
    """
    Optional OpenAI response generator.

    OpenAI is imported only when this class is instantiated. Local tests and
    deterministic execution therefore do not require the OpenAI package or an
    API key.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        temperature: float = 0.0,
        client: Any | None = None,
    ) -> None:
        if not isinstance(model, str):
            raise TypeError(
                "model must be a string."
            )

        model = model.strip()

        if not model:
            raise ValueError(
                "model cannot be empty."
            )

        if isinstance(temperature, bool) or not isinstance(
            temperature,
            (int, float),
        ):
            raise TypeError(
                "temperature must be numeric."
            )

        if not 0.0 <= float(temperature) <= 2.0:
            raise ValueError(
                "temperature must be between 0.0 and 2.0."
            )

        self._model = model
        self._temperature = float(temperature)
        self._client = (
            client
            if client is not None
            else self._create_client(api_key)
        )

    @property
    def model(self) -> str:
        """Return configured model name."""

        return self._model

    @property
    def temperature(self) -> float:
        """Return configured temperature."""

        return self._temperature

    def generate(self, messages: PromptMessages) -> str:
        """Generate a response through OpenAI."""

        if not isinstance(messages, PromptMessages):
            raise TypeError(
                "messages must be PromptMessages."
            )

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages.as_dicts(),
                temperature=self._temperature,
            )

            content = response.choices[0].message.content

            if not isinstance(content, str):
                raise GenerationError(
                    "OpenAI returned a non-text response."
                )

            content = content.strip()

            if not content:
                raise GenerationError(
                    "OpenAI returned an empty response."
                )

            return content

        except GenerationError:
            raise
        except Exception as exc:
            raise GenerationError(
                f"OpenAI response generation failed: {exc}"
            ) from exc

    @staticmethod
    def _create_client(
        api_key: str | None,
    ) -> Any:
        """Create an OpenAI client."""

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise GenerationError(
                "The OpenAI package is not installed."
            ) from exc

        try:
            return (
                OpenAI(api_key=api_key)
                if api_key
                else OpenAI()
            )
        except Exception as exc:
            raise GenerationError(
                f"Unable to initialize OpenAI client: {exc}"
            ) from exc


class FinanceRAGAgent:
    """
    Reusable RAG execution layer.

    Execution flow:

    1. Validate request
    2. Retrieve relevant reference documents
    3. Build prompt messages
    4. Generate deterministic or LLM response
    5. Apply deterministic fallback when configured
    6. Return structured result
    """

    def __init__(
        self,
        retriever: FinanceRetriever,
        config: RAGAgentConfig | None = None,
        response_generator: ResponseGenerator | None = None,
        fallback_generator: ResponseGenerator | None = None,
    ) -> None:
        if not isinstance(
            retriever,
            FinanceRetriever,
        ):
            raise TypeError(
                "retriever must be a FinanceRetriever."
            )

        self._retriever = retriever
        self._config = config or RAGAgentConfig()
        self._response_generator = response_generator
        self._fallback_generator = (
            fallback_generator
            or DeterministicResponseGenerator()
        )

        if (
            self._config.execution_mode
            is ExecutionMode.LLM
            and self._response_generator is None
        ):
            raise ValueError(
                "response_generator is required for LLM mode."
            )

    @property
    def retriever(self) -> FinanceRetriever:
        """Return configured retriever."""

        return self._retriever

    @property
    def config(self) -> RAGAgentConfig:
        """Return agent configuration."""

        return self._config

    def run(
        self,
        request: RAGRequest | None = None,
        *,
        user_request: str | None = None,
        finance_analysis: Any = None,
        prompt_type: PromptType | str | None = None,
        retrieval_query: str | None = None,
        metadata_filter: Mapping[str, Any] | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
        additional_instructions: str | None = None,
    ) -> RAGResult:
        """
        Execute the full RAG workflow.

        Callers may provide a complete ``RAGRequest`` or use keyword
        arguments. These approaches cannot be mixed.
        """

        resolved_request = self._resolve_request(
            request=request,
            user_request=user_request,
            finance_analysis=finance_analysis,
            prompt_type=prompt_type,
            retrieval_query=retrieval_query,
            metadata_filter=metadata_filter,
            top_k=top_k,
            score_threshold=score_threshold,
            additional_instructions=additional_instructions,
        )

        resolved_prompt_type = (
            resolved_request.prompt_type
            if resolved_request.prompt_type is not None
            else self._config.prompt_type
        )

        retrieval_query_text = (
            resolved_request.retrieval_query
            or resolved_request.user_request
        )

        resolved_top_k = (
            resolved_request.top_k
            if resolved_request.top_k is not None
            else self._config.top_k
        )

        resolved_threshold = (
            resolved_request.score_threshold
            if resolved_request.score_threshold is not None
            else self._config.score_threshold
        )

        try:
            retrieval_result = self._retriever.retrieve(
                query=retrieval_query_text,
                top_k=resolved_top_k,
                score_threshold=resolved_threshold,
                metadata_filter=(
                    resolved_request.metadata_filter
                ),
            )
        except RetrieverError as exc:
            raise RAGAgentError(
                f"RAG retrieval failed: {exc}"
            ) from exc

        if (
            self._config.require_context
            and not retrieval_result.has_results
        ):
            raise RAGAgentError(
                "No reference context was retrieved, but "
                "require_context is enabled."
            )

        try:
            prompt_messages = build_prompt_messages(
                resolved_prompt_type,
                user_request=resolved_request.user_request,
                finance_analysis=(
                    resolved_request.finance_analysis
                ),
                retrieved_context=retrieval_result.context,
                additional_instructions=(
                    resolved_request.additional_instructions
                ),
            )
        except PromptTemplateError as exc:
            raise RAGAgentError(
                f"RAG prompt creation failed: {exc}"
            ) from exc

        errors: list[str] = []
        used_fallback = False
        actual_mode = self._config.execution_mode

        if (
            self._config.execution_mode
            is ExecutionMode.DETERMINISTIC
        ):
            response = self._generate_with_fallback(
                generator=self._fallback_generator,
                messages=prompt_messages,
                error_prefix=(
                    "Deterministic response generation failed"
                ),
            )
        else:
            assert self._response_generator is not None

            try:
                response = self._generate_response(
                    self._response_generator,
                    prompt_messages,
                )
            except Exception as exc:
                if not self._config.deterministic_fallback:
                    raise GenerationError(
                        f"LLM response generation failed: {exc}"
                    ) from exc

                errors.append(str(exc))
                used_fallback = True
                actual_mode = ExecutionMode.DETERMINISTIC

                response = self._generate_with_fallback(
                    generator=self._fallback_generator,
                    messages=prompt_messages,
                    error_prefix=(
                        "Deterministic fallback generation failed"
                    ),
                )

        stored_prompt_messages = (
            prompt_messages
            if self._config.include_prompt_messages
            else None
        )

        return RAGResult(
            response=response,
            user_request=resolved_request.user_request,
            prompt_type=resolved_prompt_type,
            execution_mode=actual_mode,
            retrieval_result=retrieval_result,
            prompt_messages=stored_prompt_messages,
            used_fallback=used_fallback,
            success=True,
            errors=tuple(errors),
        )

    def ask(
        self,
        user_request: str,
        finance_analysis: Any,
        *,
        prompt_type: PromptType | str | None = None,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> str:
        """Run the agent and return only the generated response."""

        result = self.run(
            user_request=user_request,
            finance_analysis=finance_analysis,
            prompt_type=prompt_type,
            metadata_filter=metadata_filter,
        )

        return result.response

    def _resolve_request(
        self,
        *,
        request: RAGRequest | None,
        user_request: str | None,
        finance_analysis: Any,
        prompt_type: PromptType | str | None,
        retrieval_query: str | None,
        metadata_filter: Mapping[str, Any] | None,
        top_k: int | None,
        score_threshold: float | None,
        additional_instructions: str | None,
    ) -> RAGRequest:
        """Resolve object-based or keyword-based request input."""

        keyword_values_supplied = any(
            value is not None
            for value in (
                user_request,
                prompt_type,
                retrieval_query,
                metadata_filter,
                top_k,
                score_threshold,
                additional_instructions,
            )
        ) or finance_analysis is not None

        if request is not None:
            if not isinstance(request, RAGRequest):
                raise TypeError(
                    "request must be a RAGRequest."
                )

            if keyword_values_supplied:
                raise ValueError(
                    "Provide either request or keyword arguments, "
                    "not both."
                )

            return request

        if user_request is None:
            raise ValueError(
                "user_request is required."
            )

        resolved_metadata_filter: dict[str, Any]

        if metadata_filter is None:
            resolved_metadata_filter = {}
        else:
            if not isinstance(metadata_filter, Mapping):
                raise TypeError(
                    "metadata_filter must be a mapping."
                )

            resolved_metadata_filter = copy.deepcopy(
                dict(metadata_filter)
            )

        return RAGRequest(
            user_request=user_request,
            finance_analysis=finance_analysis,
            prompt_type=prompt_type,
            retrieval_query=retrieval_query,
            metadata_filter=resolved_metadata_filter,
            top_k=top_k,
            score_threshold=score_threshold,
            additional_instructions=additional_instructions,
        )

    @staticmethod
    def _generate_response(
        generator: ResponseGenerator,
        messages: PromptMessages,
    ) -> str:
        """Generate and validate response text."""

        try:
            response = generator.generate(messages)
        except Exception as exc:
            raise GenerationError(
                f"Response generator failed: {exc}"
            ) from exc

        if not isinstance(response, str):
            raise GenerationError(
                "Response generator returned a non-string value."
            )

        response = response.strip()

        if not response:
            raise GenerationError(
                "Response generator returned an empty response."
            )

        return response

    def _generate_with_fallback(
        self,
        *,
        generator: ResponseGenerator,
        messages: PromptMessages,
        error_prefix: str,
    ) -> str:
        """Generate a required deterministic response."""

        try:
            return self._generate_response(
                generator,
                messages,
            )
        except Exception as exc:
            raise GenerationError(
                f"{error_prefix}: {exc}"
            ) from exc


def create_rag_agent(
    retriever: FinanceRetriever,
    config: RAGAgentConfig | None = None,
    response_generator: ResponseGenerator | None = None,
    fallback_generator: ResponseGenerator | None = None,
) -> FinanceRAGAgent:
    """
    Create a configured FinanceRAGAgent.
    """

    return FinanceRAGAgent(
        retriever=retriever,
        config=config,
        response_generator=response_generator,
        fallback_generator=fallback_generator,
    )


def build_deterministic_rag_response(
    *,
    user_request: str,
    finance_analysis: Any,
    retrieval_result: RetrievalResult,
    prompt_type: PromptType | str = PromptType.FINANCE_QA,
    additional_instructions: str | None = None,
) -> str:
    """
    Build a deterministic RAG response without creating an agent.

    This helper is useful for tests and offline workflows.
    """

    if not isinstance(
        retrieval_result,
        RetrievalResult,
    ):
        raise TypeError(
            "retrieval_result must be a RetrievalResult."
        )

    messages = build_prompt_messages(
        prompt_type,
        user_request=user_request,
        finance_analysis=finance_analysis,
        retrieved_context=retrieval_result.context,
        additional_instructions=additional_instructions,
    )

    return DeterministicResponseGenerator().generate(
        messages
    )


def summarize_rag_result(
    result: RAGResult,
) -> dict[str, Any]:
    """
    Convert a RAG result into a serializable summary.
    """

    if not isinstance(result, RAGResult):
        raise TypeError(
            "result must be a RAGResult."
        )

    return {
        "success": result.success,
        "user_request": result.user_request,
        "prompt_type": result.prompt_type.value,
        "execution_mode": result.execution_mode.value,
        "used_fallback": result.used_fallback,
        "has_context": result.has_context,
        "retrieved_document_count": (
            result.retrieval_result.total_results
        ),
        "retrieved_document_ids": [
            document.id
            for document in result.retrieved_documents
        ],
        "response": result.response,
        "errors": list(result.errors),
    }


def _resolve_execution_mode(
    execution_mode: ExecutionMode | str,
) -> ExecutionMode:
    """Resolve execution-mode input."""

    if isinstance(
        execution_mode,
        ExecutionMode,
    ):
        return execution_mode

    if not isinstance(execution_mode, str):
        raise TypeError(
            "execution_mode must be an ExecutionMode or string."
        )

    cleaned_mode = execution_mode.strip().lower()

    if not cleaned_mode:
        raise ValueError(
            "execution_mode cannot be empty."
        )

    try:
        return ExecutionMode(cleaned_mode)
    except ValueError as exc:
        raise ValueError(
            "execution_mode must be 'deterministic' or 'llm'."
        ) from exc


def _resolve_prompt_type(
    prompt_type: PromptType | str,
) -> PromptType:
    """Resolve prompt-type input."""

    if isinstance(prompt_type, PromptType):
        return prompt_type

    if not isinstance(prompt_type, str):
        raise TypeError(
            "prompt_type must be a PromptType or string."
        )

    cleaned_type = prompt_type.strip().lower()

    if not cleaned_type:
        raise ValueError(
            "prompt_type cannot be empty."
        )

    try:
        return PromptType(cleaned_type)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported prompt type '{cleaned_type}'."
        ) from exc


def _validate_required_text(
    value: Any,
    *,
    field_name: str,
) -> str:
    """Validate required text."""

    if not isinstance(value, str):
        raise TypeError(
            f"{field_name} must be a string."
        )

    cleaned_value = value.strip()

    if not cleaned_value:
        raise ValueError(
            f"{field_name} cannot be empty."
        )

    return cleaned_value


def _validate_optional_text(
    value: Any,
    *,
    field_name: str,
) -> str:
    """Validate optional text."""

    if value is None:
        return ""

    if not isinstance(value, str):
        raise TypeError(
            f"{field_name} must be a string."
        )

    return value.strip()