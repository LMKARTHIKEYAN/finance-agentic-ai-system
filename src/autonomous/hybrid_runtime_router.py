"""Cost-aware routing between the working Python path and LangGraph."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable, Literal

from src.config.settings import Settings, settings


COMPLEX_MARKERS = re.compile(r"\b(why|investigate|root cause|driver|recommend|evidence|explain decline|risk)\b", re.I)


def classify_runtime(request: str) -> Literal["simple", "complex"]:
    if not request.strip():
        raise ValueError("request cannot be empty.")
    return "complex" if COMPLEX_MARKERS.search(request) else "simple"


@dataclass(frozen=True)
class HybridRuntimeResult:
    response: Any
    execution_mode: Literal["simple", "complex"]
    runtime: Literal["python", "langgraph"]
    shadow_executed: bool = False
    shadow_result: Any = None
    shadow_error: str | None = None


class HybridRuntimeRouter:
    """Cost-aware router that introduces LangGraph without risking production."""

    def __init__(
        self,
        simple_runtime: Any,
        complex_runtime: Any,
        *,
        context_factory: Callable[[str], dict[str, Any]] | None = None,
        app_settings: Settings = settings,
    ) -> None:
        self.simple_runtime = simple_runtime
        self.complex_runtime = complex_runtime
        self.context_factory = context_factory or (lambda _request: {})
        self.enabled = app_settings.LANGGRAPH_ENABLED
        self.shadow_mode = app_settings.LANGGRAPH_SHADOW_MODE

    def run(self, request: str, **kwargs: Any) -> HybridRuntimeResult:
        mode = classify_runtime(request)
        if mode == "simple" or not self.enabled:
            return HybridRuntimeResult(
                response=_invoke_simple(self.simple_runtime, request),
                execution_mode=mode,
                runtime="python",
            )

        trusted_context = self.context_factory(request)
        if self.shadow_mode:
            response = _invoke_simple(self.simple_runtime, request)
            try:
                shadow = self.complex_runtime.run(
                    request, context=trusted_context, **kwargs
                )
                return HybridRuntimeResult(
                    response=response, execution_mode="complex",
                    runtime="python", shadow_executed=True,
                    shadow_result=shadow,
                )
            except Exception as exc:  # existing answer must remain available
                return HybridRuntimeResult(
                    response=response, execution_mode="complex",
                    runtime="python", shadow_executed=True,
                    shadow_error=f"{type(exc).__name__}: shadow execution failed.",
                )

        return HybridRuntimeResult(
            response=self.complex_runtime.run(
                request, context=trusted_context, **kwargs
            ),
            execution_mode="complex",
            runtime="langgraph",
        )


def _invoke_simple(runtime: Any, request: str) -> Any:
    if hasattr(runtime, "ask") and callable(runtime.ask):
        return runtime.ask(request)
    if hasattr(runtime, "run") and callable(runtime.run):
        return runtime.run(request)
    if callable(runtime):
        return runtime(request)
    raise TypeError("simple_runtime must provide ask(), run(), or be callable.")
