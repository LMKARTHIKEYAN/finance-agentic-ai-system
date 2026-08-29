from src.autonomous.hybrid_runtime_router import HybridRuntimeRouter, classify_runtime


class Config:
    LANGGRAPH_ENABLED = True
    LANGGRAPH_SHADOW_MODE = True


class Simple:
    def ask(self, request):
        return {"answer": "existing", "request": request}


class Complex:
    def __init__(self, fail=False):
        self.fail = fail

    def run(self, request, **kwargs):
        if self.fail:
            raise RuntimeError("secret provider detail")
        return {"answer": "graph", "context": kwargs.get("context")}


def test_routes_standard_report_to_simple_runtime():
    assert classify_runtime("Show P&L for August 2026") == "simple"


def test_routes_investigation_to_complex_runtime():
    assert classify_runtime("Why did July revenue decline?") == "complex"


def test_shadow_mode_preserves_existing_answer_and_runs_graph():
    router = HybridRuntimeRouter(
        Simple(), Complex(), app_settings=Config(),
        context_factory=lambda request: {"safe": request},
    )
    result = router.run("Why did revenue decline?")
    assert result.response["answer"] == "existing"
    assert result.runtime == "python"
    assert result.shadow_executed is True
    assert result.shadow_result["answer"] == "graph"


def test_shadow_failure_does_not_break_existing_answer_or_leak_detail():
    result = HybridRuntimeRouter(
        Simple(), Complex(fail=True), app_settings=Config(),
    ).run("Investigate revenue decline")
    assert result.response["answer"] == "existing"
    assert result.shadow_error == "RuntimeError: shadow execution failed."


def test_simple_question_never_spends_on_langgraph():
    complex_runtime = Complex(fail=True)
    result = HybridRuntimeRouter(
        Simple(), complex_runtime, app_settings=Config(),
    ).run("Show P&L for August 2026")
    assert result.execution_mode == "simple"
    assert result.shadow_executed is False
