# finance-agentic-ai-system
Multi-Agent Financial Planning &amp; Analysis (FP&amp;A) System built using Python, LangGraph, LangChain, PostgreSQL, FastAPI, RAG, OpenAI API, Streamlit, and Power BI.
# Autonomous Finance Agentic AI System

This repository implements a bounded autonomous FP&A agent. A CFO supplies a
goal; the supervisor decides the next approved action, executes a tool,
observes the result, and continues until the evidence is sufficient or a stop
condition is reached.

```text
User/CFO or scheduler -> Goal Builder -> Autonomous Supervisor
    -> Decide -> Act -> Observe -> Decide
    -> PostgreSQL / Finance / Vector RAG / Reporting tools
    -> Shared state -> Validation and reconciliation
    -> clarification | approval | final cited answer
```

Company revenue is always `commission_amount`. `fare` is never treated as
company revenue.

Core implementation:

- `src/autonomous/`: goals, decisions, agent loop, evidence, guardrails,
  scheduling and release control.
- `src/autonomous/tools/`: PostgreSQL-backed finance, deterministic finance, RAG, validation,
  PDF and email adapters.
- `src/services/autonomous_finance_service.py`: application-level autonomous
  finance execution.
- `src/api/autonomous_routes.py`: autonomous HTTP interface.
- `src/evaluation/`: evaluation cases and autonomy metrics.

Run the API with `uvicorn src.api.app:app --reload`, or ask a local question
with `python main.py "Show P&L for August 2026"`.
