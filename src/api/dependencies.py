"""Dependency wiring for the autonomous finance application."""

from __future__ import annotations

from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

from src.config.settings import settings
from src.integrations.snowflake_connection import SnowflakeConnectionConfig, SnowflakeConnectionFactory
from src.repositories.snowflake_reporting_repository import SnowflakeReportingRepository
from src.services.autonomous_finance_service import AutonomousFinanceService
from src.autonomous.hybrid_runtime_router import HybridRuntimeRouter
from src.autonomous.langgraph_runtime.context_factory import ComplexFinanceContextFactory
from src.autonomous.langgraph_runtime.runtime_service import LangGraphRuntimeService
from src.rag.ingestion import PDFIngestionService
from src.rag.retriever import create_retriever
from src.rag.vector_store import InMemoryVectorStore


@lru_cache(maxsize=1)
def get_reporting_repository() -> SnowflakeReportingRepository:
    config = SnowflakeConnectionConfig.from_settings(settings)
    return SnowflakeReportingRepository(SnowflakeConnectionFactory(config))


@lru_cache(maxsize=1)
def get_autonomous_finance_service() -> AutonomousFinanceService:
    return AutonomousFinanceService(
        repository=get_reporting_repository(),
        retriever=get_company_retriever(),
    )


@lru_cache(maxsize=1)
def get_hybrid_runtime_router() -> HybridRuntimeRouter:
    simple = get_autonomous_finance_service()
    repository = get_reporting_repository()
    retriever = get_company_retriever()
    complex_runtime = (
        LangGraphRuntimeService(max_steps=settings.LANGGRAPH_MAX_STEPS)
        if settings.LANGGRAPH_ENABLED
        else None
    )
    return HybridRuntimeRouter(
        simple,
        complex_runtime,
        context_factory=ComplexFinanceContextFactory(
            repository=repository,
            retriever=retriever,
        ),
        app_settings=settings,
    )


@lru_cache(maxsize=1)
def get_company_retriever():
    store = InMemoryVectorStore()
    source = "data/assumptions/sample_budget_assumptions_fy2026_2027.pdf"
    PDFIngestionService(vector_store=store).ingest_pdf(
        source,
        metadata={"title": "FY2026-27 Budget Assumptions", "approved": True},
    )
    return create_retriever(store)
