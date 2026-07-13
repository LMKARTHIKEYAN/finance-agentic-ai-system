"""
FastAPI application entry point for the Finance Agentic AI System.

This module creates the FastAPI application and registers API routes.

It must not contain:

- Finance business logic
- LangGraph workflow logic
- RAG retrieval logic
- PostgreSQL queries
"""

from __future__ import annotations

from fastapi import FastAPI

from src.api.routes import router


def create_app() -> FastAPI:
    """
    Create and configure the Finance Agentic AI API application.

    Returns:
        Configured FastAPI application.
    """

    application = FastAPI(
        title="Finance Agentic AI API",
        description=(
            "Enterprise-style Finance Agentic AI backend for FP&A, "
            "LangGraph orchestration and RAG-based document grounding."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    application.include_router(router)

    return application


app = create_app()