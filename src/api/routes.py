"""
HTTP routes for the Finance Agentic AI API.

This module is responsible only for:

- Receiving HTTP requests
- Validating API input
- Calling the application service
- Returning HTTP responses
- Translating service errors into HTTP errors

Finance calculations, LangGraph logic, retrieval logic and database logic
must remain outside this module.
"""

from __future__ import annotations

from typing import Annotated, TypeAlias

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from src.api.dependencies import get_finance_service
from src.api.schemas import (
    AskRequest,
    AskResponse,
    DashboardPayload,
    SourceResponse,
)
from src.api.service import (
    FinanceAskService,
    FinanceAskServiceError,
)


router = APIRouter()


FinanceServiceDependency: TypeAlias = Annotated[
    FinanceAskService,
    Depends(get_finance_service),
]


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Check API health",
    description=(
        "Confirm that the Finance Agentic AI API "
        "application is running."
    ),
)
def health_check() -> dict[str, str]:
    """
    Return the basic API health status.

    This endpoint does not connect to PostgreSQL or execute finance agents.
    """

    return {
        "status": "healthy",
        "service": "finance-agentic-ai-api",
    }


@router.post(
    "/ask",
    response_model=AskResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask the Finance AI Assistant",
    description=(
        "Route a finance question through the existing "
        "LangGraph, finance agents and RAG system."
    ),
)
def ask_finance_question(
    request: AskRequest,
    service: FinanceServiceDependency,
) -> AskResponse:
    """
    Process one finance question.

    Args:
        request:
            Validated POST /ask request body.

        service:
            FinanceAskService provided through FastAPI dependency injection.

    Returns:
        Structured answer, dashboard data, sources and workflow
        execution information.

    Raises:
        HTTPException:
            400 when the submitted question is invalid.
            500 when finance workflow or RAG processing fails.
    """

    try:
        result = service.ask(
            question=request.question,
            top_k=request.top_k,
            metadata_filter=request.metadata_filter,
        )

    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except FinanceAskServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "An unexpected error occurred while "
                "processing the finance request."
            ),
        ) from exc

    sources = [
        SourceResponse(
            id=str(source["id"]),
            score=source.get("score"),
            rank=int(source["rank"]),
            metadata=dict(
                source.get(
                    "metadata",
                    {},
                )
            ),
            excerpt=str(
                source.get(
                    "excerpt",
                    "",
                )
            ),
        )
        for source in result.sources
    ]

    dashboard = DashboardPayload.model_validate(
        result.dashboard
    )

    return AskResponse(
    answer=result.answer,
    sources=result.sources,
    selected_flow=result.selected_flow,
    execution_status=result.execution_status,
    used_fallback=result.used_fallback,
    dashboard=result.dashboard,
    clarification_required=result.clarification_required,
    intent=result.intent,
 
    )