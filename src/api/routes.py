"""
HTTP routes for the Finance Agentic AI API.

This module is responsible only for:

- Receiving HTTP requests
- Validating API input
- Calling the application service
- Returning HTTP responses
- Translating service errors into HTTP errors

Finance calculations, LangGraph logic, retrieval logic, Snowflake logic,
RAG logic and database logic must remain outside this module.
"""

from __future__ import annotations

from typing import Annotated, Any, Mapping, TypeAlias

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
        "Confirm that the Finance Agentic AI API application "
        "is running."
    ),
)
def health_check() -> dict[str, str]:
    """
    Return the basic API health status.

    This endpoint performs only an application-level health check.

    It does not:

    - Query Snowflake
    - Connect to PostgreSQL
    - Execute LangGraph
    - Run finance agents
    - Call the RAG system
    - Call an LLM

    Returns:
        Basic API service health information.
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
        "Route a finance question through FinanceAskService, "
        "LangGraph, finance agents, Snowflake-backed data access "
        "and the RAG system."
    ),
)
def ask_finance_question(
    request: AskRequest,
    service: FinanceServiceDependency,
) -> AskResponse:
    """
    Process one finance question.

    The route validates the HTTP request, delegates execution to
    FinanceAskService and converts the service result into the public
    API response schema.

    Args:
        request:
            Validated request body containing the finance question and
            optional retrieval parameters.

        service:
            FinanceAskService provided through FastAPI dependency
            injection.

    Returns:
        Structured finance answer containing:

        - Natural-language answer
        - RAG sources
        - Selected workflow
        - Workflow execution status
        - Dashboard payload
        - Intent information
        - Clarification status
        - Fallback information

    Raises:
        HTTPException:
            HTTP 400 when the request or service input is invalid.

            HTTP 500 when the finance workflow fails or an unexpected
            application error occurs.
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
                "An unexpected error occurred while processing "
                "the finance request."
            ),
        ) from exc

    sources = _build_source_responses(result.sources)
    dashboard = _build_dashboard_payload(result.dashboard)

    return AskResponse(
        answer=str(result.answer),
        sources=sources,
        selected_flow=result.selected_flow,
        execution_status=result.execution_status,
        used_fallback=bool(result.used_fallback),
        dashboard=dashboard,
        clarification_required=bool(
            result.clarification_required
        ),
        intent=result.intent,
    )


def _build_source_responses(
    raw_sources: Any,
) -> list[SourceResponse]:
    """
    Convert service-layer source values into API source schemas.

    The service may return dictionaries, dictionary-like mappings or
    already-validated SourceResponse objects. This helper normalizes
    those values without introducing retrieval logic into the route.

    Args:
        raw_sources:
            Source records returned by FinanceAskService.

    Returns:
        Validated SourceResponse objects.

    Raises:
        ValueError:
            If a source record cannot be converted into the required
            API schema.
    """

    if raw_sources is None:
        return []

    sources: list[SourceResponse] = []

    for index, source in enumerate(raw_sources, start=1):
        if isinstance(source, SourceResponse):
            sources.append(source)
            continue

        source_mapping = _convert_to_mapping(source)

        source_id = source_mapping.get(
            "id",
            source_mapping.get(
                "document_id",
                source_mapping.get(
                    "source_id",
                    f"source-{index}",
                ),
            ),
        )

        source_rank = source_mapping.get("rank", index)

        source_metadata = source_mapping.get("metadata", {})
        if source_metadata is None:
            source_metadata = {}

        if not isinstance(source_metadata, Mapping):
            raise ValueError(
                "Source metadata must be a mapping."
            )

        source_excerpt = source_mapping.get(
            "excerpt",
            source_mapping.get(
                "content",
                source_mapping.get(
                    "text",
                    "",
                ),
            ),
        )

        sources.append(
            SourceResponse(
                id=str(source_id),
                score=_convert_optional_float(
                    source_mapping.get("score")
                ),
                rank=int(source_rank),
                metadata=dict(source_metadata),
                excerpt=str(source_excerpt or ""),
            )
        )

    return sources


def _build_dashboard_payload(
    raw_dashboard: Any,
) -> DashboardPayload:
    """
    Convert the service dashboard result into DashboardPayload.

    Args:
        raw_dashboard:
            Dashboard information returned by FinanceAskService.

    Returns:
        Validated DashboardPayload instance.

    Raises:
        ValueError:
            If the dashboard value cannot be validated.
    """

    if isinstance(raw_dashboard, DashboardPayload):
        return raw_dashboard

    if raw_dashboard is None:
        return DashboardPayload.model_validate({})

    if hasattr(raw_dashboard, "model_dump"):
        raw_dashboard = raw_dashboard.model_dump()

    elif hasattr(raw_dashboard, "__dict__") and not isinstance(
        raw_dashboard,
        Mapping,
    ):
        raw_dashboard = vars(raw_dashboard)

    return DashboardPayload.model_validate(raw_dashboard)


def _convert_to_mapping(
    value: Any,
) -> Mapping[str, Any]:
    """
    Convert a source object into a mapping.

    Args:
        value:
            Dictionary, Pydantic model, dataclass-like object or regular
            object returned by the service layer.

    Returns:
        Mapping representation of the value.

    Raises:
        ValueError:
            If the value cannot be represented as a mapping.
    """

    if isinstance(value, Mapping):
        return value

    if hasattr(value, "model_dump"):
        model_data = value.model_dump()

        if isinstance(model_data, Mapping):
            return model_data

    if hasattr(value, "__dict__"):
        object_data = vars(value)

        if isinstance(object_data, Mapping):
            return object_data

    raise ValueError(
        "Finance source record must be a mapping or a supported "
        "model object."
    )


def _convert_optional_float(
    value: Any,
) -> float | None:
    """
    Convert an optional source score into a float.

    Args:
        value:
            Score returned by the retrieval layer.

    Returns:
        Float score when provided; otherwise None.

    Raises:
        ValueError:
            If a non-null score cannot be converted to float.
    """

    if value is None:
        return None

    return float(value)