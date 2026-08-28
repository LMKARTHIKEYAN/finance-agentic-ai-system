"""Public routes for the autonomous finance API."""

from fastapi import APIRouter

from src.api.autonomous_routes import router as autonomous_router


router = APIRouter()
router.include_router(autonomous_router)


@router.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "healthy",
        "service": "autonomous-finance-agentic-ai-api",
        "architecture": "decide-act-observe",
    }
