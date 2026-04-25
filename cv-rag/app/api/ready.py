from typing import Annotated

from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.services.search import is_index_ready

router = APIRouter()


@router.get("/ready")
def ready(settings: Annotated[Settings, Depends(get_settings)]):
    openrouter_ready = bool(settings.openrouter.api_key)
    index_ready = is_index_ready()
    return {
        "status": "ok" if openrouter_ready and index_ready else "degraded",
        "index": index_ready,
        "openrouter": openrouter_ready,
    }
