from time import perf_counter

from fastapi import APIRouter, HTTPException

from app.core.logging import get_logger
from app.exceptions import CvRagError
from app.models import ChatRequest, ChatResponse
from app.services.rag import answer_with_rag

logger = get_logger(__name__)
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    started_at = perf_counter()
    try:
        response = answer_with_rag(req.message)
        logger.info(
            "Chat request completed",
            extra={
                "extra_data": {
                    "message_chars": len(req.message),
                    "sources": len(response["sources"]),
                    "duration_ms": round((perf_counter() - started_at) * 1000, 2),
                }
            },
        )
        return response

    except CvRagError as e:
        logger.exception(
            "Chat request failed",
            extra={
                "extra_data": {
                    "message_chars": len(req.message),
                    "duration_ms": round((perf_counter() - started_at) * 1000, 2),
                }
            },
        )
        raise HTTPException(status_code=500, detail="LLM/RAG error") from e
