from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import chat_router, health_router, ready_router
from app.core.logging import get_logger, setup_logging
from app.services.search import RAW_DATA_DIR, ensure_index

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info(
        "Preparing vector database",
        extra={"extra_data": {"source_dir": str(RAW_DATA_DIR)}},
    )
    ensure_index()
    logger.info("Vector database is ready")
    yield


app = FastAPI(title="Resume RAG API", lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(ready_router)
app.include_router(chat_router)


def main():
    print("Hello from cv-rag!")


if __name__ == "__main__":
    main()
