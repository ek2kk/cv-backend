from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag import answer_with_rag


app = FastAPI(title="Resume RAG API")


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


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)


class Source(BaseModel):
    title: str
    file: str
    score: float
    text: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        return answer_with_rag(req.message)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM/RAG error: {str(e)}",
        )


def main():
    print("Hello from cv-rag!")


if __name__ == "__main__":
    main()
