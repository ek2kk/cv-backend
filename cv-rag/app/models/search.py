from pydantic import BaseModel


class SearchResult(BaseModel):
    score: float
    file: str
    title: str
    text: str
