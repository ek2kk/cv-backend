from app.models import SearchResult
from app.services import rag


def test_answer_with_rag_rejects_low_score(monkeypatch) -> None:
    monkeypatch.setattr(
        rag,
        "search",
        lambda question, k: [
            SearchResult(score=0.7, file="skills.md", title="НАВЫКИ", text="Python")
        ],
    )
    monkeypatch.setattr(
        rag,
        "call_llm",
        lambda messages: (_ for _ in ()).throw(AssertionError("LLM must not be called")),
    )

    response = rag.answer_with_rag("Какой любимый цвет?")

    assert response == {
        "answer": "Не нашёл релевантной информации в резюме.",
        "sources": [],
    }


def test_answer_with_rag_uses_source_blocks(monkeypatch) -> None:
    captured = {}

    monkeypatch.setattr(
        rag,
        "search",
        lambda question, k: [
            SearchResult(score=0.9, file="experience.md", title="ОПЫТ", text="Работал с RAG.")
        ],
    )

    def fake_call_llm(messages):
        captured["messages"] = messages
        return "Есть опыт с RAG."

    monkeypatch.setattr(rag, "call_llm", fake_call_llm)

    response = rag.answer_with_rag("Есть ли опыт с RAG?")

    assert response["answer"] == "Есть опыт с RAG."
    assert "SOURCE[1]: ОПЫТ (experience.md)" in captured["messages"][1]["content"]
