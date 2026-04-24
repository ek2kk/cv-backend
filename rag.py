from llm import call_llm
from search import search


def answer_with_rag(question: str) -> dict:
    articles = search(question, k=2)

    if not articles or articles[0]["score"] < 0.3:
        return {
            "answer": "Не нашёл релевантной информации в резюме.",
            "sources": [],
        }

    context = "\n\n---\n\n".join(
        f"Источник: {a['title']} ({a['file']})\n\n{a['text']}" for a in articles
    )

    messages = [
        {
            "role": "system",
            "content": (
                "Ты RAG-ассистент по резюме ML/AI разработчика. "
                "Отвечай кратко, по-русски и только по контексту. "
                "Не выдумывай факты. Если информации нет, скажи об этом."
            ),
        },
        {
            "role": "user",
            "content": f"""
Контекст:
{context}

Вопрос:
{question}
""",
        },
    ]

    answer = call_llm(messages)

    return {
        "answer": answer,
        "sources": [
            {
                "title": a["title"],
                "file": a["file"],
                "score": round(float(a["score"]), 3),
                "text": a["text"],
            }
            for a in articles
        ],
    }
