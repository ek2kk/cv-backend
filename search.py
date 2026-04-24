import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-small"


def build_index(md_dir: str, index_path="resume.index", meta_path="resume_meta.json"):
    model = SentenceTransformer(MODEL_NAME)

    docs = []
    metadata = []

    for path in Path(md_dir).glob("*.md"):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        title = path.stem

        # Берём первый markdown heading как title, если есть
        for line in text.splitlines():
            if line.startswith("# "):
                title = line.replace("# ", "").strip()
                break

        docs.append(text)
        metadata.append(
            {
                "file": path.name,
                "title": title,
                "text": text,
            }
        )

    passages = [f"passage: {doc}" for doc in docs]

    embeddings = model.encode(
        passages,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    dim = embeddings.shape[1]

    # cosine similarity через inner product, потому что embeddings нормализованы
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, index_path)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Indexed {len(docs)} documents")


def search(query: str, k=3, index_path="resume.index", meta_path="resume_meta.json"):
    model = SentenceTransformer(MODEL_NAME)

    index = faiss.read_index(index_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    query_emb = model.encode(
        [f"query: {query}"],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    scores, ids = index.search(query_emb, k)

    results = []

    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue

        item = metadata[idx]

        results.append(
            {
                "score": float(score),
                "file": item["file"],
                "title": item["title"],
                "text": item["text"],
            }
        )

    return results
