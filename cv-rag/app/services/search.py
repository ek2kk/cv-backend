import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any

import faiss
from pydantic import BaseModel

from app.core.logging import get_logger
from app.dependencies import get_embedding_model
from app.exceptions import IndexBuildError
from app.models import SearchResult

APP_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = APP_DIR / "data" / "raw"
PROCESSED_DATA_DIR = APP_DIR / "data" / "processed"
DEFAULT_INDEX_PATH = PROCESSED_DATA_DIR / "resume.index"
DEFAULT_META_PATH = PROCESSED_DATA_DIR / "resume_meta.json"
INDEX_VERSION = 2
logger = get_logger(__name__)


class IndexedDocument(BaseModel):
    chunk_id: str = ""
    file: str
    title: str
    text: str


def _raw_markdown_files(md_dir: Path) -> list[Path]:
    return sorted(path for path in md_dir.glob("*.md") if path.read_text(encoding="utf-8").strip())


def _index_needs_rebuild(
    md_dir: str | Path = RAW_DATA_DIR,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    meta_path: str | Path = DEFAULT_META_PATH,
) -> bool:
    md_dir = Path(md_dir)
    index_path = Path(index_path)
    meta_path = Path(meta_path)

    if not index_path.is_file() or not meta_path.is_file():
        return True

    raw_files = _raw_markdown_files(md_dir)
    newest_raw_mtime = max((path.stat().st_mtime for path in raw_files), default=0)
    oldest_index_mtime = min(index_path.stat().st_mtime, meta_path.stat().st_mtime)

    if newest_raw_mtime > oldest_index_mtime:
        return True

    try:
        with meta_path.open("r", encoding="utf-8") as f:
            raw_metadata = json.load(f)
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return True

    if not raw_metadata or any(item.get("index_version") != INDEX_VERSION for item in raw_metadata):
        return True

    indexed_files = {item["file"] for item in raw_metadata}
    return indexed_files != {path.name for path in raw_files}


def _extract_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line.replace("# ", "").strip()
    return fallback


def _split_markdown_into_chunks(path: Path) -> list[IndexedDocument]:
    text = path.read_text(encoding="utf-8").strip()
    document_title = _extract_title(text, path.stem)
    chunks: list[IndexedDocument] = []
    current_heading = document_title
    current_lines: list[str] = []

    def flush() -> None:
        chunk_text = "\n".join(current_lines).strip()
        if not chunk_text:
            return
        chunk_number = len(chunks) + 1
        chunks.append(
            IndexedDocument(
                chunk_id=f"{path.stem}:{chunk_number}",
                file=path.name,
                title=current_heading,
                text=chunk_text,
            )
        )

    for line in text.splitlines():
        if line.startswith("## "):
            flush()
            current_heading = line.replace("## ", "").strip()
            current_lines = [f"# {document_title}", "", line]
            continue

        current_lines.append(line)

    flush()
    return chunks


def _normalize_token(token: str) -> str:
    if re.fullmatch(r"[а-яё]+", token) and len(token) > 5:
        return token[:5]
    return token


def _tokenize(text: str) -> list[str]:
    stop_words = {"что", "как", "это", "или", "для", "про", "его", "она", "они"}
    return [
        _normalize_token(token)
        for token in re.findall(r"[a-zа-яё0-9]+", text.lower())
        if len(token) > 2 and token not in stop_words
    ]


def _bm25_scores(query: str, documents: list[IndexedDocument]) -> list[float]:
    query_terms = _tokenize(query)
    if not query_terms:
        return [0.0 for _ in documents]

    tokenized_docs = [_tokenize(f"{doc.title}\n{doc.text}") for doc in documents]
    doc_count = len(tokenized_docs)
    avg_doc_len = sum(len(doc) for doc in tokenized_docs) / doc_count if doc_count else 0.0
    doc_freq: Counter[str] = Counter()

    for doc in tokenized_docs:
        doc_freq.update(set(doc))

    k1 = 1.5
    b = 0.75
    scores: list[float] = []

    for doc in tokenized_docs:
        term_freq = Counter(doc)
        doc_len = len(doc)
        score = 0.0

        for term in query_terms:
            if term_freq[term] == 0:
                continue

            idf = math.log(1 + (doc_count - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            numerator = term_freq[term] * (k1 + 1)
            denominator = term_freq[term] + k1 * (1 - b + b * doc_len / avg_doc_len)
            score += idf * numerator / denominator

        scores.append(score)

    return scores


def _normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []

    max_score = max(scores)
    if max_score <= 0:
        return [0.0 for _ in scores]

    return [score / max_score for score in scores]


def build_index(
    md_dir: str | Path = RAW_DATA_DIR,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    meta_path: str | Path = DEFAULT_META_PATH,
) -> None:
    started_at = perf_counter()
    md_dir = Path(md_dir)
    index_path = Path(index_path)
    meta_path = Path(meta_path)

    docs: list[str] = []
    metadata: list[IndexedDocument] = []

    for path in _raw_markdown_files(md_dir):
        chunks = _split_markdown_into_chunks(path)
        docs.extend(chunk.text for chunk in chunks)
        metadata.extend(chunks)

    passages = [f"passage: {doc}" for doc in docs]
    if not passages:
        logger.error(
            "Index build failed: no markdown files",
            extra={"extra_data": {"md_dir": str(md_dir)}},
        )
        raise IndexBuildError(f"No markdown articles found in {md_dir}")

    model = get_embedding_model()
    embeddings = model.encode(
        passages,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    dim = embeddings.shape[1]

    # cosine similarity через inner product, потому что embeddings нормализованы
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_path))

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    **item.model_dump(),
                    "index_version": INDEX_VERSION,
                }
                for item in metadata
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )

    _load_index.cache_clear()
    logger.info(
        "Search index built",
        extra={
            "extra_data": {
                "chunks": len(docs),
                "index_path": str(index_path),
                "meta_path": str(meta_path),
                "duration_ms": round((perf_counter() - started_at) * 1000, 2),
            }
        },
    )


def is_index_ready(
    index_path: str | Path = DEFAULT_INDEX_PATH,
    meta_path: str | Path = DEFAULT_META_PATH,
) -> bool:
    return Path(index_path).is_file() and Path(meta_path).is_file()


def ensure_index(
    md_dir: str | Path = RAW_DATA_DIR,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    meta_path: str | Path = DEFAULT_META_PATH,
) -> None:
    if _index_needs_rebuild(md_dir, index_path, meta_path):
        logger.info("Search index is missing or stale, building it")
        build_index(md_dir, index_path=index_path, meta_path=meta_path)
    else:
        logger.info("Search index is ready")


@lru_cache(maxsize=1)
def _load_index(index_path: str, meta_path: str) -> tuple[Any, list[IndexedDocument]]:
    started_at = perf_counter()
    index = faiss.read_index(index_path)
    with Path(meta_path).open("r", encoding="utf-8") as f:
        metadata = [IndexedDocument.model_validate(item) for item in json.load(f)]
    logger.info(
        "Search index loaded",
        extra={
            "extra_data": {
                "documents": len(metadata),
                "duration_ms": round((perf_counter() - started_at) * 1000, 2),
            }
        },
    )
    return index, metadata


def search(
    query: str,
    k: int = 3,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    meta_path: str | Path = DEFAULT_META_PATH,
) -> list[SearchResult]:
    started_at = perf_counter()
    index, metadata = _load_index(str(index_path), str(meta_path))
    model = get_embedding_model()
    query_emb = model.encode(
        [f"query: {query}"],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    candidate_count = min(index.ntotal, max(k * 5, 20))
    scores, ids = index.search(query_emb, candidate_count)
    dense_scores = [0.0 for _ in metadata]

    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        dense_scores[idx] = float(score)

    bm25_scores = _bm25_scores(query, metadata)
    dense_norm = _normalize_scores(dense_scores)
    bm25_norm = _normalize_scores(bm25_scores)
    hybrid_scores = [
        0.5 * dense_score + 0.5 * bm25_score
        for dense_score, bm25_score in zip(dense_norm, bm25_norm)
    ]
    ranked_ids = sorted(range(len(metadata)), key=lambda idx: hybrid_scores[idx], reverse=True)[:k]

    results = [
        SearchResult(
            score=float(hybrid_scores[idx]),
            file=metadata[idx].file,
            title=metadata[idx].title,
            text=metadata[idx].text,
        )
        for idx in ranked_ids
        if hybrid_scores[idx] > 0
    ]

    logger.info(
        "Search completed",
        extra={
            "extra_data": {
                "k": k,
                "results": len(results),
                "top_score": round(results[0].score, 3) if results else None,
                "files": [item.file for item in results],
                "retrieval": "hybrid_dense_bm25",
                "duration_ms": round((perf_counter() - started_at) * 1000, 2),
            }
        },
    )
    return results
