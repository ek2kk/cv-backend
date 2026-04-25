import json
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
logger = get_logger(__name__)


class IndexedDocument(BaseModel):
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
            indexed_files = {item["file"] for item in json.load(f)}
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return True

    return indexed_files != {path.name for path in raw_files}


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
        text = path.read_text(encoding="utf-8").strip()

        title = path.stem

        # Берём первый markdown heading как title, если есть
        for line in text.splitlines():
            if line.startswith("# "):
                title = line.replace("# ", "").strip()
                break

        docs.append(text)
        metadata.append(IndexedDocument(file=path.name, title=title, text=text))

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
        json.dump([item.model_dump() for item in metadata], f, ensure_ascii=False, indent=2)

    _load_index.cache_clear()
    logger.info(
        "Search index built",
        extra={
            "extra_data": {
                "documents": len(docs),
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

    scores, ids = index.search(query_emb, k)

    results: list[SearchResult] = []

    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue

        item = metadata[idx]

        results.append(
            SearchResult(
                score=float(score),
                file=item.file,
                title=item.title,
                text=item.text,
            )
        )

    logger.info(
        "Search completed",
        extra={
            "extra_data": {
                "k": k,
                "results": len(results),
                "top_score": round(results[0].score, 3) if results else None,
                "files": [item.file for item in results],
                "duration_ms": round((perf_counter() - started_at) * 1000, 2),
            }
        },
    )
    return results
