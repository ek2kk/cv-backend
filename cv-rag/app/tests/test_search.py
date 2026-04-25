from pathlib import Path

from app.models import SearchResult
from app.services.search import RAW_DATA_DIR, build_index, search


def test_search_returns_indexed_articles(tmp_path: Path) -> None:
    index_path = tmp_path / "resume.index"
    meta_path = tmp_path / "resume_meta.json"

    build_index(RAW_DATA_DIR, index_path=index_path, meta_path=meta_path)
    results = search("Какой линкедин у тебя?", k=3, index_path=index_path, meta_path=meta_path)

    assert results
    assert isinstance(results[0], SearchResult)
    assert results[0].score > 0
