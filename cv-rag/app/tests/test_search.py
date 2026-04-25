import json
from pathlib import Path

import faiss

from app.models import SearchResult
from app.services.search import RAW_DATA_DIR, _index_needs_rebuild, _load_index, build_index, search


def test_search_returns_indexed_articles(tmp_path: Path) -> None:
    index_path = tmp_path / "resume.index"
    meta_path = tmp_path / "resume_meta.json"

    build_index(RAW_DATA_DIR, index_path=index_path, meta_path=meta_path)
    results = search("Какой линкедин у тебя?", k=3, index_path=index_path, meta_path=meta_path)

    assert results
    assert isinstance(results[0], SearchResult)
    assert results[0].score > 0


def test_load_index_accepts_legacy_metadata_without_score(tmp_path: Path) -> None:
    index_path = tmp_path / "resume.index"
    meta_path = tmp_path / "resume_meta.json"

    index = faiss.IndexFlatIP(2)
    faiss.write_index(index, str(index_path))
    meta_path.write_text(
        json.dumps(
            [
                {
                    "file": "contacts.md",
                    "title": "КОНТАКТЫ",
                    "text": "# КОНТАКТЫ\n\n- LinkedIn: https://www.linkedin.com/in/ek2kk/",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    _load_index.cache_clear()
    _, metadata = _load_index(str(index_path), str(meta_path))

    assert metadata[0].file == "contacts.md"


def test_index_needs_rebuild_when_raw_files_change(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    index_path = tmp_path / "resume.index"
    meta_path = tmp_path / "resume_meta.json"

    (raw_dir / "about.md").write_text("# ОБО МНЕ\n\nML-инженер", encoding="utf-8")
    index = faiss.IndexFlatIP(2)
    faiss.write_index(index, str(index_path))
    meta_path.write_text(
        json.dumps(
            [{"file": "contacts.md", "title": "КОНТАКТЫ", "text": "# КОНТАКТЫ"}],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    assert _index_needs_rebuild(raw_dir, index_path, meta_path)
