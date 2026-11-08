"""Chunking and result merging (pure Python, no database)."""

from pathlib import Path

from app.agent.prompts import available_versions, load_prompt
from app.retrieval.chunking import chunk_markdown, parse_title, split_paragraphs
from app.retrieval.search import SearchHit, keyword_query, reciprocal_rank_fusion

POLICIES = Path(__file__).resolve().parent.parent / "data" / "policies"

SAMPLE = """# Returns Policy

Intro text.

## Return window

You can return items within 30 days.

## Who pays

Free when it's our fault.
"""


def test_chunk_per_section():
    chunks = chunk_markdown(SAMPLE, "Returns Policy")
    assert [c.heading for c in chunks] == [
        "Returns Policy > Overview",
        "Returns Policy > Return window",
        "Returns Policy > Who pays",
    ]
    assert chunks[1].content == "You can return items within 30 days."
    assert [c.index for c in chunks] == [0, 1, 2]
    assert chunks[1].text_for_embedding.startswith("Returns Policy > Return window\n")


def test_long_sections_split_at_paragraphs():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 30 for i in range(6))
    pieces = split_paragraphs(text, max_chars=400)
    assert len(pieces) > 1
    assert all(len(p) <= 400 for p in pieces)
    assert "".join(pieces).count("Paragraph") == 6


def test_parse_title():
    assert parse_title(SAMPLE, "x") == "Returns Policy"
    assert parse_title("no heading", "fallback") == "fallback"


def test_every_policy_file_chunks_cleanly():
    files = sorted(POLICIES.glob("*.md"))
    assert 8 <= len(files) <= 10
    for path in files:
        text = path.read_text()
        chunks = chunk_markdown(text, parse_title(text, path.stem))
        assert chunks, path.name
        assert all(c.content.strip() for c in chunks), path.name


def hit(cid):
    return SearchHit(cid, "f.md", "h", "text", 0.0)


def test_rrf_rewards_hits_found_by_both_lists():
    vector = [hit(1), hit(2), hit(3)]
    keyword = [hit(3), hit(4), hit(1)]
    fused = reciprocal_rank_fusion([vector, keyword])
    assert [h.chunk_id for h in fused][:2] in ([1, 3], [3, 1])
    assert {h.chunk_id for h in fused} == {1, 2, 3, 4}
    assert fused[0].score > fused[-1].score


def test_keyword_query_ors_words():
    assert keyword_query("What's the return window?") == "what | s | the | return | window"
    assert keyword_query("?!") is None


def test_prompts_load():
    assert available_versions() == ["v1", "v2"]
    assert "Today's date is 2026-01-02" in load_prompt("v1", today=__import__("datetime").date(2026, 1, 2))


# --- Loading the search models ---------------------------------------------------


def test_models_load_from_the_cache_first():
    from app.retrieval.models import load_cached_first

    tried = []

    class Model:
        def __init__(self, name, local_files_only=False):
            tried.append(local_files_only)

    load_cached_first(Model, "some/model")
    assert tried == [True]  # found locally: no online check


def test_models_are_downloaded_when_not_cached():
    from app.retrieval.models import load_cached_first

    tried = []

    class Model:
        def __init__(self, name, local_files_only=False):
            tried.append(local_files_only)
            if local_files_only:
                raise OSError("not in the cache")

    load_cached_first(Model, "some/model")
    assert tried == [True, False]


def test_only_the_models_the_search_mode_needs_are_loaded(monkeypatch):
    from app.retrieval import models

    loaded = []
    monkeypatch.setattr(models, "get_embedder", lambda: loaded.append("embedder"))
    monkeypatch.setattr(models, "get_reranker", lambda: loaded.append("reranker"))

    models.load_search_models("hybrid")
    assert loaded == ["embedder"]
    models.load_search_models("rerank")
    assert loaded == ["embedder", "embedder", "reranker"]
