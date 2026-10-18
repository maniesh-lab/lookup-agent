"""Search over the policy chunks, in three layers.

1. vector:  compare meaning (embedding similarity). Good at paraphrases
            ("send it back" finds "returns"), weaker on exact words.
2. hybrid:  vector results plus keyword results (Postgres full-text
            search), merged with Reciprocal Rank Fusion. Keywords catch
            exact terms like "express" or "6.95".
3. rerank:  take the hybrid top candidates and let a cross-encoder
            re-score each one against the question.

eval/run_retrieval_eval.py measures the hit rate of each layer.
"""

import re
from dataclasses import dataclass

import numpy as np

from app.db import connect
from app.retrieval.models import Embedder, Reranker, get_embedder, get_reranker

MODES = ("vector", "hybrid", "rerank")
CANDIDATES = 20  # how many results each method fetches before merging


@dataclass
class SearchHit:
    chunk_id: int
    source: str
    heading: str
    content: str
    score: float

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "section": self.heading,
            "text": self.content,
            "score": round(self.score, 4),
        }


SELECT_COLUMNS = "c.id, d.source, c.heading, c.content"


def vector_search(conn, query_vector, k: int) -> list[SearchHit]:
    vec = np.array(query_vector, dtype=np.float32)
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT {SELECT_COLUMNS}, 1 - (c.embedding <=> %s) AS score"
            " FROM chunks c JOIN documents d ON d.id = c.document_id"
            " ORDER BY c.embedding <=> %s LIMIT %s",
            (vec, vec, k),
        )
        return [SearchHit(*row) for row in cur.fetchall()]


def keyword_query(text: str) -> str | None:
    """Turn a question into an OR full-text query: "return | shipping | ...".

    Postgres' own websearch_to_tsquery() requires every word to match,
    which almost never happens for a whole question, so words are OR-ed
    and ranking decides which chunk matches best.
    """
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return " | ".join(words) if words else None


def keyword_search(conn, query: str, k: int) -> list[SearchHit]:
    tsquery = keyword_query(query)
    if not tsquery:
        return []
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT {SELECT_COLUMNS}, ts_rank_cd(c.tsv, q) AS score"
            " FROM chunks c JOIN documents d ON d.id = c.document_id,"
            " to_tsquery('english', %s) q"
            " WHERE c.tsv @@ q ORDER BY score DESC, c.id LIMIT %s",
            (tsquery, k),
        )
        return [SearchHit(*row) for row in cur.fetchall()]


def reciprocal_rank_fusion(result_lists: list[list[SearchHit]], k: int = 60) -> list[SearchHit]:
    """Merge ranked lists. Each hit earns 1 / (k + rank) from every list it
    appears in, so a chunk ranked well by both methods rises to the top.
    k=60 is the value from the original RRF paper; it stops the top rank
    of one list from outweighing everything else."""
    scores: dict[int, float] = {}
    hits: dict[int, SearchHit] = {}
    for results in result_lists:
        for rank, hit in enumerate(results, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
            hits.setdefault(hit.chunk_id, hit)
    ordered = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [
        SearchHit(hits[c].chunk_id, hits[c].source, hits[c].heading, hits[c].content, scores[c])
        for c in ordered
    ]


def search(
    query: str,
    *,
    mode: str = "rerank",
    k: int = 5,
    conn=None,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
) -> list[SearchHit]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")

    embedder = embedder or get_embedder()
    own_conn = conn is None
    conn = conn or connect(vector=True)
    try:
        qvec = embedder.embed_query(query)
        if mode == "vector":
            return vector_search(conn, qvec, k)

        fused = reciprocal_rank_fusion(
            [vector_search(conn, qvec, CANDIDATES), keyword_search(conn, query, CANDIDATES)]
        )
        if mode == "hybrid":
            return fused[:k]

        reranker = reranker or get_reranker()
        candidates = fused[:CANDIDATES]
        scores = reranker.score(query, [f"{h.heading}\n{h.content}" for h in candidates])
        for hit, score in zip(candidates, scores, strict=True):
            hit.score = score
        return sorted(candidates, key=lambda h: h.score, reverse=True)[:k]
    finally:
        if own_conn:
            conn.close()
