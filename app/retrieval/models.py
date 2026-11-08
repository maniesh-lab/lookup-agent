"""The two local ML models used by search, loaded once and reused.

- Embedder: turns text into a list of 384 numbers (a vector). Texts with
  similar meaning get vectors that point in similar directions.
- Reranker: reads a question and one chunk together and scores how well
  the chunk answers it. Slower than vectors but more accurate, so it is
  only used on the top results.

Both come from sentence-transformers and are downloaded the first time
they are used (about 220 MB together). Tests swap in fakes, so the test
suite never downloads them.

Loading is the slow part of search, not using the models. Measured on
2026-11-08, the first search after a start took 10.6 s, and 9.6 s of it
was the Hugging Face library checking online for newer versions of two
models that were already downloaded. So models load from the local
cache first and only go online when they aren't there yet, and the
server loads them at startup (load_search_models) instead of during the
first question.
"""

from functools import lru_cache
from typing import Protocol

import structlog

from app.core.config import settings

log = structlog.get_logger(__name__)

# bge models were trained to expect this prefix on search queries
# (not on the documents being searched).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


def load_cached_first(model_class, model_name: str):
    """Load a model from the local cache, and download it only if it isn't there.

    Without local_files_only, every load asks huggingface.co whether the
    model changed, which took about 5 s per model even with it on disk.
    The trade-off: a newer upstream revision of a model is not picked up
    until the cached copy is deleted (or HF_HOME points somewhere empty).
    """
    try:
        return model_class(model_name, local_files_only=True)
    except OSError:
        log.info("model_not_cached", model=model_name)
        return model_class(model_name)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # heavy import, done lazily

        self.model_name = model_name
        self.model = load_cached_first(SentenceTransformer, model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True, batch_size=32)
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        prefix = BGE_QUERY_PREFIX if "bge" in self.model_name.lower() else ""
        return self.model.encode(prefix + text, normalize_embeddings=True).tolist()


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.model = load_cached_first(CrossEncoder, model_name)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        return [float(s) for s in self.model.predict([(query, t) for t in texts])]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return SentenceTransformerEmbedder(settings.embedding_model)


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    return CrossEncoderReranker(settings.reranker_model)


def load_search_models(mode: str) -> None:
    """Load the models a search in this mode needs, so no question waits for them."""
    get_embedder()
    if mode == "rerank":
        get_reranker()
