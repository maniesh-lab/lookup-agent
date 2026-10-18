"""The two local ML models used by search, loaded once and reused.

- Embedder: turns text into a list of 384 numbers (a vector). Texts with
  similar meaning get vectors that point in similar directions.
- Reranker: reads a question and one chunk together and scores how well
  the chunk answers it. Slower than vectors but more accurate, so it is
  only used on the top results.

Both come from sentence-transformers and are downloaded the first time
they are used (about 100 MB together). Tests swap in fakes, so the test
suite never downloads them.
"""

from functools import lru_cache
from typing import Protocol

from app.core.config import settings

# bge models were trained to expect this prefix on search queries
# (not on the documents being searched).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # heavy import, done lazily

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True, batch_size=32)
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        prefix = BGE_QUERY_PREFIX if "bge" in self.model_name.lower() else ""
        return self.model.encode(prefix + text, normalize_embeddings=True).tolist()


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

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
