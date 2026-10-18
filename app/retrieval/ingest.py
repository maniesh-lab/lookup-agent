"""Ingestion: read policy files, chunk them, embed the chunks, store them.

A file is skipped when its content hasn't changed since the last run
(same sha256), so running ingestion again is cheap. A changed file has
its old chunks deleted and replaced. A file that was deleted from the
folder is removed from the database too.
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import structlog

from app.core.config import settings
from app.db import connect
from app.retrieval.chunking import chunk_markdown, parse_title
from app.retrieval.models import Embedder, get_embedder

log = structlog.get_logger(__name__)


@dataclass
class IngestStats:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    chunks_written: int = 0

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def ingest_folder(
    folder: str | Path | None = None,
    *,
    embedder: Embedder | None = None,
    conn=None,
) -> IngestStats:
    folder = Path(folder or settings.policies_dir)
    files = sorted(folder.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No .md files found in {folder}")

    embedder = embedder or get_embedder()
    stats = IngestStats()
    own_conn = conn is None
    conn = conn or connect(vector=True)

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT source, id, content_hash FROM documents")
            existing = {source: (doc_id, h) for source, doc_id, h in cur.fetchall()}

            for path in files:
                text = path.read_text(encoding="utf-8")
                digest = hashlib.sha256(text.encode()).hexdigest()
                source = path.name
                old = existing.pop(source, None)

                if old and old[1] == digest:
                    stats.unchanged.append(source)
                    continue

                title = parse_title(text, fallback=path.stem)
                chunks = chunk_markdown(text, title)
                vectors = embedder.embed_documents([c.text_for_embedding for c in chunks])

                if old:
                    doc_id = old[0]
                    cur.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))
                    cur.execute(
                        "UPDATE documents SET title = %s, content_hash = %s, ingested_at = now()"
                        " WHERE id = %s",
                        (title, digest, doc_id),
                    )
                    stats.updated.append(source)
                else:
                    cur.execute(
                        "INSERT INTO documents (source, title, content_hash) VALUES (%s, %s, %s)"
                        " RETURNING id",
                        (source, title, digest),
                    )
                    doc_id = cur.fetchone()[0]
                    stats.added.append(source)

                cur.executemany(
                    "INSERT INTO chunks (document_id, chunk_index, heading, content, embedding)"
                    " VALUES (%s, %s, %s, %s, %s)",
                    [
                        (doc_id, c.index, c.heading, c.content, np.array(v, dtype=np.float32))
                        for c, v in zip(chunks, vectors, strict=True)
                    ],
                )
                stats.chunks_written += len(chunks)

            # Files that were in the database but are no longer in the folder.
            for source, (doc_id, _h) in existing.items():
                cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
                stats.removed.append(source)

        conn.commit()
    finally:
        if own_conn:
            conn.close()

    log.info("ingest_done", **stats.as_dict())
    return stats
