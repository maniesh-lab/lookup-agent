"""
Chunk, embed and store the policy documents.

    python -m scripts.ingest_docs                 # uses data/policies
    python -m scripts.ingest_docs path/to/folder
"""

import sys

from app.retrieval.ingest import ingest_folder


def main() -> None:
    folder = sys.argv[1] if len(sys.argv) > 1 else None
    stats = ingest_folder(folder)
    print(f"Added:     {stats.added}")
    print(f"Updated:   {stats.updated}")
    print(f"Unchanged: {stats.unchanged}")
    print(f"Removed:   {stats.removed}")
    print(f"Chunks written: {stats.chunks_written}")


if __name__ == "__main__":
    main()
