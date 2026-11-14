"""
Chunk, embed and store the policy documents.

    python -m scripts.ingest_docs                          # data/policies: the whole index
    python -m scripts.ingest_docs path/to/folder           # add or update that folder's files
    python -m scripts.ingest_docs path/to/folder --replace # make that folder the whole index

The default folder is the whole index: files no longer in it are removed. Another
folder only adds and updates, unless --replace says to remove every document
that isn't in it (which would drop data/policies).
"""

import argparse

from app.retrieval.ingest import ingest_folder


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("folder", nargs="?", help="a folder of .md files (default: data/policies)")
    parser.add_argument("--replace", action="store_true", help="remove documents that aren't in the folder")
    args = parser.parse_args(argv)
    stats = ingest_folder(args.folder, remove_missing=args.folder is None or args.replace)
    print(f"Added:     {stats.added}")
    print(f"Updated:   {stats.updated}")
    print(f"Unchanged: {stats.unchanged}")
    print(f"Removed:   {stats.removed}")
    print(f"Chunks written: {stats.chunks_written}")


if __name__ == "__main__":
    main()
