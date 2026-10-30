"""
Set up the database in one go: schema, seed data, then policy ingestion.

    python -m scripts.setup_db              # rebuild everything (wipes the data)
    python -m scripts.setup_db --if-empty   # only build if the shop data isn't there yet;
                                            # otherwise just re-ingest changed policy files
    python -m scripts.setup_db --skip-ingest
"""

import argparse
from pathlib import Path

from app.core.config import settings
from app.db import connect

SCHEMA = Path(__file__).resolve().parent.parent / "db" / "schema.sql"


def data_exists() -> bool:
    """True when the tables exist and the seed finished (a start stopped
    between the schema and the seed leaves empty tables behind)."""
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT to_regclass('public.support_tickets') IS NOT NULL")
        if not cur.fetchone()[0]:
            return False
        cur.execute("SELECT EXISTS (SELECT 1 FROM support_tickets)")
        return cur.fetchone()[0]


def apply_schema() -> None:
    with connect() as conn:
        conn.execute(SCHEMA.read_text(encoding="utf-8"))
        conn.commit()
    print("Schema loaded.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--if-empty", action="store_true")
    parser.add_argument("--skip-ingest", action="store_true")
    args = parser.parse_args()

    if args.if_empty and data_exists():
        print("Shop data already exists; keeping it.")
    else:
        from db import seed

        apply_schema()
        seed.main()

    if not args.skip_ingest:
        from app.retrieval.ingest import ingest_folder

        stats = ingest_folder()
        print(f"Ingested policies: {stats.as_dict()}")

        if settings.search_mode == "rerank":
            from app.retrieval.models import get_reranker

            get_reranker()  # download it now, not during the first question
            print("Reranker model ready.")


if __name__ == "__main__":
    main()
