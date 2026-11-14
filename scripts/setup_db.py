"""
Set up the database in one go: schema, seed data, the read-only user, then policy ingestion.

    python -m scripts.setup_db              # build an empty database (refuses one that has data)
    python -m scripts.setup_db --reset      # wipe the shop data and build it again
    python -m scripts.setup_db --if-empty   # only build if the shop data isn't there yet;
                                            # otherwise just re-ingest changed policy files
    python -m scripts.setup_db --skip-ingest

Every run also applies db/roles.sql (the read-only user's rules). It touches
no data, so rule changes reach a database that already exists, for example
the Docker volume, whose start uses --if-empty. Conversation memory is in
its own tables, which neither file drops.
"""

import argparse
import sys
from pathlib import Path

from app.core.config import settings
from app.db import connect

DB_DIR = Path(__file__).resolve().parent.parent / "db"
SCHEMA = DB_DIR / "schema.sql"
ROLES = DB_DIR / "roles.sql"


def data_exists() -> bool:
    """True when the tables exist and the seed finished (a start stopped
    between the schema and the seed leaves empty tables behind)."""
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT to_regclass('public.support_tickets') IS NOT NULL")
        if not cur.fetchone()[0]:
            return False
        cur.execute("SELECT EXISTS (SELECT 1 FROM support_tickets)")
        return cur.fetchone()[0]


def apply_sql(path: Path) -> None:
    with connect() as conn:
        conn.execute(path.read_text(encoding="utf-8"))
        conn.commit()
    print(f"Loaded {path.name}.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    when = parser.add_mutually_exclusive_group()
    when.add_argument("--if-empty", action="store_true", help="keep existing shop data")
    when.add_argument("--reset", action="store_true", help="wipe existing shop data and rebuild it")
    parser.add_argument("--skip-ingest", action="store_true")
    args = parser.parse_args(argv)

    has_data = data_exists()
    if has_data and args.if_empty:
        print("Shop data already exists; keeping it.")
    elif has_data and not args.reset:
        sys.exit(
            "The database already has shop data. Run again with --reset to wipe and rebuild it, "
            "or with --if-empty to keep it."
        )
    else:
        from db import seed

        apply_sql(SCHEMA)
        seed.main()
    apply_sql(ROLES)

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
