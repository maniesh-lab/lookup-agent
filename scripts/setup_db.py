"""
Set up the database in one go: schema, seed data, then policy ingestion.

    python -m scripts.setup_db              # rebuild everything (wipes the data)
    python -m scripts.setup_db --if-empty   # only build if the tables don't exist yet;
                                            # otherwise just re-ingest changed policy files
    python -m scripts.setup_db --skip-ingest
"""

import argparse
from pathlib import Path

from app.db import connect

SCHEMA = Path(__file__).resolve().parent.parent / "db" / "schema.sql"


def tables_exist() -> bool:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT to_regclass('public.customers') IS NOT NULL")
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

    if args.if_empty and tables_exist():
        print("Tables already exist; keeping the data.")
    else:
        from db import seed

        apply_schema()
        seed.main()

    if not args.skip_ingest:
        from app.retrieval.ingest import ingest_folder

        stats = ingest_folder()
        print(f"Ingested policies: {stats.as_dict()}")


if __name__ == "__main__":
    main()
