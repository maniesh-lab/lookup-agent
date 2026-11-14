"""
Delete conversations nobody has used for a while.

    python -m scripts.prune_threads --older-than-days 30 --dry-run   # only list them
    python -m scripts.prune_threads --older-than-days 30

Conversation memory (LangGraph's checkpoint tables) never expires on its own,
and anyone who can reach /chat can start new threads. Run this now and then
(a daily cron job, for example) to keep the tables from growing without end.
A thread counts as unused when its newest saved step is older than the cutoff.
"""

import argparse
import asyncio

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection
from psycopg.rows import dict_row

from app.core.config import settings
from app.db import plain_url

STALE_THREADS_SQL = """
SELECT thread_id FROM checkpoints
GROUP BY thread_id
HAVING max((checkpoint ->> 'ts')::timestamptz) < now() - make_interval(days => %s)
ORDER BY thread_id
"""


async def prune(days: int, *, dry_run: bool = False, url: str | None = None) -> list[str]:
    """Delete (or with dry_run only list) threads unused for `days` days; returns their ids."""
    async with await AsyncConnection.connect(
        plain_url(url or settings.database_url), autocommit=True, prepare_threshold=0, row_factory=dict_row
    ) as conn:
        saver = AsyncPostgresSaver(conn)
        await saver.setup()  # the tables exist even if the server never started
        cur = await conn.execute(STALE_THREADS_SQL, (days,))
        stale = [row["thread_id"] for row in await cur.fetchall()]
        if not dry_run:
            for thread_id in stale:
                await saver.adelete_thread(thread_id)
    return stale


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--older-than-days", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true", help="list the threads without deleting them")
    args = parser.parse_args(argv)
    if args.older_than_days < 1:
        parser.error("--older-than-days must be at least 1")
    stale = asyncio.run(prune(args.older_than_days, dry_run=args.dry_run))
    verb = "Would delete" if args.dry_run else "Deleted"
    print(f"{verb} {len(stale)} conversation(s) unused for {args.older_than_days}+ days.")


if __name__ == "__main__":
    main()
