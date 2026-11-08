"""A small in-memory rate limit: at most N requests per client in any 60 seconds.

Every question to /chat costs LLM tokens, and Groq's free tier has a daily
token limit, so one client sending questions in a loop could use it up for
everyone. This keeps the timestamps of each client's recent requests and
refuses a request when the client already has N inside the last 60
seconds (a sliding window, so there is no burst allowed at a minute
boundary).

Written by hand instead of adding a dependency, and it has the limits that
come with that:

- The counters live in this process's memory, so they reset when the
  server restarts.
- It only works for a single server process. With several uvicorn workers
  or several containers, each one counts separately, so a client gets N
  per process.
- Clients are told apart by IP address. Behind a reverse proxy (nginx, a
  cloud load balancer) every request comes from the proxy's IP, so all
  users would share one limit, unless uvicorn is told to trust the
  proxy's X-Forwarded-For header: `--forwarded-allow-ips=<proxy address>`
  (uvicorn reads the header by default, but only trusts 127.0.0.1). The
  same happens with Docker port publishing where requests arrive from the
  Docker gateway (Docker Desktop on Mac and Windows). Several users behind
  one office or mobile network IP also share a limit, and so do requests
  with no client address at all (counted as "unknown").

For more than one process, use a shared store such as Redis instead.
"""

import math
import threading
import time
from collections import deque

WINDOW_SECONDS = 60.0


class SlidingWindowRateLimiter:
    def __init__(self, window_seconds: float = WINDOW_SECONDS, clock=time.monotonic):
        self.window = window_seconds
        self.clock = clock  # tests pass a fake clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()  # sync FastAPI dependencies run in worker threads
        self._last_sweep = clock()

    def hit(self, client: str, limit: int) -> int | None:
        """Count one request from `client`.

        Returns None when it is allowed, or the whole seconds to wait until
        the oldest request in the window expires when it is over the limit.
        Refused requests are not counted, so a client that waits gets in.
        """
        now = self.clock()
        with self._lock:
            self._sweep(now)
            hits = self._hits.setdefault(client, deque())
            while hits and hits[0] <= now - self.window:
                hits.popleft()
            if len(hits) >= limit:
                return max(1, math.ceil(hits[0] + self.window - now))
            hits.append(now)
            return None

    def _sweep(self, now: float) -> None:
        """Forget clients with no request in the last window, at most once per window,
        so memory doesn't grow with every IP address that ever called."""
        if now - self._last_sweep < self.window:
            return
        self._last_sweep = now
        for client in [c for c, hits in self._hits.items() if not hits or hits[-1] <= now - self.window]:
            del self._hits[client]

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()

    def tracked_clients(self) -> int:
        with self._lock:
            return len(self._hits)
